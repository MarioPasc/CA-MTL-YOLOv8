"""Bounding box label standardization utilities.

Convert CADICA pixel bbox files (x y w h label) -> YOLO normalized (class xc yc w h).
Normalization uses the original image size when available (preferred).
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Dict

from PIL import Image

from camtl_yolo.data.preprocess.modules.cadica import PROHIBITED_LABELS


def _parse_line(line: str) -> Optional[Tuple[int, int, int, int, str]]:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        x = int(parts[0]); y = int(parts[1]); w = int(parts[2]); h = int(parts[3]); label = parts[4]
    except ValueError:
        return None
    return x, y, w, h, label


def _find_corresponding_image(path: Path) -> Optional[Path]:
    """
    Given a label file path like .../detect/labels/<stem>.txt try to find the
    image at .../detect/images/<stem>.(png|jpg|jpeg|tif|tiff).
    Returns Path or None.
    """
    stem = path.stem
    # Expect labels dir structure: .../detect/labels
    labels_dir = path.parent
    detect_dir = labels_dir.parent if labels_dir.name == 'labels' else labels_dir
    images_dir = detect_dir / 'images'
    exts = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    if images_dir.exists() and images_dir.is_dir():
        for ext in exts:
            candidate = images_dir / (stem + ext)
            if candidate.exists():
                return candidate
        # try case-insensitive or any file with same stem
        for fp in images_dir.iterdir():
            if fp.is_file() and fp.stem == stem:
                return fp
    # fallback: look in same parent directories for a matching stem
    for parent in [path.parent, path.parent.parent, path.parent.parent.parent]:
        if parent is None:
            continue
        for ext in exts:
            candidate = parent / (stem + ext)
            if candidate.exists():
                return candidate
    return None


def _cadica_file_to_yolo(path: Path, target_w: int, target_h: int) -> Tuple[str, bool, bool, Optional[str]]:
    """Convert one CADICA label file to YOLO.
    Returns (file_path, success, became_empty, error)
    """
    try:
        if not path.is_file():
            return str(path), False, False, "not-a-file"

        # read raw CADICA lines
        with path.open('r') as f:
            lines = [l for l in (ln.strip() for ln in f) if l]

        # try to find original image to get original size
        orig_w = target_w
        orig_h = target_h
        img_path = _find_corresponding_image(path)
        if img_path:
            try:
                with Image.open(img_path) as im:
                    ow, oh = im.size
                    if ow > 0 and oh > 0:
                        orig_w, orig_h = int(ow), int(oh)
            except Exception:
                # if image cannot be opened, fallback to target
                pass

        yolo_rows: List[str] = []
        for ln in lines:
            parsed = _parse_line(ln)
            if not parsed:
                continue
            x, y, w, h, label = parsed
            if label in PROHIBITED_LABELS:
                continue

            # Normalize using original image size (preferred).
            # xc, yc are center coords normalized to original. This equals scaling then normalizing to target.
            xc = (x + w / 2.0) / float(orig_w)
            yc = (y + h / 2.0) / float(orig_h)
            nw = w / float(orig_w)
            nh = h / float(orig_h)

            # Clamp to [0,1]
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            yolo_rows.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        became_empty = len(yolo_rows) == 0
        # Overwrite the label file with YOLO lines (empty file if no objects)
        with path.open('w') as f:
            if yolo_rows:
                f.write("\n".join(yolo_rows) + "\n")
            else:
                # truncate to zero length (empty file)
                pass

        return str(path), True, became_empty, None

    except Exception as e:  # pragma: no cover
        return str(path), False, False, str(e)


def convert_cadica_labels(label_paths: Iterable[str | Path], target_size: Tuple[int, int], logger=None, workers: int | None = None) -> Tuple[int, int, int]:
    """Batch convert CADICA style labels to YOLO format.

    Args:
        label_paths: iterable of .txt paths.
        target_size: (width,height) used for normalization if original image not found.
        logger: optional logger.
        workers: process pool size (None -> cpu count)
    Returns:
        (processed_ok, empty_files, failed)
    """
    paths = [Path(p) for p in label_paths]
    if logger:
        logger.info(f"[bbox] Converting {len(paths)} CADICA label files to YOLO")
    processed = 0
    emptied = 0
    failed = 0
    W, H = int(target_size[0]), int(target_size[1])
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_cadica_file_to_yolo, p, W, H): p for p in paths}
        for fut in as_completed(futs):
            fp, ok, became_empty, err = fut.result()
            if ok:
                processed += 1
                if became_empty:
                    emptied += 1
            else:
                failed += 1
                if logger:
                    logger.error(f"[bbox] Failed {fp}: {err}")
    if logger:
        logger.info(f"[bbox] CADICA conversion done. OK:{processed} Empty:{emptied} Failed:{failed}")
    return processed, emptied, failed


############################################################
# ARCADE CONVERSION                                        #
############################################################

def _load_arcade_json(json_path: Path) -> Dict:
    with json_path.open('r') as f:
        return json.load(f)


def _arcade_build_index(meta: Dict) -> Tuple[Dict[int, Dict], Dict[int, list]]:
    """Return (image_index, annotations_by_image_id).
    Expects COCO-like keys: images (list of {id,width,height,file_name,...}), annotations (list with image_id, bbox[XYWH]).
    Some provided JSON excerpt shows missing fields (truncated). We defensively handle absent fields.
    """
    image_index: Dict[int, Dict] = {}
    for img in meta.get('images', []):
        iid = img.get('id')
        if iid is None:
            continue
        image_index[int(iid)] = img
    ann_by_img: Dict[int, list] = {}
    for ann in meta.get('annotations', []):
        iid = ann.get('image_id')
        if iid is None:
            # Some schemas may embed image reference differently; skip
            continue
        ann_by_img.setdefault(int(iid), []).append(ann)
    return image_index, ann_by_img


def _infer_arcade_image_id(label_stem: str) -> Optional[int]:
    """Label files are ARCADE{split}_{num}.txt -> want {num} as int.
    Accepts stems like ARCADEtrain_12, ARCADEval_003, ARCADEtest_7
    """
    # Find last underscore and parse the tail as int
    if '_' not in label_stem:
        return None
    tail = label_stem.split('_')[-1]
    # Strip leading zeros
    try:
        return int(tail)
    except ValueError:
        return None


def _arcade_label_to_yolo(label_path: Path, image_index: Dict[int, Dict], ann_by_img: Dict[int, list]) -> Tuple[str, bool, bool, Optional[str]]:
    """Convert a single ARCADE label file (placeholder content) to YOLO by
    looking up its bbox(es) in the JSON. Currently: assume at most one stenosis
    annotation relevant (class 0). If multiple, we keep all as separate lines.

    Returns (file_path, success, became_empty, error)
    """
    try:
        stem = label_path.stem
        img_id = _infer_arcade_image_id(stem)
        if img_id is None:
            return str(label_path), False, False, "cannot-parse-image-id"
        img_meta = image_index.get(img_id)
        if not img_meta:
            return str(label_path), False, False, "image-id-not-in-json"
        w = img_meta.get('width') or img_meta.get('W') or img_meta.get('w')
        h = img_meta.get('height') or img_meta.get('H') or img_meta.get('h')
        if not w or not h:
            # Without dimensions cannot normalize
            return str(label_path), False, False, "missing-image-dimensions"
        w = int(w); h = int(h)
        anns = ann_by_img.get(img_id, [])
        # Filter to those with bbox
        yolo_lines: List[str] = []
        for ann in anns:
            bbox = ann.get('bbox')
            if not bbox or len(bbox) < 4:
                continue
            x, y, bw, bh = bbox[0], bbox[1], bbox[2], bbox[3]
            # Convert XYWH (top-left + width/height) to YOLO normalized center format
            xc = (x + bw / 2.0) / float(w)
            yc = (y + bh / 2.0) / float(h)
            nw = bw / float(w)
            nh = bh / float(h)
            # Clamp
            xc = max(0.0, min(1.0, xc)); yc = max(0.0, min(1.0, yc))
            nw = max(0.0, min(1.0, nw)); nh = max(0.0, min(1.0, nh))
            yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        became_empty = len(yolo_lines) == 0
        with label_path.open('w') as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines) + "\n")
            else:
                # leave empty file
                pass
        return str(label_path), True, became_empty, None
    except Exception as e:  # pragma: no cover
        return str(label_path), False, False, str(e)


def convert_arcade_labels(label_paths: Iterable[str | Path], json_path: str | Path, logger=None, workers: int | None = None) -> Tuple[int, int, int]:
    """Convert ARCADE placeholder label files to YOLO using a COCO-like JSON.

    Args:
        label_paths: iterable of ARCADE*.txt paths in a split labels dir.
        json_path: path to the annotations JSON file of that split.
        logger: optional logger
        workers: parallelism
    Returns:
        (processed_ok, empty_files, failed)
    """
    json_path = Path(json_path)
    if not json_path.exists():
        if logger:
            logger.error(f"[bbox] ARCADE JSON not found: {json_path}")
        return 0, 0, 0
    try:
        meta = _load_arcade_json(json_path)
    except Exception as e:
        if logger:
            logger.error(f"[bbox] Failed to parse ARCADE JSON {json_path}: {e}")
        return 0, 0, 0
    image_index, ann_by_img = _arcade_build_index(meta)
    paths = [Path(p) for p in label_paths]
    if logger:
        logger.info(f"[bbox] Converting {len(paths)} ARCADE label files using {json_path.name}")
    processed = 0
    emptied = 0
    failed = 0
    # Process sequentially to avoid pickling issues with large shared dicts / local closures.
    for p in paths:
        fp, ok, became_empty, err = _arcade_label_to_yolo(p, image_index, ann_by_img)
        if ok:
            processed += 1
            if became_empty:
                emptied += 1
        else:
            failed += 1
            if logger:
                logger.error(f"[bbox] ARCADE failed {fp}: {err}")
    if logger:
        logger.info(f"[bbox] ARCADE conversion done. OK:{processed} Empty:{emptied} Failed:{failed}")
    return processed, emptied, failed


__all__ = ["convert_cadica_labels", "convert_arcade_labels"]
