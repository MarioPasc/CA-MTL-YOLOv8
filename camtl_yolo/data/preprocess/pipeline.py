"""Preprocessing pipeline orchestrator.

Currently implemented stage(s):
  1. Image format standardization (convert all images to configured format)

Design:
  - Pipeline reads the same data.yaml used by generator.
  - Accepts already-standardized output directory root and applies steps in-place.
  - Parallelization: each image is processed in a process pool using the
    low-level `convert_image` utility. Masks (segmentation/retinography labels)
    are treated as images for conversion purposes.

Function:
    run_preprocessing(output_dir: str, config: dict, logger) -> None

Config keys used (under preprocessing):
    img_format: target extension (e.g., 'png')
    to_gray: (reserved for future) if True convert to grayscale BEFORE save
    img_size: (reserved for future) (w,h) resize

Notes:
  - Only the image-format step is active now; placeholders for future steps.
  - The function is idempotent: images already in target format are re-saved
    only if overwrite=True (to normalize modes). This is currently True.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from camtl_yolo.data.preprocess.tools.img_format import convert_image
from camtl_yolo.data.preprocess.tools.bbox_converter import convert_cadica_labels, convert_arcade_labels

# Dataset modality subpaths relative to output_dir
SEG_IMG = ["angiographies", "segment", "images"]
SEG_LBL = ["angiographies", "segment", "labels"]
DET_IMG = ["angiographies", "detect", "images"]  # labels for detection are text, skip
RET_IMG = ["retinography", "images"]
RET_LBL = ["retinography", "labels"]

IMAGE_DIRS = [SEG_IMG, SEG_LBL, DET_IMG, RET_IMG, RET_LBL]

def _gather_all_images(root: Path) -> List[Path]:
    paths: List[Path] = []
    for parts in IMAGE_DIRS:
        d = root.joinpath(*parts)
        if not d.exists():
            continue
        for fp in d.rglob('*'):
            if fp.is_file():
                # Skip detection label txt
                if fp.suffix.lower() == '.txt':
                    continue
                paths.append(fp)
    return paths

def _chunk(iterable, n):
    iterable = list(iterable)
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def run_preprocessing(output_dir: str, config: Dict, logger, num_workers: int | None = None) -> None:
    prep_cfg = (config or {}).get('preprocessing', {})
    target_fmt = prep_cfg.get('img_format', 'png')
    if not target_fmt.startswith('.'):
        target_fmt = '.' + target_fmt

    out_root = Path(output_dir)
    if not out_root.exists():
        logger.warning(f"[pipeline] Output directory does not exist: {out_root}")
        return

    logger.info("[pipeline] Starting preprocessing pipeline")
    logger.info(f"[pipeline] Target image format: {target_fmt}")
    # ---------- Step 1: Detection label conversion (CADICA + ARCADE) ----------
    if prep_cfg.get('detect_labels_to_yolo', False):
        detect_lbl_dir = out_root / 'angiographies' / 'detect' / 'labels'
        if detect_lbl_dir.exists():
            # CADICA
            cadica_label_files = [p for p in detect_lbl_dir.glob('CADICAp*.txt')]
            if cadica_label_files:
                img_size = prep_cfg.get('img_size', [640,640])
                if not (isinstance(img_size, (list, tuple)) and len(img_size)==2):
                    logger.warning('[pipeline] Invalid img_size in config; using (640,640)')
                    img_size = (640,640)
                else:
                    img_size = (int(img_size[0]), int(img_size[1]))
                logger.info(f"[pipeline] Converting CADICA labels to YOLO using target size {img_size}")
                convert_cadica_labels(cadica_label_files, img_size, logger=logger, workers=num_workers)
            else:
                logger.info('[pipeline] No CADICA label files found for conversion.')

            # ARCADE: Need original JSONs in raw root. We try to locate them under config root/angiography/detect/ARCADE/{split}/annotations/*.json
            # Label file pattern: ARCADE{split}_{num}.txt
            arcade_label_files = [p for p in detect_lbl_dir.glob('ARCADE*.txt')]
            if arcade_label_files:
                # group by split based on stem prefix, expecting ARCADEtrain_, ARCADEtest_, ARCADEval_
                from typing import Dict, List as _List
                groups: Dict[str, _List[Path]] = {'train': [], 'test': [], 'val': []}
                for p in arcade_label_files:
                    stem = p.stem.lower()
                    if stem.startswith('arcadetrain_'):
                        groups['train'].append(p)
                    elif stem.startswith('arcadetest_'):
                        groups['test'].append(p)
                    elif stem.startswith('arcadeval_'):
                        groups['val'].append(p)
                raw_root = Path(config.get('root', ''))
                if not raw_root.exists():
                    logger.warning(f"[pipeline] Raw root from config does not exist; cannot perform ARCADE conversion: {raw_root}")
                else:
                    arcade_base = raw_root / 'angiography' / 'detect' / 'ARCADE'
                    for split, files in groups.items():
                        if not files:
                            continue
                        # Try typical json filename patterns within annotations dir
                        ann_dir = arcade_base / split / 'annotations'
                        if not ann_dir.exists():
                            logger.warning(f"[pipeline] ARCADE annotations dir missing for split {split}: {ann_dir}")
                            continue
                        # Heuristics: prefer test.json/train.json/val.json else any *.json
                        candidate = ann_dir / f"{split}.json"
                        if not candidate.exists():
                            # fallback: first *.json
                            json_candidates = list(ann_dir.glob('*.json'))
                            if not json_candidates:
                                logger.warning(f"[pipeline] No JSON file found in {ann_dir} for ARCADE {split}")
                                continue
                            candidate = json_candidates[0]
                        logger.info(f"[pipeline] Converting ARCADE {split} labels ({len(files)}) using {candidate.name}")
                        convert_arcade_labels(files, candidate, logger=logger, workers=num_workers)
            else:
                logger.info('[pipeline] No ARCADE label files found for conversion.')
        else:
            logger.info('[pipeline] Detection labels directory missing; skipping label conversion.')

    all_images = _gather_all_images(out_root)
    logger.info(f"[pipeline] Collected {len(all_images)} files for potential format conversion")
    if not all_images:
        logger.info("[pipeline] Nothing to process.")
        return

    # ---------- Step 2: Image format conversion ----------
    # Parallel convert
    successes = 0
    failures = 0
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(convert_image, str(img), target_fmt, True, True, None): img for img in all_images}
        for fut in as_completed(futures):
            img = futures[fut]
            try:
                ok, new_path, err = fut.result()
            except Exception as e:  # pragma: no cover
                failures += 1
                logger.error(f"[pipeline] Unexpected failure {img}: {e}")
                continue
            if ok:
                successes += 1
            else:
                failures += 1
                logger.error(f"[pipeline] Conversion failed {img}: {err}")

    logger.info(f"[pipeline] Format conversion done. Success: {successes} Failed: {failures}")
    logger.info("[pipeline] Preprocessing pipeline finished")

__all__ = ["run_preprocessing"]

