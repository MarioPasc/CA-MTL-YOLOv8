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
from camtl_yolo.data.preprocess.tools.upsample_images import UpscaleConfig, upscale_image
from camtl_yolo.data.preprocess.tools.downsample_images import DownsampleConfig, downsample_image
from camtl_yolo.data.preprocess.tools.upsample_masks import MaskUpscaleConfig, upscale_mask
from camtl_yolo.data.preprocess.tools.downsample_masks import MaskDownsampleConfig, downsample_mask
from camtl_yolo.data.preprocess.tools.preprocess_fundus import preprocess as preprocess_fundus_fn, Cfg as FundusCfg
from camtl_yolo.data.preprocess.tools.preprocess_xca import preprocess_xca, CfgXCA
import numpy as np
from PIL import Image

# Dataset modality subpaths relative to output_dir
SEG_IMG = ["angiography", "segment", "images"]
SEG_LBL = ["angiography", "segment", "labels"]
DET_IMG = ["angiography", "detect", "images"]  # labels for detection are text, skip
RET_IMG = ["retinography", "images"]
RET_LBL = ["retinography", "labels"]

IMAGE_DIRS = [SEG_IMG, SEG_LBL, DET_IMG, RET_IMG, RET_LBL]

def _gather_all_images(root: Path) -> List[Path]:
    """Return all image-like files (including masks) regardless of format."""
    paths: List[Path] = []
    for parts in IMAGE_DIRS:
        d = root.joinpath(*parts)
        if not d.exists():
            continue
        for fp in d.rglob('*'):
            if not fp.is_file():
                continue
            if fp.suffix.lower() == '.txt':  # skip detection label text files
                continue
            paths.append(fp)
    return paths

def _prepare_mask_array(arr, strategy: str):
    """Ensure mask is 2D binary when required by skeleton-related strategies.
    Strategies 'skeleton_preserve' (down) and 'skeleton_upscale' (up) expect a 2D binary mask.
    If multi-channel, convert via mean across channels; then binarize >0.
    """
    strat = (strategy or '').lower()
    if strat in ('skeleton_preserve', 'skeleton_upscale'):
        import numpy as _np
        if arr.ndim == 3:
            # average across channels to get intensity map
            arr = arr.mean(axis=2)
        # Binarize: any positive value -> 1
        arr = (_np.asarray(arr) > 0).astype('uint8')
    return arr

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
        detect_lbl_dir = out_root / 'angiography' / 'detect' / 'labels'
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
    if not all_images:
        logger.info("[pipeline] No image files found; skipping resize + format stages.")
        logger.info("[pipeline] Preprocessing pipeline finished")
        return

    # ---------- Step 2: Modality-specific preprocessing (fundus, XCA) BEFORE resize ----------
    # Fundus preprocessing (retinography images) if enabled
    if prep_cfg.get('preprocess_fundus', False):
        fundus_dir = out_root.joinpath(*RET_IMG)
        if fundus_dir.exists():
            logger.info('[pipeline] Applying fundus preprocessing to retinography images')
            # Iterate PNG/JPG/TIF etc
            fundus_exts = {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'}
            fcfg = FundusCfg()  # defaults; can later expose parameters
            for img_path in fundus_dir.rglob('*'):
                if not img_path.is_file() or img_path.suffix.lower() not in fundus_exts:
                    continue
                try:
                    # read and process
                    import cv2, numpy as _np
                    raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                    if raw is None:
                        logger.warning(f"[pipeline] Fundus read failed {img_path}")
                        continue
                    out = preprocess_fundus_fn(raw, fcfg, logger)
                    cv2.imwrite(str(img_path), out)
                except Exception as e:
                    logger.error(f"[pipeline] Fundus preprocess failed {img_path}: {e}")
        else:
            logger.info('[pipeline] Retinography images directory missing; skipping fundus preprocessing.')

    # XCA preprocessing (angiography) depending on config: none|segment|detect|both
    xca_mode = prep_cfg.get('preprocess_xca', 'none')
    if xca_mode and xca_mode.lower() != 'none':
        targets = []
        if xca_mode in ('segment','both'):
            targets.append(out_root.joinpath(*SEG_IMG))
        if xca_mode in ('detect','both'):
            targets.append(out_root.joinpath(*DET_IMG))
        xcfg = CfgXCA()  # defaults; could be extended via config later
        for tdir in targets:
            if not tdir.exists():
                logger.info(f"[pipeline] XCA target dir missing ({tdir}); skipping")
                continue
            logger.info(f"[pipeline] Applying XCA preprocessing to {tdir}")
            exts = {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'}
            for img_path in tdir.rglob('*'):
                if not img_path.is_file() or img_path.suffix.lower() not in exts:
                    continue
                try:
                    import cv2, numpy as _np
                    g = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                    if g is None:
                        logger.warning(f"[pipeline] XCA read failed {img_path}")
                        continue
                    if g.ndim == 3:
                        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
                    out = preprocess_xca(g, xcfg)
                    cv2.imwrite(str(img_path), out)
                except Exception as e:
                    logger.error(f"[pipeline] XCA preprocess failed {img_path}: {e}")

    # ---------- Step 3: Resize (upsample or downsample) BEFORE format conversion ----------
    # Config structure (new): img_size: { target: [W,H], upsampling_strategy: str, mask_upsampling_strategy: str, mask_downsampling_strategy: str }
    img_size_cfg = prep_cfg.get('img_size')
    target_size = None
    if isinstance(img_size_cfg, dict) and 'target' in img_size_cfg:
        ts = img_size_cfg.get('target')
        if isinstance(ts, (list, tuple)) and len(ts) == 2:
            try:
                target_size = (int(ts[0]), int(ts[1]))
            except Exception:
                target_size = None
    elif isinstance(img_size_cfg, (list, tuple)) and len(img_size_cfg) == 2:  # backward compatibility
        try:
            target_size = (int(img_size_cfg[0]), int(img_size_cfg[1]))
        except Exception:
            target_size = None

    if target_size:
        tw, th = target_size
        upsample_queue: List[Path] = []
        downsample_queue: List[Path] = []
        # Determine if a path is mask (segmentation or retinography labels) by directory part 'labels' under segment or retinography
        def is_mask(p: Path) -> bool:
            parts = p.parts
            return 'labels' in parts and ('segment' in parts or 'retinography' in parts)

        for p in all_images:
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:
                logger.warning(f"[pipeline] Could not read size for {p}; skipping resize")
                continue
            if (w, h) == (tw, th):
                continue
            if w < tw or h < th:
                upsample_queue.append(p)
            else:
                downsample_queue.append(p)

        logger.info(f"[pipeline] Resize target={target_size}; upsample {len(upsample_queue)} | downsample {len(downsample_queue)}")

        # Strategies
        up_strategy: str = 'lanczos'
        down_strategy: str = 'progressive'
        mask_up_strategy: str = 'nearest'
        mask_down_strategy: str = 'nearest'
        if isinstance(img_size_cfg, dict):
            up_strategy = str(img_size_cfg.get('upsampling_strategy', 'lanczos')).lower()
            down_strategy = 'progressive'  # fixed as per comment
            mask_down_strategy = str(img_size_cfg.get('mask_downsampling_strategy', 'nearest'))
            mask_up_strategy = str(img_size_cfg.get('mask_upsampling_strategy', 'nearest'))

        # Process upsample queue
        for src in upsample_queue:
            try:
                if is_mask(src):
                    # load as numpy
                    with Image.open(src) as im:
                        arr = np.array(im)
                        arr = _prepare_mask_array(arr, mask_up_strategy)
                        mu_cfg = MaskUpscaleConfig(target_size=target_size, strategy=mask_up_strategy)
                        out_arr = upscale_mask(arr, (im.size[0], im.size[1]), mu_cfg)
                        # save back (preserve single-channel if possible)
                        out_im = Image.fromarray(out_arr)
                        out_im.save(src)
                else:
                    # image upscale
                    # Normalize possible enum-like name 'RES_LANCZOS'
                    norm_up = 'lanczos' if up_strategy.lower().replace('res_', '') == 'lanczos' else up_strategy
                    up_cfg = UpscaleConfig(target_size=target_size, strategy=norm_up)
                    # Upscale writes to destination; use temp then replace
                    tmp_dst = src.with_suffix('.tmp_upscale' + src.suffix)
                    upscale_image(src, tmp_dst, up_cfg)
                    # replace original
                    tmp_dst.replace(src)
            except Exception as e:
                logger.error(f"[pipeline] Upscale failed for {src}: {e}")

        # Process downsample queue
        for src in downsample_queue:
            try:
                if is_mask(src):
                    with Image.open(src) as im:
                        arr = np.array(im)
                        arr = _prepare_mask_array(arr, mask_down_strategy)
                        md_cfg = MaskDownsampleConfig(target_size=target_size, method=mask_down_strategy)
                        out_arr = downsample_mask(arr, (im.size[0], im.size[1]), md_cfg)  # type: ignore
                        out_im = Image.fromarray(out_arr)
                        out_im.save(src)
                else:
                    down_cfg = DownsampleConfig(target_size=target_size, strategy=down_strategy)
                    tmp_dst = src.with_suffix('.tmp_downscale' + src.suffix)
                    downsample_image(src, tmp_dst, down_cfg)
                    tmp_dst.replace(src)
            except Exception as e:
                logger.error(f"[pipeline] Downscale failed for {src}: {e}")
    else:
        logger.info('[pipeline] No valid target img_size specified; skipping resize stage.')

    # Filter for images not already in target format (after potential resize modifications)
    to_convert: List[Path] = [p for p in all_images if p.suffix.lower() != target_fmt.lower()]
    logger.info(f"[pipeline] Found {len(all_images)} total images; {len(to_convert)} require format conversion ({target_fmt})")
    if not to_convert:
        logger.info("[pipeline] All images already in target format; skipping conversion stage.")
        logger.info("[pipeline] Preprocessing pipeline finished")
        return

    successes = 0
    failures = 0
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(convert_image, str(img), target_fmt, True, True, None): img for img in to_convert}
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

    logger.info(f"[pipeline] Format conversion done. Converted: {successes} Failed: {failures}")
    logger.info("[pipeline] Preprocessing pipeline finished")

__all__ = ["run_preprocessing"]

