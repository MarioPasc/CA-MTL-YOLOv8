"""Preprocessing pipeline orchestrator with fast parallel resize and tqdm progress.

Implemented stages (in required order):
    1. Detection label conversion (CADICA + ARCADE -> YOLO txt)
    2. Image format standardization (convert images/masks to target extension)
    3. Spatial resizing (upsample / downsample images & masks to target size)  ← optimized
    4. Modality-specific enhancement (fundus, XCA) on already resized & formatted images

Design:
    - Reads the same data.yaml used by the generator.
    - Applies steps in-place on the standardized dataset root.
    - Format conversion and resize are parallelised with process pools.
    - Progress shown with tqdm for long loops.
    - Fundus/XCA stages iterate their modality folders directly.

Function:
    run_preprocessing(output_dir: str, config: dict, logger, num_workers: int | None = None) -> None

Key config (under preprocessing):
    detect_labels_to_yolo: bool
    img_format: target extension (e.g. 'png')
    img_size: {
        target: [W,H],
        upsampling_strategy: 'lanczos' | 'bicubic' | ...,
        mask_upsampling_strategy: 'nearest' | ...,
        mask_downsampling_strategy: 'nearest' | ...
    }
    preprocess_fundus: bool
    extract_fov: bool   # if false, skip fundus FOV extraction entirely
    preprocess_xca: one of 'none'|'segment'|'detect'|'both'

Notes:
    - Enhancement steps operate after resizing to avoid double interpolation.
    - Fundus stage adheres to: skip FOV on demand, no mask saving, stricter quarantine, and paired-mask removal.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Callable
import cv2  # type: ignore
cv2.setNumThreads(0)  # avoid thread oversubscription under multiprocessing

from camtl_yolo.data.preprocess.tools.img_format import convert_image
from camtl_yolo.data.preprocess.tools.bbox_converter import convert_cadica_labels, convert_arcade_labels
from camtl_yolo.data.preprocess.tools.upsample_images import UpscaleConfig, upscale_image
from camtl_yolo.data.preprocess.tools.downsample_images import DownsampleConfig, downsample_image
from camtl_yolo.data.preprocess.tools.upsample_masks import MaskUpscaleConfig, upscale_mask
from camtl_yolo.data.preprocess.tools.downsample_masks import MaskDownsampleConfig, downsample_mask
from camtl_yolo.data.preprocess.tools.preprocess_fundus import (
    preprocess as preprocess_fundus_fn,
    preprocess_with_mask,
    fov_mask_from_raw,
    fov_mask_fallback_strict,
    mask_metrics,
    Cfg as FundusCfg,
)
from camtl_yolo.data.preprocess.tools.preprocess_xca import preprocess_xca, CfgXCA
import numpy as np
from PIL import Image

from dataclasses import dataclass
import csv, shutil
from tqdm.auto import tqdm


# Dataset modality subpaths relative to output_dir
SEG_IMG = ["angiography", "segment", "images"]
SEG_LBL = ["angiography", "segment", "labels"]
DET_IMG = ["angiography", "detect", "images"]  # labels for detection are text, skip
RET_IMG = ["retinography", "images"]
RET_LBL = ["retinography", "labels"]

IMAGE_DIRS = [SEG_IMG, SEG_LBL, DET_IMG, RET_IMG, RET_LBL]


@dataclass(frozen=True)
class FOVQC:
    """Robust stats for FOV area QC."""
    median: float
    mad: float                # scaled MAD (σ_MAD = 1.4826 * median(|x - median|))
    threshold: float          # modified z-score cutoff, default 3.5


def _robust_stats(values: List[float]) -> FOVQC:
    """Compute robust statistics for a list of values."""
    vals = np.asarray(values, dtype=float)
    med = float(np.median(vals)) if vals.size else 0.0
    mad_raw = float(np.median(np.abs(vals - med))) if vals.size else 0.0
    mad = 1.4826 * mad_raw
    return FOVQC(median=med, mad=mad, threshold=3.5)


def _modified_z(x: float, qc: FOVQC) -> float:
    """Modified z-score using MAD."""
    if qc.mad <= 1e-9:
        return 0.0
    return 0.6745 * (x - qc.median) / qc.mad


def _write_qc_csv(rows: List[dict], out_csv: Path) -> None:
    """Write QC rows to CSV."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


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
            if fp.suffix.lower() == '.txt':
                continue
            paths.append(fp)
    return paths


def _prepare_mask_array(arr, strategy: str):
    """Ensure mask is 2D binary for skeleton-related strategies."""
    strat = (strategy or '').lower()
    if strat in ('skeleton_preserve', 'skeleton_upscale'):
        import numpy as _np
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        arr = (_np.asarray(arr) > 0).astype('uint8')
    return arr


def _is_mask(p: Path) -> bool:
    """Detect mask files by directory semantics."""
    parts = p.parts
    return 'labels' in parts and ('segment' in parts or 'retinography' in parts)


def _image_size(p: Path) -> Optional[Tuple[int, int]]:
    """Safely read image size without loading full data into Python objects."""
    try:
        with Image.open(p) as im:
            return im.size
    except Exception:
        return None


def _parallel_submit(
    executor: ProcessPoolExecutor,
    tasks: Iterable[tuple],
    func: Callable[..., tuple],
    desc: str,
    total: int,
) -> Tuple[int, int]:
    """Run tasks in parallel with tqdm. Each task returns (ok:bool, msg:str)."""
    successes = 0
    failures = 0
    futures = {executor.submit(func, *args): args for args in tasks}
    for fut in tqdm(as_completed(futures), total=total, desc=desc, unit="file"):
        try:
            ok, _ = fut.result()
            if ok:
                successes += 1
            else:
                failures += 1
        except Exception:
            failures += 1
    return successes, failures


# --- Worker functions for process pool -------------------------------------------------

def _wf_convert(img_path: str, target_fmt: str) -> tuple:
    """Worker: convert single image to target format."""
    ok, _, err = convert_image(img_path, target_fmt, True, True, None)
    return (ok, "" if ok else str(err))


def _wf_up_image(src: str, target_size: Tuple[int, int], strategy: str) -> tuple:
    """Worker: upscale single image using temp file then replace."""
    try:
        norm = 'lanczos' if strategy.lower().replace('res_', '') == 'lanczos' else strategy
        cfg = UpscaleConfig(target_size=target_size, strategy=norm)
        src_path = Path(src)
        tmp_dst = src_path.with_suffix('.tmp_upscale' + src_path.suffix)
        upscale_image(src_path, tmp_dst, cfg)
        tmp_dst.replace(src_path)
        return (True, "")
    except Exception as e:
        return (False, f"{src}: {e}")


def _wf_down_image(src: str, target_size: Tuple[int, int], strategy: str) -> tuple:
    """Worker: downscale single image using temp file then replace."""
    try:
        cfg = DownsampleConfig(target_size=target_size, strategy=strategy)
        src_path = Path(src)
        tmp_dst = src_path.with_suffix('.tmp_downscale' + src_path.suffix)
        downsample_image(src_path, tmp_dst, cfg)
        tmp_dst.replace(src_path)
        return (True, "")
    except Exception as e:
        return (False, f"{src}: {e}")


def _wf_up_mask(src: str, target_size: Tuple[int, int], strategy: str) -> tuple:
    """Worker: upscale single mask in-place."""
    try:
        src_path = Path(src)
        with Image.open(src_path) as im:
            arr = np.array(im)
            arr = _prepare_mask_array(arr, strategy)
            cfg = MaskUpscaleConfig(target_size=target_size, strategy=strategy)
            out = upscale_mask(arr, (im.size[0], im.size[1]), cfg)
            Image.fromarray(out).save(src_path)
        return (True, "")
    except Exception as e:
        return (False, f"{src}: {e}")


def _wf_down_mask(src: str, target_size: Tuple[int, int], strategy: str) -> tuple:
    """Worker: downscale single mask in-place."""
    try:
        src_path = Path(src)
        with Image.open(src_path) as im:
            arr = np.array(im)
            arr = _prepare_mask_array(arr, strategy)
            cfg = MaskDownsampleConfig(target_size=target_size, method=strategy)
            out = downsample_mask(arr, (im.size[0], im.size[1]), cfg)  # type: ignore
            Image.fromarray(out).save(src_path)
        return (True, "")
    except Exception as e:
        return (False, f"{src}: {e}")


# --- Main entry -----------------------------------------------------------------------

def run_preprocessing(output_dir: str, config: Dict, logger, num_workers: int | None = None) -> None:
    """Run the full preprocessing pipeline."""
    prep_cfg = (config or {}).get('preprocessing', {})
    out_root = Path(output_dir)
    if not out_root.exists():
        logger.warning(f"[pipeline] Output directory does not exist: {out_root}")
        return

    if num_workers is None:
        # Cap workers to avoid oversubscription in heavy I/O environments
        num_workers = max(1, (os.cpu_count() or 4) - 1)

    logger.info(f"[pipeline] Starting preprocessing pipeline with {num_workers} workers")

    # ---------- Step 1: Detection label conversion (CADICA + ARCADE) ----------
    if prep_cfg.get('detect_labels_to_yolo', False):
        detect_lbl_dir = out_root / 'angiography' / 'detect' / 'labels'
        if detect_lbl_dir.exists():
            # CADICA
            cadica_label_files = [p for p in detect_lbl_dir.glob('CADICAp*.txt')]
            if cadica_label_files:
                img_size = prep_cfg.get('img_size', None)
                img_size = img_size.get('target', [512, 512]) if isinstance(img_size, dict) else img_size
                if not (isinstance(img_size, (list, tuple)) and len(img_size) == 2):
                    logger.warning('[pipeline] Invalid img_size in config; using (640,640)')
                    img_size = (640, 640)
                else:
                    img_size = (int(img_size[0]), int(img_size[1]))
                logger.info(f"[pipeline] Converting CADICA labels to YOLO using target size {img_size}")
                convert_cadica_labels(cadica_label_files, img_size, logger=logger, workers=num_workers)
            else:
                logger.info('[pipeline] No CADICA label files found for conversion.')

            # ARCADE
            arcade_label_files = [p for p in detect_lbl_dir.glob('ARCADE*.txt')]
            if arcade_label_files:
                from typing import Dict as _Dict, List as _List
                groups: _Dict[str, _List[Path]] = {'train': [], 'test': [], 'val': []}
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
                        ann_dir = arcade_base / split / 'annotations'
                        if not ann_dir.exists():
                            logger.warning(f"[pipeline] ARCADE annotations dir missing for split {split}: {ann_dir}")
                            continue
                        candidate = ann_dir / f"{split}.json"
                        if not candidate.exists():
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

    # Collect all files once
    all_images = _gather_all_images(out_root)
    if not all_images:
        logger.info("[pipeline] No image files found; skipping resize + format stages.")
        logger.info("[pipeline] Preprocessing pipeline finished")
        return

    # ---------- Step 2: Image format conversion (to target extension) ----------
    target_fmt = prep_cfg.get('img_format', 'png')
    if not target_fmt.startswith('.'):
        target_fmt = '.' + target_fmt
    logger.info(f"[pipeline] Target image format: {target_fmt}")

    to_convert: List[Path] = [p for p in all_images if p.suffix.lower() != target_fmt.lower()]
    logger.info(f"[pipeline] Found {len(all_images)} total images; {len(to_convert)} require format conversion ({target_fmt})")

    if to_convert:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            ok, fail = _parallel_submit(
                ex,
                ((str(p), target_fmt) for p in to_convert),
                _wf_convert,
                desc="Format conversion",
                total=len(to_convert),
            )
        logger.info(f"[pipeline] Format conversion done. Converted: {ok} Failed: {fail}")
        # Refresh list after conversions
        all_images = _gather_all_images(out_root)
    else:
        logger.info("[pipeline] All images already in target format; skipping conversion stage.")

    # ---------- Step 3: Spatial resizing (upsample or downsample) ----------
    img_size_cfg = prep_cfg.get('img_size')
    target_size: Optional[Tuple[int, int]] = None
    if isinstance(img_size_cfg, dict) and 'target' in img_size_cfg:
        ts = img_size_cfg.get('target')
        if isinstance(ts, (list, tuple)) and len(ts) == 2:
            try:
                target_size = (int(ts[0]), int(ts[1]))
            except Exception:
                target_size = None
    elif isinstance(img_size_cfg, (list, tuple)) and len(img_size_cfg) == 2:
        try:
            target_size = (int(img_size_cfg[0]), int(img_size_cfg[1]))
        except Exception:
            target_size = None

    logger.info(f"[pipeline] [resize] Target image size: {target_size}")

    if target_size:
        tw, th = target_size
        up_img: List[Path] = []
        dn_img: List[Path] = []
        up_msk: List[Path] = []
        dn_msk: List[Path] = []

        # Pre-scan sizes with tqdm for transparency
        for p in tqdm(all_images, desc="Scanning sizes", unit="file"):
            sz = _image_size(p)
            if sz is None:
                logger.warning(f"[pipeline] Could not read size for {p}; skipping resize")
                continue
            w, h = sz
            if (w, h) == (tw, th):
                continue
            if _is_mask(p):
                if w < tw or h < th:
                    up_msk.append(p)
                else:
                    dn_msk.append(p)
            else:
                if w < tw or h < th:
                    up_img.append(p)
                else:
                    dn_img.append(p)

        logger.info(f"[pipeline] [Resize] target={target_size}; up_img {len(up_img)} | dn_img {len(dn_img)} | up_mask {len(up_msk)} | dn_mask {len(dn_msk)}")

        # Strategies
        up_strategy: str = 'lanczos'
        down_strategy: str = 'progressive'
        mask_up_strategy: str = 'nearest'
        mask_down_strategy: str = 'nearest'
        if isinstance(img_size_cfg, dict):
            up_strategy = str(img_size_cfg.get('upsampling_strategy', 'lanczos')).lower()
            down_strategy = 'progressive'  # fixed by design
            mask_down_strategy = str(img_size_cfg.get('mask_downsampling_strategy', 'nearest'))
            mask_up_strategy = str(img_size_cfg.get('mask_upsampling_strategy', 'nearest'))

        # Execute in parallel pools with tqdm
        if up_img or dn_img or up_msk or dn_msk:
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                if up_img:
                    ok, fail = _parallel_submit(
                        ex,
                        ((str(p), target_size, up_strategy) for p in up_img),
                        _wf_up_image,
                        desc="Upscaling images",
                        total=len(up_img),
                    )
                    logger.info(f"[pipeline] Upscaled images: {ok} Failed: {fail}")

                if dn_img:
                    ok, fail = _parallel_submit(
                        ex,
                        ((str(p), target_size, down_strategy) for p in dn_img),
                        _wf_down_image,
                        desc="Downscaling images",
                        total=len(dn_img),
                    )
                    logger.info(f"[pipeline] Downscaled images: {ok} Failed: {fail}")

                if up_msk:
                    ok, fail = _parallel_submit(
                        ex,
                        ((str(p), target_size, mask_up_strategy) for p in up_msk),
                        _wf_up_mask,
                        desc="Upscaling masks",
                        total=len(up_msk),
                    )
                    logger.info(f"[pipeline] Upscaled masks: {ok} Failed: {fail}")

                if dn_msk:
                    ok, fail = _parallel_submit(
                        ex,
                        ((str(p), target_size, mask_down_strategy) for p in dn_msk),
                        _wf_down_mask,
                        desc="Downscaling masks",
                        total=len(dn_msk),
                    )
                    logger.info(f"[pipeline] Downscaled masks: {ok} Failed: {fail}")
        else:
            logger.info("[pipeline] All files already at target size; skipping resize work.")
    else:
        logger.info('[pipeline] No valid target img_size specified; skipping resize stage.')

    # ---------- Step 4: Modality-specific preprocessing (fundus, XCA) ----------
    preprocessing_subconfig = prep_cfg.get('preprocessing', prep_cfg)
    if preprocessing_subconfig.get('fundus', False):
        if preprocessing_subconfig.get('extract_fov', True) is False:
            logger.info('[pipeline] Skipping fundus stage (extract_fov=false). Images left untouched.')
        else:
            fundus_dir  = out_root.joinpath(*RET_IMG)
            labels_dir  = out_root.joinpath(*RET_LBL)
            if fundus_dir.exists():
                logger.info('[pipeline] Fundus preprocessing with dataset-aware QC')
                exts = {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'}
                fcfg = FundusCfg()
                # ---- First pass: compute FOV + metrics only (no writes) ----
                rec = []
                for p in tqdm(list(fundus_dir.rglob('*')), desc="Fundus FOV pass", unit="file"):
                    if not p.is_file() or p.suffix.lower() not in exts:
                        continue
                    raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                    if raw is None:
                        logger.warning(f"[pipeline] Fundus read failed {p}")
                        continue
                    try:
                        fov = fov_mask_from_raw(raw)
                        a, c, h = mask_metrics(fov)
                        rec.append({'path': p, 'a': a, 'c': c, 'h': h})
                    except Exception as e:
                        logger.error(f"[pipeline] FOV compute failed {p}: {e}")
                if rec:
                    vals = np.array([r['a'] for r in rec], dtype=float)
                    med = float(np.median(vals)); mad = float(np.median(np.abs(vals - med)))
                    sigma = 1.4826*mad if mad>0 else 0.0
                    def mz(x: float) -> float: return 0.0 if sigma==0.0 else 0.6745*(x - med)/sigma
                    ABS_MIN, ABS_MAX = 0.50, 0.86
                    COMP_MIN, HOLE_MAX, MZ_MAX = 0.83, 0.005, 2.5
                    kept = quarantined = 0
                    qdir = out_root / 'retinography' / '_qc_excluded'
                    qdir.mkdir(parents=True, exist_ok=True)
                    for r in tqdm(rec, desc="Fundus apply+QC", unit="img"):
                        ok = (ABS_MIN <= r['a'] <= ABS_MAX) and (abs(mz(r['a'])) <= MZ_MAX) and (r['c'] >= COMP_MIN) and (r['h'] <= HOLE_MAX)
                        if not ok:
                            raw = cv2.imread(str(r['path']), cv2.IMREAD_UNCHANGED)
                            if raw is not None:
                                f2 = fov_mask_fallback_strict(raw)
                                a2, c2, h2 = mask_metrics(f2)
                                ok = (ABS_MIN <= a2 <= ABS_MAX) and (abs(mz(a2)) <= MZ_MAX) and (c2 >= COMP_MIN) and (h2 <= HOLE_MAX)
                        if ok:
                            raw = cv2.imread(str(r['path']), cv2.IMREAD_UNCHANGED)
                            if raw is not None:
                                out, _ = preprocess_with_mask(raw, fcfg, logger)
                                cv2.imwrite(str(r['path']), out)
                                kept += 1
                        else:
                            try:
                                (qdir / r['path'].name).write_bytes(r['path'].read_bytes())
                                r['path'].unlink()
                            except Exception:
                                pass
                            if labels_dir.exists():
                                stem = r['path'].stem
                                for m in labels_dir.glob(stem + '.*'):
                                    try:
                                        (qdir / m.name).write_bytes(m.read_bytes())
                                        m.unlink()
                                    except Exception:
                                        pass
                            quarantined += 1
                    logger.info(f"[pipeline] Fundus kept={kept} quarantined={quarantined} (median={med:.4f} MAD={mad:.4f})")
            else:
                logger.info('[pipeline] Retinography images directory missing; skipping fundus preprocessing.')

    # XCA preprocessing (angiography)
    xca_mode = preprocessing_subconfig.get('xca', 'none')
    if xca_mode and xca_mode.lower() != 'none':
        targets = []
        if xca_mode in ('segment','both'):
            targets.append(out_root.joinpath(*SEG_IMG))
        if xca_mode in ('detect','both'):
            targets.append(out_root.joinpath(*DET_IMG))
        xcfg = CfgXCA()
        for tdir in targets:
            if not tdir.exists():
                logger.info(f"[pipeline] XCA target dir missing ({tdir}); skipping")
                continue
            logger.info(f"[pipeline] Applying XCA preprocessing to {tdir}")
            exts = {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'}
            for img_path in tqdm(list(tdir.rglob('*')), desc=f"XCA preprocess {tdir.name}", unit="file"):
                if not img_path.is_file() or img_path.suffix.lower() not in exts:
                    continue
                try:
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

    logger.info("[pipeline] Preprocessing pipeline finished")


__all__ = ["run_preprocessing"]
