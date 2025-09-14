"""Unified data standardization generator (optimized, parallel, IO‑efficient).

Key speedups:
- Avoid Pillow re‑encoding when source is already PNG: hardlink/symlink/copy fast path.
- When conversion is required (non‑PNG → PNG), use Pillow once with low compression.
- Parallelize all file operations with ThreadPoolExecutor (IO‑bound speedup).
- Zero redundant imports in inner loops. Single Image import.
- Atomic writes and skip if exists to support re‑runs.
- Buffered label copies via shutil.copyfile.

Target standardized structure (under output_dir):

output_dir/
    angiography/
        segment/
            images/
            labels/
        detect/
            images/
            labels/
    retinography/
        images/
        labels/

Config (data.yaml) expected keys:
    root: base path containing subfolders:
        angiography/
            detect/<DATASET_NAME>
            segment/<DATASET_NAME>
        retinography/<DATASET_NAME>
    detect_datasets: list[str]
    segment_datasets: list[str]
    retinography_datasets: list[str]
    output_dir: destination root

Each dataset module must expose the appropriate collector function:
    segmentation: collect_image_mask_pairs(input_dir) -> List[(img, mask)]
    detection: collect_image_label_pairs(input_dir) -> List[(img, label, ...)]
    retinography: collect_image_mask_pairs(input_dir) (same signature as segmentation)
"""
from __future__ import annotations

import os
import sys
import io
import shutil
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple, Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from tqdm import tqdm

# Single import of Pillow. We keep it optional at runtime.
try:
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception as _:
    Image = None
    ImageFile = None


def main() -> None:
    # NOTE: This sys.path logic is only needed for running scripts directly.
    PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from camtl_yolo.utils.logger import get_logger
    from camtl_yolo.data.preprocess.modules import dca1, fs_cad, xcav, chasedb1, drive, fives
    from camtl_yolo.data.preprocess.modules import arcade_stenosis, cadica
    from camtl_yolo.data.preprocess.pipeline import run_preprocessing

    logger = get_logger("generator_optimized")

    # ---------------- Config ----------------
    def load_config(config_path: str) -> Dict[str, Any]:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    config_path = Path(__file__).parent / "config" / "data.yaml"
    config = load_config(str(config_path))

    root = config.get("root")
    output_dir = config.get("output_dir")
    segment_datasets: List[str] = config.get("segment_datasets", [])
    detect_datasets: List[str] = config.get("detect_datasets", [])
    retino_datasets: List[str] = config.get("retinography_datasets", [])

    # Tuning knobs
    max_workers = int(config.get("io_max_workers", max(8, (os.cpu_count() or 2) * 4)))
    png_compress_level = int(config.get("png_compress_level", 1))  # 0..9, 1 is fast
    allow_hardlinks = bool(config.get("allow_hardlinks", True))
    allow_symlinks = bool(config.get("allow_symlinks", False))  # off by default
    skip_if_exists = bool(config.get("skip_if_exists", True))

    if not root or not output_dir:
        logger.error("Config must include 'root' and 'output_dir'. Exiting.")
        return

    # ---------------- Target tree ----------------
    angiography_segment_img = Path(output_dir) / "angiography" / "segment" / "images"
    angiography_segment_lbl = Path(output_dir) / "angiography" / "segment" / "labels"
    angiography_detect_img = Path(output_dir) / "angiography" / "detect" / "images"
    angiography_detect_lbl = Path(output_dir) / "angiography" / "detect" / "labels"
    retino_img_dir = Path(output_dir) / "retinography" / "images"
    retino_lbl_dir = Path(output_dir) / "retinography" / "labels"
    for p in [angiography_segment_img, angiography_segment_lbl,
              angiography_detect_img, angiography_detect_lbl,
              retino_img_dir, retino_lbl_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # ---------------- Helpers ----------------
    def _resolve_path(category: str, subcat: str | None, dataset: str) -> str:
        """Return absolute path for a dataset, trying new layout first then old."""
        if category == "retinography":
            return os.path.join(root, "retinography", dataset)
        assert subcat in {"segment", "detect"}
        new_path = os.path.join(root, "angiography", subcat, dataset)
        if os.path.isdir(new_path):
            return new_path
        # fallback old path
        old_path = os.path.join(root, subcat, dataset)
        return old_path

    # Optimized image writer: link/copy if already PNG, otherwise convert once.
    def _fast_export_to_png(src: str, dst_png: Path) -> None:
        if skip_if_exists and dst_png.exists():
            return

        src_path = Path(src)
        # Ensure parent exists for atomic replace pattern
        dst_png.parent.mkdir(parents=True, exist_ok=True)

        if src_path.suffix.lower() == ".png":
            # Try hardlink → symlink → copy2
            try:
                if allow_hardlinks:
                    os.link(src_path, dst_png)
                    return
            except Exception:
                pass
            try:
                if allow_symlinks:
                    os.symlink(src_path, dst_png)
                    return
            except Exception:
                pass
            # copy (fast, metadata preserved)
            shutil.copy2(src_path, dst_png)
            return

        # Fallback: convert once using Pillow with low compression.
        if Image is None:
            raise RuntimeError("Pillow not available for non‑PNG conversion.")
        with Image.open(src_path) as im:
            # Ensure mode that is safe for PNG masks and images
            if im.mode in ("I;16", "I"):
                im = im.convert("I;16")
            elif im.mode == "P":
                im = im.convert("RGBA")
            else:
                # leave RGB/L/L A as is
                pass
            # Atomic write: write to tmp then replace
            tmp = dst_png.with_suffix(".tmp.png")
            im.save(tmp, format="PNG", compress_level=png_compress_level)
            os.replace(tmp, dst_png)

    # Buffered label copy
    def _copy_label(src: str, dst: Path) -> None:
        if skip_if_exists and dst.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    def _process_seg_pair(img_path: str, mask_path: str, dataset_name: str, idx: int) -> tuple[bool, str]:
        out_name = f"{dataset_name}_{idx:06d}.png"
        img_out_path = angiography_segment_img / out_name
        mask_out_path = angiography_segment_lbl / out_name
        try:
            _fast_export_to_png(img_path, img_out_path)
            _fast_export_to_png(mask_path, mask_out_path)
            return True, out_name
        except Exception as e:
            return False, f"[SEG] {img_path} | {e}"

    def _process_arcade_item(img_path: str, label_path: str, split: str, number: str) -> tuple[bool, str]:
        out_img = f"ARCADE{split}_{number}.png"
        out_lbl = f"ARCADE{split}_{number}.txt"
        try:
            _fast_export_to_png(img_path, angiography_detect_img / out_img)
            _copy_label(label_path, angiography_detect_lbl / out_lbl)
            return True, out_img
        except Exception as e:
            return False, f"[DETECT][ARCADE] {img_path} | {e}"

    def _process_cadica_item(img_path: str, label_path: str, unique_name: str) -> tuple[bool, str]:
        out_img = f"{unique_name}.png"
        out_lbl = f"{unique_name}.txt"
        try:
            _fast_export_to_png(img_path, angiography_detect_img / out_img)
            _copy_label(label_path, angiography_detect_lbl / out_lbl)
            return True, out_img
        except Exception as e:
            return False, f"[DETECT][CADICA] {img_path} | {e}"

    def _process_retino_generic(img_path: str, mask_path: str, dataset: str, order_idx: int) -> tuple[bool, str]:
        out_name = f"{dataset}_{order_idx:06d}.png"
        try:
            _fast_export_to_png(img_path, retino_img_dir / out_name)
            _fast_export_to_png(mask_path, retino_lbl_dir / out_name)
            return True, out_name
        except Exception as e:
            return False, f"[RETINO][{dataset}] {img_path} | {e}"

    def _process_fives_item(img_path: str, mask_path: str, split: str, number: str) -> tuple[bool, str]:
        out_name = f"FIVES{split}_{number}.png"
        try:
            _fast_export_to_png(img_path, retino_img_dir / out_name)
            _fast_export_to_png(mask_path, retino_lbl_dir / out_name)
            return True, out_name
        except Exception as e:
            return False, f"[RETINO][FIVES] {img_path} | {e}"

    # Determine if anything already processed
    ordering_already_done = any(angiography_segment_img.glob('*.png')) or any(angiography_detect_img.glob('*.png')) or any(retino_img_dir.glob('*.png'))
    if ordering_already_done:
        logger.info("[generator] Detected existing processed structure -> skipping reordering phase.")
    else:
        logger.info("[generator] Starting dataset reordering phase.")

    # ---------------- Segment (angiography) ----------------
    if not ordering_already_done:
        all_seg_pairs: List[Tuple[str, str, str]] = []
        for dataset in segment_datasets:
            ds_input = _resolve_path("angiography", "segment", dataset)
            if dataset == "DCA1":
                logger.info(f"Collecting DCA1: {ds_input}")
                pairs = dca1.collect_image_mask_pairs(ds_input)
            elif dataset == "FS-CAD":
                logger.info(f"Collecting FS-CAD: {ds_input}")
                pairs = fs_cad.collect_image_mask_pairs(ds_input)
            elif dataset == "XCAV":
                logger.info(f"Collecting XCAV: {ds_input}")
                pairs = xcav.collect_image_mask_pairs(ds_input, logger=logger)
            else:
                logger.warning(f"Unknown segmentation dataset: {dataset}")
                continue
            for img, mask in pairs:
                all_seg_pairs.append((img, mask, dataset))

        # Parallel processing
        seg_ok = seg_fail = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for idx, (img_path, mask_path, dataset_name) in enumerate(all_seg_pairs, 1):
                futures.append(ex.submit(_process_seg_pair, img_path, mask_path, dataset_name, idx))
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Segmentation"):
                ok, msg = fut.result()
                if ok:
                    seg_ok += 1
                else:
                    seg_fail += 1
                    logger.error(msg)
        logger.info(f"Segmentation (angiography): {seg_ok} ok, {seg_fail} failed.")

    # ---------------- Detection (angiography) ----------------
    if not ordering_already_done:
        detection_total = detection_ok = detection_fail = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = []

            if "ARCADE" in detect_datasets:
                arcade_dir = _resolve_path("angiography", "detect", "ARCADE")
                logger.info(f"Collecting ARCADE detection: {arcade_dir}")
                try:
                    arcade_pairs = arcade_stenosis.collect_image_label_pairs(arcade_dir)
                    for img_path, label_path, split, number in arcade_pairs:
                        futures.append(ex.submit(_process_arcade_item, img_path, label_path, split, number))
                    detection_total += len(arcade_pairs)
                except Exception as e:
                    logger.error(f"Failed collecting ARCADE pairs: {e}")

            if "CADICA" in detect_datasets:
                cadica_dir = _resolve_path("angiography", "detect", "CADICA")
                logger.info(f"Collecting CADICA detection: {cadica_dir}")
                try:
                    cadica_pairs = cadica.collect_image_label_pairs(cadica_dir, logger=logger)
                    for img_path, label_path, unique_name in cadica_pairs:
                        futures.append(ex.submit(_process_cadica_item, img_path, label_path, unique_name))
                    detection_total += len(cadica_pairs)
                except Exception as e:
                    logger.error(f"Failed collecting CADICA pairs: {e}")

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Detection"):
                ok, msg = fut.result()
                if ok:
                    detection_ok += 1
                else:
                    detection_fail += 1
                    logger.error(msg)

        logger.info(f"Detection (angiography): {detection_ok}/{detection_total} ok, {detection_fail} failed.")

    # ---------------- Retinography ----------------
    if not ordering_already_done:
        retino_ok = retino_fail = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for dataset in retino_datasets:
                ds_input = _resolve_path("retinography", None, dataset)
                if dataset == "CHASEDB1":
                    logger.info(f"Collecting CHASEDB1: {ds_input}")
                    try:
                        pairs = chasedb1.collect_image_mask_pairs(ds_input)
                        for idx, (img_path, mask_path) in enumerate(pairs, 1):
                            futures.append(ex.submit(_process_retino_generic, img_path, mask_path, dataset, idx))
                    except Exception as e:
                        logger.error(f"Failed collecting CHASEDB1 pairs: {e}")
                        continue
                elif dataset == "DRIVE":
                    logger.info(f"Collecting DRIVE: {ds_input}")
                    try:
                        pairs = drive.collect_image_mask_pairs(ds_input)
                        for idx, (img_path, mask_path) in enumerate(pairs, 1):
                            futures.append(ex.submit(_process_retino_generic, img_path, mask_path, dataset, idx))
                    except Exception as e:
                        logger.error(f"Failed collecting DRIVE pairs: {e}")
                        continue
                elif dataset == "FIVES":
                    logger.info(f"Collecting FIVES: {ds_input}")
                    try:
                        fives_pairs = fives.collect_image_mask_pairs(ds_input)  # (img, mask, split, number)
                        for img_path, mask_path, split, number in fives_pairs:
                            futures.append(ex.submit(_process_fives_item, img_path, mask_path, split, number))
                    except Exception as e:
                        logger.error(f"Failed collecting FIVES pairs: {e}")
                        continue
                else:
                    logger.warning(f"Retinography dataset not implemented: {dataset}")
                    continue

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Retinography"):
                ok, msg = fut.result()
                if ok:
                    retino_ok += 1
                else:
                    retino_fail += 1
                    logger.error(msg)

        logger.info(f"Retinography: {retino_ok} ok, {retino_fail} failed.")

    # ---------------- Preprocessing Pipeline ----------------
    try:
        run_preprocessing(output_dir, config, logger)
    except Exception as e:
        logger.error(f"[generator] Preprocessing pipeline failed: {e}")

    logger.info("Data standardization + preprocessing completed.")


if __name__ == "__main__":
    main()