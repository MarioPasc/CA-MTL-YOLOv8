"""Unified data standardization generator.

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
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

def main():
    # NOTE: This sys.path logic is only needed for running scripts directly.
    # It should NOT be placed in __init__.py, as it can cause import issues in production or when used as a package.
    PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from camtl_yolo.utils.logger import get_logger
    from camtl_yolo.data.preprocess.modules import dca1, fs_cad, xcav, chasedb1, drive, fives

    logger = get_logger("generator")

    def load_config(config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    config_path = Path(__file__).parent / "config" / "data.yaml"
    config = load_config(str(config_path))

    root = config.get("root")
    output_dir = config.get("output_dir")
    segment_datasets: List[str] = config.get("segment_datasets", [])
    detect_datasets: List[str] = config.get("detect_datasets", [])
    retino_datasets: List[str] = config.get("retinography_datasets", [])

    if not root or not output_dir:
        logger.error("Config must include 'root' and 'output_dir'. Exiting.")
        return

    # Prepare target directory tree
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

    # Helper to resolve dataset directory with backward compatibility
    def _resolve_path(category: str, subcat: str | None, dataset: str) -> str:
        """Return absolute path for a dataset, trying new layout first then old.

        New layout: root/angiography/{segment|detect}/dataset
        Old layout: root/{segment|detect}/dataset
        Retinography: root/retinography/dataset (unchanged)
        """
        if category == "retinography":
            return os.path.join(root, "retinography", dataset)
        # category == 'angiography'
        assert subcat in {"segment", "detect"}
        new_path = os.path.join(root, "angiography", subcat, dataset)
        if os.path.isdir(new_path):
            return new_path
        # fallback old path
        old_path = os.path.join(root, subcat, dataset)
        return old_path

    # Determine if ordering step has already been performed (simple heuristic: any images exist)
    ordering_already_done = any(angiography_segment_img.glob('*.png')) or any(angiography_detect_img.glob('*.png')) or any(retino_img_dir.glob('*.png'))
    if ordering_already_done:
        logger.info("[generator] Detected existing processed structure -> skipping reordering phase.")
    else:
        logger.info("[generator] Starting dataset reordering phase.")

    # ================= Segment (angiography) =================
    all_seg_pairs: List[Tuple[str, str, str]] = []  # (img, mask, dataset_name)
    if not ordering_already_done:
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

        for idx, (img_path, mask_path, dataset_name) in enumerate(all_seg_pairs, 1):
            out_name = f"{dataset_name}_{idx:06d}.png"
            img_out_path = angiography_segment_img / out_name
            mask_out_path = angiography_segment_lbl / out_name
            try:
                from PIL import Image
                with Image.open(img_path) as img:
                    img.save(img_out_path)
                with Image.open(mask_path) as mask:
                    mask.save(mask_out_path)
            except Exception as e:
                logger.error(f"[SEG] Error processing {img_path} or {mask_path}: {e}")
        logger.info(f"Segmentation (angiography): {len(all_seg_pairs)} pairs saved.")

    # ================= Detection (angiography) =================
    from camtl_yolo.data.preprocess.modules import arcade_stenosis, cadica
    if not ordering_already_done:
        # ARCADE
        detection_counter: int = 0
        if "ARCADE" in detect_datasets:
            arcade_dir = _resolve_path("angiography", "detect", "ARCADE")
            logger.info(f"Collecting ARCADE detection: {arcade_dir}")
            try:
                arcade_pairs = arcade_stenosis.collect_image_label_pairs(arcade_dir)
                detection_counter += len(arcade_pairs)
                for img_path, label_path, split, number in arcade_pairs:
                    out_img = f"ARCADE{split}_{number}.png"
                    out_lbl = f"ARCADE{split}_{number}.txt"
                    img_out_path = angiography_detect_img / out_img
                    lbl_out_path = angiography_detect_lbl / out_lbl
                    try:
                        from PIL import Image
                        with Image.open(img_path) as img:
                            img.save(img_out_path)
                        with open(label_path) as fin, open(lbl_out_path, 'w') as fout:
                            fout.write(fin.read())
                    except Exception as e:
                        logger.error(f"[DETECT][ARCADE] Error processing {img_path}: {e}")
            except Exception as e:
                logger.error(f"Failed collecting ARCADE pairs: {e}")
        # CADICA
        if "CADICA" in detect_datasets:
            cadica_dir = _resolve_path("angiography", "detect", "CADICA")
            logger.info(f"Collecting CADICA detection: {cadica_dir}")
            try:
                cadica_pairs = cadica.collect_image_label_pairs(cadica_dir, logger=logger)
                detection_counter += len(cadica_pairs)
                for img_path, label_path, unique_name in cadica_pairs:
                    out_img = f"{unique_name}.png"
                    out_lbl = f"{unique_name}.txt"
                    img_out_path = angiography_detect_img / out_img
                    lbl_out_path = angiography_detect_lbl / out_lbl
                    try:
                        from PIL import Image
                        with Image.open(img_path) as img:
                            img.save(img_out_path)
                        with open(label_path) as fin, open(lbl_out_path, 'w') as fout:
                            fout.write(fin.read())
                    except Exception as e:
                        logger.error(f"[DETECT][CADICA] Error processing {img_path}: {e}")
            except Exception as e:
                logger.error(f"Failed collecting CADICA pairs: {e}")
        logger.info(f"Detection (angiography): {detection_counter} pairs saved.")
    # ================= Retinography =================
    # For now, only datasets with segmentation-like structure (image + mask) use collect_image_mask_pairs
    retino_counter = 0
    if not ordering_already_done:
        for dataset in retino_datasets:
            ds_input = _resolve_path("retinography", None, dataset)
            if dataset == "CHASEDB1":
                logger.info(f"Collecting CHASEDB1: {ds_input}")
                try:
                    pairs = chasedb1.collect_image_mask_pairs(ds_input)
                except Exception as e:
                    logger.error(f"Failed collecting CHASEDB1 pairs: {e}")
                    continue
            elif dataset == "DRIVE":
                logger.info(f"Collecting DRIVE: {ds_input}")
                try:
                    pairs = drive.collect_image_mask_pairs(ds_input)
                except Exception as e:
                    logger.error(f"Failed collecting DRIVE pairs: {e}")
                    continue
            elif dataset == "FIVES":
                logger.info(f"Collecting FIVES: {ds_input}")
                try:
                    # FIVES returns tuples with split and number
                    fives_pairs = fives.collect_image_mask_pairs(ds_input)  # (img, mask, split, number)
                except Exception as e:
                    logger.error(f"Failed collecting FIVES pairs: {e}")
                    continue
                # Naming: FIVES{split}_{number:originalDigits}
                for img_path, mask_path, split, number in fives_pairs:
                    out_name = f"FIVES{split}_{number}.png"
                    img_out_path = retino_img_dir / out_name
                    mask_out_path = retino_lbl_dir / out_name
                    try:
                        from PIL import Image
                        with Image.open(img_path) as img:
                            img.save(img_out_path)
                        with Image.open(mask_path) as mask:
                            mask.save(mask_out_path)
                    except Exception as e:
                        logger.error(f"[RETINO][FIVES] Error processing {img_path}: {e}")
                retino_counter += len(fives_pairs)
                continue  # Skip generic handler below
            else:
                logger.warning(f"Retinography dataset not implemented: {dataset}")
                continue
            for idx, (img_path, mask_path) in enumerate(pairs, 1):
                out_name = f"{dataset}_{retino_counter + idx:06d}.png"
                img_out_path = retino_img_dir / out_name
                mask_out_path = retino_lbl_dir / out_name
                try:
                    from PIL import Image
                    with Image.open(img_path) as img:
                        img.save(img_out_path)
                    with Image.open(mask_path) as mask:
                        mask.save(mask_out_path)
                except Exception as e:
                    logger.error(f"[RETINO][{dataset}] Error processing {img_path}: {e}")
            retino_counter += len(pairs)
        logger.info(f"Retinography: {retino_counter} pairs saved.")

    # ================= Preprocessing Pipeline =================
    from camtl_yolo.data.preprocess.pipeline import run_preprocessing
    try:
        run_preprocessing(output_dir, config, logger)
    except Exception as e:
        logger.error(f"[generator] Preprocessing pipeline failed: {e}")

    logger.info("Data standardization + preprocessing completed.")

if __name__ == "__main__":
    main()