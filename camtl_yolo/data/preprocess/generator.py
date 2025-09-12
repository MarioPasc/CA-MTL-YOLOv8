"""
Main generator script for dataset processing.
Reads data.yaml and dynamically calls dataset-specific processing modules.
"""
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

def main():
    # NOTE: This sys.path logic is only needed for running scripts directly.
    # It should NOT be placed in __init__.py, as it can cause import issues in production or when used as a package.
    PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from camtl_yolo.utils.logger import get_logger
    from camtl_yolo.data.preprocess.modules import dca1, fs_cad, xcav

    logger = get_logger("generator")

    def load_config(config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    config_path = Path(__file__).parent / "config" / "data.yaml"
    config = load_config(str(config_path))

    root = config["root"]
    output_dir = config["output_dir"]
    segment_datasets = config.get("segment_datasets", [])

    # --- Segmentation datasets ---
    seg_img_dir = Path(output_dir) / "segmentation" / "images"
    seg_lbl_dir = Path(output_dir) / "segmentation" / "labels"
    seg_img_dir.mkdir(parents=True, exist_ok=True)
    seg_lbl_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    for dataset in segment_datasets:
        if dataset == "DCA1":
            input_dir = os.path.join(root, "segment", "DCA1")
            logger.info(f"Collecting DCA1: {input_dir}")
            pairs = dca1.collect_image_mask_pairs(input_dir)
            for img, mask in pairs:
                all_pairs.append((img, mask, "DCA1"))
        elif dataset == "FS-CAD":
            input_dir = os.path.join(root, "segment", "FS-CAD")
            logger.info(f"Collecting FS-CAD: {input_dir}")
            pairs = fs_cad.collect_image_mask_pairs(input_dir)
            for img, mask in pairs:
                all_pairs.append((img, mask, "FS-CAD"))
        elif dataset == "XCAV":
            input_dir = os.path.join(root, "segment", "XCAV")
            logger.info(f"Collecting XCAV: {input_dir}")
            pairs = xcav.collect_image_mask_pairs(input_dir, logger=logger)
            for img, mask in pairs:
                all_pairs.append((img, mask, "XCAV"))
        else:
            logger.warning(f"No processing script for dataset: {dataset}")

    for idx, (img_path, mask_path, dataset_name) in enumerate(all_pairs, 1):
        out_name = f"{dataset_name}_{idx:06d}.png"
        img_out_path = seg_img_dir / out_name
        mask_out_path = seg_lbl_dir / out_name
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                img.save(img_out_path)
            with Image.open(mask_path) as mask:
                mask.save(mask_out_path)
            logger.debug(f"Saved {out_name}")
        except Exception as e:
            logger.error(f"Error processing {img_path} or {mask_path}: {e}")

    # --- Detection datasets ---
    from camtl_yolo.data.preprocess.modules import arcade_stenosis, cadica
    detect_img_dir = Path(output_dir) / "detection" / "images"
    detect_lbl_dir = Path(output_dir) / "detection" / "labels"
    detect_img_dir.mkdir(parents=True, exist_ok=True)
    detect_lbl_dir.mkdir(parents=True, exist_ok=True)

    # ARCADE detection dataset
    arcade_dir = os.path.join(root, "detect", "ARCADE")
    logger.info(f"Collecting ARCADE detection: {arcade_dir}")
    arcade_pairs = arcade_stenosis.collect_image_label_pairs(arcade_dir)
    for img_path, label_path, split, number in arcade_pairs:
        out_name_img = f"ARCADE{split}_{number}.png"
        out_name_lbl = f"ARCADE{split}_{number}.txt"
        img_out_path = detect_img_dir / out_name_img
        lbl_out_path = detect_lbl_dir / out_name_lbl
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                img.save(img_out_path)
            with open(label_path, 'r') as fin, open(lbl_out_path, 'w') as fout:
                fout.write(fin.read())
            logger.debug(f"Saved {out_name_img} and {out_name_lbl}")
        except Exception as e:
            logger.error(f"Error processing {img_path} or {label_path}: {e}")

    # CADICA detection dataset
    cadica_dir = os.path.join(root, "detect", "CADICA")
    logger.info(f"Collecting CADICA detection: {cadica_dir}")
    cadica_pairs = cadica.collect_image_label_pairs(cadica_dir, logger=logger)
    for img_path, label_path, unique_name in cadica_pairs:
        out_name_img = f"{unique_name}.png"
        out_name_lbl = f"{unique_name}.txt"
        img_out_path = detect_img_dir / out_name_img
        lbl_out_path = detect_lbl_dir / out_name_lbl
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                img.save(img_out_path)
            with open(label_path, 'r') as fin, open(lbl_out_path, 'w') as fout:
                fout.write(fin.read())
            logger.debug(f"Saved {out_name_img} and {out_name_lbl}")
        except Exception as e:
            logger.error(f"Error processing {img_path} or {label_path}: {e}")

if __name__ == "__main__":
    main()