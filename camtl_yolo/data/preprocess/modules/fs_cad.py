"""
FS-CAD dataset processing module.
Standardizes segmentation images and masks into the required output format.
"""
import os
from pathlib import Path
from typing import List, Tuple, Optional, Type
from PIL import Image

def collect_image_mask_pairs(input_dir: str) -> List[Tuple[str, str]]:
    a_dir = Path(input_dir) / "A"
    gt_dir = Path(input_dir) / "GT"
    pairs = []
    for fname in sorted(os.listdir(a_dir)):
        if fname.endswith('.png'):
            img_path = a_dir / fname
            mask_path = gt_dir / fname
            if mask_path.exists():
                pairs.append((str(img_path), str(mask_path)))
    return pairs

def process_fs_cad(
    input_dir: str,
    output_dir: str,
    dataset_name: str = "FS-CAD",
    logger=None,
    tqdm_cls=None
) -> None:
    """
    Process the FS-CAD dataset and save standardized images and masks.
    Args:
        input_dir: Path to the raw FS-CAD dataset directory.
        output_dir: Path to the output directory for processed data.
        dataset_name: Name of the dataset (default: FS-CAD).
        logger: Logger instance.
        tqdm_cls: tqdm class for progress bar.
    """
    input_path = Path(input_dir)
    seg_img_dir = Path(output_dir) / "segmentation" / "images"
    seg_lbl_dir = Path(output_dir) / "segmentation" / "labels"
    seg_img_dir.mkdir(parents=True, exist_ok=True)
    seg_lbl_dir.mkdir(parents=True, exist_ok=True)

    a_dir = input_path / "A"
    gt_dir = input_path / "GT"
    if not a_dir.exists() or not gt_dir.exists():
        if logger:
            logger.error(f"FS-CAD: Missing 'A' or 'GT' directory in {input_path}")
        return

    image_files = sorted([f for f in os.listdir(a_dir) if f.endswith('.png')])
    pbar = tqdm_cls(image_files, desc=f"{dataset_name} images") if tqdm_cls else image_files
    for file in pbar:
        img_path = a_dir / file
        mask_path = gt_dir / file
        out_name = f"{dataset_name}_{file}"
        img_out_path = seg_img_dir / out_name
        mask_out_path = seg_lbl_dir / out_name
        # Copy or convert image
        try:
            with Image.open(img_path) as img:
                img.save(img_out_path)
            with Image.open(mask_path) as mask:
                mask.save(mask_out_path)
            if logger:
                logger.debug(f"Processed {file}")
        except Exception as e:
            if logger:
                logger.error(f"Error processing {file}: {e}")