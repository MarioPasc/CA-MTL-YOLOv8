"""
DCA1 dataset processing module.
Standardizes segmentation images and masks into the required output format.
"""
import os
from pathlib import Path
from typing import List, Tuple
from PIL import Image

def is_mask(filename: str) -> bool:
    """Return True if the filename corresponds to a mask file."""
    return filename.endswith('_gt.pgm')

def get_image_number(filename: str) -> str:
    """Extract the image number from the filename (e.g., DB134_100_gt.pgm -> 100)."""
    base = os.path.basename(filename)
    parts = base.split('_')
    if len(parts) < 2:
        raise ValueError(f"Unexpected filename format: {filename}")
    number = parts[1].split('.')[0]
    if number.endswith('gt'):
        number = number[:-2]
    return number

def process_dca1(
    input_dir: str,
    output_dir: str,
    dataset_name: str = "DCA1",
    logger=None,
    tqdm_cls=None
) -> None:
    """
    Process the DCA1 dataset and save standardized images and masks.
    Args:
        input_dir: Path to the raw DCA1 dataset directory.
        output_dir: Path to the output directory for processed data.
        dataset_name: Name of the dataset (default: DCA1).
        logger: Logger instance.
        tqdm_cls: tqdm class for progress bar.
    """
    input_path = Path(input_dir)
    seg_img_dir = Path(output_dir) / "segmentation" / "images"
    seg_lbl_dir = Path(output_dir) / "segmentation" / "labels"
    seg_img_dir.mkdir(parents=True, exist_ok=True)
    seg_lbl_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in os.listdir(input_path) if f.endswith('.pgm')])
    pbar = tqdm_cls(files, desc=f"{dataset_name} files") if tqdm_cls else files
    for file in pbar:
        file_path = input_path / file
        try:
            if is_mask(file):
                # Mask file
                number = get_image_number(file)
                out_name = f"{dataset_name}_{number}.png"
                out_path = seg_lbl_dir / out_name
                # Convert mask to PNG
                with Image.open(file_path) as img:
                    img.save(out_path)
            else:
                # Image file
                number = get_image_number(file)
                out_name = f"{dataset_name}_{number}.png"
                out_path = seg_img_dir / out_name
                # Convert image to PNG
                with Image.open(file_path) as img:
                    img.save(out_path)
            if logger:
                logger.debug(f"Processed {file}")
        except Exception as e:
            if logger:
                logger.error(f"Error processing {file}: {e}")

def collect_image_mask_pairs(input_dir: str) -> List[Tuple[str, str]]:
    input_path = Path(input_dir)
    files = sorted([f for f in os.listdir(input_path) if f.endswith('.pgm')])
    images = {}
    masks = {}
    for file in files:
        if is_mask(file):
            number = get_image_number(file)
            masks[number] = str(input_path / file)
        else:
            number = get_image_number(file)
            images[number] = str(input_path / file)
    pairs = []
    for number in images:
        if number in masks:
            pairs.append((images[number], masks[number]))
    return pairs
