"""FIVES dataset processing module.

Structure (input_dir):
  FIVES/
    train/
      Original/*.png
      Ground truth/*.png
    test/
      Original/*.png
      Ground truth/*.png

We pair each image in `Original` with mask of identical filename in `Ground truth`.
Return list of (image_path, mask_path, split, number) where:
  split  -> 'train' or 'test'
  number -> numeric prefix before first underscore (e.g., 100 from 100_A.png)

Generator will use naming pattern: FIVES{split}_{number}.png / .txt (for segmentation masks just .png)
"""
from pathlib import Path
from typing import List, Tuple

VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

def _extract_number(filename: str) -> str:
    # filename like 100_A.png -> number = 100
    base = Path(filename).name
    first = base.split('_')[0]
    # Strip non-digit just in case, keep digits
    digits = ''.join(ch for ch in first if ch.isdigit())
    return digits or first  # fallback to original segment if no digits

def _collect_split(split_dir: Path, split: str) -> List[Tuple[str, str, str, str]]:
    orig_dir = split_dir / "Original"
    gt_dir = split_dir / "Ground truth"
    if not orig_dir.exists() or not gt_dir.exists():
        return []
    pairs: List[Tuple[str, str, str, str]] = []
    for img_file in orig_dir.iterdir():
        if not img_file.is_file():
            continue
        if img_file.suffix.lower() not in VALID_IMAGE_SUFFIXES:
            continue
        mask_file = gt_dir / img_file.name
        if mask_file.exists():
            number = _extract_number(img_file.name)
            pairs.append((str(img_file), str(mask_file), split, number))
    pairs.sort(key=lambda x: Path(x[0]).name)
    return pairs

def collect_image_mask_pairs(input_dir: str) -> List[Tuple[str, str, str, str]]:
    root = Path(input_dir)
    all_pairs: List[Tuple[str, str, str, str]] = []
    for split in ["train", "test"]:
        split_dir = root / split
        if split_dir.exists():
            all_pairs.extend(_collect_split(split_dir, split))
    # Overall sort on split then image name
    all_pairs.sort(key=lambda x: (x[2], Path(x[0]).name))
    return all_pairs

__all__ = ["collect_image_mask_pairs"]

