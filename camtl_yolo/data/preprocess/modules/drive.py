"""DRIVE dataset processing module.

We standardize the DRIVE retinal vessel segmentation dataset by pairing each image
with its first manual annotation (located in `1st_manual/`). We ignore the second
manual annotations, masks, and any additional metadata.

Expected folder layout (root passed to this module):
  DRIVE/
    training/
      images/*.tif
      1st_manual/*_manual1.gif
      mask/ (ignored)
    test/ (optional, same structure as training)

The function `collect_image_mask_pairs(input_dir)` walks both `training` and `test`
subdirectories if present and yields (image_path, mask_path) tuples. Images are
`.tif` and masks are `.gif`. We return only pairs where both files exist.
"""
from pathlib import Path
from typing import List, Tuple

IMAGE_EXTS = {".tif", ".tiff"}
MASK_SUFFIX = "_manual1.gif"

def _collect_split(split_dir: Path) -> List[Tuple[str, str]]:
    images_dir = split_dir / "images"
    manual_dir = split_dir / "1st_manual"
    if not images_dir.exists() or not manual_dir.exists():
        return []
    # Build dict of masks keyed by numeric id (e.g., 21 from 21_manual1.gif)
    masks = {}
    for mf in manual_dir.iterdir():
        if mf.is_file() and mf.name.endswith(MASK_SUFFIX):
            # Example: 21_manual1.gif -> id = 21
            id_part = mf.name.split("_", 1)[0]
            masks[id_part] = str(mf)
    pairs: List[Tuple[str, str]] = []
    for img_file in images_dir.iterdir():
        if not img_file.is_file() or img_file.suffix.lower() not in IMAGE_EXTS:
            continue
        # Example: 21_training.tif -> id = 21
        stem = img_file.stem  # 21_training
        id_part = stem.split("_", 1)[0]
        mask_path = masks.get(id_part)
        if mask_path:
            pairs.append((str(img_file), mask_path))
    # Sort deterministically
    pairs.sort(key=lambda x: Path(x[0]).name)
    return pairs

def collect_image_mask_pairs(input_dir: str) -> List[Tuple[str, str]]:
    root = Path(input_dir)
    splits = [d for d in [root / "training", root / "test"] if d.exists()]
    all_pairs: List[Tuple[str, str]] = []
    for split in splits:
        all_pairs.extend(_collect_split(split))
    # Global sort
    all_pairs.sort(key=lambda x: Path(x[0]).name)
    return all_pairs

__all__ = ["collect_image_mask_pairs"]

