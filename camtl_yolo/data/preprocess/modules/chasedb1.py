"""CHASEDB1 dataset processing module.

This dataset contains retinal fundus images and multiple human observer (HO) annotations.
We only keep the masks ending with `1stHO.png` (first human observer) as ground truth.

File naming examples:
  Image_01L.jpg          -> original image
  Image_01L_1stHO.png    -> first observer mask (keep)
  Image_01L_2ndHO.png    -> second observer mask (discard)

The standard interface exposed is `collect_image_mask_pairs(input_dir)` returning
`List[Tuple[image_path, mask_path]]` where mask corresponds to the *1stHO.png file.
"""
from pathlib import Path
from typing import List, Tuple

VALID_MASK_SUFFIX = "_1stHO.png"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def collect_image_mask_pairs(input_dir: str) -> List[Tuple[str, str]]:
	"""Return list of (image_path, mask_path) for CHASEDB1.

	Args:
		input_dir: Path to folder containing CHASEDB1 images and HO masks all together.
	Returns:
		List of tuples where each original image has a corresponding 1stHO mask.
	"""
	root = Path(input_dir)
	if not root.exists():
		return []

	# Index masks by their base stem without the _1stHO suffix
	masks = {}
	images = {}
	for file in root.iterdir():
		if not file.is_file():
			continue
		stem = file.stem  # e.g., Image_01L_1stHO or Image_01L
		suffix = file.suffix.lower()
		name = file.name
		if name.endswith(VALID_MASK_SUFFIX):
			# Remove suffix from stem to get base image stem
			base_stem = name[:-len(VALID_MASK_SUFFIX)]  # Image_01L
			masks[base_stem] = str(file)
		else:
			# Potential image file
			if suffix in IMAGE_EXTENSIONS and not (stem.endswith("_1stHO") or stem.endswith("_2ndHO")):
				images[stem] = str(file)

	pairs: List[Tuple[str, str]] = []
	for base_stem, img_path in images.items():
		mask_path = masks.get(base_stem)
		if mask_path:
			pairs.append((img_path, mask_path))
	# Sort deterministically by image name
	pairs.sort(key=lambda x: Path(x[0]).name)
	return pairs

__all__ = ["collect_image_mask_pairs"]
