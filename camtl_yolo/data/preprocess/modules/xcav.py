from pathlib import Path
from typing import List, Tuple

def collect_image_mask_pairs(input_dir: str, logger = None) -> List[Tuple[str, str]]:
    """
    Return list of (image_path, mask_path) pairs for XCAV dataset.
    Only pairs where both image and mask exist are returned.
    """
    input_path = Path(input_dir)
    pairs = []
    # Iterate over all first-level folders (e.g., CVAI-1207)
    for patient_dir in sorted(input_path.iterdir()):
        if not patient_dir.is_dir():
            continue
        gt_root = patient_dir / "ground_truth"
        img_root = patient_dir / "images"
        if not gt_root.exists() or not img_root.exists():
            if logger:
                logger.warning(f"Missing ground_truth or images in {patient_dir}")
            continue
        # Iterate over all subfolders in ground_truth (skip those ending with CATH)
        for gt_subdir in sorted(gt_root.iterdir()):
            if not gt_subdir.is_dir() or gt_subdir.name.endswith("CATH"):
                continue
            img_subdir = img_root / gt_subdir.name
            if not img_subdir.exists():
                if logger:
                    logger.warning(f"Missing image subdir {img_subdir} for ground truth {gt_subdir}")
                continue
            # For each mask in ground_truth subdir, find corresponding image
            for mask_file in sorted(gt_subdir.glob("*.png")):
                frame = mask_file.name
                img_file = img_subdir / frame
                if img_file.exists():
                    pairs.append((str(img_file), str(mask_file)))
                    if logger:
                        logger.info(f"Pair found: {img_file} <-> {mask_file}")
                else:
                    if logger:
                        logger.warning(f"Image not found for mask: {mask_file}")
    if logger:
        logger.info(f"Total pairs found in XCAV: {len(pairs)}")
    return pairs