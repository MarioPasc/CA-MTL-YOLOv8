import os
from pathlib import Path
from typing import List, Tuple

def collect_image_label_pairs(input_dir: str) -> List[Tuple[str, str, str, str]]:
    """
    Return list of (image_path, label_path, split, number) for ARCADE detection dataset.
    split is one of 'train', 'val', 'test'.
    number is the image number (without extension).
    """
    pairs = []
    for split in ["train", "val", "test"]:
        split_dir = Path(input_dir) / split
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        for img_file in sorted(img_dir.glob("*.png")):
            number = img_file.stem
            label_file = lbl_dir / f"{number}.txt"
            if label_file.exists():
                pairs.append((str(img_file), str(label_file), split, number))
    return pairs
