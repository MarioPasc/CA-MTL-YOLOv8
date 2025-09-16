"""Generate unified train/val/test splits JSON for multi-task datasets.

Reads configuration from the project data YAML (e.g. `camtl_yolo/data/config/data.yaml`).

Expected processed dataset directory structure (paths come from YAML):

  output_dir/
    angiography/
      detect/
        images/*.png|jpg
        labels/*.txt              (YOLO txt, same stem as images)
      segment/
        images/*.png|jpg
        labels/*.png|jpg          (segmentation masks, same stem)
    retinography/
      images/*.png|jpg
      labels/*.png|jpg            (segmentation masks, same stem)

The YAML contains the fractions for each logical task:
  holdout:
    retinography_segmentation: {train: 0.7, val: 0.2, test: 0.1}
    angiography_segmentation:  {train: 0.7, val: 0.3, test: 0.0}
    angiography_detection:     {train: 0.7, val: 0.3, test: 0.0}

Output JSON schema (per task):
{
  "retinography_segmentation": {
    "train": [ {"image": "/abs/.../img.png", "mask": "/abs/.../mask.png"}, ...],
    "val":   [ ... ],
    "test":  [ ... ]
  },
  "angiography_segmentation": { ... },
  "angiography_detection": {
     "train": [ {"image": "/abs/.../img.png", "label": "/abs/.../labels/img.txt"}, ... ]
  }
}

Detection task entries DO NOT embed boxes (as per requirement); only absolute paths.

Run:
  python -m camtl_yolo.data.holdout --config camtl_yolo/data/config/data.yaml \
      --seed 42 --json-name splits.json

If the JSON already exists and --force is not used, the script aborts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import yaml

IMAGE_EXTS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp"}
MASK_EXTS = IMAGE_EXTS  # treat same set for now

import sys
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from camtl_yolo.utils.logger import get_logger

@dataclass
class HoldoutFractions:
    train: float
    val: float
    test: float

    def normalized(self) -> "HoldoutFractions":
        total = self.train + self.val + self.test
        if not math.isclose(total, 1.0, rel_tol=1e-6):
            if total == 0:
                raise ValueError("All holdout fractions are zero.")
            return HoldoutFractions(self.train / total, self.val / total, self.test / total)
        return self


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_files(directory: Path, exts: Sequence[str]) -> List[Path]:
    return sorted([p for p in directory.rglob("*") if p.is_file() and p.suffix.lower().lstrip('.') in exts])


def match_label(image_path: Path, labels_dir: Path) -> Path:
    cand = labels_dir / (image_path.stem + "." + image_path.suffix.lstrip('.').split('.')[0])
    # For YOLO detection: labels are .txt regardless of image extension
    txt = labels_dir / f"{image_path.stem}.txt"
    return txt


def match_mask(image_path: Path, labels_dir: Path) -> Path:
    # Assume same stem + any allowed mask extension. Prefer same extension first.
    preferred = labels_dir / (image_path.stem + image_path.suffix)
    if preferred.exists():
        return preferred
    # fallback: any ext match
    for ext in MASK_EXTS:
        alt = labels_dir / f"{image_path.stem}.{ext}"
        if alt.exists():
            return alt
    raise FileNotFoundError(f"Mask for image {image_path} not found in {labels_dir}")


def random_split(items: List[Path], fractions: HoldoutFractions, seed: int) -> Dict[str, List[Path]]:
    fracs = fractions.normalized()
    n = len(items)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    rng.shuffle(items)
    n_train = int(round(fracs.train * n))
    n_val = int(round(fracs.val * n))
    # adjust to ensure total == n
    while n_train + n_val > n:
        n_val -= 1
    n_test = n - n_train - n_val
    return {
        "train": items[:n_train],
        "val": items[n_train : n_train + n_val],
        "test": items[n_train + n_val :],
    }


def build_retinography_segmentation(root: Path, fractions: HoldoutFractions, seed: int) -> Dict[str, List[dict]]:
    images_dir = root / "retinography" / "images"
    masks_dir = root / "retinography" / "labels"
    images = list_files(images_dir, list(IMAGE_EXTS))
    split_paths = random_split(images, fractions, seed)
    out = {}
    for split, paths in split_paths.items():
        out[split] = [
            {"image": str(p.resolve()), "mask": str(match_mask(p, masks_dir).resolve()), "is_segmentation": True, "domain": "source"}
            for p in paths
        ]
    return out


def build_angiography_segmentation(root: Path, fractions: HoldoutFractions, seed: int) -> Dict[str, List[dict]]:
    images_dir = root / "angiography" / "segment" / "images"
    masks_dir = root / "angiography" / "segment" / "labels"
    images = list_files(images_dir, list(IMAGE_EXTS))
    split_paths = random_split(images, fractions, seed)
    out = {}
    for split, paths in split_paths.items():
        out[split] = [
            {"image": str(p.resolve()), "mask": str(match_mask(p, masks_dir).resolve()), "is_segmentation": True, "domain": "target"}
            for p in paths
        ]
    return out


def build_angiography_detection(root: Path, fractions: HoldoutFractions, seed: int) -> Dict[str, List[dict]]:
    images_dir = root / "angiography" / "detect" / "images"
    labels_dir = root / "angiography" / "detect" / "labels"
    images = list_files(images_dir, list(IMAGE_EXTS))
    split_paths = random_split(images, fractions, seed)
    out = {}
    for split, paths in split_paths.items():
        out[split] = [
            {"image": str(p.resolve()), "label": str(match_label(p, labels_dir).resolve()), "is_segmentation": False, "domain": "target"}
            for p in paths
        ]
    return out


def build_all(config: dict, seed: int) -> dict:
    output_dir = Path(config["output_dir"]).expanduser()
    if not output_dir.exists():
        raise FileNotFoundError(f"output_dir '{output_dir}' does not exist; run preprocessing first.")

    holdout_cfg = config.get("holdout", {})
    def extract(name: str) -> HoldoutFractions:
        frac = holdout_cfg.get(name, {})
        return HoldoutFractions(
            train=float(frac.get("train", 0.0)),
            val=float(frac.get("val", 0.0)),
            test=float(frac.get("test", 0.0)),
        )

    return {
        "retinography_segmentation": build_retinography_segmentation(output_dir, extract("retinography_segmentation"), seed),
        "angiography_segmentation": build_angiography_segmentation(output_dir, extract("angiography_segmentation"), seed),
        "angiography_detection": build_angiography_detection(output_dir, extract("angiography_detection"), seed),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate multi-task holdout JSON splits")
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[0] / "config" / "data.yaml", help="Path to data.yaml configuration")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--json-name", type=str, default="splits.json", help="Output JSON filename inside output_dir")
    ap.add_argument("--force", action="store_true", help="Overwrite existing JSON")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger("holdout")
    logger.info(f"Using config: {args.config}")
    logger.info(f"Output JSON: {args.json_name} (force={args.force})")

    output_dir = Path(cfg["output_dir"]).expanduser()
    json_path = output_dir / args.json_name
    if json_path.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing {json_path}. Use --force to override.")
    splits = build_all(cfg, args.seed)
    json_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    logger.info(f"Wrote splits JSON: {json_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
