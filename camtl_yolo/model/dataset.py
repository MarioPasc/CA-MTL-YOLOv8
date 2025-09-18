# camtl_yolo/data/dataset.py
"""
Multi-task dataset for CA-MTL-YOLOv8 using Ultralytics BaseDataset.

- Reads a JSON manifest under the dataset root defined in data.yaml.
- Supports three logical sources:
  * retinography_segmentation    (image, mask)
  * angiography_segmentation     (image, mask)
  * angiography_detection        (image, YOLO-txt labels in normalized xywh)
- Returns per-sample dicts compatible with Ultralytics training.
- Applies flips from hyp consistently to images, masks, and boxes.

Notes
-----
Ultralytics BaseDataset calls:
  - get_img_files(img_path)   -> list of image paths
  - get_labels()              -> list[dict] with im_file, cls, bboxes, etc.
  - build_transforms(hyp)     -> callable(dict) -> dict

We override all three. We pass a sentinel img_path="__json__" and ignore it.

Augmentations
-------------
This version wires:
  - fliplr, flipud for both tasks.
Future extension (TODO): degrees/translate/scale/shear with affine-consistent remapping
of boxes and masks, and color jitter.

Batching
--------
Use your trainer’s DataLoader. If you need a custom collate, `multitask_collate_fn`
is provided.
"""
from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

try:
    # Vendored Ultralytics base
    from camtl_yolo.external.ultralytics.ultralytics.data.dataset import BaseDataset
except Exception as e:  # pragma: no cover
    raise ImportError("Ultralytics BaseDataset not found under camtl_yolo.external.ultralytics") from e

LOGGER = logging.getLogger(__name__)


# ----------------------------- Records ----------------------------- #

@dataclass(frozen=True)
class SampleRec:
    im_file: str         # absolute image path
    tgt_file: Optional[str]  # mask or label .txt path
    is_seg: bool
    domain: str          # 'retinography' | 'angiography'
    task_key: str        # top-level JSON key


# ----------------------------- IO helpers ----------------------------- #

def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_data_yaml(path: Path) -> Dict[str, Any]:
    import yaml # type: ignore
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _abspath(root: Path, p: str) -> str:
    q = Path(p)
    return str(q if q.is_absolute() else (root / q).resolve())


def _read_yolo_txt(p: Path) -> np.ndarray:
    """Nx5 array [cls, x, y, w, h] in normalized xywh, empty if missing."""
    if not p.exists():
        return np.zeros((0, 5), dtype=np.float32)
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            raise ValueError(f"Malformed label line in {p}: '{ln}'")
        cls = float(parts[0])
        x, y, w, h = map(float, parts[1:5])
        rows.append([cls, x, y, w, h])
    return np.asarray(rows, dtype=np.float32)


# ----------------------------- Dataset ----------------------------- #

class MultiTaskJSONDataset(BaseDataset):
    """
    JSON-driven multi-task dataset.

    Parameters
    ----------
    data_yaml : str | Path
        Path to data.yaml. Must include: root: <dataset_root>.
    split : {'train','val','test'}
        Which split to load from the JSON.
    splits_json : str | Path | None
        Overrides default <root>/splits.json.
    tasks : list[str] | None
        Subset of JSON top-level keys to include. If None, include all known keys.
    imgsz, cache, augment, hyp, ... : passed through to BaseDataset.
    """

    JSON_KEYS = (
        "retinography_segmentation",
        "angiography_segmentation",
        "angiography_detection",
    )

    def __init__(
        self,
        data_yaml: str | Path,
        split: Literal["train", "val", "test"] = "train",
        splits_json: str | Path | None = None,
        tasks: Optional[List[str]] = None,
        imgsz: int = 512,
        cache: bool | str = False,
        augment: bool = True,
        hyp: Dict[str, Any] | None = None,
        prefix: str = "",
        rect: bool = False,
        batch_size: int = 16,
        stride: int = 32,
        pad: float = 0.5,
        single_cls: bool = False,
        classes: List[int] | None = None,
        fraction: float = 1.0,
        channels: int = 3,
    ) -> None:
        # Parse config and stash JSON paths BEFORE calling super().__init__()
        meta = _parse_data_yaml(Path(data_yaml))
        self.root = Path(meta["root"]).expanduser().resolve()
        self.split = split
        self.splits_json = Path(splits_json) if splits_json else (self.root / "splits.json")
        self.include_keys = list(tasks) if tasks else list(self.JSON_KEYS)
        self._records: List[SampleRec] = []  # filled in get_img_files/get_labels

        super().__init__(
            img_path="__json__",  # sentinel, ignored by our overrides
            imgsz=imgsz,
            cache=cache,
            augment=augment,
            hyp=hyp or {},
            prefix=prefix,
            rect=rect,
            batch_size=batch_size,
            stride=stride,
            pad=pad,
            single_cls=single_cls,
            classes=classes,
            fraction=fraction,
            channels=channels,
        )

    # ---------- BaseDataset virtuals ---------- #

    def get_img_files(self, img_path: str | List[str]) -> List[str]:
        """Populate self._records from JSON and return image list aligned with labels."""
        if not self.splits_json.exists():
            raise FileNotFoundError(f"Splits JSON not found: {self.splits_json}")
        manifest = _load_json(self.splits_json)

        recs: List[SampleRec] = []
        for key in self.include_keys:
            if key not in manifest:
                continue
            entries = manifest[key].get(self.split, [])
            for e in entries:
                if key.endswith("segmentation"):
                    im = _abspath(self.root, e["image"])
                    mk = _abspath(self.root, e["mask"])
                    recs.append(SampleRec(im_file=im, tgt_file=mk, is_seg=True,
                                          domain=str(e.get("domain", "")), task_key=key))
                else:
                    im = _abspath(self.root, e["image"])
                    lb = _abspath(self.root, e["label"])
                    recs.append(SampleRec(im_file=im, tgt_file=lb, is_seg=False,
                                          domain=str(e.get("domain", "")), task_key=key))

        if self.fraction < 1.0 and len(recs) > 0:
            keep = max(1, int(round(len(recs) * self.fraction)))
            recs = recs[:keep]

        if not recs:
            raise RuntimeError(f"No samples found for split='{self.split}' in {self.splits_json}")

        self._records = recs
        im_files = [r.im_file for r in recs]
        # BaseDataset will compute cache structures from this list
        return im_files

    def get_labels(self) -> List[Dict[str, Any]]:
        """
        Build labels aligned with `im_files`. Keep normalized xywh for detection.
        Segmentation items carry a 'mask_file' and will create a binary mask in transforms.
        """
        labels: List[Dict[str, Any]] = []
        for r in self._records:
            item: Dict[str, Any] = {
                "im_file": r.im_file,
                "shape": None,                 # not used unless rect=True
                "cls": np.zeros((0, 1), dtype=np.float32),
                "bboxes": np.zeros((0, 4), dtype=np.float32),
                "segments": [],                # polygon segs not used here
                "keypoints": None,
                "normalized": True,
                "bbox_format": "xywh",
                "is_seg": r.is_seg,
                "domain": r.domain,
                "task_key": r.task_key,
                "mask_file": None,
                "label_file": None,
            }
            if r.is_seg:
                item["mask_file"] = r.tgt_file
            else:
                item["label_file"] = r.tgt_file
                rows = _read_yolo_txt(Path(r.tgt_file)) if r.tgt_file else np.zeros((0, 5), dtype=np.float32)
                if rows.size:
                    item["cls"] = rows[:, 0:1]
                    item["bboxes"] = rows[:, 1:5]
            labels.append(item)
        return labels

    def build_transforms(self, hyp: Dict[str, Any] | None = None):
        """
        Return a callable that converts BaseDataset label dict -> final training dict.
        Applies flips, converts to torch tensors, prepares mask if present.
        """
        hyp = hyp or {}
        p_hflip = float(hyp.get("fliplr", 0.0))
        p_vflip = float(hyp.get("flipud", 0.0))

        def _transform(label: Dict[str, Any]) -> Dict[str, Any]:
            # Unpack image loaded by BaseDataset
            img_bgr: np.ndarray = label["img"]  # HxWxC, uint8, BGR
            H, W = img_bgr.shape[:2]

            # Optional mask
            mask = None
            if label.get("is_seg", False) and label.get("mask_file"):
                m = cv2.imread(str(label["mask_file"]), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    raise FileNotFoundError(f"Mask not found: {label['mask_file']}")
                # Resize to the same size as 'label["resized_shape"]' or current image
                if "resized_shape" in label and label["resized_shape"] is not None:
                    h_r, w_r = label["resized_shape"]
                    m = cv2.resize(m, (w_r, h_r), interpolation=cv2.INTER_NEAREST)
                    if (h_r, w_r) != (H, W):
                        img_bgr = cv2.resize(img_bgr, (w_r, h_r), interpolation=cv2.INTER_LINEAR)
                        H, W = h_r, w_r
                else:
                    # align to image
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                mask = (m > 0).astype(np.float32)[None, ...]  # 1xHxW

            # Detection targets (normalized xywh)
            bboxes = label.get("bboxes", np.zeros((0, 4), dtype=np.float32))
            cls = label.get("cls", np.zeros((0, 1), dtype=np.float32))

            # Random flips
            if p_hflip > 0.0 and np.random.rand() < p_hflip:
                img_bgr = np.ascontiguousarray(img_bgr[:, ::-1, :])
                if mask is not None:
                    mask = np.ascontiguousarray(mask[:, :, ::-1])
                if bboxes.size:
                    bboxes[:, 0] = 1.0 - bboxes[:, 0]  # x -> 1-x

            if p_vflip > 0.0 and np.random.rand() < p_vflip:
                img_bgr = np.ascontiguousarray(img_bgr[::-1, :, :])
                if mask is not None:
                    mask = np.ascontiguousarray(mask[:, ::-1, :])
                if bboxes.size:
                    bboxes[:, 1] = 1.0 - bboxes[:, 1]  # y -> 1-y

            # Convert to tensors
            img_rgb = img_bgr[..., ::-1]  # BGR->RGB
            img = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0  # [3,H,W]
            out: Dict[str, Any] = {
                "img": img,  # [3,H,W], float32 in [0,1]
                "cls": torch.from_numpy(cls).long() if cls.size else torch.zeros((0, 1), dtype=torch.long),
                "bboxes": torch.from_numpy(bboxes).float() if bboxes.size else torch.zeros((0, 4), dtype=torch.float32),
                "is_seg": bool(label.get("is_seg", False)),
                "domain": str(label.get("domain", "")),
                "path": str(label.get("im_file", "")),
            }
            if mask is not None:
                out["mask"] = torch.from_numpy(mask).float()
            else:
                out["mask"] = torch.zeros((1, H, W), dtype=torch.float32)

            return out

        return _transform

    # ---------- Optional: custom collate ---------- #

def multitask_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate mixed seg/det items with a unified interface."""
    imgs = torch.stack([b["img"] for b in batch], 0)
    is_seg = torch.tensor([b["is_seg"] for b in batch], dtype=torch.bool)
    domains = [b["domain"] for b in batch]
    paths = [b["path"] for b in batch]
    masks = torch.stack([b["mask"] for b in batch], 0)  # always present (zeros for det)

    # concatenate detection labels with batch indices
    all_boxes, all_cls, all_bi = [], [], []
    for i, b in enumerate(batch):
        if b["bboxes"].numel():
            n = b["bboxes"].shape[0]
            all_boxes.append(b["bboxes"])
            all_cls.append(b["cls"])
            all_bi.append(torch.full((n,), i, dtype=torch.long))
    bboxes = torch.cat(all_boxes, 0) if all_boxes else torch.zeros((0, 4), dtype=torch.float32)
    cls = torch.cat(all_cls, 0) if all_cls else torch.zeros((0, 1), dtype=torch.long)
    batch_idx = torch.cat(all_bi, 0) if all_bi else torch.zeros((0,), dtype=torch.long)

    return {
        "img": imgs, "is_seg": is_seg, "domain": domains, "mask": masks,
        "bboxes": bboxes, "cls": cls, "batch_idx": batch_idx, "paths": paths,
    }


def build_dataloader(
    dataset: MultiTaskJSONDataset,
    batch_size: int,
    workers: int = 8,
    shuffle: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Wrap the dataset with a DataLoader using our collate."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        collate_fn=multitask_collate_fn,
        drop_last=drop_last,
    )
