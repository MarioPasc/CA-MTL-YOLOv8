from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from camtl_yolo.external.ultralytics.ultralytics.data.dataset import BaseDataset
except Exception as e:  # pragma: no cover
    raise ImportError("Ultralytics BaseDataset not found under camtl_yolo.external.ultralytics") from e

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampleRec:
    im_file: str
    tgt_file: Optional[str]
    is_seg: bool
    domain: str
    task_key: str


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_data_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # type: ignore
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _abspath(root: Path, p: str) -> str:
    q = Path(p)
    return str(q if q.is_absolute() else (root / q).resolve())


def _read_yolo_txt(p: Path) -> np.ndarray:
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
        cls = float(parts[0]); x, y, w, h = map(float, parts[1:5])
        rows.append([cls, x, y, w, h])
    return np.asarray(rows, dtype=np.float32)


def _map_bn_domain(dom: str) -> str:
    d = (dom or "").lower()
    if "retino" in d or "fundus" in d or "source" in d:
        return "source"
    return "target"


class MultiTaskJSONDataset(BaseDataset):
    JSON_KEYS = ("retinography_segmentation", "angiography_segmentation", "angiography_detection")

    def __init__(
        self,
        data_yaml: str | Path,
        split: Literal["train", "val", "test"] = "train",
        splits_json: str | Path | None = None,
        tasks: Optional[List[str]] = None,
        filter_is: Optional[Literal["seg", "det"]] = None,  # NEW: quick filter by task type
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
        meta = _parse_data_yaml(Path(data_yaml))
        self.root = Path(meta["root"]).expanduser().resolve()
        self.split = split
        self.splits_json = Path(splits_json) if splits_json else (self.root / "splits.json")
        self.include_keys = list(tasks) if tasks else list(self.JSON_KEYS)
        self.filter_is = filter_is
        self._records: List[SampleRec] = []
        super().__init__(
            img_path="__json__", imgsz=imgsz, cache=cache, augment=augment, hyp=hyp or {}, prefix=prefix,
            rect=rect, batch_size=batch_size, stride=stride, pad=pad, single_cls=single_cls,
            classes=classes, fraction=fraction, channels=channels,
        )

    # BaseDataset hooks
    def get_img_files(self, img_path: str | List[str]) -> List[str]:
        if not self.splits_json.exists():
            raise FileNotFoundError(f"Splits JSON not found: {self.splits_json}")
        manifest = _load_json(self.splits_json)

        recs: List[SampleRec] = []
        for key in self.include_keys:
            if key not in manifest:
                continue
            for e in manifest[key].get(self.split, []):
                if key.endswith("segmentation"):
                    if self.filter_is == "det":
                        continue
                    im = _abspath(self.root, e["image"])
                    mk = _abspath(self.root, e["mask"])
                    recs.append(SampleRec(im_file=im, tgt_file=mk, is_seg=True,
                                          domain=str(e.get("domain", "")), task_key=key))
                else:
                    if self.filter_is == "seg":
                        continue
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
        return [r.im_file for r in recs]

    def get_labels(self) -> List[Dict[str, Any]]:
        labels: List[Dict[str, Any]] = []
        for r in self._records:
            item: Dict[str, Any] = {
                "im_file": r.im_file,
                "shape": None,
                "cls": np.zeros((0, 1), dtype=np.float32),
                "bboxes": np.zeros((0, 4), dtype=np.float32),
                "segments": [],
                "keypoints": None,
                "normalized": True,
                "bbox_format": "xywh",
                "is_seg": r.is_seg,
                "domain": r.domain,
                "bn_domain": _map_bn_domain(r.domain),  # NEW: explicit BN domain tag
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
        hyp = hyp or {}
        p_hflip = float(hyp.get("fliplr", 0.0))
        p_vflip = float(hyp.get("flipud", 0.0))

        def _transform(label: Dict[str, Any]) -> Dict[str, Any]:
            img_bgr: np.ndarray = label["img"]
            H, W = img_bgr.shape[:2]
            mask = None
            if label.get("is_seg", False) and label.get("mask_file"):
                m = cv2.imread(str(label["mask_file"]), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    raise FileNotFoundError(f"Mask not found: {label['mask_file']}")
                if "resized_shape" in label and label["resized_shape"] is not None:
                    h_r, w_r = label["resized_shape"]
                    m = cv2.resize(m, (w_r, h_r), interpolation=cv2.INTER_NEAREST)
                    if (h_r, w_r) != (H, W):
                        img_bgr = cv2.resize(img_bgr, (w_r, h_r), interpolation=cv2.INTER_LINEAR)
                        H, W = h_r, w_r
                else:
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                mask = (m > 0).astype(np.float32)[None, ...]

            bboxes = label.get("bboxes", np.zeros((0, 4), dtype=np.float32))
            cls = label.get("cls", np.zeros((0, 1), dtype=np.float32))

            if p_hflip > 0.0 and np.random.rand() < p_hflip:
                img_bgr = np.ascontiguousarray(img_bgr[:, ::-1, :])
                if mask is not None:
                    mask = np.ascontiguousarray(mask[:, :, ::-1])
                if bboxes.size:
                    bboxes[:, 0] = 1.0 - bboxes[:, 0]
            if p_vflip > 0.0 and np.random.rand() < p_vflip:
                img_bgr = np.ascontiguousarray(img_bgr[::-1, :, :])
                if mask is not None:
                    mask = np.ascontiguousarray(mask[:, ::-1, :])
                if bboxes.size:
                    bboxes[:, 1] = 1.0 - bboxes[:, 1]

            # Normalize channel layout to contiguous RGB without negative strides
            if img_bgr.ndim == 2:  # grayscale → BGR
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
            elif img_bgr.shape[2] == 4:  # BGRA → BGR
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # contiguous by construction
            LOGGER.info(
                f"Image {label.get('im_file', '')}: shape={img_rgb.shape}, "
                f"bboxes={bboxes.shape if bboxes is not None else None}, "
                f"mask={mask.shape if mask is not None else None}"
            )
            chw = np.ascontiguousarray(img_rgb.transpose(2, 0, 1))  # ensure positive strides
            img = torch.from_numpy(chw).float() / 255.0
            out: Dict[str, Any] = {
                "img": img,
                "cls": torch.from_numpy(cls).long() if cls.size else torch.zeros((0, 1), dtype=torch.long),
                "bboxes": torch.from_numpy(bboxes).float() if bboxes.size else torch.zeros((0, 4), dtype=torch.float32),
                "is_seg": bool(label.get("is_seg", False)),
                "domain": str(label.get("domain", "")),
                "bn_domain": str(label.get("bn_domain", _map_bn_domain(label.get("domain", "")))),  # propagate
                "path": str(label.get("im_file", "")),
            }
            out["mask"] = torch.from_numpy(mask).float() if mask is not None else torch.zeros((1, H, W), dtype=torch.float32)
            return out

        return _transform


def multitask_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    imgs = torch.stack([b["img"] for b in batch], 0)
    is_seg = torch.tensor([b["is_seg"] for b in batch], dtype=torch.bool)
    domains = [b["domain"] for b in batch]
    bn_domains = [b.get("bn_domain", _map_bn_domain(b["domain"])) for b in batch]
    paths = [b["path"] for b in batch]
    masks = torch.stack([b["mask"] for b in batch], 0)

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
        "img": imgs, "is_seg": is_seg, "domain": domains, "bn_domain": bn_domains, "mask": masks,
        "bboxes": bboxes, "cls": cls, "batch_idx": batch_idx, "paths": paths,
    }


def build_dataloader(dataset: MultiTaskJSONDataset, batch_size: int, workers: int = 8,
                     shuffle: bool = True, drop_last: bool = False) -> DataLoader:
    from torch.utils.data import DataLoader
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True,
        collate_fn=multitask_collate_fn, drop_last=drop_last,
    )
