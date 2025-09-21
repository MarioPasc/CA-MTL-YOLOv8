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

from camtl_yolo.model.utils.downsample_mask import downsample_mask_prob 
from camtl_yolo.model.utils import plotting as plot_utils

try:
    from camtl_yolo.external.ultralytics.ultralytics.data.dataset import BaseDataset
except Exception as e:  # pragma: no cover
    raise ImportError("Ultralytics BaseDataset not found under camtl_yolo.external.ultralytics") from e

import os

from camtl_yolo.external.ultralytics.ultralytics.utils import LOGGER

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
        if os.path.exists(self.splits_json):
            self.splits_json = self.splits_json.resolve()
        if not self.splits_json.exists():
            from camtl_yolo.data import holdout
            LOGGER.warning(f" 'splits.json' not found under {self.splits_json}")
            LOGGER.warning("Generating splits.json with default 70/30 train/val fractions for all tasks.")
            json_path = self.splits_json
            LOGGER.warning(f"Writing splits JSON to: {json_path}")
            splits = holdout.build_all(
                config={
                    'output_dir': self.root,
                    'holdout': {
                        'retinography_segmentation': {
                            'train': 0.7,
                            'val': 0.3,
                            'test': 0.0
                        },
                        'angiography_segmentation': {
                            'train': 0.7,
                            'val': 0.3,
                            'test': 0.0
                        },
                        'angiography_detection': {
                            'train': 0.7,
                            'val': 0.3,
                            'test': 0.0
                        }
                    }
                    }, 
                seed=42)
            json_path = self.splits_json
            json_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
        self.include_keys = list(tasks) if tasks else list(self.JSON_KEYS)
        self.filter_is = filter_is
        self._records: List[SampleRec] = []
        # Debug counter for optional augmentation saves
        self._aug_save_count: int = 0
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

            # Keep pre-augmentation copies (after any initial size alignment, before flips)
            img_bgr_pre = img_bgr.copy()
            mask_pre = None if mask is None else mask.copy()

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

            # Optional debugging: save pre/post augmented image and mask
            try:
                out_dir = os.getenv("CAMTL_SAVE_AUG_MASKS", "")
                if out_dir:
                    max_saves = int(os.getenv("CAMTL_SAVE_MAX", "0") or 0)
                    do_save = max_saves <= 0 or (self._aug_save_count < max_saves)
                    if do_save:
                        Path(out_dir).mkdir(parents=True, exist_ok=True)
                        stem = Path(label.get("im_file", "img")).stem
                        # Compute downsampled masks for pre and post (if segmentation)
                        pre_m2d = mask_pre[0] if mask_pre is not None else None
                        post_m2d = mask[0] if mask is not None else None

                        def ds_all(m2d: Optional[np.ndarray]):
                            if m2d is None:
                                return None, None, None
                            m3 = downsample_mask_prob(m2d, stride=8,  method="avgpool")
                            m4 = downsample_mask_prob(m2d, stride=16, method="avgpool")
                            m5 = downsample_mask_prob(m2d, stride=32, method="avgpool")
                            return m3, m4, m5

                        pre_p3, pre_p4, pre_p5 = ds_all(pre_m2d)
                        post_p3, post_p4, post_p5 = ds_all(post_m2d)

                        # Save composite 2x4 grid visualization
                        out_grid = Path(out_dir) / f"{stem}_aug_grid.png"
                        saved_path = plot_utils.save_aug_debug_grid(
                            out_grid,
                            pre_img_bgr=img_bgr_pre,
                            pre_mask2d=pre_m2d,
                            pre_p3=pre_p3,
                            pre_p4=pre_p4,
                            pre_p5=pre_p5,
                            post_img_bgr=img_bgr,
                            post_mask2d=post_m2d,
                            post_p3=post_p3,
                            post_p4=post_p4,
                            post_p5=post_p5,
                        )
                        if saved_path and Path(saved_path).exists():
                            LOGGER.debug(f"[aug-debug] Saved grid to: {saved_path}")
                        else:
                            LOGGER.warning(f"[aug-debug] Failed to save grid to: {out_grid}")
                        self._aug_save_count += 1
                else:
                    # Saving is enabled but we've reached the cap; log only once per epoch if needed
                    pass
            except Exception as e:  # pragma: no cover
                LOGGER.warning(f"[aug-debug] Exception while saving aug grid for {label.get('im_file','')}: {e}")

            # Normalize channel layout to contiguous RGB
            if img_bgr.ndim == 2:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
            elif img_bgr.shape[2] == 4:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            chw = np.ascontiguousarray(img_rgb.transpose(2, 0, 1))
            img = torch.from_numpy(chw).float() / 255.0

            out: Dict[str, Any] = {
                "img": img,
                "cls": torch.from_numpy(cls).long() if cls.size else torch.zeros((0,1), dtype=torch.long),
                "bboxes": torch.from_numpy(bboxes).float() if bboxes.size else torch.zeros((0,4), dtype=torch.float32),
                "is_seg": bool(label.get("is_seg", False)),
                "domain": str(label.get("domain","")),
                "bn_domain": str(label.get("bn_domain", _map_bn_domain(label.get("domain","")))),
                "path": str(label.get("im_file","")),
            }

            if mask is None:
                out["mask"]   = torch.zeros((1, H, W), dtype=torch.float32)
                out["mask_p3"]= torch.zeros((1, H//8,  W//8 ), dtype=torch.float32)
                out["mask_p4"]= torch.zeros((1, H//16, W//16), dtype=torch.float32)
                out["mask_p5"]= torch.zeros((1, H//32, W//32), dtype=torch.float32)
            else:
                out["mask"] = torch.from_numpy(mask).float()  # (1,H,W)
                m2d = mask[0]  # (H,W) np.float32 in {0,1}
                # NOTE: Hardcoded strides, please refactor if changed in model. I know, bad practice.
                m3 = downsample_mask_prob(m2d, stride=8,  method="avgpool")   # (H/8, W/8) float
                m4 = downsample_mask_prob(m2d, stride=16, method="avgpool")
                m5 = downsample_mask_prob(m2d, stride=32, method="avgpool")
                out["mask_p3"] = torch.from_numpy(m3[None, ...]).float()
                out["mask_p4"] = torch.from_numpy(m4[None, ...]).float()
                out["mask_p5"] = torch.from_numpy(m5[None, ...]).float()

            return out

        return _transform


def multitask_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    imgs = torch.stack([b["img"] for b in batch], 0)
    is_seg = torch.tensor([b["is_seg"] for b in batch], dtype=torch.bool)
    domains = [b["domain"] for b in batch]
    bn_domains = [b.get("bn_domain", _map_bn_domain(b["domain"])) for b in batch]
    paths = [b["path"] for b in batch]
    masks    = torch.stack([b["mask"]    for b in batch], 0)
    masks_p3 = torch.stack([b["mask_p3"] for b in batch], 0)
    masks_p4 = torch.stack([b["mask_p4"] for b in batch], 0)
    masks_p5 = torch.stack([b["mask_p5"] for b in batch], 0)

    all_boxes, all_cls, all_bi = [], [], []
    for i, b in enumerate(batch):
        if b["bboxes"].numel():
            n = b["bboxes"].shape[0]
            all_boxes.append(b["bboxes"]); all_cls.append(b["cls"])
            all_bi.append(torch.full((n,), i, dtype=torch.long))
    bboxes = torch.cat(all_boxes, 0) if all_boxes else torch.zeros((0,4), dtype=torch.float32)
    cls = torch.cat(all_cls, 0) if all_cls else torch.zeros((0,1), dtype=torch.long)
    batch_idx = torch.cat(all_bi, 0) if all_bi else torch.zeros((0,), dtype=torch.long)

    return {
        "img": imgs, "is_seg": is_seg, "domain": domains, "bn_domain": bn_domains, "paths": paths,
        "mask": masks, "mask_p3": masks_p3, "mask_p4": masks_p4, "mask_p5": masks_p5,
        "bboxes": bboxes, "cls": cls, "batch_idx": batch_idx,
    }

def build_dataloader(dataset: MultiTaskJSONDataset, batch_size: int, workers: int = 8,
                     shuffle: bool = True, drop_last: bool = False) -> DataLoader:
    from torch.utils.data import DataLoader
    dl = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True,
        collate_fn=multitask_collate_fn, drop_last=drop_last, persistent_workers=(workers > 0),
    )
    # Ultralytics calls train_loader.reset() at close_mosaic. Provide a no-op.
    if not hasattr(dl, "reset"):
        def _reset() -> None:
            # If you later add dataset.close_mosaic(), it will be honored here.
            if hasattr(dl, "dataset") and hasattr(dl.dataset, "close_mosaic"):
                try:
                    dl.dataset.close_mosaic()
                except Exception:
                    pass
            return None
        setattr(dl, "reset", _reset)
    return dl
