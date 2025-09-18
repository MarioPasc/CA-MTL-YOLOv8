# camtl_yolo/engine/train.py
from __future__ import annotations

import math
import random
from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from camtl_yolo.model.dataset import MultiTaskJSONDataset, build_dataloader
from camtl_yolo.model.engine.val import CAMTLValidator
from camtl_yolo.model.model import CAMTL_YOLO
from camtl_yolo.model.utils.normalization import set_bn_domain
from camtl_yolo.model.utils.samplers import AlternatingLoader, map_domain_name
from camtl_yolo.model.tasks import configure_task, select_dataset_tasks_for_mode
from camtl_yolo.external.ultralytics.ultralytics.engine.trainer import BaseTrainer
from camtl_yolo.external.ultralytics.ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from camtl_yolo.external.ultralytics.ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


class CAMTLTrainer(BaseTrainer):
    """
    Trainer for CA-MTL-YOLOv8.

    - Task-aware datasets: DomainShift1 (seg-only) and CAMTL (det:seg ratio).
    - DualBN domain switching per batch.
    - Uses BaseTrainer infra (AMP, EMA, schedulers, checkpointing).
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks=None):
        overrides = overrides or {}
        super().__init__(cfg, overrides, _callbacks)

    # ------------------------ Dataset & Loader ------------------------ #

    def get_dataset(self) -> Dict[str, Any]:
        """Return dataset meta dict expected by BaseTrainer."""
        import yaml

        data_yaml = Path(self.args.data)
        with data_yaml.open("r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)

        names = meta.get("names") or meta.get("classes") or {0: "object"}
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        channels = int(meta.get("channels", 3))
        nc = int(meta.get("nc", len(names)))

        # We pass the path and let get_dataloader build proper datasets/loaders
        return {
            "data_yaml": str(data_yaml),
            "train": str(data_yaml),
            "val": str(data_yaml),
            "names": names,
            "nc": nc,
            "channels": channels,
        }

    def _build_mt_dataset(
        self,
        data_yaml: str,
        split: str,
        tasks: List[str],
        filter_is: str | None,
        imgsz: int,
        batch: int,
        shuffle: bool,
    ):
        ds = MultiTaskJSONDataset(
            data_yaml=data_yaml,
            split=split,
            tasks=tasks,
            filter_is=filter_is,  # "seg" or "det" or None
            imgsz=imgsz,
            augment=(split == "train"),
            hyp=self.args,  # reuse same hyp dict
            batch_size=batch,
            rect=False,
            channels=self.data["channels"],
        )
        dl = build_dataloader(
            ds, batch_size=batch, workers=self.args.workers if split == "train" else max(1, self.args.workers // 2), shuffle=shuffle
        )
        return ds, dl

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Construct dataloader(s) per active task. Returns an AlternatingLoader in CAMTL train."""
        assert mode in {"train", "val"}
        with torch_distributed_zero_first(rank):
            from camtl_yolo.model.tasks import select_dataset_tasks_for_mode

            task = str(getattr(self, "active_task", None) or self._get_active_task_from_model())
            task_keys = select_dataset_tasks_for_mode(task)[mode]
            imgsz = self.args.imgsz
            shuffle = mode == "train"

            if task == "DomainShift1":
                # Segmentation-only stream from source (retinography)
                _, seg_loader = self._build_mt_dataset(
                    data_yaml=dataset_path,
                    split=mode,
                    tasks=task_keys,
                    filter_is="seg",
                    imgsz=imgsz,
                    batch=batch_size,
                    shuffle=shuffle,
                )
                seg_loader.ratio = (0, 1)  # for logging
                return seg_loader

            if task == "CAMTL" and mode == "train":
                # Two loaders: detection(target) and segmentation(target)
                _, det_loader = self._build_mt_dataset(
                    data_yaml=dataset_path,
                    split="train",
                    tasks=["angiography_detection"],
                    filter_is="det",
                    imgsz=imgsz,
                    batch=batch_size,
                    shuffle=True,
                )
                _, seg_loader = self._build_mt_dataset(
                    data_yaml=dataset_path,
                    split="train",
                    tasks=["angiography_segmentation"],
                    filter_is="seg",
                    imgsz=imgsz,
                    batch=batch_size,
                    shuffle=True,
                )
                # Ratio X:Y from args or YAML
                ratio = getattr(self.args, "camtl_ratio", None) or getattr(self.model, "yaml", {}).get("CAMTL_RATIO", [1, 1])
                rx, ry = int(ratio[0]), int(ratio[1])
                alt = AlternatingLoader(det_loader, seg_loader, ratio=(rx, ry))
                alt.num_workers = getattr(det_loader, "num_workers", 0) + getattr(seg_loader, "num_workers", 0)
                alt.batch_size = batch_size
                alt.ratio = (rx, ry)
                return alt

            # For CAMTL val: single union loader, no alternation
            _, union_loader = self._build_mt_dataset(
                data_yaml=dataset_path, split="val", tasks=task_keys, filter_is=None, imgsz=imgsz, batch=batch_size, shuffle=False
            )
            union_loader.ratio = (1, 1)
            return union_loader

    # ------------------------ Model & Attrs ------------------------ #

    def _get_active_task_from_model(self) -> str:
        try:
            return str(getattr(self.model, "yaml", {}).get("TASK", "DomainShift1"))
        except Exception:
            return "DomainShift1"

    def get_model(self, cfg: str | dict | None = None, weights: str | Path | None = None, verbose: bool = True):
        """Instantiate CAMTL model."""
        model = CAMTL_YOLO(cfg=cfg or self.args.model, ch=self.data["channels"], nc=self.data["nc"], verbose=verbose and RANK in {-1, 0})
        # keep a copy of active task to drive dataloaders
        self.active_task = str(model.yaml.get("TASK", "DomainShift1"))
        return model

    def set_model_attributes(self):
        """Attach common attributes for logging."""
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args

    # ------------------------ Preprocess & Loss Routing ------------------------ #

    def preprocess_batch(self, batch: dict) -> dict:
        """Move to device, normalize, and set BN domain based on batch."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        batch["img"] = batch["img"].float() / 255.0

        # Expect single-domain batches; set DualBN switch
        if "bn_domain" in batch and len(batch["bn_domain"]):
            # choose the first; enforce consistency
            dom = map_domain_name(batch["bn_domain"][0])
            set_bn_domain(dom)

        return batch

    # ------------------------ Validator & Loss Names ------------------------ #

    def get_validator(self):
        """Return CAMTL validator."""
        # losses: det, seg, cons, align, l2sp, total
        self.loss_names = ("det", "seg", "cons", "align", "l2sp", "total")
        return CAMTLValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def label_loss_items(self, loss_items: torch.Tensor | None = None, prefix: str = "train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            vals = [round(float(x), 6) for x in (loss_items.tolist() if torch.is_tensor(loss_items) else loss_items)]
            return dict(zip(keys, vals))
        return keys

    # ------------------------ Save Hook ------------------------ #

    def save_model(self):
        """Augment BaseTrainer saving with task-aware checkpoint."""
        super().save_model()
        try:
            unwrap_model(self.model).save_task_checkpoint(self.wdir, epoch=self.epoch, optimizer=self.optimizer)
        except Exception as e:
            LOGGER.warning(f"Task checkpoint save failed: {e}")

    # ------------------------ Progress String ------------------------ #

    def progress_string(self):
        s = super().progress_string()
        if hasattr(self.train_loader, "ratio"):
            rx, ry = getattr(self.train_loader, "ratio", (1, 1))
            s += f"  [det:seg={rx}:{ry}]"
        return s
