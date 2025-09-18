# camtl_yolo/engine/val.py
from __future__ import annotations

from copy import copy
from typing import Any, Dict

import torch
import torch.nn.functional as F

from camtl_yolo.train.normalization import set_bn_domain
from camtl_yolo.train.samplers import map_domain_name
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import smart_inference_mode, unwrap_model


class CAMTLValidator(BaseValidator):
    """
    Minimal validator for CA-MTL-YOLOv8.

    Computes:
      - detection loss on det batches
      - segmentation loss and Dice on seg batches
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "detect+segment"

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255.0
        # set BN domain for this batch
        if "bn_domain" in batch and len(batch["bn_domain"]):
            set_bn_domain(map_domain_name(batch["bn_domain"][0]))
        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        self.names = getattr(model, "names", {0: "object"})
        self.nc = len(self.names)
        self.loss_det = 0.0
        self.loss_seg = 0.0
        self.loss_cons = 0.0
        self.loss_align = 0.0
        self.loss_l2sp = 0.0
        self.count_det = 0
        self.count_seg = 0
        self.seg_dice = 0.0
        self.seen = 0

    def _dice(self, p: torch.Tensor, y: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        # p: (B,1,H,W) or (B,*,H,W), sigmoid not applied
        p = (p.sigmoid() > 0.5).float()
        y = (y > 0.5).float()
        inter = (p * y).sum(dim=(1, 2, 3))
        union = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
        return ((2 * inter + eps) / (union + eps)).mean()

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        self.training = trainer is not None
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            if trainer.args.compile and hasattr(model, "_orig_mod"):
                model = model._orig_mod
            model = model.half() if self.args.half else model.float()
            self.loss = torch.zeros(len(trainer.loss_names), device=self.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            raise RuntimeError("Standalone CAMTLValidator inference is not implemented")

        self.run_callbacks("on_val_start")
        self.init_metrics(unwrap_model(model))

        for i, batch in enumerate(self.dataloader):
            self.run_callbacks("on_val_batch_start")
            batch = self.preprocess(batch)
            preds = model(batch["img"])
            # Model must expose .loss(batch, preds) -> (loss_total, items_tensor)
            loss_total, items = unwrap_model(model).loss(batch, preds)
            # items order: det, seg, cons, align, l2sp, total
            det, seg, cons, align, l2sp, total = [float(x) for x in items.tolist()]
            is_seg_batch = bool(batch.get("is_seg", torch.tensor([False])).any().item())
            if is_seg_batch:
                self.loss_seg += seg
                self.count_seg += 1
                # try to compute dice on primary mask head
                if isinstance(preds, (tuple, list)) and len(preds) >= 2:
                    seg_pred = preds[1]
                else:
                    seg_pred = preds
                # expect (B,1,H,W)
                if isinstance(seg_pred, (list, tuple)):
                    seg_pred = seg_pred[0]
                if seg_pred.ndim == 4 and "mask" in batch:
                    self.seg_dice += float(self._dice(seg_pred, batch["mask"]))
            else:
                self.loss_det += det
                self.count_det += 1
            self.loss_cons += cons
            self.loss_align += align
            self.loss_l2sp += l2sp
            self.seen += batch["img"].shape[0]
            self.run_callbacks("on_val_batch_end")

        # Averages
        md = {
            "val/det": (self.loss_det / max(1, self.count_det)),
            "val/seg": (self.loss_seg / max(1, self.count_seg)),
            "val/cons": (self.loss_cons / max(1, self.count_det + self.count_seg)),
            "val/align": (self.loss_align / max(1, self.count_det + self.count_seg)),
            "val/l2sp": (self.loss_l2sp / max(1, self.count_det + self.count_seg)),
            "val/total": (
                (self.loss_det + self.loss_seg + self.loss_cons + self.loss_align + self.loss_l2sp)
                / max(1, self.count_det + self.count_seg)
            ),
            "val/dice": (self.seg_dice / max(1, self.count_seg)),
        }

        # Pack results for trainer
        self.run_callbacks("on_val_end")
        if self.training:
            # Reuse BaseTrainer labeler for consistency of CSV columns
            return {**md}
        return md
