"""
Wrapper around Ultralytics YOLOv8 detection loss to fit our batch dict.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple

import torch
from torch import Tensor, nn
from camtl_yolo.external.ultralytics.ultralytics.utils.loss import v8DetectionLoss


class DetectionLoss(nn.Module):
    """
    Thin wrapper to call Ultralytics' v8DetectionLoss with our (preds, batch) convention.
    Expects batch keys: img, bboxes [M,4] xywh normalized, cls [M,1], batch_idx [M].
    """

    def __init__(self, model, hyp: Dict[str, Any] | None = None) -> None:
        super().__init__()
        # Ultralytics loss reads model attributes (assigner, stride, dfl, etc.)
        self.crit = v8DetectionLoss(model)
        self.hyp = hyp or {}

    def forward(self, det_preds: Any, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Returns total_loss, item_dict. Delegates to Ultralytics implementation.
        """
        loss, loss_items_raw = self.crit(det_preds, batch)  # type: ignore[arg-type]
        if isinstance(loss_items_raw, dict):
            # ensure tensors
            loss_items: Dict[str, Tensor] = {
                k: (v if isinstance(v, torch.Tensor) else torch.tensor(v, device=loss.device))
                for k, v in loss_items_raw.items()
            }
        else:
            # YOLOv8 returns box, cls, and dfl, in that order, inside a Tensor. Convert to dict.
            li = loss_items_raw.detach()
            loss_items: Dict[str, Tensor] = {"box": li[0], "cls": li[1], "dfl": li[2]}
        return loss, loss_items
