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
        loss, loss_items = self.crit(det_preds, batch)  # type: ignore[arg-type]
        if isinstance(loss_items, dict):
            # ensure tensors
            loss_items = {k: (v if isinstance(v, torch.Tensor) else torch.tensor(v, device=loss.device))
                          for k, v in loss_items.items()}
        else:
            loss_items = {"det_loss": loss.detach()} # type: ignore[assignment]
        return loss, loss_items # type: ignore[return-value]
