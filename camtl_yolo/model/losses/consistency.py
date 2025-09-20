# camtl_yolo/model/losses/consistency.py
"""
Consistency loss: encourage segmentation to be foreground within detection boxes.
Uses ground-truth detection boxes in the batch to build pseudo-masks.
"""
from __future__ import annotations
from typing import Dict, Tuple, Sequence, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _boxes_to_masks(
    boxes_xywh: torch.Tensor,  # [N,4] normalized
    batch_idx: torch.Tensor,   # [N]
    B: int,
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Rasterize normalized xywh boxes into binary masks per image.
    Returns Tensor [B,1,H,W].
    """
    device = boxes_xywh.device
    masks = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)
    if boxes_xywh.numel() == 0:
        return masks
    # denormalize to pixel coords
    cx = boxes_xywh[:, 0] * W
    cy = boxes_xywh[:, 1] * H
    bw = boxes_xywh[:, 2] * W
    bh = boxes_xywh[:, 3] * H
    x1 = (cx - bw / 2.0).clamp(0, W - 1).long()
    y1 = (cy - bh / 2.0).clamp(0, H - 1).long()
    x2 = (cx + bw / 2.0).clamp(0, W - 1).long()
    y2 = (cy + bh / 2.0).clamp(0, H - 1).long()

    for i in range(boxes_xywh.shape[0]):
        b = int(batch_idx[i].item())
        masks[b, 0, y1[i] : y2[i] + 1, x1[i] : x2[i] + 1] = 1.0
    return masks


class ConsistencyMaskFromBoxes(nn.Module):
    """
    Builds pseudo foreground masks from detection boxes and aligns the segmentation output.

    Parameters
    ----------
    weight : float
        λ for this loss term.
    loss : {"bce","dice"}
        Loss used to match seg probabilities to pseudo mask.
    """

    def __init__(self, weight: float = 0.1, loss: str = "bce") -> None:
        super().__init__()
        self.weight = float(weight)
        self.use_bce = (loss.lower() == "bce")
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self,
        seg_preds: Union[torch.Tensor, Sequence[torch.Tensor], Dict[str, torch.Tensor]],
        batch: Dict[str, torch.Tensor | list],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        seg_preds:
            - logits Tensor [B,1,H,W], or
            - list/tuple of logits at various scales, or
            - dict from SegHeadMulti with keys like {'p3','p4','p5','full'} (logits).
        We use 'full' if available, else the largest spatial tensor.

        batch: needs keys 'img' [B,3,H,W], 'bboxes' [M,4] xywh, 'batch_idx' [M], 'is_seg' [B]
        """
        # --- pick prediction tensor ---
        if isinstance(seg_preds, dict):
            if "full" in seg_preds and torch.is_tensor(seg_preds["full"]):
                preds_list = [seg_preds["full"]]
            else:
                preds_list = [v for v in seg_preds.values() if torch.is_tensor(v)]
        elif isinstance(seg_preds, (list, tuple)):
            preds_list = [t for t in seg_preds if torch.is_tensor(t)]
        elif torch.is_tensor(seg_preds):
            preds_list = [seg_preds]
        else:
            raise TypeError(f"Unsupported seg_preds type: {type(seg_preds)}")

        if not preds_list:
            # nothing to align
            z = torch.zeros((), device=batch["img"].device)
            return z, {"cons_loss": z}

        # ensure 4D and pick highest resolution
        preds_list = [t if t.ndim == 4 else t.unsqueeze(1) for t in preds_list]
        preds_list.sort(key=lambda t: int(t.shape[-1]) * int(t.shape[-2]), reverse=True)
        pred = preds_list[0]  # logits
        B, _, H, W = pred.shape
        device = pred.device

        # --- select detection-domain images ---
        is_seg_raw = batch.get("is_seg", None)
        if isinstance(is_seg_raw, torch.Tensor):
            is_seg = is_seg_raw.to(device=device, dtype=torch.bool)
        else:
            is_seg = torch.as_tensor(is_seg_raw, device=device, dtype=torch.bool)
        det_mask = ~is_seg
        if det_mask.sum() == 0:
            z = torch.zeros((), device=device)
            return z, {"cons_loss": z}

        # --- boxes and per-image presence ---
        boxes = batch["bboxes"].to(device=device) if batch["bboxes"].numel() else torch.zeros((0, 4), device=device)
        bi = batch["batch_idx"].to(device=device) if batch["batch_idx"].numel() else torch.zeros((0,), dtype=torch.long, device=device)

        # pseudo-mask at pred resolution
        pseudo = _boxes_to_masks(boxes, bi, B=B, H=H, W=W)  # [B,1,H,W]

        has_box = torch.zeros(B, dtype=torch.bool, device=device)
        if bi.numel():
            has_box.scatter_(0, bi.clamp_min(0).clamp_max(B - 1), True)

        eff = det_mask & has_box  # effective images: detection domain and with boxes
        if not eff.any():
            z = torch.zeros((), device=device)
            return z, {"cons_loss": z}

        eff4 = eff.view(B, 1, 1, 1).float()
        pred_logits = pred * eff4
        pseudo = pseudo * eff4

        # --- loss ---
        if self.use_bce:
            loss_raw = self.bce(pred_logits, pseudo)
        else:
            prob = torch.sigmoid(pred_logits)
            eps = 1e-6
            inter = (prob * pseudo).sum(dim=(1, 2, 3))
            denom = prob.sum(dim=(1, 2, 3)) + pseudo.sum(dim=(1, 2, 3)) + eps
            dice = (2.0 * inter + eps) / denom
            loss_raw = (1.0 - dice[eff]).mean()

        loss = self.weight * loss_raw
        return loss, {"cons_loss": loss.detach()}
