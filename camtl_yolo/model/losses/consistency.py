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
        seg_preds: Union[torch.Tensor, Sequence[torch.Tensor]],
        batch: Dict[str, torch.Tensor | list],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        seg_preds : Tensor [B,1,H,W] or list of Tensors (any scales). We use the highest resolution.
        batch : needs keys img [B,3,H,W], bboxes [M,4] xywh, batch_idx [M], is_seg [B]
        """
        preds_list: List[torch.Tensor] = [seg_preds] if isinstance(seg_preds, torch.Tensor) else list(seg_preds)
        # pick finest resolution
        preds_list.sort(key=lambda t: t.shape[-1] * t.shape[-2], reverse=True)
        pred = preds_list[0]
        B, _, H, W = pred.shape
        device = pred.device

        # select images coming from detection domain (is_seg == False)
        is_seg = batch["is_seg"].to(device=device)
        det_mask = (~is_seg).to(torch.bool)
        if det_mask.sum() == 0:
            return torch.zeros((), device=device), {"cons_loss": torch.zeros((), device=device)}

        boxes = batch["bboxes"].to(device=device)
        bi = batch["batch_idx"].to(device=device)
        # build pseudo mask and restrict to detection images only
        pseudo = _boxes_to_masks(boxes, bi, B=B, H=H, W=W)

        # mask out non-detection images from loss by zeroing targets and predictions contribution
        # keep computation over full batch to avoid index gymnastics
        det_mask_4d = det_mask.view(B, 1, 1, 1).float()
        pseudo = pseudo * det_mask_4d
        pred_logits = pred  # expect logits

        if self.use_bce:
            loss_raw = self.bce(pred_logits, pseudo)
        else:
            prob = torch.sigmoid(pred_logits)
            # Dice on detection images only
            eps = 1e-6
            inter = torch.sum(prob * pseudo, dim=(1, 2, 3))
            denom = torch.sum(prob, dim=(1, 2, 3)) + torch.sum(pseudo, dim=(1, 2, 3)) + eps
            dice = (2.0 * inter + eps) / denom
            loss_raw = (1.0 - dice[det_mask]).mean() if det_mask.any() else torch.zeros((), device=device)

        loss = self.weight * loss_raw
        return loss, {"cons_loss": loss.detach()}
