# camtl_yolo/model/losses/segmentation.py
"""
Multi-scale BCE + Dice loss for binary masks.
Handles Tensor or List[Tensor] predictions; expects logits by default.
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Dice loss = 1 - Dice. Expects probabilities in [0,1].
    pred, target: [B,1,H,W]
    """
    pred = pred.contiguous()
    target = target.contiguous()
    inter = torch.sum(pred * target, dim=(1, 2, 3))
    denom = torch.sum(pred, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3)) + eps
    dice = (2.0 * inter + eps) / denom
    return 1.0 - dice.mean()


class MultiScaleBCEDiceLoss(nn.Module):
    """
    BCE(with logits) + Dice across one or more scales.

    Parameters
    ----------
    bce_weight : float
        λ for BCE term.
    dice_weight : float
        λ for Dice term.
    scale_weights : Sequence[float] | None
        Per-scale weights. If None, equal weights.
    from_logits : bool
        If True, applies sigmoid to compute Dice; BCE uses logits.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        scale_weights: Sequence[float] | None = None,
        from_logits: bool = True,
    ) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.scale_weights = scale_weights
        self.from_logits = bool(from_logits)

    def _ensure_list(self, preds: Union[torch.Tensor, Sequence[torch.Tensor]]) -> List[torch.Tensor]:
        return [preds] if isinstance(preds, torch.Tensor) else list(preds)

    def forward(self, seg_preds: Union[torch.Tensor, Sequence[torch.Tensor]], batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        seg_preds : Tensor [B,1,H,W] or list of Tensors, finest scale last or first (any order).
        batch['mask'] : Tensor [B,1,H,W] binary {0,1}
        """
        preds_list = self._ensure_list(seg_preds)
        B = batch["img"].shape[0]
        gt_mask = batch["mask"].float()  # [B,1,H,W]

        # build per-scale weights
        if self.scale_weights is None:
            w = [1.0 / len(preds_list)] * len(preds_list)
        else:
            assert len(self.scale_weights) == len(preds_list), "scale_weights length must match number of predictions"
            w = [float(x) for x in self.scale_weights]

        total = torch.zeros((), device=gt_mask.device)
        items: Dict[str, torch.Tensor] = {}

        for i, (pred, wi) in enumerate(zip(preds_list, w)):
            # resize gt to pred size
            _, _, h, w_ = pred.shape
            gt_i = F.interpolate(gt_mask, size=(h, w_), mode="nearest")

            # BCE component (expects logits)
            bce_i = self.bce(pred, gt_i)
            # Dice on probabilities
            prob = torch.sigmoid(pred) if self.from_logits else pred.clamp(0, 1)
            dice_i = _dice_loss(prob, gt_i)

            loss_i = wi * (self.bce_weight * bce_i + self.dice_weight * dice_i)
            total = total + loss_i

            items[f"seg_bce_s{i}"] = bce_i.detach()
            items[f"seg_dice_s{i}"] = dice_i.detach()
            items[f"seg_s{i}"] = loss_i.detach()

        items["seg_loss"] = total.detach()
        return total, items
