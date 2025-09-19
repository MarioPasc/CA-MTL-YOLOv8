# segmentation.py
from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLossProb(nn.Module):
    """Soft Dice on probabilities."""
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # p, y: [B,1,H,W] in [0,1]
        inter = (p * y).sum(dim=(1,2,3))
        denom = p.sum(dim=(1,2,3)) + y.sum(dim=(1,2,3)) + self.eps
        dice = (2.0 * inter + self.eps) / denom
        return 1.0 - dice.mean()

class DeepSupervisionBCEDiceLoss(nn.Module):
    """
    L = Σ_i w_i * [BCE(sig(logits_i), y_i) + Dice(sig(logits_i), y_i)]
    Expects preds as dict with keys {'p3','p4','p5'} (and optional 'full').
    Uses batch['mask_p{3,4,5}'] if present; falls back to area interpolate from batch['mask'].
    """
    def __init__(self, w_p3: float = 1.0, w_p4: float = 0.7, w_p5: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLossProb()
        self.weights = {"p3": w_p3, "p4": w_p4, "p5": w_p5}

    @staticmethod
    def _resize_mask(mask: torch.Tensor, size: Tuple[int,int]) -> torch.Tensor:
        return F.interpolate(mask, size=size, mode="area")

    def forward(self, preds: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss = 0.0
        items: Dict[str, float] = {}
        for key in ("p3","p4","p5"):
            if key not in preds:
                continue
            logit = preds[key]                   # [B,1,h,w]
            B, _, h, w = logit.shape
            # target
            if f"mask_{key}" in batch:
                tgt = batch[f"mask_{key}"].to(logit.device).float()  # [B,1,h,w]
            else:
                tgt = self._resize_mask(batch["mask"].to(logit.device).float(), (h,w))
            prob = torch.sigmoid(logit)
            bce = self.bce(logit, tgt)
            dice = self.dice(prob, tgt)
            w = self.weights[key]
            items[f"{key}_bce"] = float(bce.detach())
            items[f"{key}_dice"] = float(dice.detach())
            loss = loss + w * (bce + dice)
        items["seg_loss"] = float(loss.detach())
        return loss, items
