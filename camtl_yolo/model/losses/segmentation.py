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

class DeepSupervisionConfigurableLoss(nn.Module):
    """
    Per-scale configurable combination of {BCE, DICE, LOVASZ} with learnable scale weights.
    Loss = sum_s  w_s * sum_{ℓ in cfg[s]} ℓ_s

    config: dict like {"p3": ["LOVASZ"], "p4": ["BCE","DICE"], "p5": ["BCE","DICE"]}
    init_weights: dict like {"p3":1.0,"p4":1.0,"p5":1.0} used to initialize learnable weights
    """
    def __init__(self, config: dict[str, list[str]], init_weights: dict[str, float]):
        super().__init__()
        self.config = {k.lower(): [x.upper() for x in v] for k, v in config.items()}
        # primitives
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLossProb()
        # Lovasz expects logits [B,H,W]
        from camtl_yolo.model.losses.lovasz_losses import lovasz_hinge as lovasz_hinge_bin  # binary
        self.lovasz = lovasz_hinge_bin

        # learnable positive weights via softplus
        def inv_softplus(y: float) -> float:
            y = max(y, 1e-6); return float(torch.log(torch.exp(torch.tensor(y)) - 1.0))
        self._alpha = nn.ParameterDict({
            "p3": nn.Parameter(torch.tensor(inv_softplus(float(init_weights.get("p3", 1.0))), dtype=torch.float32)),
            "p4": nn.Parameter(torch.tensor(inv_softplus(float(init_weights.get("p4", 1.0))), dtype=torch.float32)),
            "p5": nn.Parameter(torch.tensor(inv_softplus(float(init_weights.get("p5", 1.0))), dtype=torch.float32)),
        })

    @staticmethod
    def _resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return torch.nn.functional.interpolate(mask, size=size, mode="area")

    def get_weights(self) -> dict[str, float]:
        w = {k: torch.nn.functional.softplus(p).item() for k, p in self._alpha.items()}
        return {"w_p3": w["p3"], "w_p4": w["p4"], "w_p5": w["p5"]}

    def _scale_loss(self, key: str, logit: torch.Tensor, tgt: torch.Tensor) -> dict[str, torch.Tensor]:
        # returns dict of component losses for this scale
        prob = torch.sigmoid(logit)
        comps: dict[str, torch.Tensor] = {}
        for name in self.config.get(key, []):
            if name == "BCE":
                comps["bce"] = self.bce(logit, tgt)
            elif name == "DICE":
                comps["dice"] = self.dice(prob, tgt)
            elif name == "LOVASZ":
                comps["lovasz"] = self.lovasz(logit.squeeze(1), tgt.squeeze(1).long())
            else:
                continue
        return comps

    def forward(self, preds: dict[str, torch.Tensor], batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        total = torch.zeros((), device=preds["p3"].device if isinstance(preds, dict) and "p3" in preds else batch["img"].device)
        items: dict[str, float] = {}
        for key in ("p3", "p4", "p5"):
            if not isinstance(preds, dict) or key not in preds:
                continue
            logit = preds[key]                           # [B,1,h,w]
            h, w = logit.shape[-2:]
            tgt = batch.get(f"mask_{key}", None)
            if tgt is None:
                tgt = self._resize_mask(batch["mask"].to(logit.device).float(), (h, w))
            else:
                tgt = tgt.to(logit.device).float()
            comps = self._scale_loss(key, logit, tgt)
            # sum components for this scale
            if not comps:
                continue
            loss_s = torch.stack([v if v.ndim == 0 else v.mean() for v in comps.values()]).sum()
            w_s = torch.nn.functional.softplus(self._alpha[key])  # positive
            total = total + w_s * loss_s

            # log components
            for cname, v in comps.items():
                items[f"{key}_{cname}"] = float(v.detach())
        # expose current weights
        w = self.get_weights()
        items.update(w)
        items["seg_loss"] = float(total.detach())
        return total, items
