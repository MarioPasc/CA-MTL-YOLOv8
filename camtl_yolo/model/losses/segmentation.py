# segmentation.py
from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from camtl_yolo.model.losses.lovasz_losses import lovasz_hinge as lovasz_hinge_bin  # binary


class DiceLossProb(nn.Module):
    """Soft Dice on probabilities. NaN-safe."""
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # sanitize (keep in float32 for numerical headroom under AMP)
        p = p.float()
        y = y.float()
        p = torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0, 1)
        y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0, 1)
        inter = (p * y).sum(dim=(1, 2, 3))
        denom = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) + self.eps
        dice = (2.0 * inter + self.eps) / denom
        loss = 1.0 - dice.mean()
        return torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=1.0)


class DeepSupervisionConfigurableLoss(nn.Module):
    """
    Per-scale configurable combination of {BCE, DICE, LOVASZ} with learnable scale weights.
    Loss = sum_s  w_s * sum_{ℓ in cfg[s]} ℓ_s
    """
    def __init__(self, config: dict[str, list[str]], init_weights: dict[str, float]):
        super().__init__()
        self.config = {k.lower(): [x.upper() for x in v] for k, v in config.items()}
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.dice = DiceLossProb()
        from camtl_yolo.model.losses.lovasz_losses import lovasz_hinge as lovasz_hinge_bin
        self.lovasz = lovasz_hinge_bin

        # learnable positive weights via softplus; init in the safe inverse space
        def inv_softplus(y: float) -> float:
            y = max(float(y), 1e-6)
            # numerically stable inverse softplus
            t = torch.tensor(y, dtype=torch.float32)
            return float(torch.log(torch.expm1(t)))
        self._alpha = nn.ParameterDict({
            "p3": nn.Parameter(torch.tensor(inv_softplus(init_weights.get("p3", 1.0)), dtype=torch.float32)),
            "p4": nn.Parameter(torch.tensor(inv_softplus(init_weights.get("p4", 1.0)), dtype=torch.float32)),
            "p5": nn.Parameter(torch.tensor(inv_softplus(init_weights.get("p5", 1.0)), dtype=torch.float32)),
        })

    @staticmethod
    def _resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(mask, size=size, mode="area")

    def get_weights(self) -> dict[str, float]:
        w3 = F.softplus(self._alpha["p3"]).detach().item()
        w4 = F.softplus(self._alpha["p4"]).detach().item()
        w5 = F.softplus(self._alpha["p5"]).detach().item()
        return {"w_p3": float(w3), "w_p4": float(w4), "w_p5": float(w5)}

    def _scale_loss(self, key: str, logit: torch.Tensor, tgt: torch.Tensor) -> dict[str, torch.Tensor]:
        # sanitize targets
        if torch.is_floating_point(tgt) and tgt.max() > 1.5:
            tgt = (tgt / 255.0).clamp_(0, 1)
        tgt = torch.nan_to_num(tgt, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0, 1).float()

        # sanitize logits before any op
        logit = torch.nan_to_num(logit, nan=0.0, posinf=20.0, neginf=-20.0).clamp_(-20.0, 20.0).float()
        prob = torch.sigmoid(logit)

        comps: dict[str, torch.Tensor] = {}
        for name in self.config.get(key, []):
            if name == "BCE":
                v = self.bce(logit, tgt)
            elif name == "DICE":
                v = self.dice(prob, tgt)
            elif name == "LOVASZ":
                # expects logits [B,H,W] and binary {0,1} targets
                v = self.lovasz(logit.squeeze(1), tgt.squeeze(1).round().long())
            else:
                continue
            comps[name.lower()] = torch.nan_to_num(v, nan=0.0, posinf=1e3, neginf=1e3)
        return comps

    def forward(self, preds: dict[str, torch.Tensor], batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        # total starts as finite scalar on the right device
        dev = (preds["p3"].device if isinstance(preds, dict) and "p3" in preds else batch["img"].device)
        total = torch.zeros((), device=dev, dtype=torch.float32)
        items: dict[str, float] = {}

        for key in ("p3", "p4", "p5"):
            if not (isinstance(preds, dict) and key in preds):
                continue
            logit = preds[key]
            logit = logit.float()
            h, w = logit.shape[-2:]
            tgt = batch.get(f"mask_{key}", None)
            if tgt is None:
                tgt = self._resize_mask(batch["mask"].to(logit.device).float(), (h, w))
            else:
                tgt = tgt.to(logit.device).float()

            comps = self._scale_loss(key, logit, tgt)
            
            if not comps:  # no components selected for this scale
                continue

            # sum components for this scale
            loss_s = torch.stack([v if v.ndim == 0 else v.mean() for v in comps.values()]).sum().float()
            # Keep weights in a safe dynamic range for FP16 checkpoints (~6e4 max)
            w_s = F.softplus(self._alpha[key]).clamp(1e-6, 6e4).float()
            total = total + (w_s * loss_s)

            # expose components
            for cname, v in comps.items():
                items[f"{key}_{cname}"] = float(torch.nan_to_num(v.detach(), nan=0.0))

        # expose current weights and total
        items.update(self.get_weights())
        items["seg_loss"] = float(torch.nan_to_num(total.detach(), nan=0.0))
        # final guard
        total = torch.nan_to_num(total, nan=0.0, posinf=1e4, neginf=1e4).float()
        return total, items

