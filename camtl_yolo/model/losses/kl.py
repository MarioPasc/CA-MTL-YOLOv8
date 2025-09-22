from __future__ import annotations

import weakref
import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import Optional

class AttentionGuidanceKLLoss(nn.Module):
    """
    Teach attention where to look by aligning spatial attention with the vessel probability map.
    For each attention map A (per-scale), compute KL( A || seg_prob ), both normalized to distributions.

    Implementation:
      - CTAM: expects each CTAM module to cache self.last_attn_map = [B,1,H,W] (key-side spatial weights)
      - CSAM (optional): expects self.last_attn_maps = dict(scale-> [B,1,H,W])

    YAML knobs under LOSS:
      consistency: float  # global weight
      consistency_include_csam: bool  # also use CSAM maps if available
      consistency_eps: float  # numerical stability
    """
    def __init__(self, model: nn.Module, weight: float = 1e-3, include_csam: bool = True, eps: float = 1e-8):
        super().__init__()
        # IMPORTANT: do NOT register a strong nn.Module reference here, it creates a module graph cycle
        # when this loss is attached to the model (model -> loss -> model -> ...).
        # Keep only a weak reference so train()/eval() recursion over children doesn't revisit the root model.
        self._model_ref = weakref.ref(model)
        self.weight = float(weight)
        self.include_csam = bool(include_csam)
        self.eps = float(eps)

    @staticmethod
    def _pick_seg_prob_for_size(seg_preds: dict | torch.Tensor, hw: tuple[int, int]) -> torch.Tensor | None:
        if seg_preds is None:
            return None
        if isinstance(seg_preds, dict):
            # try exact size match first
            cand = []
            for k in ("full", "p3", "p4", "p5"):
                v = seg_preds.get(k, None)
                if torch.is_tensor(v):
                    cand.append(v)
                    if v.shape[-2:] == hw:
                        p = v.sigmoid()
                        return p
            # fallback: closest by L1 size distance
            if cand:
                sizes = torch.tensor([abs(v.shape[-2] - hw[0]) + abs(v.shape[-1] - hw[1]) for v in cand])
                v = cand[int(torch.argmin(sizes))]
                return v.sigmoid()
            return None
        # single tensor
        v = seg_preds
        return v.sigmoid() if torch.is_tensor(v) else None

    @staticmethod
    def _kl(p: torch.Tensor, q: torch.Tensor, eps: float) -> torch.Tensor:
        # p,q: [B,1,H,W] non-negative; normalize to sum=1 per image, then KL(p||q)
        B = p.shape[0]
        p = p.clamp_min(0)
        q = q.clamp_min(0)
        p = p / (p.sum(dim=(1,2,3), keepdim=True) + eps)
        q = q / (q.sum(dim=(1,2,3), keepdim=True) + eps)
        return (p * (p.add(eps).log() - q.add(eps).log())).sum(dim=(1,2,3)).mean()  # scalar

    @staticmethod
    def _collect_attn_maps(model: Optional[nn.Module], include_csam: bool) -> list[torch.Tensor]:
        """Return cached spatial attention maps from CTAM/CSAM. Safe on None."""
        if model is None:
            return []
        maps: list[torch.Tensor] = []
        for m in model.modules():
            if hasattr(m, "last_attn_map") and torch.is_tensor(m.last_attn_map):
                maps.append(m.last_attn_map)
            if include_csam and hasattr(m, "last_attn_maps") and isinstance(m.last_attn_maps, dict):
                for v in m.last_attn_maps.values():
                    if torch.is_tensor(v):
                        maps.append(v)
        return maps

    def forward(self, seg_preds: dict | torch.Tensor | None, batch: dict
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Dereference weak model reference at call time to avoid graph recursion.
        Works whether `self.model` is a strong nn.Module or a weakref.ref(nn.Module).
        """
        dev = batch["img"].device
        if self.weight <= 0.0:
            z = torch.zeros((), device=dev)
            return z, {"cons_kl": z}

        # --- weakref-safe model access ---
        model_attr = getattr(self, "model", None)
        model: Optional[nn.Module]
        if isinstance(model_attr, weakref.ReferenceType):
            model = model_attr()  # may be None if GC'ed
        else:
            model = model_attr if isinstance(model_attr, nn.Module) else None

        maps = self._collect_attn_maps(model, include_csam=self.include_csam)
        if not maps:
            z = torch.zeros((), device=dev)
            return z, {"cons_kl": z}

        total = torch.zeros((), device=dev)
        count = 0
        for a in maps:
            if a.ndim == 3:
                a = a.unsqueeze(1)
            if a.ndim != 4:
                continue
            B, C, H, W = a.shape
            if C != 1:
                a = a.mean(dim=1, keepdim=True)
            a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            if (a.abs().sum(dim=(1, 2, 3)) == 0).any():
                a = a + 1e-6

            seg_p = self._pick_seg_prob_for_size(seg_preds, (H, W))
            if seg_p is None:
                continue
            if seg_p.shape[-2:] != (H, W):
                seg_p = F.interpolate(seg_p, size=(H, W), mode="bilinear", align_corners=False)
            seg_p = seg_p.detach()
            seg_p = torch.nan_to_num(seg_p, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0, 1)
            if (seg_p.sum(dim=(1, 2, 3)) == 0).any():
                seg_p = seg_p + 1e-6

            # normalize to distributions and compute KL
            p = a / (a.sum(dim=(1, 2, 3), keepdim=True) + self.eps)
            q = seg_p / (seg_p.sum(dim=(1, 2, 3), keepdim=True) + self.eps)
            kl = (p * (p.add(self.eps).log() - q.add(self.eps).log())).sum(dim=(1, 2, 3)).mean()
            kl = torch.nan_to_num(kl, nan=0.0, posinf=1e6, neginf=1e6)

            total = total + kl
            count += 1

        if count == 0:
            z = torch.zeros((), device=dev)
            return z, {"cons_kl": z}
        loss = self.weight * (total / count)
        return loss, {"cons_kl": loss.detach()}
