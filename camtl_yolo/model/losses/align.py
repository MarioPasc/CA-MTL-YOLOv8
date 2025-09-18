# camtl_yolo/model/losses/align.py
"""
Attention alignment loss over CTAM attention maps between domains.
Requires CTAM modules to expose 'last_attn' tensors per forward:
  shape typically [B, heads, Tq, Tk] or compatible.
If unavailable, returns zero loss.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from camtl_yolo.model.nn import CTAM


def _collect_ctam_attn(model: nn.Module) -> List[torch.Tensor]:
    """
    Gather all available CTAM.last_attn tensors from the current forward.
    """
    atts: List[torch.Tensor] = []
    for m in model.modules():
        if isinstance(m, CTAM) and hasattr(m, "last_attn") and isinstance(m.last_attn, torch.Tensor):
            atts.append(m.last_attn)  # [B, H, Tq, Tk] or similar
    return atts


class AttentionAlignmentLoss(nn.Module):
    """
    L2 alignment between mean attention of source and target domains in the current batch.

    Parameters
    ----------
    model : nn.Module
        The CAMTL model containing CTAM modules.
    source_name : str
        Domain string for source (e.g., "retinography").
    target_name : str
        Domain string for target (e.g., "angiography").
    weight : float
        λ for this loss term.
    """

    def __init__(self, model: nn.Module, source_name: str = "retinography", target_name: str = "angiography", weight: float = 0.1) -> None:
        super().__init__()
        self.model = model
        self.source_name = str(source_name)
        self.target_name = str(target_name)
        self.weight = float(weight)
        self._warned = False

    def forward(self, batch: Dict[str, torch.Tensor | list]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Expects batch['domain'] as list[str] with length B.
        """
        device = batch["img"].device
        atts = _collect_ctam_attn(self.model)
        if not atts:
            if not self._warned:
                # warn only once to avoid spam; rely on trainer/logger externally if needed
                self._warned = True
            return torch.zeros((), device=device), {"align_loss": torch.zeros((), device=device)}

        # Reduce each attention to per-sample scalar summary to avoid shape mismatch issues.
        # Mean over heads and tokens -> [B]
        per_module_means: List[torch.Tensor] = []
        for A in atts:
            # A: [B, H, Tq, Tk] or compatible -> mean over all dims except batch
            while A.dim() > 1:
                A = A.mean(dim=-1)
            per_module_means.append(A)  # [B]

        # Average across modules -> [B]
        mean_att = torch.stack(per_module_means, dim=0).mean(dim=0)  # [B]
        # domain masks
        domains = batch["domain"]
        src_mask = torch.tensor([d == self.source_name for d in domains], device=device, dtype=torch.bool)
        tgt_mask = torch.tensor([d == self.target_name for d in domains], device=device, dtype=torch.bool)

        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return torch.zeros((), device=device), {"align_loss": torch.zeros((), device=device)}

        src_mean = mean_att[src_mask].mean()
        tgt_mean = mean_att[tgt_mask].mean()
        loss_raw = (src_mean - tgt_mean).pow(2)
        loss = self.weight * loss_raw
        return loss, {"align_loss": loss.detach()}
