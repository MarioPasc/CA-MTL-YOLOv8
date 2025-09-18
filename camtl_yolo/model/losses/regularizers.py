# camtl_yolo/train/regularizers.py
"""
Regularizers used by CA-MTL-YOLOv8 training.
Currently implements L2-SP (Li et al., 2018).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class L2SPRegularizer:
    """
    L2-SP penalty: sum_i ||w_i - w_i^0||^2 over selected parameters.

    Parameters
    ----------
    ref : dict[str, Tensor]
        Reference weights snapshot. Typically taken at initialization.
    include : callable | None
        Predicate on (name, parameter) to decide inclusion.
    weight : float
        λ coefficient for the penalty.
    device : torch.device | None
        Move references to this device for efficient subtraction.
    """
    ref: Dict[str, torch.Tensor] = field(default_factory=dict)
    include: Optional[Callable[[str, nn.Parameter], bool]] = None
    weight: float = 1e-4
    device: Optional[torch.device] = None

    def __post_init__(self) -> None:
        if self.device is not None:
            for k, v in list(self.ref.items()):
                self.ref[k] = v.to(self.device)

    @torch.no_grad()
    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.ref.items()}

    def __call__(self, model: nn.Module) -> torch.Tensor:
        if self.weight <= 0.0:
            p0 = next(model.parameters())
            return torch.zeros((), device=p0.device)
        total = None
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if self.include is not None and not self.include(name, p):
                continue
            ref_w = self.ref.get(name)
            if ref_w is None or ref_w.shape != p.shape:
                continue
            diff = (p - ref_w.to(p.device)) ** 2
            total = diff.sum() if total is None else (total + diff.sum())
        if total is None:
            p0 = next(model.parameters())
            return torch.zeros((), device=p0.device)
        return self.weight * total


@torch.no_grad()
def snapshot_reference(
    model: nn.Module,
    predicate: Callable[[str, nn.Parameter], bool],
    device=None,
) -> Dict[str, torch.Tensor]:
    """Capture a detached copy of parameters that satisfy `predicate`."""
    out: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if predicate(name, p):
            out[name] = p.detach().clone().to(device or p.device)
    return out
