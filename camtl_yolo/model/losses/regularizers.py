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
    """L2-SP: sum_i ||w_i - w_i^0||^2 over selected parameters."""
    ref: Dict[str, torch.Tensor] = field(default_factory=dict)
    include: Optional[Callable[[str, nn.Parameter], bool]] = None
    weight: float = 1e-4
    device: Optional[torch.device] = None

    def __post_init__(self) -> None:
        if self.device is not None:
            for k, v in list(self.ref.items()):
                self.ref[k] = v.to(self.device)

    __slots__ = ("anchors", "lam")

    def __init__(self, anchors: dict[str, torch.Tensor], lam: float) -> None:
        # store CPU copies to keep checkpoints portable; move to device at use-time
        self.anchors = {k: v.detach().cpu() for k, v in anchors.items()}
        self.lam = float(lam)

    def __call__(self, model: nn.Module) -> torch.Tensor:
        if self.lam <= 0.0 or not self.anchors:
            # return a scalar 0 on the model's device
            dev = next(model.parameters()).device
            return torch.zeros((), device=dev)
        cur = dict(model.named_parameters())
        dev = next(model.parameters()).device
        loss = torch.zeros((), device=dev, dtype=torch.float32)
        for n, ref_cpu in self.anchors.items():
            p = cur.get(n, None)
            if p is None or not p.requires_grad:
                continue
            ref = ref_cpu.to(device=p.device, dtype=p.dtype)
            loss = loss + torch.sum((p.float() - ref.float()) ** 2)
        return loss * self.lam


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
