# camtl_yolo/train/grad.py
"""
Gradient utilities.

- GradientAccumulator: accumulate N steps before optimizer.step().
- GradientMonitor: compute group-wise gradient norms.
- LossWeightBalancer: optional automatic λ tuning to equalize gradient norms.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import torch
import torch.nn as nn


class GradientAccumulator:
    """Accumulate gradients over `steps` backward() calls."""
    def __init__(self, steps: int = 1):
        self.steps = max(1, int(steps))
        self._counter = 0

    def backward(self, loss: torch.Tensor) -> bool:
        loss = loss / self.steps
        loss.backward()
        self._counter += 1
        if self._counter >= self.steps:
            self._counter = 0
            return True
        return False

    def ready(self) -> bool:
        return self._counter == 0


class GradientMonitor:
    """Compute gradient norms per parameter group."""
    @staticmethod
    def group_norms(param_groups: Dict[str, List[nn.Parameter]], norm: float = 2.0) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, params in param_groups.items():
            total = 0.0
            for p in params:
                if p.grad is None:
                    continue
                total += float(p.grad.detach().data.norm(norm).cpu())
            out[name] = total
        return out


class LossWeightBalancer:
    """
    Simple λ tuner to equalize two loss streams (e.g., det vs seg) by gradient norms.
    Use conservatively; default momentum keeps updates stable.
    """
    def __init__(self, lambda_det: float = 1.0, lambda_seg: float = 1.0, momentum: float = 0.9):
        self.lambda_det = float(lambda_det)
        self.lambda_seg = float(lambda_seg)
        self.momentum = float(momentum)

    def update(self, g_det: float, g_seg: float, eps: float = 1e-8) -> None:
        if g_det <= eps or g_seg <= eps:
            return
        ratio = g_det / (g_seg + eps)
        # target ratio = 1 -> scale det down if too large, seg up if too small
        new_det = self.lambda_det / ratio**0.5
        new_seg = self.lambda_seg * ratio**0.5
        self.lambda_det = self.momentum * self.lambda_det + (1 - self.momentum) * new_det
        self.lambda_seg = self.momentum * self.lambda_seg + (1 - self.momentum) * new_seg
