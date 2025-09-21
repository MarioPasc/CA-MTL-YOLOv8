# camtl_yolo/model/utils/ema.py  (or your current EMA helper path)
from __future__ import annotations
from copy import deepcopy
from typing import Dict
import torch
import torch.nn as nn
from camtl_yolo.model.model import CAMTL_YOLO
from typing import Dict, Iterable, Optional
import math

def _flat_state_dict_cycle_safe(src: nn.Module) -> Dict[str, torch.Tensor]:
    """Collect tensors without recursing into cycles."""
    sd: Dict[str, torch.Tensor] = {}
    visited: set[int] = set()

    def dfs(m: nn.Module, prefix: str = ""):
        mid = id(m)
        if mid in visited:
            return
        visited.add(mid)
        # parameters
        for name, p in m._parameters.items():
            if p is None:
                continue
            sd[prefix + name] = p.detach().float().cpu()
        # buffers
        for name, b in m._buffers.items():
            if b is None:
                continue
            sd[prefix + name] = b.detach().float().cpu()
        # children
        for name, ch in m._modules.items():
            if ch is None:
                continue
            dfs(ch, prefix + name + ".")

    dfs(src)
    return sd


def _clone_model_from_yaml(src: nn.Module) -> nn.Module:
    yaml_cfg = getattr(src, "yaml", None)
    ch = getattr(src, "ch", 3) if hasattr(src, "ch") else 3
    nc = getattr(src, "nc", None)
    m = CAMTL_YOLO(cfg=yaml_cfg, ch=ch, nc=nc, verbose=False)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    # Load weights without calling src.state_dict()
    sd = _flat_state_dict_cycle_safe(src)
    m.load_state_dict(sd, strict=False)
    return m


class SafeModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = float(decay)
        # Build EMA clone and move it to the same device as the training model
        self.ema = _clone_model_from_yaml(model)
        try:
            model_device = next(model.parameters()).device  # may raise StopIteration if no params
        except StopIteration:
            model_device = torch.device("cpu")
        # Keep EMA in float32 for numerical stability, located on the training device
        self.ema.to(device=model_device, dtype=torch.float32)
        self.updates = 0
        self._ema_params: Dict[str, torch.Tensor] = dict(self.ema.named_parameters())
        self._ema_buffers: Dict[str, torch.Tensor] = dict(self.ema.named_buffers())

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.updates += 1
        d = self._current_decay()
        for n, p in model.named_parameters():
            if n in self._ema_params:
                dst = self._ema_params[n]
                src = p.detach().to(device=dst.device, dtype=dst.dtype)
                dst.mul_(d).add_(src, alpha=1.0 - d)
        for n, b in model.named_buffers():
            if n in self._ema_buffers:
                dstb = self._ema_buffers[n]
                srcb = b.detach().to(device=dstb.device, dtype=dstb.dtype)
                dstb.copy_(srcb)

    def _current_decay(self, warmup_updates: int = 2000) -> float:
        """Linear warmup of EMA decay over `warmup_updates` steps."""
        if self.updates < warmup_updates:
            return 0.0 + (self.decay - 0.0) * (self.updates / float(warmup_updates))
        return self.decay

    @torch.no_grad()
    def hard_sync(self, model: nn.Module) -> None:
        """Copy model → EMA exactly once. Use at resume or phase switch."""
        sd = model.state_dict()
        # ensure dtype/device match EMA tensors
        for k, v in sd.items():
            if k in self._ema_params:
                self._ema_params[k].copy_(v.to(self._ema_params[k].device, dtype=self._ema_params[k].dtype))
            elif k in self._ema_buffers:
                self._ema_buffers[k].copy_(v.to(self._ema_buffers[k].device, dtype=self._ema_buffers[k].dtype))
        # reset warmup so subsequent updates blend smoothly
        self.updates = 0

    def set_decay(self, decay: float) -> None:
        """Adjust target decay dynamically if needed."""
        self.decay = float(decay)

    @torch.no_grad()
    def load_state_dicts(self, ema_sd: Dict[str, torch.Tensor]) -> None:
        """Restore EMA weights if available in a checkpoint."""
        self.ema.load_state_dict(ema_sd, strict=False)
        # rebuild fast views
        self._ema_params = dict(self.ema.named_parameters())
        self._ema_buffers = dict(self.ema.named_buffers())

    def update_attr(self, model: nn.Module, include: list[str] | None = None, exclude: list[str] | None = None):
        """Update simple attributes on the EMA clone from the source model.

        This mirrors the Ultralytics ModelEMA API used by BaseTrainer.

        Args:
            model: Source model to copy attributes from.
            include: List of attribute names to copy (e.g., ["yaml", "nc", "args", "names", "stride", "class_weights"]).
            exclude: Optional list of attribute names to skip (not used here but kept for API compatibility).
        """
        if not include:
            return
        excl = set(exclude or ())
        for k in include:
            if k in excl:
                continue
            if hasattr(model, k):
                try:
                    setattr(self.ema, k, getattr(model, k))
                except Exception:
                    # Be tolerant to non-copyable attrs (e.g., torch.compile wrappers)
                    pass
