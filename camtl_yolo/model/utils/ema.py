# camtl_yolo/model/utils/ema.py  (or your current EMA helper path)
from __future__ import annotations
from copy import deepcopy
from typing import Dict
import torch
import torch.nn as nn
from camtl_yolo.model.model import CAMTL_YOLO


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
        d = self.decay
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
