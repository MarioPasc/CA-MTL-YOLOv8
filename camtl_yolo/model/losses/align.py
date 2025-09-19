# camtl_yolo/model/losses/align.py
from __future__ import annotations
from typing import Dict, List, Tuple, Union, Optional
import weakref
import torch
import torch.nn as nn
from camtl_yolo.model.nn import CTAM

ProxyLike = Union[weakref.ProxyType, weakref.CallableProxyType]

def _collect_ctam_attn(root: nn.Module) -> List[torch.Tensor]:
    atts: List[torch.Tensor] = []
    for m in root.modules():
        if isinstance(m, CTAM):
            att = getattr(m, "last_attn", None)
            if isinstance(att, torch.Tensor):
                atts.append(att)
    return atts

class AttentionAlignmentLoss(nn.Module):
    """
    Aligns CTAM attention statistics across domains.
    Holds ONLY a weak proxy to the parent. Never registers it as a child.
    """

    __slots__ = ("_model_ref", "source_name", "target_name", "weight", "_warned")

    def __init__(
        self,
        model: Union[nn.Module, ProxyLike],
        source_name: str = "retinography",
        target_name: str = "angiography",
        weight: float = 0.1,
    ) -> None:
        super().__init__()

        # 1) hard purge if these names were ever registered as children
        for k in ("model", "_model_ref"):
            if k in self._modules:  # type: ignore[attr-defined]
                del self._modules[k]  # break stale back-edge

        # 2) store weak proxy via object.__setattr__ to bypass registration
        if isinstance(model, (weakref.ProxyType, weakref.CallableProxyType)):
            proxy = model
        elif isinstance(model, nn.Module):
            proxy = weakref.proxy(model)
        else:
            proxy = model  # for tests only

        object.__setattr__(self, "_model_ref", proxy)

        self.source_name = str(source_name)
        self.target_name = str(target_name)
        self.weight = float(weight)
        self._warned = False

    # 3) forbid future registrations under forbidden names
    def __setattr__(self, name: str, value) -> None:
        if name in {"model", "_model_ref"} and isinstance(value, nn.Module):
            raise TypeError(f"Refusing to register Module under '{name}' to avoid cycles.")
        return super().__setattr__(name, value)

    # read-only accessor; returns proxy, NOT a Module child
    @property
    def model(self) -> object:
        return object.__getattribute__(self, "_model_ref")

    def forward(self, batch: Dict[str, torch.Tensor | list]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Expects batch['img'] (tensor) and batch['domain'] (list[str]).
        """
        device = batch["img"].device
        atts = _collect_ctam_attn(self.model)  # proxy ok
        if not atts:
            if not self._warned:
                self._warned = True
            z = torch.zeros((), device=device)
            return z, {"align_loss": z}

        per_module_means: List[torch.Tensor] = []
        for att in atts:
            per_module_means.append(att.float().mean(dim=tuple(range(1, att.ndim))))  # [B]

        mean_att = torch.stack(per_module_means, dim=0).mean(dim=0)  # [B]
        domains = batch["domain"]

        src_mask = torch.tensor([d == self.source_name for d in domains], device=device, dtype=torch.bool)
        tgt_mask = torch.tensor([d == self.target_name for d in domains], device=device, dtype=torch.bool)

        if not src_mask.any() or not tgt_mask.any():
            z = torch.zeros((), device=device)
            return z, {"align_loss": z}

        src_mean = mean_att[src_mask].mean()
        tgt_mean = mean_att[tgt_mask].mean()
        loss = self.weight * (src_mean - tgt_mean).pow(2)
        return loss, {"align_loss": loss.detach()}
