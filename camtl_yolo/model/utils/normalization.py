"""
Normalization utilities.

- DualBatchNorm2d: domain-specific BN with a thread-local domain switch.
- Functions to convert a model:
    * convert_backbone_and_detect_to_dual_bn(...)
    * replace_seg_stream_bn_with_groupnorm(...)

Usage in training:
    from camtl_yolo.train.normalization import bn_domain, set_bn_domain
    with bn_domain("source"):  # or "target"
        loss, items = forward_and_loss(...)
"""
from __future__ import annotations

import contextlib
import threading
from typing import Callable, Optional

import torch
import torch.nn as nn


# ----------------------------- Domain switch ----------------------------- #

_THREAD_LOCAL = threading.local()
_DEFAULT_DOMAIN = "source"


def get_bn_domain() -> str:
    return getattr(_THREAD_LOCAL, "bn_domain", _DEFAULT_DOMAIN)


def set_bn_domain(domain: str) -> None:
    setattr(_THREAD_LOCAL, "bn_domain", str(domain))


@contextlib.contextmanager
def bn_domain(domain: str):
    prev = get_bn_domain()
    try:
        set_bn_domain(domain)
        yield
    finally:
        set_bn_domain(prev)

def _dfs_cycle_detect(root: nn.Module) -> str | None:
    """Return a dotted path to a node where a cycle is detected, else None."""
    visited, stack = set(), set()

    def dfs(m: nn.Module, path: list[str]) -> str | None:
        mid = id(m)
        if mid in stack:
            return ".".join(path + ["<cycle>"])
        if mid in visited:
            return None
        stack.add(mid)
        for name, ch in m._modules.items():
            if ch is None:
                continue
            got = dfs(ch, path + [name])
            if got is not None:
                return got
        stack.remove(mid)
        visited.add(mid)
        return None

    return dfs(root, ["root"])


def assert_no_module_cycles(root: nn.Module) -> None:
    p = _dfs_cycle_detect(root)
    if p is not None:
        raise RuntimeError(f"Module graph contains a cycle along path: {p}")


# ----------------------------- Dual BN ----------------------------- #

class DualBatchNorm2d(nn.Module):
    """
    Two independent BN2d branches. The active branch is selected by `get_bn_domain()`.

    Notes
    -----
    - Carries two sets of buffers and affine params.
    - `from_bn` copies weights/buffers from a single BN into both branches.
    """
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.src = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.tgt = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    @classmethod
    def from_bn(cls, bn: nn.BatchNorm2d) -> "DualBatchNorm2d":
        m = cls(
            bn.num_features,
            eps=bn.eps,
            momentum=bn.momentum,
            affine=bn.affine,
            track_running_stats=bn.track_running_stats,
        )
        with torch.no_grad():
            for b in (m.src, m.tgt):
                if bn.affine:
                    b.weight.copy_(bn.weight.data)
                    b.bias.copy_(bn.bias.data)
                if bn.track_running_stats:
                    b.running_mean.copy_(bn.running_mean)
                    b.running_var.copy_(bn.running_var)
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.src(x) if get_bn_domain() == "source" else self.tgt(x)


# ----------------------------- Model conversion helpers ----------------------------- #

def _replace_module_attr(m: nn.Module, attr: str, replace_fn: Callable[[nn.Module], nn.Module]) -> bool:
    """
    If m.<attr> exists and is a Module, replace it with replace_fn(m.<attr>).
    Returns True if replaced.
    """
    if hasattr(m, attr):
        sub = getattr(m, attr)
        if isinstance(sub, nn.Module):
            new = replace_fn(sub)
            if new is not sub:
                setattr(m, attr, new)
                return True
    return False


def _pick_gn_groups(channels: int, max_groups: int = 32) -> int:
    # Choose the largest divisor of channels not exceeding max_groups.
    for g in [max_groups, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


def replace_seg_stream_bn_with_groupnorm(model: nn.Module, max_groups: int = 32) -> int:
    """
    Replace nn.BatchNorm2d under SegHeadMulti/FPMA/CTAM subtrees with GroupNorm.
    Returns number of layers replaced.
    """
    from camtl_yolo.model.nn import SegHeadMulti, FPMA, CTAM  # local types

    def _convert_bn_to_gn(bn: nn.BatchNorm2d) -> nn.GroupNorm:
        gn = nn.GroupNorm(num_groups=_pick_gn_groups(bn.num_features, max_groups),
                          num_channels=bn.num_features, eps=bn.eps, affine=True)
        with torch.no_grad():
            if bn.affine:
                gn.weight.copy_(bn.weight.data)
                gn.bias.copy_(bn.bias.data)
        return gn

    replaced = 0
    visited: set[int] = set()

    def visit(root: nn.Module):
        nonlocal replaced
        mid = id(root)
        if mid in visited:
            return
        visited.add(mid)

        for name, child in list(root.named_children()):
            # 1) common Ultralytics blocks have .bn attribute
            if _replace_module_attr(child, "bn",
                                    lambda old: _convert_bn_to_gn(old) if isinstance(old, nn.BatchNorm2d) else old):
                replaced += 1
            # 2) raw BN as direct child
            if isinstance(child, nn.BatchNorm2d):
                setattr(root, name, _convert_bn_to_gn(child))
                replaced += 1
                # stop here; new GN has no children
                continue
            # 3) descend
            visit(child)

    # Only start from segmentation roots to avoid touching backbone/detect
    seg_roots = [m for m in model.modules() if isinstance(m, (SegHeadMulti, FPMA, CTAM))]
    for r in seg_roots:
        visit(r)

    # sanity
    assert_no_module_cycles(model)
    return replaced


def convert_backbone_and_detect_to_dual_bn(model: nn.Module) -> int:
    """
    Convert BatchNorm2d in backbone and Detect head to DualBatchNorm2d.
    Skips SegHeadMulti/FPMA/CTAM entirely. Returns number of layers converted.
    """
    from camtl_yolo.model.nn import SegHeadMulti, FPMA, CTAM
    from camtl_yolo.external.ultralytics.ultralytics.nn.modules import Detect

    def _should_skip(m: nn.Module) -> bool:
        return isinstance(m, (SegHeadMulti, FPMA, CTAM))

    converted = 0
    visited: set[int] = set()

    def visit(root: nn.Module, skip: bool = False):
        nonlocal converted
        mid = id(root)
        if mid in visited:
            return
        visited.add(mid)

        skip_here = skip or _should_skip(root)
        for name, child in list(root.named_children()):
            if not skip_here:
                if _replace_module_attr(child, "bn",
                                        lambda old: DualBatchNorm2d.from_bn(old) if isinstance(old, nn.BatchNorm2d) else old):
                    converted += 1
                if isinstance(child, nn.BatchNorm2d):
                    setattr(root, name, DualBatchNorm2d.from_bn(child))
                    converted += 1
                    continue
            visit(child, skip_here)

    # single pass from the top
    visit(model, skip=False)
    # ensure Detect submodules are handled even if nested oddly
    for m in model.modules():
        if isinstance(m, Detect):
            visit(m, skip=False)

    assert_no_module_cycles(model)
    return converted
