# camtl_yolo/train/tasks.py
"""
Task configuration and model freezing utilities.

Implements "DomainShift1":
- Start from COCO weights (already loaded by CAMTL_YOLO).
- Use only retinography segmentation data.
- Freeze Detect head parameters; set BN under Detect to eval().
- Keep backbone+neck+FPMA+CTAM+CSAM trainable.
- Attach L2-SP over backbone+neck to reduce forgetting.

Exposes
-------
- select_dataset_tasks_for_mode(mode) -> dict[split -> list[str]]
- configure_task(model, mode, l2sp_lambda) -> TaskState
- parameter_groups(model) -> dict[str, list[nn.Parameter]]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from camtl_yolo.external.ultralytics.ultralytics.nn.modules import Detect
from camtl_yolo.model.nn import CTAM, CSAM, FPMA, SegHead
from camtl_yolo.model.losses.regularizers import L2SPRegularizer, snapshot_reference
from camtl_yolo.external.ultralytics.ultralytics.utils import LOGGER    

# ----------------------------- Grouping ----------------------------- #

def _is_attention(m: nn.Module) -> bool:
    return isinstance(m, (CTAM, CSAM, FPMA))


def _set_bn_eval(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def freeze_detect_head(model: nn.Module, bn_eval: bool = True) -> List[str]:
    """Freeze parameters of Ultralytics Detect head(s)."""
    frozen: List[str] = []
    for mod in model.modules():
        if isinstance(mod, Detect):
            if bn_eval:
                _set_bn_eval(mod)
            for n, p in mod.named_parameters(recurse=True):
                p.requires_grad = False
                frozen.append(n)
    LOGGER.info(f"Frozen Detect-head params: {len(frozen)}")
    return frozen


def parameter_groups(model: nn.Module) -> Dict[str, List[nn.Parameter]]:
    """
    Split parameters into semantic groups for optimizer config.
    """
    groups: Dict[str, List[nn.Parameter]] = {
        "backbone_neck": [], "attention": [], "seg_head": [], "detect_head": [], "other": []
    }
    for mod in model.modules():
        if isinstance(mod, Detect):
            key = "detect_head"
        elif isinstance(mod, SegHead):
            key = "seg_head"
        elif _is_attention(mod):
            key = "attention"
        else:
            key = "backbone_neck" if any(isinstance(m, nn.Conv2d) for m in mod.modules()) else "other"
        for p in getattr(mod, "parameters", lambda: [])():
            if p.requires_grad:
                groups[key].append(p)
    return groups


# ----------------------------- Task selection ----------------------------- #

def select_dataset_tasks_for_mode(mode: str) -> Dict[str, List[str]]:
    """
    Map training mode -> list of JSON keys to include per split.
    """
    mode = str(mode)
    if mode == "DomainShift1":
        keys = ["retinography_segmentation"]
        return {"train": keys, "val": keys, "test": keys}
    # Default: include everything
    return {
        "train": ["retinography_segmentation", "angiography_segmentation", "angiography_detection"],
        "val":   ["retinography_segmentation", "angiography_segmentation", "angiography_detection"],
        "test":  ["retinography_segmentation", "angiography_segmentation", "angiography_detection"],
    }


# ----------------------------- Task application ----------------------------- #

@dataclass
class TaskState:
    mode: str
    l2sp: Optional[L2SPRegularizer]
    frozen_names: List[str]
    param_groups: Dict[str, List[nn.Parameter]]


def configure_task(
    model: nn.Module,
    mode: str = "DomainShift1",
    l2sp_lambda: float = 1e-4,
    device: Optional[torch.device] = None,
) -> TaskState:
    """
    Apply a task mode configuration in-place and return the TaskState.

    DomainShift1:
        - Freeze Detect head.
        - L2-SP over backbone+neck parameters using current weights as reference.
          Exclude attention modules and heads from L2-SP.
    """
    mode = str(mode)
    l2sp: Optional[L2SPRegularizer] = None
    frozen: List[str] = []

    if mode == "DomainShift1":
        # 1) Freeze detector
        frozen = freeze_detect_head(model, bn_eval=True)

        # 2) Build inclusion set for L2-SP: include conv layers in backbone/neck only
        include_names: set[str] = set()
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # Exclude heads and attention by name hints
            if any(tag in n for tag in (".cv2.", ".cv3.", ".dfl.", "SegHead")):
                continue
            if any(tag in n for tag in ("CTAM", "CSAM", "FPMA")):
                continue
            include_names.add(n)

        def _include(name: str, param: nn.Parameter) -> bool:
            return name in include_names

        ref = snapshot_reference(model, _include, device=device)
        l2sp = L2SPRegularizer(ref=ref, include=_include, weight=float(l2sp_lambda), device=device)

    else:
        LOGGER.warning(f"Unknown task mode '{mode}'. No changes applied.")

    groups = parameter_groups(model)
    return TaskState(mode=mode, l2sp=l2sp, frozen_names=frozen, param_groups=groups)
