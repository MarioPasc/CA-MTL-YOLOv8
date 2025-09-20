# camtl_yolo/train/tasks.py  (updated)
"""
Task configuration for DomainShift1 and CAMTL.
- Dataset key selection per mode.
- Normalization setup:
    * DomainShift1: keep default BN in backbone; set GN in Seg stream.
    * CAMTL: Dual BN in backbone+Detect, GN in Seg stream.
- L2-SP attachment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from camtl_yolo.external.ultralytics.ultralytics.nn.modules import Detect
from camtl_yolo.model.nn import CTAM, CSAM, FPMA, SegHeadMulti
from camtl_yolo.model.losses.regularizers import L2SPRegularizer, snapshot_reference
from camtl_yolo.external.ultralytics.ultralytics.utils import LOGGER
from camtl_yolo.model.utils.normalization import (
    convert_backbone_and_detect_to_dual_bn, 
    replace_seg_stream_bn_with_groupnorm,
    assert_no_module_cycles
)


# ----------------------------- Dataset selection ----------------------------- #

def select_dataset_tasks_for_mode(mode: str) -> Dict[str, List[str]]:
    m = str(mode)
    if m == "DomainShift1":
        keys = ["retinography_segmentation"]
        return {"train": keys, "val": keys, "test": keys}
    if m == "CAMTL":
        # Angiography-only in phase 2; both tasks
        tr = ["angiography_detection", "angiography_segmentation"]
        return {"train": tr, "val": tr, "test": tr}
    # Fallback: everything
    allk = ["retinography_segmentation", "angiography_segmentation", "angiography_detection"]
    return {"train": allk, "val": allk, "test": allk}


# ----------------------------- Groups ----------------------------- #

def _is_attention(m: nn.Module) -> bool:
    return isinstance(m, (CTAM, CSAM, FPMA))


def _set_bn_eval(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def freeze_detect_head(model: nn.Module, bn_eval: bool = True) -> List[str]:
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
    groups: Dict[str, List[nn.Parameter]] = {
        "backbone_neck": [], "attention": [], "seg_head": [], "detect_head": [], "other": []
    }
    for mod in model.modules():
        if isinstance(mod, Detect):
            key = "detect_head"
        elif isinstance(mod, SegHeadMulti):
            key = "seg_head"
        elif _is_attention(mod):
            key = "attention"
        else:
            key = "backbone_neck" if any(isinstance(m, nn.Conv2d) for m in mod.modules()) else "other"
        for p in getattr(mod, "parameters", lambda: [])():
            if p.requires_grad:
                groups[key].append(p)
    return groups


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
    gn_groups: int = 32,
) -> TaskState:
    """
    Apply mode-specific configuration and attach L2-SP.
    Also applies normalization strategy per mode.
    """
    mode = str(mode)
    l2sp: Optional[L2SPRegularizer] = None
    frozen: List[str] = []
    include_names: set[str] = set()


    if mode == "DomainShift1":
        # Normalization: GN in segmentation stream; keep single BN elsewhere
        gn_repl = replace_seg_stream_bn_with_groupnorm(model, max_groups=gn_groups)
        assert_no_module_cycles(model)
        LOGGER.info(f"Seg-stream GroupNorm replacements: {gn_repl}")
        # Freeze Detect
        frozen = freeze_detect_head(model, bn_eval=True)
        # L2-SP over backbone/neck
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(tag in n for tag in (".cv2.", ".cv3.", ".dfl.", "SegHeadMulti", "CTAM", "CSAM", "FPMA")):
                continue
            include_names.add(n)
        def _include(name: str, param: nn.Parameter) -> bool: return name in include_names
        ref = snapshot_reference(model, _include, device=device)
        l2sp = L2SPRegularizer(ref=ref, include=_include, weight=float(l2sp_lambda), device=device)

    elif mode == "CAMTL":
        # Normalization: Dual BN in backbone+Detect; GN in segmentation stream
        gn_repl = replace_seg_stream_bn_with_groupnorm(model, max_groups=gn_groups)
        dualbn_repl = convert_backbone_and_detect_to_dual_bn(model)
        assert_no_module_cycles(model)
        LOGGER.info(f"Seg-stream GroupNorm replacements: {gn_repl}; DualBN conversions: {dualbn_repl}")
        # Unfreeze Detect head for joint training
        # (If previously frozen, ensure requires_grad=True)
        for mod in model.modules():
            if isinstance(mod, Detect):
                for p in mod.parameters(recurse=True):
                    p.requires_grad = True
        # L2-SP still on backbone/neck
        for n, p in model.named_parameters():
            if any(tag in n for tag in (".cv2.", ".cv3.", ".dfl.", "SegHeadMulti", "CTAM", "CSAM", "FPMA")):
                continue
            include_names.add(n)
        def _include(name: str, param: nn.Parameter) -> bool: return name in include_names
        ref = snapshot_reference(model, _include, device=device)
        l2sp = L2SPRegularizer(ref=ref, include=_include, weight=float(l2sp_lambda), device=device)

    else:
        LOGGER.warning(f"Unknown task mode '{mode}'. No changes applied.")

    setattr(model, "_l2sp", l2sp)
    groups = parameter_groups(model)
    return TaskState(mode=mode, l2sp=l2sp, frozen_names=frozen, param_groups=groups)
