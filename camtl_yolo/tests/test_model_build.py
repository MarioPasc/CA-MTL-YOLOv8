# --- tests/test_model_build.py: replace test_pretrained_weight_mapping ---
import os
import yaml
import torch
import pytest
from pathlib import Path
import copy
import sys

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from camtl_yolo.model.model import CAMTL_YOLO
from camtl_yolo.external.ultralytics.ultralytics.nn.modules import Detect

def _ckpt(path):
    return torch.load(path, map_location="cpu", weights_only=False)["model"].float().state_dict()

def _ckpt_nc(path):
    m = torch.load(path, map_location="cpu", weights_only=False)["model"]
    return int(getattr(m, "nc", 80))

def _detect_head_idx(model):
    for i, m in enumerate(model.model):
        if isinstance(m, Detect):
            return i
    raise AssertionError("Detect head not found")

@pytest.fixture(scope="module")
def cfg():
    cfg_path = Path(PROJECT_ROOT) / "camtl_yolo/model/configs/models/camtl_yolov8.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

@pytest.mark.parametrize("scale", ["n","s","m","l","x"])
def test_pretrained_weight_mapping(cfg, scale):
    cfg = copy.deepcopy(cfg)
    cfg["SCALE"] = scale
    p = Path(cfg["PRETRAINED_MODELS_PATH"])
    seg_pt = p / f"yolov8{scale}-seg.pt"
    det_pt = p / f"yolov8{scale}.pt"

    if not seg_pt.exists() or not det_pt.exists():
        pytest.skip(f"Missing checkpoints for scale {scale}: {seg_pt} or {det_pt}")

    model = CAMTL_YOLO(cfg=cfg, ch=3, nc=cfg["nc"])
    sd = model.state_dict()

    csd_seg = _ckpt(seg_pt)
    csd_det = _ckpt(det_pt)
    coco_nc = _ckpt_nc(det_pt)

    # 1) Early backbone conv must match exactly
    k0 = "model.0.conv.weight"
    assert k0 in csd_seg and k0 in sd, f"Key {k0} missing"
    assert torch.allclose(sd[k0], csd_seg[k0]), "Backbone conv mismatch vs seg checkpoint"

    # 2) Detect head remap coverage excluding classification tensors when nc differs
    det_idx = _detect_head_idx(model)
    model_nc = int(getattr(model.model[det_idx], "nc", cfg["nc"]))
    skip_cls = model_nc != coco_nc

    eligible_keys = []
    matched = 0
    for k_det, v in csd_det.items():
        if not k_det.startswith("model.22."):
            continue
        sub = k_det.split("model.22.", 1)[1]
        if skip_cls and sub.startswith("cv3"):  # classification branch
            continue
        eligible_keys.append(k_det)
        new_k = f"model.{det_idx}.{sub}"
        if new_k in sd and sd[new_k].shape == v.shape and torch.allclose(sd[new_k], v):
            matched += 1

    assert len(eligible_keys) > 0, "No eligible detect-head tensors found in det checkpoint"

    # 99% coverage for eligible tensors
    coverage = matched / len(eligible_keys)
    assert coverage >= 0.99, f"Detect-head mapping coverage too low: {matched}/{len(eligible_keys)} ({coverage:.3f})"

    # 3) Ensure seg-head tensors from seg checkpoint were not copied
    leaked = [k for k in csd_seg if k.startswith("model.22.") and k in sd and torch.allclose(sd[k], csd_seg[k])]
    assert not leaked, f"Seg-head tensors from seg checkpoint leaked into the model: {leaked}"

    # 4) Sanity: if model_nc==coco_nc, classification weights must also match
    if not skip_cls:
        cls_keys = [k for k in csd_det if k.startswith("model.22.cv3")]
        assert cls_keys, "No classification tensors in det checkpoint"
        cls_matched = 0
        for k in cls_keys:
            sub = k.split("model.22.", 1)[1]
            new_k = f"model.{det_idx}.{sub}"
            if new_k in sd and sd[new_k].shape == csd_det[k].shape and torch.allclose(sd[new_k], csd_det[k]):
                cls_matched += 1
        assert cls_matched == len(cls_keys), f"Classification tensors mismatch: {cls_matched}/{len(cls_keys)}"
