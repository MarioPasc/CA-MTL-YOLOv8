# camtl_yolo/model/losses/align.py
from __future__ import annotations
from typing import Dict, List, Tuple, Union, Optional
import weakref
import torch
import torch.nn as nn
from camtl_yolo.model.nn import CTAM

# ADD this new class near the top-level of align.py

class VesselContainmentAlignLoss(nn.Module):
    """
    Penalize detections that lie outside the predicted vessel mask.
    Only active on detection batches. Gradients flow to the segmentation stream.

    Config:
      - weight: global scalar applied to the penalty
      - margin: desired minimum average mask probability inside a box (e.g., 0.7)
      - box_source: 'gt' (default) uses ground-truth boxes; 'pred' would need decoded det boxes (not recommended)
      - upsample_to: spatial size to upsample the seg map to; defaults to batch['img'] size
    """
    __slots__ = ("weight", "margin", "box_source", "upsample_to")

    def __init__(self, weight: float = 0.1, margin: float = 0.7,
                 box_source: str = "gt", upsample_to: str = "img") -> None:
        super().__init__()
        self.weight = float(weight)
        self.margin = float(margin)
        self.box_source = str(box_source)
        self.upsample_to = str(upsample_to)

    @staticmethod
    def _pick_seg_map(seg_preds: dict | torch.Tensor | None) -> torch.Tensor | None:
        if seg_preds is None:
            return None
        if isinstance(seg_preds, dict):
            # prefer full-resolution fused map if present
            if "full" in seg_preds and torch.is_tensor(seg_preds["full"]):
                return seg_preds["full"]
            # otherwise fall back to P3
            if "p3" in seg_preds and torch.is_tensor(seg_preds["p3"]):
                return seg_preds["p3"]
            # any single tensor-valued entry
            for v in seg_preds.values():
                if torch.is_tensor(v):
                    return v
            return None
        return seg_preds if torch.is_tensor(seg_preds) else None

    @torch.no_grad()
    def _gt_rois_xyxy(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Build ROI tensor [N, 5] = [batch_idx, x1, y1, x2, y2] in pixel space from
        normalized XYWH targets in `batch['bboxes']` and per-box `batch_idx`.
        """
        if ("bboxes" not in batch) or ("batch_idx" not in batch):
            return torch.zeros((0, 5), device=batch["img"].device)

        bxywh = batch["bboxes"]  # normalized xywh
        bi = batch["batch_idx"].to(dtype=torch.float32)
        if bxywh.numel() == 0:
            return torch.zeros((0, 5), device=batch["img"].device)

        # image size in pixels
        _, _, H, W = batch["img"].shape
        scale = torch.tensor([W, H, W, H], device=bxywh.device, dtype=bxywh.dtype)
        from camtl_yolo.external.ultralytics.ultralytics.utils import ops as ULops
        xyxy = ULops.xywh2xyxy(bxywh) * scale
        rois = torch.cat([bi.unsqueeze(1), xyxy], dim=1)  # [N,5]
        return rois

    def forward(self, seg_preds: dict | torch.Tensor | None,
                det_preds: torch.Tensor | None,
                batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        seg_preds: dict/tensor from SegHeadMulti
        det_preds: unused here (we rely on GT by default)
        batch: training batch dict with 'img', 'is_seg'/'is_det', 'bboxes', 'batch_idx'
        """
        device = batch["img"].device
        # Only run on detection batches
        is_det = bool(torch.as_tensor(batch.get("is_det", False)).any().item())
        if not is_det or self.weight <= 0.0:
            z = torch.zeros((), device=device)
            return z, {"align_loss": z}

        seg = self._pick_seg_map(seg_preds)
        if seg is None:
            z = torch.zeros((), device=device)
            return z, {"align_loss": z}

        # upsample seg logits/prob to input resolution
        target_hw = batch["img"].shape[-2:]
        if seg.shape[-2:] != target_hw:
            seg = torch.nn.functional.interpolate(seg.float(), size=target_hw, mode="bilinear", align_corners=False)
        # use probabilities, not hard mask
        seg_prob = seg.sigmoid()  # [B,1,H,W] expected # type: ignore

        # choose boxes
        if self.box_source != "gt":
            # Decoding det predictions in training graph would require custom decode; not advised here.
            # Fall back to GT to keep a clean gradient path to the seg stream.
            raise NotImplementedError("VesselContainmentAlignLoss(box_source='pred') is not implemented. Use 'gt'.")

        rois = self._gt_rois_xyxy(batch)  # [N,5]
        if rois.numel() == 0:
            z = torch.zeros((), device=device)
            return z, {"align_loss": z}

        # differentiable average mask prob inside each ROI via ROIAlign
        from torchvision.ops import roi_align  # type: ignore[import]
        # seg_prob must be [B, C=1, H, W]; roi_align returns [N, 1, 1, 1]
        pooled = roi_align(seg_prob, rois, output_size=(1, 1), spatial_scale=1.0, aligned=True)
        coverage = pooled.view(-1)  # [N] in [0,1]

        # hinge penalty towards margin
        per_box = torch.relu(self.margin - coverage)
        loss = self.weight * per_box.mean()
        return loss, {"align_loss": loss.detach()}


def build_alignment_loss(cfg: dict, model: nn.Module) -> nn.Module:
    """
    Factory for alignment losses.

    YAML options under LOSS:
      align_enable: bool
      align_mode: 'containment' | 'ctam'
      align: float                       # global weight used inside each loss
      align_margin: float                # for 'containment'
      align_boxes: 'gt' | 'pred'         # for 'containment'
      source_domain / target_domain      # for 'ctam'
    """
    enable = bool(cfg.get("align_enable", True))
    if not enable:
        # no-op module
        return nn.Identity()

    mode = str(cfg.get("align_mode", "containment")).lower()
    w = float(cfg.get("align", 0.1))
    if mode == "containment":
        return VesselContainmentAlignLoss(
            weight=w,
            margin=float(cfg.get("align_margin", 0.7)),
            box_source=str(cfg.get("align_boxes", "gt")),
        )
    else:
        # keep your existing CTAM statistics matching as fallback
        return AttentionAlignmentLoss(
            model=model,
            source_name=str(cfg.get("source_domain", "retinography")),
            target_name=str(cfg.get("target_domain", "angiography")),
            weight=w,
        )


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


    def forward(self, seg_preds: dict | torch.Tensor | None,
                det_preds: torch.Tensor | None,
                batch: Dict[str, torch.Tensor | list]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Expects batch['img'] (tensor) and batch['domain'] (list[str]).
        """
        device = batch["img"].device # type: ignore[union-attr]
        atts = _collect_ctam_attn(self.model)  # type: ignore[attr-defined]
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
