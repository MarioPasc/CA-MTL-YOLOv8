"""
Alignment losses for multi-task learning between detection and segmentation.

This module provides small, plug-and-play losses that couple the behavior of
the detection and segmentation heads during training:

- ZeroStenosisSuppressionLoss: On segmentation batches (i.e., batches used to
  train the segmentation branch where we don't expect detections), suppress
  spurious detection logits by enforcing a small per-image max class score.

- VesselContainmentAlignLoss: On detection batches (i.e., batches used to
  train the detection branch), encourage detected objects (e.g., stenosis) to
  lie within the vessel mask predicted by the segmentation branch by maximizing
  mask coverage within each ROI.

- CompositeAlignmentLoss: A thin wrapper that combines both behaviors, running
  each loss on its respective batch type and summing the results.

All losses return a tuple of (loss, logs_dict) to integrate seamlessly with the
training loop and metrics logging.
"""
# camtl_yolo/model/losses/align.py
from __future__ import annotations
from typing import Dict, List, Tuple, Union, Optional, cast, Callable
import weakref
import torch
import torch.nn as nn
from camtl_yolo.model.nn import CTAM

class ZeroStenosisSuppressionLoss(nn.Module):
    """
    Suppress detection logits on segmentation batches.

    Intended for multi-task training where some batches optimize segmentation
    only (no positive detection targets expected). On such batches, this loss
    penalizes the maximum predicted detection class probability per image if it
    rises above a margin, discouraging spurious detections on negative or
    segmentation-only data.

    Args:
        nc: Number of detection classes (C). Used to select the classification
            slice from detection predictions when layout includes extra channels
            (e.g., box/obj + cls).
        weight: Scalar multiplier applied to the computed loss.
        margin: Allowed per-image maximum class probability without penalty.

    Expected inputs to forward:
        det_preds: Either
            - Tensor of shape [B, A, C_total], or
            - List/tuple of pyramid tensors. Each tensor may be
              [B, A_i, C_total] or [B, H_i, W_i, C_total]; in the latter case,
              anchors A_i = H_i * W_i. Tensors are flattened and concatenated
              along the anchors dimension to shape [B, sum(A_i), C_total].
        batch: Dict containing at least:
            - "img": Tensor with device/shape context [B, C, H, W].
            - "is_seg": Bool or tensor-like flag; True indicates a segmentation
              batch and activates this loss. When False, the loss returns 0.

    Returns:
        Tuple[Tensor, Dict[str, Tensor]]: A scalar loss tensor and a logging
        dictionary with key "align_suppress" containing a detached copy.

    Notes:
        - If the predictions don't expose at least `nc` class channels in the
          last dimension, the loss is a no-op (returns 0).
        - The loss is inactive on non-segmentation batches.
    """
    __slots__ = ("nc","weight","margin")
    def __init__(self, nc:int, weight:float=0.05, margin:float=0.1):
        super().__init__(); self.nc=int(nc); self.weight=float(weight); self.margin=float(margin)
    @staticmethod
    def _is_seg(batch:dict)->bool:
        """Return True if the current batch should be treated as a segmentation batch."""
        return bool(torch.as_tensor(batch.get("is_seg", False)).any().item())
    @staticmethod
    def _stack_preds(p):  # -> [B, A, C]
        """
        Flatten and concatenate multi-scale detection predictions.

        Accepts either a single tensor [B, A, C] or [B, H, W, C], or a list of
        such tensors for pyramid outputs. 4D tensors are reshaped to [B, H*W, C]
        before concatenation along anchors A.
        """
        if isinstance(p,(list,tuple)):
            parts=[]; 
            for t in p: parts.append(t.reshape(t.shape[0], -1, t.shape[-1]) if t.ndim==4 else t)
            return torch.cat(parts,1)
        return p.reshape(p.shape[0], -1, p.shape[-1]) if p.ndim==4 else p
    def forward(self, det_preds, batch):
        """Compute suppression loss for segmentation batches.

        The loss penalizes per-image max class probability above `margin`:

            loss = weight * mean(relu(max_prob_per_image - margin))

        Args:
            det_preds: Detection predictions (see class docstring).
            batch: Training batch dict (must include "img"; uses "is_seg").

        Returns:
            loss: Scalar tensor.
            logs: {"align_suppress": detached loss}
        """
        dev = batch["img"].device
        if not self._is_seg(batch) or self.weight<=0: z=torch.zeros((),device=dev); return z,{"align_suppress":z}
        P = self._stack_preds(det_preds)
        if P.shape[-1] < self.nc: z=torch.zeros((),device=dev); return z,{"align_suppress":z}
        cls_prob = P[..., -self.nc:].sigmoid()             # [B,A,nc]
        per_img_max = cls_prob.amax(dim=(1,2))             # [B]
        loss = self.weight * torch.relu(per_img_max - self.margin).mean()
        return loss, {"align_suppress": loss.detach()}

class VesselContainmentAlignLoss(nn.Module):
    """
    Encourage detections to lie within the vessel segmentation.

    On detection batches, this loss samples the segmentation probabilities
    within Regions of Interest (ROIs) derived from ground-truth (or other
    sources in the future) and penalizes low coverage. Intuitively, a stenosis
    detection should be contained by the vessel mask.

    Args:
        weight: Scalar multiplier for the loss.
        margin: Desired minimum vessel coverage in each ROI; lower coverage is
            penalized with ReLU(margin - coverage).
        box_source: Source for ROIs. Currently "gt" is supported; hooks exist
            for future extensions (e.g., "pred").

    Expected inputs to forward:
        seg_preds: Segmentation logits tensor or dict. If dict, the first
            available tensor among {"full", "p3", "p4", "p5"} is used. Shape
            should be [B, C_seg=1, H, W]. If spatial size doesn't match the
            batch image size, it is bilinearly resized.
        det_preds: Unused at the moment; present for API symmetry.
        batch: Dict with keys:
            - "img": Tensor [B, C, H, W] used to infer spatial size and device.
            - "bboxes": Tensor [N, 4] in normalized XYWH format.
            - "batch_idx": Tensor [N] with image indices for each bbox.

    Returns:
        Tuple[Tensor, Dict[str, Tensor]] with keys:
            - loss: Scalar loss = weight * mean(relu(margin - coverage)).
            - logs: {"align_contain": detached loss}.
    """
    __slots__=("weight","margin","box_source")
    def __init__(self, weight:float=0.1, margin:float=0.7, box_source:str="gt"):
        super().__init__(); self.weight=float(weight); self.margin=float(margin); self.box_source=str(box_source)
    @staticmethod
    def _pick_seg(seg):
        """Select a usable segmentation tensor from various container types."""
        if seg is None: return None
        if isinstance(seg,dict):
            for k in ("full","p3","p4","p5"):
                v=seg.get(k,None)
                if torch.is_tensor(v): return v
            return None
        return seg if torch.is_tensor(seg) else None
    @torch.no_grad()
    def _gt_rois(self, batch):
        """
        Build ROIs from ground-truth boxes in the batch.

        Expects normalized XYWH boxes in `batch["bboxes"]` and indices in
        `batch["batch_idx"]`. Converts to absolute XYXY and returns a tensor of
        shape [M, 5] where each row is [batch_index, x1, y1, x2, y2].
        """
        if "bboxes" not in batch or "batch_idx" not in batch: 
            return torch.zeros((0,5), device=batch["img"].device)
        xywh = batch["bboxes"]; bi = batch["batch_idx"].to(dtype=torch.float32)
        if xywh.numel()==0: return torch.zeros((0,5), device=batch["img"].device)
        H,W = batch["img"].shape[-2:]
        from camtl_yolo.external.ultralytics.ultralytics.utils import ops as ULops
        # Ensure type stability for static analyzers; UL returns Tensor at runtime.
        xyxy = cast(torch.Tensor, ULops.xywh2xyxy(xywh)) * torch.tensor([W, H, W, H], device=xywh.device, dtype=xywh.dtype)
        return torch.cat([bi[:, None], xyxy], 1)
    def forward(self, seg_preds, det_preds, batch):
        """Compute containment loss on detection batches.

        Steps:
        1. Skip if not a detection batch or no usable segmentation available.
        2. Resize seg logits to the image size and apply sigmoid to get probs.
        3. Build ROIs (currently from ground-truth boxes).
        4. Use ROI Align to sample a 1x1 pooled coverage per ROI.
        5. Penalize coverage below `margin`.

        Returns a tuple (loss, {"align_contain": detached loss}).
        """
        dev = batch["img"].device
        is_det = not bool(torch.as_tensor(batch.get("is_seg", False)).any().item())
        if not is_det or self.weight<=0: z=torch.zeros((),device=dev); return z,{"align_contain":z}
        seg = self._pick_seg(seg_preds)
        if seg is None: z=torch.zeros((),device=dev); return z,{"align_contain":z}
        size = batch["img"].shape[-2:]
        if seg.shape[-2:]!=size: seg = torch.nn.functional.interpolate(seg.float(), size=size, mode="bilinear", align_corners=False)
        prob = seg.sigmoid()
        rois = self._gt_rois(batch)
        if rois.numel()==0: z=torch.zeros((),device=dev); return z,{"align_contain":z}
        import torchvision.ops as tv_ops
        roi_align_fn: Callable[..., torch.Tensor] = getattr(tv_ops, "roi_align")
        coverage = roi_align_fn(prob, rois, output_size=(1,1), spatial_scale=1.0, aligned=True).view(-1)
        loss = self.weight * torch.relu(self.margin - coverage).mean()
        return loss, {"align_contain": loss.detach()}

class CompositeAlignmentLoss(nn.Module):
    """
    Containment on detection batches + suppression on segmentation batches.

    This wrapper keeps each sub-loss active only on the appropriate batch type
    and aggregates the results. Any missing component is treated as a no-op.

    Args:
        contain: Module implementing the containment loss on detection batches.
        suppress: Module implementing the suppression loss on segmentation batches.

    Returns (forward):
        Tuple[Tensor, Dict[str, Tensor]] with the summed loss and a merged logs
        dictionary that includes "align_loss" as the detached total.
    """
    def __init__(self, contain:Optional[nn.Module], suppress:Optional[nn.Module]): 
        super().__init__(); self.contain=contain; self.suppress=suppress
    def forward(self, seg_preds, det_preds, batch):
        dev = batch["img"].device; z=torch.zeros((),device=dev); logs={}
        lc, ls = z, z
        if self.contain is not None: lc, ic = self.contain(seg_preds, det_preds, batch); logs.update(ic)
        if self.suppress is not None: ls, is_ = self.suppress(det_preds, batch); logs.update(is_)
        loss = lc + ls; logs["align_loss"] = loss.detach(); return loss, logs

def build_alignment_loss(cfg:dict, model:nn.Module)->nn.Module:
    """
    Factory for alignment losses controlled by a config dict.

    Config keys (all optional with defaults):
        align_enable (bool, default True): Enable alignment losses globally.
        align_contain_enable (bool, default True): Enable containment loss.
        align_suppress_enable (bool, default True): Enable suppression loss.
        align (float, default 0.1): Default weight if align_contain not provided.
        align_contain (float): Containment loss weight (overrides `align`).
        align_margin (float, default 0.7): Minimum desired ROI coverage.
        align_boxes (str, default "gt"): Source for boxes (currently "gt").
        align_suppress (float, default 0.05): Suppression loss weight.
        align_suppress_margin (float, default 0.1): Allowed max cls prob on seg batches.
        nc (int): Number of detection classes; falls back to `model.nc` if present.

    Returns:
        nn.Module: Either an nn.Identity (when disabled) or a CompositeAlignmentLoss
        possibly wrapping one or both component losses.
    """
    if not bool(cfg.get("align_enable", True)): return nn.Identity()
    contain = VesselContainmentAlignLoss(
        weight=float(cfg.get("align_contain", cfg.get("align", 0.1))),
        margin=float(cfg.get("align_margin", 0.7)),
        box_source=str(cfg.get("align_boxes", "gt")),
    ) if bool(cfg.get("align_contain_enable", True)) else None
    suppress = ZeroStenosisSuppressionLoss(
        nc=int(getattr(model,"nc", cfg.get("nc",1))),
        weight=float(cfg.get("align_suppress", 0.05)),
        margin=float(cfg.get("align_suppress_margin", 0.1)),
    ) if bool(cfg.get("align_suppress_enable", True)) else None
    return nn.Identity() if (contain is None and suppress is None) else CompositeAlignmentLoss(contain, suppress)
