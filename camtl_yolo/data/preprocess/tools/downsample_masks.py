#!/usr/bin/env python3
"""
mask_downsample.py

Robust downsampling for segmentation masks.

Strategies:
 - 'nearest'        : nearest-neighbor (label-preserving, fast)
 - 'area'           : area-averaging of probabilities then threshold (recommended for binary)
 - 'majority'       : block-wise majority vote (integer downscale only; recommended if factors are integer)
 - 'perclass_area'  : per-class area pooling then argmax (multi-class)
 - 'skeleton_preserve': skeletonize -> downsample -> dilate (preserve thin structures)

Design:
 - Atomic functions
 - Typed signatures
 - Dataclass config
 - Lightweight logging
 - Uses numpy, skimage, scipy
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

# Optional imports; raise informative error if missing when used
try:
    from skimage.transform import resize, downscale_local_mean
    from skimage.morphology import skeletonize, binary_dilation, disk
    from skimage.util import img_as_float32
except Exception as e:
    resize = None  # type: ignore
    downscale_local_mean = None  # type: ignore
    skeletonize = None  # type: ignore
    binary_dilation = None  # type: ignore
    disk = None  # type: ignore

try:
    import scipy.ndimage as ndi  # for zoom and generic filters
except Exception:
    ndi = None  # type: ignore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass(frozen=True)
class MaskDownsampleConfig:
    target_size: Tuple[int, int]  # (width, height)
    method: str = "area"          # 'nearest'|'area'|'majority'|'perclass_area'|'skeleton_preserve'
    threshold: float = 0.5        # threshold for probabilistic -> binary
    preserve_dtype: bool = True   # try to return integer labels if input was integer
    majority_block_limit: int = 64  # avoid huge blocks with majority algorithm


class MaskDownsampleError(RuntimeError):
    """Raised for failures in mask downsampling."""


def _check_dependencies():
    if resize is None or downscale_local_mean is None or ndi is None:
        raise MaskDownsampleError("skimage and scipy are required. Install scikit-image and scipy.")


def _compute_scale_factors(src_size: Tuple[int, int], tgt_size: Tuple[int, int]) -> Tuple[float, float]:
    ow, oh = src_size
    tw, th = tgt_size
    return float(tw) / float(ow), float(th) / float(oh)


def _is_integer_factor(src_size: Tuple[int,int], tgt_size: Tuple[int,int]) -> Optional[Tuple[int,int]]:
    """Return integer block factors (fx,fy) if target is exact integer division of source."""
    ow, oh = src_size
    tw, th = tgt_size
    if ow % tw == 0 and oh % th == 0:
        return (ow // tw, oh // th)
    return None


def downsample_nearest(mask: np.ndarray, target: Tuple[int,int]) -> np.ndarray:
    """Nearest-neighbor label resize. Preserves labels; can alias thin objects."""
    _check_dependencies()
    tw, th = int(target[0]), int(target[1])
    # use order=0 (nearest). preserve_range to keep integer labels
    out = resize(mask, (th, tw), order=0, preserve_range=True, anti_aliasing=False)
    return out.astype(mask.dtype)


def downsample_area_binary(mask: np.ndarray, target: Tuple[int,int], threshold: float=0.5) -> np.ndarray:
    """
    Area-averaging for binary or probabilistic masks.
    If integer downscale factors exist use downscale_local_mean (fast & exact).
    Otherwise resize float->threshold.
    """
    _check_dependencies()
    src_h, src_w = mask.shape[:2]
    tw, th = int(target[0]), int(target[1])
    int_factors = _is_integer_factor((src_w, src_h), (tw, th))
    # ensure float representation for averaging
    prob = img_as_float32(mask) if mask.dtype != np.float32 else mask.astype(np.float32)
    if prob.ndim == 3:
        # if mask has channels, assume single-channel probability is first channel or collapse
        prob = prob[..., 0]
    if int_factors:
        fx, fy = int_factors
        if fx <= 0 or fy <= 0:
            raise MaskDownsampleError("invalid integer factors")
        if downscale_local_mean is None:
            raise MaskDownsampleError("downscale_local_mean required")
        # block mean then threshold
        # note: downscale_local_mean expects shape (H, W)
        out_mean = downscale_local_mean(prob, (fy, fx))
    else:
        # use anti-aliased resize to target (float), then threshold
        out_mean = resize(prob, (th, tw), order=1, preserve_range=True, anti_aliasing=True)
    return (out_mean >= threshold).astype(np.uint8)


def downsample_majority(mask: np.ndarray, target: Tuple[int,int]) -> np.ndarray:
    """
    Block-wise majority voting. Only valid for integer block factors.
    Returns integer labels (same dtype as input).
    """
    _check_dependencies()
    src_h, src_w = mask.shape[:2]
    tw, th = int(target[0]), int(target[1])
    int_factors = _is_integer_factor((src_w, src_h), (tw, th))
    if not int_factors:
        # fallback: use area/per-class approach
        logger.info("majority fallback to perclass_area (non-integer factors)")
        return downsample_perclass_area(mask, target)
    fx, fy = int_factors
    if fx > 1000 or fy > 1000:
        raise MaskDownsampleError("block factors too large for majority pooling")
    # reshape blocks and compute mode
    shp = mask.shape
    if mask.ndim == 2:
        arr = mask
        # reshape to (th, fy, tw, fx)
        arr = arr.reshape((th, fy, tw, fx))
        # move axes to gather blocks
        arr = arr.transpose((0,2,1,3)).reshape((th, tw, fy*fx))
        # compute mode per block
        from scipy import stats
        mode_vals, _ = stats.mode(arr, axis=1, nan_policy='omit')
        out = mode_vals.ravel().astype(mask.dtype)
        out = out.reshape((th, tw))
        # return with shape (H, W)
        return out
    else:
        # multi-channel: treat each unique vector as label by argmax over channels
        raise MaskDownsampleError("majority for multi-channel masks not implemented")


def downsample_perclass_area(mask: np.ndarray, target: Tuple[int,int]) -> np.ndarray:
    """
    For multi-class label maps. Convert to one-hot, compute area-mean per class,
    then argmax to assign final label.
    Works for non-integer factors using anti-aliased float resize.
    """
    _check_dependencies()
    if mask.ndim != 2:
        # assume channels last with one-hot or probability channels
        if mask.ndim == 3:
            chans = mask.shape[2]
            # if mask is already probabilities per class, resize each channel
            resized = np.stack([resize(mask[..., c], (target[1], target[0]), order=1, preserve_range=True, anti_aliasing=True) for c in range(chans)], axis=-1)
            out = np.argmax(resized, axis=-1).astype(np.int32)
            return out
        else:
            raise MaskDownsampleError("unsupported mask shape")
    # mask is integer labels per pixel
    labels = np.unique(mask)
    if labels.size == 2 and np.array_equal(labels, [0,1]):
        # binary -> use area
        return downsample_area_binary(mask, target)
    # build one-hot per label
    H, W = mask.shape
    tw, th = int(target[0]), int(target[1])
    onehots = [(mask == lab).astype(np.float32) for lab in labels]
    resized = [resize(ch, (th, tw), order=1, preserve_range=True, anti_aliasing=True) for ch in onehots]
    stack = np.stack(resized, axis=-1)
    out_idx = np.argmax(stack, axis=-1)
    # map index back to label values
    label_map = {i: int(lab) for i, lab in enumerate(labels)}
    vec = np.vectorize(label_map.get)(out_idx)
    return vec.astype(mask.dtype)


def downsample_skeleton_preserve(mask: np.ndarray, target: Tuple[int,int], dilation_radius: int = 1) -> np.ndarray:
    """
    Preserve thin structures:
      1) skeletonize binary mask
      2) downsample skeleton with nearest
      3) dilate to recover approximate thickness
    Works only for binary masks.
    """
    _check_dependencies()
    if skeletonize is None or binary_dilation is None or disk is None:
        raise MaskDownsampleError("skimage.morphology required for skeleton_preserve")
    if mask.ndim != 2:
        raise MaskDownsampleError("skeleton_preserve supports 2D binary masks only")
    # ensure binary dtype
    bin_mask = (mask > 0).astype(np.uint8)
    ske = skeletonize(bin_mask)
    # nearest downsample skeleton (to preserve exact pixels where possible)
    ske_ds = resize(ske.astype(np.float32), (target[1], target[0]), order=0, preserve_range=True, anti_aliasing=False)
    # dilate skeleton to re-expand thin shapes
    selem = disk(dilation_radius)
    dil = binary_dilation(ske_ds.astype(bool), selem)
    return dil.astype(np.uint8)


def downsample_mask(mask: np.ndarray, src_size: Tuple[int,int], cfg: MaskDownsampleConfig) -> np.ndarray:
    """
    Top-level function.
    - mask: numpy array. Shape (H,W) for label maps, or (H,W,C) for probabilistic/multi-channel.
    - src_size: (width, height) of source image. Used to check integer factors.
    - cfg: MaskDownsampleConfig
    Returns downsampled mask as numpy array.
    """
    _check_dependencies()
    tw, th = cfg.target_size
    # if mask provided does not match src_size, we assume mask already matches pixel grid
    # ensure shape orientation (H,W)
    if mask.ndim == 3 and mask.shape[2] in (1,):
        mask = mask[...,0]

    method = cfg.method.lower()
    logger.info("downsample_mask method=%s src_size=%s target=%s", method, src_size, cfg.target_size)

    if method == "nearest":
        return downsample_nearest(mask, cfg.target_size)
    if method == "area":
        return downsample_area_binary(mask, cfg.target_size, threshold=cfg.threshold)
    if method == "majority":
        # try majority; if not integer factors fallback to perclass_area
        try:
            return downsample_majority(mask, cfg.target_size)
        except MaskDownsampleError as e:
            logger.info("majority failed (%s). Falling back to perclass_area.", str(e))
            return downsample_perclass_area(mask, cfg.target_size)
    if method == "perclass_area":
        return downsample_perclass_area(mask, cfg.target_size)
    if method == "skeleton_preserve":
        return downsample_skeleton_preserve(mask, cfg.target_size, dilation_radius=1)
    raise MaskDownsampleError(f"unsupported method: {cfg.method}")
