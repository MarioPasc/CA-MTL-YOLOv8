#!/usr/bin/env python3
"""
mask_upscale.py

Robust small-step upscaling for segmentation masks (512->640 etc).

Strategies:
 - 'nearest'           : nearest-neighbor (label-preserving)
 - 'prob_bicubic'      : float bicubic (or lanczos) then threshold (recommended for soft masks)
 - 'perclass_prob'     : resize each class probability then argmax (multi-class)
 - 'nearest_dilate'    : nearest then morphological dilation (recover thin widths)
 - 'skeleton_upscale'  : skeletonize -> upscale -> dilate (preserve connectivity)

Dependencies: numpy, scikit-image (skimage). Use pip install scikit-image numpy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from skimage.transform import resize
from skimage.morphology import dilation, disk, skeletonize
from skimage.util import img_as_float32

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass(frozen=True)
class MaskUpscaleConfig:
    target_size: Tuple[int, int]           # (width, height)
    strategy: str = "prob_bicubic"         # see strategies above
    threshold: float = 0.5                 # for probabilistic -> binary
    dilation_radius: int = 1               # morphological radius for nearest_dilate / skeleton paths
    preserve_dtype: bool = True            # try to return integer labels if input was integer


class MaskUpscaleError(RuntimeError):
    pass


def _ensure_float(mask: np.ndarray) -> np.ndarray:
    return img_as_float32(mask)


def upscale_nearest(mask: np.ndarray, target: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor upsample. Good for discrete label maps."""
    th, tw = target[1], target[0]
    out = resize(mask, (th, tw), order=0, preserve_range=True, anti_aliasing=False)
    return out.astype(mask.dtype)


def upscale_prob_bicubic(mask: np.ndarray, target: Tuple[int, int], threshold: float = 0.5, use_lanczos: bool = False) -> np.ndarray:
    """
    Upsample soft/probabilistic mask using bicubic (order=3) or lanczos (order=4 if available).
    Return binary mask after threshold.
    """
    th, tw = target[1], target[0]
    if mask.ndim == 3 and mask.shape[2] > 1:
        # multi-channel probability -> per-channel resize then argmax externally
        raise MaskUpscaleError("use perclass_prob for multi-channel masks")
    prob = _ensure_float(mask)
    # order 3 = bicubic. use order=4 for very high-quality if supported.
    order = 3
    outf = resize(prob, (th, tw), order=order, preserve_range=True, anti_aliasing=True)
    return (outf >= threshold).astype(np.uint8)


def upscale_perclass_prob(mask: np.ndarray, target: Tuple[int, int]) -> np.ndarray:
    """
    Multi-class probability map (H,W,C) -> resize each channel (bicubic) -> argmax -> label map.
    If the input is integer label map, caller should one-hot it first.
    """
    th, tw = target[1], target[0]
    if mask.ndim != 3:
        raise MaskUpscaleError("perclass_prob expects (H,W,C) probability array")
    chans = mask.shape[2]
    probs_resized = [resize(mask[..., c], (th, tw), order=3, preserve_range=True, anti_aliasing=True) for c in range(chans)]
    stacked = np.stack(probs_resized, axis=-1)
    out = np.argmax(stacked, axis=-1).astype(np.int32)
    return out


def upscale_nearest_then_dilate(mask: np.ndarray, target: Tuple[int, int], radius: int = 1) -> np.ndarray:
    """Nearest upsample followed by morphological dilation to recover thickness."""
    up = upscale_nearest(mask, target)
    if up.ndim != 2:
        # only binary/discrete this applies to
        return up
    selem = disk(radius)
    dil = dilation((up > 0).astype(np.uint8), selem)
    return dil.astype(mask.dtype)


def upscale_skeleton_preserve(mask: np.ndarray, target: Tuple[int, int], radius: int = 1) -> np.ndarray:
    """
    For thin structures:
      1) skeletonize binary mask
      2) upscale skeleton (nearest)
      3) dilate to approximate thickness
    """
    if mask.ndim != 2:
        raise MaskUpscaleError("skeleton_upscale expects 2D binary mask")
    bin_mask = (mask > 0).astype(np.uint8)
    ske = skeletonize(bin_mask)
    th, tw = target[1], target[0]
    ske_up = resize(ske.astype(np.float32), (th, tw), order=0, preserve_range=True, anti_aliasing=False)
    selem = disk(radius)
    return dilation((ske_up > 0).astype(np.uint8), selem).astype(np.uint8)


def upscale_mask(mask: np.ndarray, src_size: Tuple[int, int], cfg: MaskUpscaleConfig) -> np.ndarray:
    """
    Top-level function.
    - mask: numpy array. Shape (H,W) integer labels or (H,W) float probabilities or (H,W,C) per-class probs.
    - src_size: (width, height) of source image (used only for checks).
    - cfg: MaskUpscaleConfig
    """
    logger.debug("upscale_mask strategy=%s src_size=%s target=%s", cfg.strategy, src_size, cfg.target_size)
    strat = cfg.strategy.lower()
    if strat == "nearest":
        return upscale_nearest(mask, cfg.target_size)
    if strat == "prob_bicubic":
        return upscale_prob_bicubic(mask, cfg.target_size, threshold=cfg.threshold)
    if strat == "perclass_prob":
        return upscale_perclass_prob(mask, cfg.target_size)
    if strat == "nearest_dilate":
        return upscale_nearest_then_dilate(mask, cfg.target_size, radius=cfg.dilation_radius)
    if strat == "skeleton_upscale":
        return upscale_skeleton_preserve(mask, cfg.target_size, radius=cfg.dilation_radius)
    raise MaskUpscaleError(f"unknown strategy: {cfg.strategy}")


# Example usage (not executed here):
# cfg = MaskUpscaleConfig(target_size=(640,640), strategy='prob_bicubic', threshold=0.5)
# out = upscale_mask(mask_array, (512,512), cfg)
