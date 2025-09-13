#!/usr/bin/env python3
"""
downsample.py

Progressive, anti-aliased image downsampling for medical images.

Design:
 - Prefer progressive integer halving (Image.reduce) while image >> target.
 - Apply mild Gaussian pre-filter proportional to scale to reduce aliasing.
 - Final resize with Lanczos (high-quality) filter.
 - Minimal memory footprint. Atomic functions. Typed. Logging.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

from PIL import Image, ImageFilter

# Use Resampling enum when available (Pillow 10+). Fall back to constants.
_RES = getattr(Image, "Resampling", Image)
RES_LANCZOS = _RES.LANCZOS
RES_BILINEAR = _RES.BILINEAR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ImageProcessingError(RuntimeError):
    """Raised when an image cannot be processed."""


@dataclass(frozen=True)
class DownsampleConfig:
    target_size: Tuple[int, int]  # (w, h)
    strategy: str = "progressive"  # "progressive"|"gauss_lanczos"|"lanczos"
    max_progressive_factor: int = 4  # max times to half before final resize
    gaussian_sigma_clamp: Tuple[float, float] = (0.5, 2.0)


def _compute_scale(orig: Tuple[int, int], target: Tuple[int, int]) -> float:
    ow, oh = orig
    tw, th = target
    return max(ow / tw, oh / th)


def _compute_gaussian_sigma(scale: float, clamp: Tuple[float, float] = (0.5, 2.0)) -> float:
    """Heuristic sigma proportional to log(scale)."""
    lo, hi = clamp
    sigma = 0.5 * math.log1p(max(1.0, scale))
    return max(lo, min(hi, sigma))


def progressive_downsample(im: Image.Image, target: Tuple[int, int], max_halves: int = 4) -> Image.Image:
    """
    Progressive integer halving using Image.reduce (box-filter) then final Lanczos.
    This reduces aliasing when downscaling by large factors.
    """
    tw, th = target
    tmp = im
    w, h = tmp.size
    halves = 0
    # reduce while still at least 2x larger than target and under max_halves
    while halves < max_halves and (w // 2) >= tw and (h // 2) >= th:
        tmp = tmp.reduce(2)  # box filter reduce by 2
        w, h = tmp.size
        halves += 1
        logger.debug("progressive reduce -> %dx%d (halves=%d)", w, h, halves)
    # Final high-quality resample
    return tmp.resize((tw, th), RES_LANCZOS)


def gaussian_then_lanczos(im: Image.Image, target: Tuple[int, int], clamp: Tuple[float, float]) -> Image.Image:
    """
    Mild Gaussian blur proportional to scale followed by Lanczos resize.
    Good when progressive halving isn't applicable or to further suppress aliasing.
    """
    scale = _compute_scale(im.size, target)
    sigma = _compute_gaussian_sigma(scale, clamp)
    logger.debug("gaussian sigma=%.3f for scale=%.3f", sigma, scale)
    blurred = im.filter(ImageFilter.GaussianBlur(radius=sigma))
    return blurred.resize(target, RES_LANCZOS)


def lanczos_direct(im: Image.Image, target: Tuple[int, int]) -> Image.Image:
    """Direct single-step Lanczos resampling."""
    return im.resize(target, RES_LANCZOS)


def downsample_image(src: Path, dst: Path, cfg: DownsampleConfig) -> None:
    """
    Load image at src, downsample according to cfg and save to dst.
    Raises ImageProcessingError on failure.
    """
    if not src.is_file():
        raise ImageProcessingError(f"source image missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(src) as im:
            im = im.convert("RGB")  # normalize mode
            orig_size = im.size
            logger.info("src=%s orig=%s target=%s strategy=%s", src.name, orig_size, cfg.target_size, cfg.strategy)
            if cfg.strategy == "progressive":
                out = progressive_downsample(im, cfg.target_size, cfg.max_progressive_factor)
            elif cfg.strategy == "gauss_lanczos":
                out = gaussian_then_lanczos(im, cfg.target_size, cfg.gaussian_sigma_clamp)
            elif cfg.strategy == "lanczos":
                out = lanczos_direct(im, cfg.target_size)
            else:
                raise ImageProcessingError(f"unknown strategy: {cfg.strategy}")
            out.save(dst, format="PNG")
            logger.info("wrote %s (size=%s)", dst, out.size)
    except Exception as e:
        raise ImageProcessingError(str(e)) from e


def batch_downsample(inputs: list[Tuple[Path, Path]], cfg: DownsampleConfig, workers: Optional[int] = None) -> None:
    """
    Process multiple images in sequence. Use a ProcessPoolExecutor externally if parallel needed.
    Keeps function atomic and testable.
    """
    for src, dst in inputs:
        downsample_image(src, dst, cfg)


if __name__ == "__main__":
    # CLI for quick runs (kept minimal).
    import argparse
    p = argparse.ArgumentParser(description="Downsample images with anti-aliasing.")
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--target", type=int, nargs=2, default=(640, 640))
    p.add_argument("--strategy", choices=("progressive", "gauss_lanczos", "lanczos"), default="progressive")
    args = p.parse_args()
    cfg = DownsampleConfig(target_size=(int(args.target[0]), int(args.target[1])), strategy=args.strategy)
    downsample_image(args.src, args.dst, cfg)
