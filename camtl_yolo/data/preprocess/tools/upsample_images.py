#!/usr/bin/env python3
"""
upscale.py

Small-step image upscaling utilities suitable for medical images (e.g. 512->640).
Strategies:
 - lanczos      : high-quality classical resample (recommended for small upscales)
 - bicubic      : classical resample with smoothness
 - sr           : optional deep-learning super-resolution (Real-ESRGAN/EDSR) if configured

Design goals:
 - Atomic functions
 - Typed signatures
 - Dataclass config
 - Clear logging and custom errors
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

from PIL import Image

# Resampling enum compatibility (Pillow 10+)
_RES = getattr(Image, "Resampling", Image)
RES_LANCZOS = _RES.LANCZOS
RES_BICUBIC = _RES.BICUBIC
RES_BILINEAR = _RES.BILINEAR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class UpscaleError(RuntimeError):
    """Raised for upscale pipeline failures."""


@dataclass(frozen=True)
class UpscaleConfig:
    target_size: Tuple[int, int]           # (width, height)
    strategy: str = "lanczos"              # "lanczos" | "bicubic" | "sr"
    sr_backend: Optional[str] = None       # "real-esrgan" | "edsr" etc. (optional)
    sr_model_path: Optional[Path] = None   # if using local model weights


def _compute_scale(orig: Tuple[int, int], target: Tuple[int, int]) -> float:
    ow, oh = orig
    tw, th = target
    return max(tw / ow, th / oh)


def _validate_target(orig: Tuple[int, int], target: Tuple[int, int]) -> None:
    ow, oh = orig
    tw, th = target
    if tw < ow or th < oh:
        logger.warning("Target is smaller than source; this script is focused on upscaling.")


def upscale_lanczos(im: Image.Image, target: Tuple[int, int]) -> Image.Image:
    """Single-step Lanczos upsample."""
    return im.resize(target, RES_LANCZOS)


def upscale_bicubic(im: Image.Image, target: Tuple[int, int]) -> Image.Image:
    """Single-step Bicubic upsample."""
    return im.resize(target, RES_BICUBIC)


def upscale_sr(im_path: Path, target: Tuple[int, int], backend: Optional[str] = None, model_path: Optional[Path] = None) -> Image.Image:
    """
    Wrapper to call a supervised SR backend. This function tries to call a backend if available.
    Implementation notes:
      - This function does NOT bundle an SR model. The environment must provide one.
      - Two safe patterns:
        1) call an installed python package (real_esrgan, rrdbnet) -> import and run
        2) call an external CLI (Real-ESRGAN) via subprocess with input/output files
    Here we prefer a conservative default: try to import `realesrgan` or raise an informative UpscaleError.
    """
    try:
        # Example with real-esrgan python package (if installed)
        from realesrgan import RealESRGAN  # type: ignore
    except Exception as exc:
        raise UpscaleError(
            "SR backend not available. Install 'realesrgan' or use strategy 'lanczos'/'bicubic'."
        ) from exc

    # If import succeeded, run inference
    # instantiate model; assume CUDA if available
    device = "cuda" if RealESRGAN.get_device_name().lower().startswith("cuda") else "cpu"
    try:
        # The following is a minimal example; actual usage depends on package version.
        with Image.open(im_path) as _im:
            with RealESRGAN(device, scale=2) as model:
                model.load_weights(str(model_path)) if model_path else None
                sr_im = model.predict(_im)
                # final resize to exact target if scale not exact
                if sr_im.size != target:
                    sr_im = sr_im.resize(target, RES_LANCZOS)
                return sr_im
    except Exception as exc:
        raise UpscaleError("SR backend failed during inference.") from exc


def upscale_image(src: Path, dst: Path, cfg: UpscaleConfig) -> None:
    """
    Load src image, upscale according to cfg, save to dst.
    Raises UpscaleError on failure.
    """
    if not src.is_file():
        raise UpscaleError(f"source missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            orig_size = im.size
            _validate_target(orig_size, cfg.target_size)
            scale = _compute_scale(orig_size, cfg.target_size)
            logger.debug("Upscale: src=%s orig=%s target=%s scale=%.3f strategy=%s", src.name, orig_size, cfg.target_size, scale, cfg.strategy)

            if cfg.strategy == "lanczos":
                out = upscale_lanczos(im, cfg.target_size)
            elif cfg.strategy == "bicubic":
                out = upscale_bicubic(im, cfg.target_size)
            elif cfg.strategy == "sr":
                out = upscale_sr(src, cfg.target_size, backend=cfg.sr_backend, model_path=cfg.sr_model_path)
            else:
                raise UpscaleError(f"unknown strategy: {cfg.strategy}")

            out.save(dst, format="PNG")
            logger.debug("Wrote upscaled image %s size=%s", dst, out.size)
    except UpscaleError:
        raise
    except Exception as exc:
        raise UpscaleError(str(exc)) from exc


if __name__ == "__main__":
    # Minimal CLI for quick testing.
    import argparse
    p = argparse.ArgumentParser(description="Upscale small-step images (e.g. 512->640).")
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--target", type=int, nargs=2, default=(640, 640))
    p.add_argument("--strategy", choices=("lanczos", "bicubic", "sr"), default="lanczos")
    p.add_argument("--sr-backend", type=str, default=None)
    p.add_argument("--sr-model", type=Path, default=None)
    args = p.parse_args()

    cfg = UpscaleConfig(target_size=(int(args.target[0]), int(args.target[1])),
                        strategy=args.strategy,
                        sr_backend=args.sr_backend,
                        sr_model_path=args.sr_model)
    upscale_image(args.src, args.dst, cfg)
