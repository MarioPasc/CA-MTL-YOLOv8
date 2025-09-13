"""Minimal image format conversion utility.

This module intentionally exposes ONLY a low-level single-image conversion
function. Parallel/job orchestration is delegated to `pipeline.py`.

Public API:
        convert_image(path, target_ext='.png', delete_original=True, overwrite=True, logger=None)

Behavior:
    * Supports input formats: png, jpg/jpeg, tif/tiff, bmp, gif.
    * If the image already has target_ext and overwrite=True, it is re-saved in
        place to normalize mode (palette/CMYK -> RGB) using an atomic temp file.
    * If overwrite=False and extension already matches, no action is taken.
    * When converting, original stem is preserved; original file optionally
        deleted after successful write.
    * Errors return (False, message) instead of raising to let the pipeline
        aggregate results.
"""
from __future__ import annotations

import tempfile
import shutil
import errno
from pathlib import Path
from typing import Tuple, Optional

from PIL import Image

SUPPORTED_INPUT_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}

def _safe_move(src: Path, dst: Path):
    """Move file atomically; fallback to copy for cross-device moves.

    If a cross-device link error (EXDEV) occurs, performs copy2 + unlink.
    """
    try:
        src.replace(dst)
    except OSError as e:
        if e.errno == errno.EXDEV:  # cross-device
            shutil.copy2(src, dst)
            try:
                src.unlink()
            except OSError:
                pass
        else:
            raise

def convert_image(
    path: str | Path,
    target_ext: str = ".png",
    delete_original: bool = True,
    overwrite: bool = True,
    logger = None,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Convert a single image to target_ext (atomic replace).

    Args:
        path: file path.
        target_ext: desired extension (with leading dot).
        delete_original: delete original if extension changes.
        overwrite: if False and extension already matches, skip conversion.
        logger: optional logger.
    Returns:
        (success, new_path|None, error|None)
    """
    p = Path(path)
    target_ext = target_ext.lower()
    if not p.exists() or p.suffix.lower() not in SUPPORTED_INPUT_EXTS:
        return (False, None, "unsupported-or-missing")
    try:
        if p.suffix.lower() == target_ext:
            if not overwrite:
                return (True, str(p), None)
            # normalize write
            with Image.open(p) as img:
                img.load()
                if img.mode in ("P", "CMYK"):
                    img = img.convert("RGB")
                with tempfile.NamedTemporaryFile(delete=False, suffix=target_ext) as tmp:
                    tmp_path = Path(tmp.name)
                img.save(tmp_path)
            _safe_move(tmp_path, p)
            return (True, str(p), None)
        out_path = p.with_suffix(target_ext)
        with Image.open(p) as img:
            img.load()
            if img.mode in ("P", "CMYK"):
                img = img.convert("RGB")
            with tempfile.NamedTemporaryFile(delete=False, suffix=target_ext) as tmp:
                tmp_path = Path(tmp.name)
            img.save(tmp_path)
        _safe_move(tmp_path, out_path)
        if delete_original and p.exists() and p != out_path:
            try:
                p.unlink()
            except OSError:
                pass
        return (True, str(out_path), None)
    except Exception as e:  # pragma: no cover - robustness
        if logger:
            logger.error(f"convert_image failed: {p} -> {target_ext}: {e}")
        return (False, None, str(e))

__all__ = ["convert_image"]

