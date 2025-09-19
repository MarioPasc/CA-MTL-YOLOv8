"""
Command-line interface to train CA-MTL-YOLOv8 models.

Usage example:
  camtl_yolo.train --cfg camtl_yolo/model/configs/hyperparams/defaults.yaml \
                   --set epochs=5 batch=2 device=cpu

Notes
- --cfg must point to a YAML file with hyperparameters (Ultralytics-style).
- Use --set key=value to override YAML fields at runtime (repeatable).
- Relative paths inside the YAML (e.g., model, data, project) are resolved
  relative to the cfg file directory for robustness.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Tuple


def _parse_key_value(s: str) -> Tuple[str, Any]:
    """Parse a key=value string and convert the value to a sensible type.

    Attempts to parse ints, floats, bools, None, and simple Python literals
    (lists/tuples/dicts) using ast.literal_eval. Falls back to raw string.
    """
    import ast

    if "=" not in s:
        raise argparse.ArgumentTypeError(f"Override '{s}' must be in key=value format")
    k, v = s.split("=", 1)
    k = k.strip()
    v = v.strip()
    if not k:
        raise argparse.ArgumentTypeError("Override key cannot be empty")
    # Try boolean/None
    lowered = v.lower()
    if lowered in {"true", "false"}:
        return k, lowered == "true"
    if lowered == "none":
        return k, None
    # Try numeric
    try:
        if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
            return k, int(v)
        return k, float(v)
    except ValueError:
        pass
    # Try literal eval for lists/tuples/dicts/complex types
    try:
        return k, ast.literal_eval(v)
    except Exception:
        return k, v


def _deep_update(dst: MutableMapping[str, Any], src: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Deep-merge mapping 'src' into 'dst' in-place and return dst.

    - For nested dicts, updates recursively.
    - For other types, overwrites.
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


def _resolve_path_fields(cfg: Dict[str, Any], base_dir: Path, fields: Iterable[str]) -> None:
    """Resolve path-like fields in the config relative to 'base_dir' if they are relative.

    Mutates cfg in-place.
    """
    for field in fields:
        if field in cfg and isinstance(cfg[field], (str, os.PathLike)) and cfg[field]:
            p = Path(cfg[field])
            if not p.is_absolute():
                cfg[field] = str((base_dir / p).resolve())


def _apply_seed_settings(cfg: Mapping[str, Any]) -> None:
    """Set Python, NumPy, and PyTorch seeds if present in cfg.

    Honors 'deterministic' flag to configure PyTorch deterministic mode.
    """
    seed = cfg.get("seed")
    if seed is None:
        return
    try:
        import random
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))

        if bool(cfg.get("deterministic", False)):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        # Seeding is best-effort; training can still proceed.
        pass


def load_cfg(path: str, overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Load a YAML config from 'path' and merge optional flat overrides.

    Also resolves relative path fields to be relative to the YAML directory.
    """
    # Local import to avoid hard dependency at module import time
    import yaml  # type: ignore[import-not-found]

    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    # Apply flat overrides (k=v) at top-level
    if overrides:
        cfg = dict(cfg)  # shallow copy
        _deep_update(cfg, dict(overrides))

    # Resolve common path-like fields
    base_dir = cfg_path.parent
    _resolve_path_fields(cfg, base_dir, fields=(
        "model",
        "data",
        "project",
        "tracker",
        "source",
        "cfg",
    ))

    return cfg


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="camtl_yolo.train",
        description="Train CA-MTL-YOLOv8 using a hyperparameter YAML (--cfg)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Path to hyperparameter YAML (Ultralytics-style). Relative paths inside will be resolved relative to this file.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        type=_parse_key_value,
        action="append",
        default=[],
        metavar="key=value",
        help="Override hyperparameters from the YAML. Can be used multiple times.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    """Entry point for console script. Returns POSIX exit code."""
    args = parse_args(argv)

    # Build overrides dict from repeated --set key=value
    overrides = {k: v for k, v in (args.overrides or [])}

    # Load config and apply seed/determinism
    cfg = load_cfg(args.cfg, overrides=overrides)
    _apply_seed_settings(cfg)

    # Import trainer lazily to avoid importing torch at module import time
    from camtl_yolo.model.engine.train import CAMTLTrainer

    # Kick off training
    trainer = CAMTLTrainer(overrides=cfg)
    trainer.train()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
