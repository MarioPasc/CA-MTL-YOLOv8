"""
CA-MTL-YOLO package initializer.

Ensures the vendored 'ultralytics' package is importable as a top-level module.
This mirrors the previous behavior where the project preferred its bundled
ultralytics over an external installation.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _ensure_vendored_ultralytics_on_path() -> None:
	"""Prepend the vendored ultralytics root to sys.path if present.

	We add '<repo>/camtl_yolo/external/ultralytics' to sys.path so that
	'import ultralytics' resolves to the bundled copy located at
	'camtl_yolo/external/ultralytics/ultralytics'.
	"""

	pkg_root: Path = Path(__file__).resolve().parent
	vendor_root: Path = pkg_root / "external" / "ultralytics"
	if vendor_root.is_dir():
		vendor_str = str(vendor_root)
		if vendor_str not in sys.path:
			sys.path.insert(0, vendor_str)


_ensure_vendored_ultralytics_on_path()

# Eagerly import to register the top-level name in sys.modules so absolute
# imports like 'from ultralytics.engine.model import Model' succeed in vendored code.
try:  # pragma: no cover - import robustness only
	if "ultralytics" not in sys.modules:
		importlib.import_module("ultralytics")
except Exception:
	# Defer failures to callers that actually require ultralytics
	pass

__all__: list[str] = []