"""Preprocessing pipeline orchestrator.

Currently implemented stage(s):
  1. Image format standardization (convert all images to configured format)

Design:
  - Pipeline reads the same data.yaml used by generator.
  - Accepts already-standardized output directory root and applies steps in-place.
  - Parallelization: each image is processed in a process pool using the
	low-level `convert_image` utility. Masks (segmentation/retinography labels)
	are treated as images for conversion purposes.

Function:
	run_preprocessing(output_dir: str, config: dict, logger) -> None

Config keys used (under preprocessing):
	img_format: target extension (e.g., 'png')
	to_gray: (reserved for future) if True convert to grayscale BEFORE save
	img_size: (reserved for future) (w,h) resize

Notes:
  - Only the image-format step is active now; placeholders for future steps.
  - The function is idempotent: images already in target format are re-saved
	only if overwrite=True (to normalize modes). This is currently True.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from camtl_yolo.data.preprocess.tools.img_format import convert_image

# Dataset modality subpaths relative to output_dir
SEG_IMG = ["angiographies", "segment", "images"]
SEG_LBL = ["angiographies", "segment", "labels"]
DET_IMG = ["angiographies", "detect", "images"]  # labels for detection are text, skip
RET_IMG = ["retinography", "images"]
RET_LBL = ["retinography", "labels"]

IMAGE_DIRS = [SEG_IMG, SEG_LBL, DET_IMG, RET_IMG, RET_LBL]

def _gather_all_images(root: Path) -> List[Path]:
	paths: List[Path] = []
	for parts in IMAGE_DIRS:
		d = root.joinpath(*parts)
		if not d.exists():
			continue
		for fp in d.rglob('*'):
			if fp.is_file():
				# Skip detection label txt
				if fp.suffix.lower() == '.txt':
					continue
				paths.append(fp)
	return paths

def _chunk(iterable, n):
	iterable = list(iterable)
	for i in range(0, len(iterable), n):
		yield iterable[i:i+n]

def run_preprocessing(output_dir: str, config: Dict, logger, num_workers: int | None = None) -> None:
	prep_cfg = (config or {}).get('preprocessing', {})
	target_fmt = prep_cfg.get('img_format', 'png')
	if not target_fmt.startswith('.'):
		target_fmt = '.' + target_fmt

	out_root = Path(output_dir)
	if not out_root.exists():
		logger.warning(f"[pipeline] Output directory does not exist: {out_root}")
		return

	logger.info("[pipeline] Starting preprocessing pipeline")
	logger.info(f"[pipeline] Target image format: {target_fmt}")

	all_images = _gather_all_images(out_root)
	logger.info(f"[pipeline] Collected {len(all_images)} files for potential format conversion")
	if not all_images:
		logger.info("[pipeline] Nothing to process.")
		return

	# Parallel convert
	successes = 0
	failures = 0
	with ProcessPoolExecutor(max_workers=num_workers) as ex:
		futures = {ex.submit(convert_image, str(img), target_fmt, True, True, None): img for img in all_images}
		for fut in as_completed(futures):
			img = futures[fut]
			try:
				ok, new_path, err = fut.result()
			except Exception as e:  # pragma: no cover
				failures += 1
				logger.error(f"[pipeline] Unexpected failure {img}: {e}")
				continue
			if ok:
				successes += 1
			else:
				failures += 1
				logger.error(f"[pipeline] Conversion failed {img}: {err}")

	logger.info(f"[pipeline] Format conversion done. Success: {successes} Failed: {failures}")
	logger.info("[pipeline] Preprocessing pipeline finished")

__all__ = ["run_preprocessing"]

