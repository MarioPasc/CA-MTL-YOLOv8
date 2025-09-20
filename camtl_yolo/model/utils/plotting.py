"""
Plotting utilities for CA-MTL-YOLO.

This module centralizes plot styling (scienceplots + LaTeX), color/style
configuration, and figure generation for training metrics.

Usage from trainer:
  from camtl_yolo.model.utils import plotting as plot_utils
  plot_utils.plot_camtl_metrics(trainer)

You can tweak COLORS, LINESTYLES, and other STYLE_* constants below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple
import logging
import csv

import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg") 

# =========================
# Style configuration block
# =========================

# Define colors and line styles here so you can adjust them easily
COLORS: Dict[str, str] = {
	"train": "#1f77b4",  # blue
	"val": "#d62728",    # red
}

LINESTYLES: Dict[str, str] = {
	"train": "-",
	"val": "--",
}

LINEWIDTH: float = 1.8
MARKERSIZE: float = 0  # 0 disables markers; set e.g. 3.0 to enable


def configure_matplotlib() -> None:
	"""
	Configure matplotlib and scienceplots with LaTeX and requested typography.
	Falls back gracefully if LaTeX or scienceplots are not available.
	"""
	try:
		import importlib
		importlib.import_module("scienceplots")
		plt.style.use(["science"])  # base science + latex style
	except Exception as e:  # pragma: no cover - env dependent
		logging.warning("scienceplots not available: %s. Continuing with default style.", e)

	# Requested typography
	plt.rcParams.update(
		{
			"figure.dpi": 600,
			"font.size": 10,
			"font.family": "serif",
			"font.serif": ["Times"],
			"axes.grid": True,
			"axes.spines.top": False,
			"axes.spines.right": False,
			"legend.frameon": False,
			"savefig.bbox": "tight",
		}
	)
	# LaTeX text rendering
	try:
		plt.rcParams["text.usetex"] = True
		plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
	except Exception as e:  # pragma: no cover - env dependent
		logging.warning("LaTeX not available: %s. Falling back to non-LaTeX text.", e)
		plt.rcParams["text.usetex"] = False


# =========================
# Data loading helpers
# =========================

def _fallback_read_results_csv(csv_path: Path) -> Dict[str, List[float]]:
	"""Lightweight CSV reader that returns {column_name: list_of_values}.

	Non-parsable values are converted to NaN to keep list lengths aligned.
	"""
	table: Dict[str, List[float]] = {}
	if not csv_path.exists():
		return table
	with csv_path.open("r", encoding="utf-8") as f:
		reader = csv.reader(f)
		try:
			header = next(reader)
		except StopIteration:
			return {}
		cols: List[List[float]] = [[] for _ in header]
		for row in reader:
			for i, v in enumerate(row):
				try:
					cols[i].append(float(v))
				except Exception:
					# keep alignment
					cols[i].append(float("nan"))
	table = {h: cols[i] for i, h in enumerate(header)}
	return table


def read_results_table(
	csv_path: Path,
	reader: Optional[Callable[[], Mapping[str, List[float]]]] = None,
) -> Dict[str, List[float]]:
	"""Read results into a dict-of-lists table.

	Attempts a trainer-provided reader first (e.g., BaseTrainer.read_results_csv),
	falling back to a small CSV parser.
	"""
	if reader is not None:
		try:
			table = dict(reader())
			if table:
				return table
		except Exception as e:
			logging.warning("read_results_csv failed: %s. Using fallback parser.", e)
	return _fallback_read_results_csv(csv_path)


def _infer_epochs(table: Mapping[str, List[float]]) -> List[float]:
	if "epoch" in table and isinstance(table["epoch"], list):
		return table["epoch"]
	# fallback: length of first column
	first_col = next(iter(table.values()), [])
	return list(range(1, len(first_col) + 1))


# =========================
# Plot generators
# =========================

def plot_camtl_losses(
	table: Mapping[str, List[float]],
	save_path: Path,
	components: Iterable[str] | None = None,
	ncols: int = 3,
) -> Optional[Path]:
	"""Create subplots, one per loss component, plotting train and val curves.

	- Legend is placed below all subplots, outside the figure area.
	- No title is added; axes are labeled.
	- Colors and line styles are taken from the module-level constants.
	"""
	if not table:
		return None

	comps = list(components or ("total", "det", "seg", "cons", "align", "l2sp"))
	# keep only components that exist in at least one of train/val
	comps = [c for c in comps if (f"train/{c}" in table or f"val/{c}" in table)]
	if not comps:
		return None

	epochs = _infer_epochs(table)
	n = len(comps)
	ncols = max(1, int(ncols))
	nrows = (n + ncols - 1) // ncols
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.0 * ncols, 2.6 * nrows), squeeze=False, sharex=True)

	handles_all: List = []
	labels_all: List[str] = []

	for idx, comp in enumerate(comps):
		r, c = divmod(idx, ncols)
		ax = axes[r][c]

		# Train
		h_train = None
		if f"train/{comp}" in table:
			h_train = ax.plot(
				epochs,
				table[f"train/{comp}"],
				label="train",
				color=COLORS.get("train", "C0"),
				linestyle=LINESTYLES.get("train", "-"),
				linewidth=LINEWIDTH,
				markersize=MARKERSIZE,
			)[0]

		# Val
		h_val = None
		if f"val/{comp}" in table:
			h_val = ax.plot(
				epochs,
				table[f"val/{comp}"],
				label="val",
				color=COLORS.get("val", "C3"),
				linestyle=LINESTYLES.get("val", "--"),
				linewidth=LINEWIDTH,
				markersize=MARKERSIZE,
			)[0]

		# Titles to identify each subplot
		ax.set_title(str(comp))

		# Label x only on last row; y only on first column
		if r == nrows - 1:
			ax.set_xlabel("epoch")
		if c == 0:
			ax.set_ylabel("loss")

		# collect handles for global legend, but only once per label
		if h_train and "train" not in labels_all:
			handles_all.append(h_train)
			labels_all.append("train")
		if h_val and "val" not in labels_all:
			handles_all.append(h_val)
			labels_all.append("val")

	# Remove any unused axes
	for j in range(len(comps), nrows * ncols):
		r, c = divmod(j, ncols)
		axes[r][c].axis("off")

	# Global legend at the bottom, outside all subplots
	if handles_all:
		fig.legend(
			handles_all,
			labels_all,
			loc="lower center",
			ncol=len(labels_all),
			bbox_to_anchor=(0.5, -0.02),
			frameon=False,
		)

	fig.tight_layout()
	fig.savefig(save_path)
	plt.close(fig)
	return save_path


def plot_camtl_dice(
	table: Mapping[str, List[float]],
	save_path: Path,
	scales: Iterable[str] | None = None,
	ncols: int = 2,
) -> Optional[Path]:
	"""Create subplots for segmentation Dice (val and train if present).

	Expected keys: 'val/dice_p3', 'val/dice_p4', 'val/dice_p5', 'val/dice_full'.
	If 'train/dice_*' exist, they will be overlaid in the same axis.
	"""
	if not table:
		return None

	scales_list = list(scales or ("p3", "p4", "p5", "full"))
	# Keep only scales that exist in at least val or train
	def has_any(s: str) -> bool:
		return (f"val/dice_{s}" in table) or (f"train/dice_{s}" in table)

	scales_list = [s for s in scales_list if has_any(s)]
	if not scales_list:
		return None

	epochs = _infer_epochs(table)
	n = len(scales_list)
	ncols = max(1, int(ncols))
	nrows = (n + ncols - 1) // ncols
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.0 * ncols, 2.6 * nrows), squeeze=False, sharex=True, sharey=True)

	handles_all: List = []
	labels_all: List[str] = []

	for idx, s in enumerate(scales_list):
		r, c = divmod(idx, ncols)
		ax = axes[r][c]

		# Train dice if available
		h_train = None
		if f"train/dice_{s}" in table:
			h_train = ax.plot(
				epochs,
				table[f"train/dice_{s}"],
				label="train",
				color=COLORS.get("train", "C0"),
				linestyle=LINESTYLES.get("train", "-"),
				linewidth=LINEWIDTH,
				markersize=MARKERSIZE,
			)[0]

		# Val dice
		h_val = None
		key = f"val/dice_{s}"
		if key in table:
			h_val = ax.plot(
				epochs,
				table[key],
				label="val",
				color=COLORS.get("val", "C3"),
				linestyle=LINESTYLES.get("val", "--"),
				linewidth=LINEWIDTH,
				markersize=MARKERSIZE,
			)[0]

		# Titles to identify each subplot
		ax.set_title(str(s))

		# Label x only on last row; y only on first column (same Dice stat across subplots)
		if r == nrows - 1:
			ax.set_xlabel("epoch")
		if c == 0:
			ax.set_ylabel("Dice")

		if h_train and "train" not in labels_all:
			handles_all.append(h_train)
			labels_all.append("train")
		if h_val and "val" not in labels_all:
			handles_all.append(h_val)
			labels_all.append("val")

	# Remove any unused axes
	for j in range(len(scales_list), nrows * ncols):
		r, c = divmod(j, ncols)
		axes[r][c].axis("off")

	# Global legend at the bottom, outside all subplots
	if handles_all:
		fig.legend(
			handles_all,
			labels_all,
			loc="lower center",
			ncol=len(labels_all),
			bbox_to_anchor=(0.5, -0.02),
			frameon=False,
		)

	fig.tight_layout()
	fig.savefig(save_path)
	plt.close(fig)
	return save_path


def plot_camtl_metrics(trainer) -> None:
	"""End-to-end plotting driven by the trainer instance.

	- Reads results via trainer.read_results_csv() with fallback to trainer.csv
	- Generates loss subplots and per-scale Dice subplots
	- Notifies trainer via on_plot callback for each artifact
	"""
	try:
		configure_matplotlib()
	except Exception:
		# Styling is best-effort; do not fail plotting if style config breaks
		pass

	# Load metrics table
	csv_path = Path(getattr(trainer, "csv", "results.csv"))
	reader_fn = getattr(trainer, "read_results_csv", None)
	if callable(reader_fn):
		table = read_results_table(csv_path, reader=reader_fn)
	else:
		table = read_results_table(csv_path, reader=None)

	if not table:
		return

	save_dir = Path(getattr(trainer, "save_dir", "."))

	# Loss components
	try:
		loss_path = save_dir / "results_camtl_losses.png"
		out1 = plot_camtl_losses(table, loss_path)
		if out1 is not None:
			try:
				trainer.on_plot(str(out1))
			except Exception:
				pass
	except Exception as e:
		logging.warning("CAMTL loss plotting failed: %s", e)

	# Dice metrics
	try:
		dice_path = save_dir / "results_camtl_dice.png"
		out2 = plot_camtl_dice(table, dice_path)
		if out2 is not None:
			try:
				trainer.on_plot(str(out2))
			except Exception:
				pass
	except Exception as e:
		logging.warning("CAMTL dice plotting failed: %s", e)


# =========================
# Augmentation debug grid (pre/post + downsampled)
# =========================

def overlay_mask_on_image_bgr(
    img_bgr: np.ndarray,
    mask2d: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.6,
) -> np.ndarray:
    """Overlay a binary/soft mask on a BGR image with given color and alpha.

    Background (mask==0) remains unchanged. Activated pixels blend towards 'color'.
    mask2d can be float in [0,1] or uint8 in {0,255}.
    """
    assert img_bgr.ndim == 3 and img_bgr.shape[2] == 3, "img_bgr must be HxWx3"
    h, w = img_bgr.shape[:2]
    if mask2d.shape != (h, w):
        mask2d = cv2.resize(mask2d, (w, h), interpolation=cv2.INTER_NEAREST)

    img = img_bgr.astype(np.float32)
    if mask2d.dtype != np.float32:
        m = mask2d.astype(np.float32)
        if m.max() > 1.0:
            m = m / 255.0
    else:
        m = mask2d
    m = np.clip(m, 0.0, 1.0)
    overlay = np.zeros_like(img) + np.array(color, dtype=np.float32)
    # Blend only where m>0
    m3 = m[..., None]
    comp = img * (1.0 - alpha * m3) + overlay * (alpha * m3)
    return np.clip(comp, 0, 255).astype(np.uint8)


def _to_vis_bgr_from_mask(mask2d: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Convert a single-channel mask to 3-channel BGR image for visualization."""
    h, w = target_hw
    if mask2d.shape != (h, w):
        mask2d = cv2.resize(mask2d, (w, h), interpolation=cv2.INTER_NEAREST)
    if mask2d.dtype != np.uint8:
        m = np.clip(mask2d, 0.0, 1.0)
        mask_u8 = (m * 255).astype(np.uint8)
    else:
        mask_u8 = mask2d
    return cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)


def save_aug_debug_grid(
    out_path: Path,
    pre_img_bgr: np.ndarray,
    pre_mask2d: Optional[np.ndarray],
    pre_p3: Optional[np.ndarray],
    pre_p4: Optional[np.ndarray],
    pre_p5: Optional[np.ndarray],
    post_img_bgr: np.ndarray,
    post_mask2d: Optional[np.ndarray],
    post_p3: Optional[np.ndarray],
    post_p4: Optional[np.ndarray],
    post_p5: Optional[np.ndarray],
) -> Path:
    """Save a 2x4 grid image using Matplotlib with annotated tiles.

    Row 0: pre-augmentation [image+full overlay] | [p3] | [p4] | [p5]
    Row 1: post-augmentation [image+full overlay] | [p3] | [p4] | [p5]
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert BGR images to RGB for Matplotlib
    pre_img_rgb = cv2.cvtColor(pre_img_bgr, cv2.COLOR_BGR2RGB)
    post_img_rgb = cv2.cvtColor(post_img_bgr, cv2.COLOR_BGR2RGB)

    # Normalize masks to [0,1] and build masked arrays so zero stays transparent
    def _norm_mask(m: Optional[np.ndarray]) -> Optional[np.ma.MaskedArray]:
        if m is None:
            return None
        mm = m.astype(np.float32)
        if mm.max() > 1.0:
            mm = mm / 255.0
        mm = np.clip(mm, 0.0, 1.0)
        return np.ma.masked_where(mm <= 0.0, mm)

    m_pre = _norm_mask(pre_mask2d)
    m_post = _norm_mask(post_mask2d)

    # For p3/p4/p5, visualize masks (resized to image shape for clarity)
    H, W = pre_img_bgr.shape[:2]
    def _resize_mask(m: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if m is None:
            return None
        return cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

    pre_p3_r = _resize_mask(pre_p3)
    pre_p4_r = _resize_mask(pre_p4)
    pre_p5_r = _resize_mask(pre_p5)
    post_p3_r = _resize_mask(post_p3)
    post_p4_r = _resize_mask(post_p4)
    post_p5_r = _resize_mask(post_p5)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6), squeeze=True, constrained_layout=True)

    # Top row: pre
    ax = axes[0, 0]
    ax.imshow(pre_img_rgb)
    if m_pre is not None:
        ax.imshow(m_pre, cmap="Reds", vmin=0, vmax=1, alpha=0.6)
    ax.set_title("pre: overlay")
    ax.axis("off")

    for j, (ax, m, title) in enumerate([
        (axes[0, 1], pre_p3_r, "pre: p3"),
        (axes[0, 2], pre_p4_r, "pre: p4"),
        (axes[0, 3], pre_p5_r, "pre: p5"),
    ]):
        if m is None:
            ax.imshow(np.zeros((H, W), dtype=np.uint8), cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(m, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    # Bottom row: post
    ax = axes[1, 0]
    ax.imshow(post_img_rgb)
    if m_post is not None:
        ax.imshow(m_post, cmap="Reds", vmin=0, vmax=1, alpha=0.6)
    ax.set_title("post: overlay")
    ax.axis("off")

    for j, (ax, m, title) in enumerate([
        (axes[1, 1], post_p3_r, "post: p3"),
        (axes[1, 2], post_p4_r, "post: p4"),
        (axes[1, 3], post_p5_r, "post: p5"),
    ]):
        if m is None:
            ax.imshow(np.zeros((H, W), dtype=np.uint8), cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(m, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)
    return out_path

