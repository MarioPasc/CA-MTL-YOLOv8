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
		plt.style.use(["science", "ieee"])  # base science + ieee style
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

