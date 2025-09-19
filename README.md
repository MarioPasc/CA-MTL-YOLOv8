# CA‑MTL‑YOLOv8

Cross‑Attention Multi‑Task Learning for YOLOv8‑based Object Detection and Segmentation.

This repository extends the YOLOv8 family with attention modules and a two‑phase training pipeline to jointly learn instance detection and semantic/instance segmentation under domain shift. It vendors a compatible copy of Ultralytics YOLO under `camtl_yolo/external/ultralytics` to ensure consistent behavior without external conflicts.


## Highlights

- Single model for detection + segmentation with shared backbone/neck
- New plug‑in attention modules:
	- CTAM (Cross‑Task Attention): fuse cues across tasks per scale
	- CSAM (Cross‑Scale Attention): self‑attention across scales within a task
	- FPMA (Feature Propagation with Attention): content‑aware decoder up‑path
- Two‑phase training for domain adaptation
	1) DomainShift1: warm‑up on source domain (segmentation‑only), detection head frozen
	2) CAMTL: alternating multi‑task batches on target (detection+segmentation)
- Auxiliary losses: cross‑task consistency from boxes→masks, attention alignment, L2‑SP
- Vendored Ultralytics ensures reproducibility without PyPI conflicts


## Repository structure

Key paths only (see tree for full layout):

- `camtl_yolo/model/configs/`
	- `models/camtl_yolov8.yaml` — model graph with backbone, neck, CTAM/CSAM/FPMA, heads
	- `hyperparams/defaults.yaml` — Ultralytics‑style hyperparameters and runtime flags
	- `data/data.yaml` — dataset root and split metadata
- `camtl_yolo/model/`
	- `model.py` — CAMTL model (builds from YAML, task‑aware weight loading/saving)
	- `engine/train.py` — trainer integrating datasets, AMP/EMA, schedulers, saving
	- `engine/val.py` — validator (metrics plumbing)
	- `nn/ctam.py`, `nn/csam.py`, `nn/fpma.py`, `nn/seghead.py` — attention + seg head
	- `losses/` — detection, segmentation (BCE+Dice), consistency, alignment, regularizers
	- `utils/` — normalization (DualBN/GN), EMA, samplers
- `camtl_yolo/cli/train.py` — command‑line entrypoint `camtl_yolo.train`
- `camtl_yolo/external/ultralytics/` — vendored Ultralytics package
- `tests/` — smoke tests for build/forward/train


## Model architecture (YOLOv8 + attention)

- Backbone/Neck: YOLOv8‑style CSP‑Darknet + C2f PAN at P3/P4/P5
- Segmentation decoder: top‑down with FPMA instead of naive upsample+add
- Detection head: decoupled YOLOv8 Detect
- Attention modules (plug‑ins; all are `nn.Module`):
	- CTAM: transformer cross‑attention at each scale. Q=target, K/V=source task. Residual to the target feature. Output shape: [B, C_tgt, H, W]
	- CSAM: cross‑scale attention within task stream across {P3,P4,P5}. Concatenate MHA outputs → 1×1 fuse → residual
	- FPMA: decoder up‑path replacing skip‑add with cross‑attention (Q=fine, K/V=coarse)

See `camtl_yolo/model/configs/models/camtl_yolov8.yaml` for the full graph and channel bookkeeping.


## Training pipeline

Two stages with shared code:

1) DomainShift1 (warm‑up)
	 - Train segmentation on source domain (e.g., retinography)
	 - Detection head frozen; Seg stream uses GroupNorm
	 - Pretrained weights: COCO segmentation (`yolov8{SCALE}-seg.pt`) for backbone/neck; COCO detection (`yolov8{SCALE}.pt`) for Detect head

2) CAMTL (multi‑task, domain adaptation)
	 - Alternate detection and segmentation batches on target (e.g., angiography)
	 - Backbone/Detect use DualBN; Seg stream uses GroupNorm
	 - Ratio det:seg configurable via `--set camtl_ratio=[X,Y]` or `CAMTL_RATIO` in model YAML

Auxiliary losses (automatically handled in `model.py`):
- Cross‑task consistency (boxes→pseudo mask on det‑only images)
- Attention alignment across domains
- L2‑SP regularization on backbone/neck
Total loss = det + seg + λ_cons + λ_align + λ_L2SP


## Datasets

Expected processed structure (see `camtl_yolo/model/configs/data/data.yaml`):

```
root/
	├─ angiography/
	│   ├─ detect/{images,labels}
	│   └─ segment/{images,labels}
	├─ retinography/{images,labels}
	└─ splits.json
```
Preprocessed datasets are provided here:
- https://drive.google.com/drive/folders/10qBYj3A_OTis2zUm55f5S0pZwHTyBeaa?usp=sharing

You can also explore the data preprocessing pipeline under `camtl_yolo/data/preprocess/`.


## Installation

Requirements:
- Python 3.10+
- PyTorch ≥ 1.12 (CUDA) and TorchVision ≥ 0.13

This project is a Python package using setuptools. It includes a vendored `ultralytics` package and should be installed in a clean virtual environment to avoid conflicts with a system `ultralytics`.

Quick install (editable):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

The `pip install -e .` step uses `pyproject.toml` and will register the console script `camtl_yolo.train`.


## Configuration

- Hyperparameters/runtime: `camtl_yolo/model/configs/hyperparams/defaults.yaml`
	- Contains Ultralytics‑style keys (epochs, batch, imgsz, lr0, lrf, augmentations, export, etc.)
- Model definition: `camtl_yolo/model/configs/models/camtl_yolov8.yaml`
	- Important keys:
		- `TASK`: DomainShift1 or CAMTL
		- `SCALE`: n|s|m|l|x
		- `PRETRAINED_MODELS_PATH`: folder with expected weights
			- DomainShift1 expects: `yolov8{SCALE}-seg.pt` and `yolov8{SCALE}.pt`
			- CAMTL expects: `yolov8{SCALE}-domainshift1.pt`
- Data root and splits: `camtl_yolo/model/configs/data/data.yaml`


## Usage

Entry point: `camtl_yolo.train` (see `camtl_yolo/cli/train.py`).

Example 1 — Train DomainShift1 warm‑up on CPU for a quick sanity check:

```bash
camtl_yolo.train \
	--cfg camtl_yolo/model/configs/hyperparams/defaults.yaml \
	--set device=cpu epochs=2 batch=2 imgsz=512
```

Example 2 — Switch to CAMTL, set det:seg ratio, and point to your preprocessed data root:

```bash
camtl_yolo.train \
	--cfg camtl_yolo/model/configs/hyperparams/defaults.yaml \
	--set model=camtl_yolo/model/configs/models/camtl_yolov8.yaml \
			 data=camtl_yolo/model/configs/data/data.yaml \
			 epochs=100 batch=4 device=0 \
			 camtl_ratio=[1,2]
```

Notes
- Use repeated `--set key=value` to override any YAML field at runtime
- Relative paths in YAML are resolved relative to the YAML file location
- The trainer creates task‑aware dataloaders automatically from `TASK`


## Pretrained weights

Place your YOLOv8 checkpoint files under the folder specified by `PRETRAINED_MODELS_PATH` in `camtl_yolov8.yaml`.

- DomainShift1:
	- `yolov8{SCALE}-seg.pt` (backbone+neck from seg ckpt)
	- `yolov8{SCALE}.pt` (Detect head mapped safely; classification branch skipped if `nc` differs)
- CAMTL:
	- `yolov8{SCALE}-domainshift1.pt` (fine‑tuned checkpoint produced after DomainShift1)

Task‑aware checkpoints saved during training use filenames like `yolov8{SCALE}-domainshift1.pt` or `yolov8{SCALE}-camtl.pt`.


## Tests

Smoke tests are included and can be run with pytest after installation:

```bash
pytest -q
```

They cover model build/forward paths and basic train/val flows.


## Visuals

Sample images are under `assets/imgs/` and can be used to sanity‑check predictions and plotting.


## Acknowledgements

- Ultralytics YOLOv8 — vendored for stability under `camtl_yolo/external/ultralytics`
- Community contributions and open datasets that enable research on cross‑domain angiography and retinography


## License

See `LICENSE` for details.


## How to cite

We are currently writing a paper on this, until it is published, please cite with:

```bibtex
@misc{pascual-gonzalez2025camtl-yolov8,
	title        = {CA-MTL-YOLOv8: Cross-Attention Multi-Task Learning for Object Detection and Segmentation},
	author       = {Mario Pascual-Gonz{\'a}lez and Ezequiel L{\'o}pez-Rubio},
	year         = {2025},
	howpublished = {GitHub repository},
	url          = {https://github.com/MarioPasc/CA-MLT-YOLOv8},
	note         = {Version 0.1}
}
```

For questions, feel free to open an issue or reach the authors via the emails listed in `pyproject.toml`.

