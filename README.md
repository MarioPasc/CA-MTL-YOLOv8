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

Additional characteristics from this codebase
- Deep‑supervision segmentation at multiple scales (P3/P4/P5) plus optional fused full‑resolution logits
- Dual BatchNorm for domain adaptation in backbone/detect with per‑batch domain switching; GroupNorm in segmentation stream
- Alternating det:seg batch scheduling with a configurable ratio (e.g., 1:2)
- Built‑in validator reports total loss breakdown and Dice per scale (P3/P4/P5/fused)


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

Segmentation head
- `SegHeadMulti` consumes [P3, P4, P5] and emits logits at those scales; the model forwards a tuple `(det_out, seg_out)` where `seg_out` can include per‑scale tensors and an optional fused full‑res map.
- Loss uses deep supervision with BCE+Dice at P3/P4/P5 (weights configurable).

Normalization strategy
- Segmentation stream (FPMA/CTAM/SegHead) uses GroupNorm at train time for stability under domain shift.
- Backbone + Detect use DualBatchNorm in CAMTL mode; the active BN branch is switched per batch using a domain tag derived from the dataset (source=retinography, target=angiography). This is handled automatically by the trainer.


## Training pipeline

Two stages with shared code:

1) DomainShift1 (warm‑up)
	 - Task: retinography segmentation only; detection head is frozen
	 - Normalization: convert segmentation stream BN→GroupNorm; keep single BN elsewhere
	 - Pretrained weights: COCO segmentation checkpoint for backbone/neck; COCO detection checkpoint mapped into Detect head (cls branch safely skipped if `nc` differs)
	 - Losses: deep‑supervision BCE+Dice on masks; optional L2‑SP on backbone/neck

2) CAMTL (multi‑task, domain adaptation)
	 - Tasks: angiography detection + angiography segmentation
	 - Alternating loader with ratio det:seg configurable via CLI `--set camtl_ratio=[X,Y]` or `CAMTL_RATIO` in model YAML
	 - Normalization: DualBN in backbone/Detect with per‑batch domain switch; GroupNorm in segmentation stream remains
	 - Initialization: loads DomainShift1 fine‑tuned checkpoint `yolov8{SCALE}-domainshift1.pt`
	 - Losses: detect + seg + consistency (boxes→pseudo mask) + attention alignment + L2‑SP

Auxiliary losses (automatically handled in `model.py`):
- Cross‑task consistency (boxes→pseudo mask on det‑only images)
- Attention alignment across domains
- L2‑SP regularization on backbone/neck
Total loss = det + seg + λ_cons + λ_align + λ_L2SP

Loss breakdown and metrics
- The trainer/validator track a 6‑component loss vector: `[det, seg, cons, align, l2sp, total]`.
- Validator also reports Dice for segmentation, including per‑scale metrics: `val/dice_p3`, `val/dice_p4`, `val/dice_p5`, and `val/dice_full`.


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

Dataset tasks and splits
- The dataset interface uses a JSON manifest `splits.json` under the root to list image/mask/label relative paths per task key and split.
- Task keys used by this project:
	- `retinography_segmentation`
	- `angiography_segmentation`
	- `angiography_detection`
- If `splits.json` is missing, a default file is generated at first run using 70/30 train/val for each task.

Batch composition and scheduling
- In CAMTL training, two loaders (det, seg) are alternated following the configured ratio; val uses a union loader.
- Batches carry a `bn_domain` tag (`source`/`target`) used to switch DualBN branches automatically.


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

Loss configuration (`LOSS` section in model YAML)
- Detection loss hyperparameters can be overridden under `det_hyp` (e.g., `box`, `cls`, `dfl`).
- Segmentation deep‑supervision weights: `w_p3`, `w_p4`, `w_p5`.
- Auxiliary losses and scalars: `consistency`, `consistency_loss` (`bce` or other), `align`, and multipliers `lambda_det`, `lambda_seg`, `lambda_cons`, `lambda_align`.


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

Tasks and modes
- DomainShift1 mode trains only retinography segmentation (Detect head frozen, Seg stream uses GroupNorm).
- CAMTL mode alternates angiography detection and segmentation (DualBN enabled, both heads trainable).

Notes
- Detectors and backbones are initialized from COCO checkpoints. Detect head mapping is safe with differing `nc`.
- DualBN branch switching is handled by the trainer based on batch domain; no manual calls needed.
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

Checkpoint save/load behavior
- During training, the trainer saves both the standard Ultralytics checkpoint and a task‑aware checkpoint via `save_task_checkpoint`, including the serialized `model` module (FP32), epoch, optimizer state, and YAML.
- CAMTL training expects the DomainShift1 task checkpoint present under `PRETRAINED_MODELS_PATH` for initialization.


## Tests

Smoke tests are included and can be run with pytest after installation:

```bash
pytest -q
```

They cover model build/forward paths and basic train/val flows.

What’s covered
- Model construction from YAML, forward returning `(det_out, seg_out)`
- Trainer dataloaders and alternating scheduler wiring
- Minimal end‑to‑end train/val loops with EMA and loss reporting


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

