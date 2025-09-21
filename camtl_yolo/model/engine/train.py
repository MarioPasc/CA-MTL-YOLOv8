# camtl_yolo/engine/train.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import math
from copy import copy
import csv
import yaml # type: ignore
import matplotlib.pyplot as plt
from camtl_yolo.model.utils import plotting as plot_utils

import torch
import torch.nn as nn
from torch import optim

from camtl_yolo.model.dataset import MultiTaskJSONDataset, build_dataloader
from camtl_yolo.model.engine.val import CAMTLValidator
from camtl_yolo.model.model import CAMTL_YOLO
from camtl_yolo.model.utils.normalization import set_bn_domain
from camtl_yolo.model.utils.samplers import AlternatingLoader, map_domain_name
from camtl_yolo.external.ultralytics.ultralytics.engine.trainer import BaseTrainer
from camtl_yolo.external.ultralytics.ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from camtl_yolo.external.ultralytics.ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model
from camtl_yolo.external.ultralytics.ultralytics.utils.torch_utils import strip_optimizer, unwrap_model

from camtl_yolo.model.utils.ema import SafeModelEMA
from camtl_yolo.external.ultralytics.ultralytics.utils.torch_utils import (
    EarlyStopping, attempt_compile, one_cycle, select_device, torch_distributed_zero_first,
    unwrap_model, TORCH_2_4
)
from camtl_yolo.external.ultralytics.ultralytics.utils.checks import check_amp
from camtl_yolo.external.ultralytics.ultralytics.utils import callbacks, LOGGER, RANK, LOCAL_RANK
from camtl_yolo.external.ultralytics.ultralytics.utils.checks import check_imgsz
from camtl_yolo.external.ultralytics.ultralytics.utils.autobatch import check_train_batch_size


class CAMTLTrainer(BaseTrainer):
    """
    Trainer for CA-MTL-YOLOv8.

    - Task-aware datasets: DomainShift1 (seg-only) and CAMTL (det:seg ratio).
    - DualBN domain switching per batch.
    - Uses BaseTrainer infra (AMP, EMA, schedulers, checkpointing).
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks=None):
        overrides = overrides or {}
        super().__init__(cfg, overrides, _callbacks)
        # One-time ratio logging and epoch loss tracking
        self._ratio_logged: bool = False
        self._prev_epoch_losses: List[float] | None = None
        # Register lightweight callbacks
        self.add_callback("on_train_start", self._log_ratio_once)
        self.add_callback("on_train_epoch_end", self._capture_epoch_losses)
        self.add_callback("on_fit_epoch_end", self._clear_cuda_cache)

    def _setup_train(self, world_size):
        """Copy of BaseTrainer._setup_train with a SafeModelEMA swap."""
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        if hasattr(self.model, "init_criterion"):
            self.model.criterion = self.model.init_criterion()

        self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)

        # Freeze policy from BaseTrainer (unchanged)
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # AMP + DDP (unchanged)
        self.amp = torch.tensor(self.args.amp).to(self.device)
        if self.amp and RANK in {-1, 0}:
            callbacks_backup = callbacks.default_callbacks.copy()
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup
        if RANK > -1 and world_size > 1:
            from torch import distributed as dist
            dist.broadcast(self.amp.int(), src=0)
        self.amp = bool(self.amp)
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # imgsz and stride
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs

        # Batch size
        if self.batch_size < 1 and RANK == -1:
            self.args.batch = self.batch_size = self.auto_batch()

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train")
        if RANK in {-1, 0}:
            self.test_loader = self.get_dataloader(
                self.data.get("val") or self.data.get("test"),
                batch_size=batch_size if self.args.task == "obb" else batch_size * 2,
                rank=-1,
                mode="val",
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            # ---- SAFE EMA here ----
            self.ema = SafeModelEMA(unwrap_model(self.model), decay=float(getattr(self.args, "ema_decay", 0.9999)))
            self.ema.update_attr(self.model, include=["yaml", "nc", "names", "args", "stride"])
             
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
        iterations = self._compute_total_iterations(
            loader=self.train_loader,
            batch_size=self.batch_size,
            nbs=self.args.nbs,
            epochs=self.epochs,
        )

        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler and resume
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False

        self.resume_training(ckpt)
        # After resume: restore EMA if present, else hard-sync from current weights
        try:
            if isinstance(ckpt, dict) and "ema" in ckpt and ckpt["ema"]:
                self.ema.load_state_dicts(ckpt["ema"])
            else:
                self.ema.hard_sync(unwrap_model(self.model))
        except Exception as e:
            LOGGER.warning(f"[EMA] restore/sync failed: {e}")

        self.scheduler.last_epoch = self.start_epoch - 1
        self.run_callbacks("on_pretrain_routine_end")

    # ------------------------ Dataset & Loader ------------------------ #

    def get_dataset(self) -> Dict[str, Any]:
        """Return dataset meta dict expected by BaseTrainer."""

        data_yaml = Path(self.args.data)
        with data_yaml.open("r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)

        names = meta.get("names") or meta.get("classes") or {0: "object"}
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        channels = int(meta.get("channels", 3))
        nc = int(meta.get("nc", len(names)))

        # We pass the path and let get_dataloader build proper datasets/loaders
        return {
            "data_yaml": str(data_yaml),
            "train": str(data_yaml),
            "val": str(data_yaml),
            "names": names,
            "nc": nc,
            "channels": channels,
        }

    def _build_mt_dataset(
        self,
        data_yaml: str,
        split: str,
        tasks: List[str],
        filter_is: str | None,
        imgsz: int,
        batch: int,
        shuffle: bool,
    ):
        ds = MultiTaskJSONDataset(
            data_yaml=data_yaml,
            split=split, # type: ignore[arg-type]
            tasks=tasks,
            filter_is=filter_is,  # type: ignore[arg-type]
            imgsz=imgsz,
            augment=(split == "train"),
            hyp=self.args,  # reuse same hyp dict
            batch_size=batch,
            rect=False,
            channels=self.data["channels"],
        )
        dl = build_dataloader(
            ds, batch_size=batch, workers=self.args.workers if split == "train" else max(1, self.args.workers // 2), shuffle=shuffle
        )
        return ds, dl

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Construct dataloader(s) per active task. Returns an AlternatingLoader in CAMTL train."""
        assert mode in {"train", "val"}
        with torch_distributed_zero_first(rank):
            from camtl_yolo.model.tasks import select_dataset_tasks_for_mode

            task = str(getattr(self, "active_task", None) or self._get_active_task_from_model())
            task_keys = select_dataset_tasks_for_mode(task)[mode]
            imgsz = self.args.imgsz
            shuffle = mode == "train"

            if task == "DomainShift1":
                # Segmentation-only stream from source (retinography)
                _, seg_loader = self._build_mt_dataset(
                    data_yaml=dataset_path,
                    split=mode,
                    tasks=task_keys,
                    filter_is="seg",
                    imgsz=imgsz,
                    batch=batch_size,
                    shuffle=shuffle,
                )
                seg_loader.ratio = (0, 1)  # for logging
                return seg_loader

            if task == "CAMTL" and mode == "train":
                # Two loaders: detection(target) and segmentation(target)
                _, det_loader = self._build_mt_dataset(
                    data_yaml=dataset_path,
                    split="train",
                    tasks=["angiography_detection"],
                    filter_is="det",
                    imgsz=imgsz,
                    batch=batch_size,
                    shuffle=True,
                )
                _, seg_loader = self._build_mt_dataset(
                    data_yaml=dataset_path,
                    split="train",
                    tasks=["angiography_segmentation"],
                    filter_is="seg",
                    imgsz=imgsz,
                    batch=batch_size,
                    shuffle=True,
                )
                # Ratio X:Y from args or YAML
                ratio = getattr(self.args, "camtl_ratio", None) or getattr(self.model.yaml, "CAMTL_RATIO", [1, 1])
                rx, ry = int(ratio[0]), int(ratio[1])
                alt = AlternatingLoader(det_loader, seg_loader, ratio=(rx, ry))
                # convenience attrs for logs
                setattr(alt, "num_workers", getattr(det_loader, "num_workers", 0) + getattr(seg_loader, "num_workers", 0))
                setattr(alt, "batch_size", batch_size)
                setattr(alt, "ratio", (rx, ry))
                if not hasattr(alt, "reset"):
                    def _alt_reset() -> None:
                        for _ldr in (det_loader, seg_loader):
                            if hasattr(_ldr, "reset"):
                                try: _ldr.reset()
                                except Exception: pass
                    setattr(alt, "reset", _alt_reset)
                return alt

            # For CAMTL val: single union loader, no alternation
            _, union_loader = self._build_mt_dataset(
                data_yaml=dataset_path, split="val", tasks=task_keys, filter_is=None, imgsz=imgsz, batch=batch_size, shuffle=False
            )
            union_loader.ratio = (1, 1)
            return union_loader

    # ------------------------ Model & Attrs ------------------------ #

    def _get_active_task_from_model(self) -> str:
        try:
            return str(getattr(self.model, "yaml", {}).get("TASK", "DomainShift1"))
        except Exception:
            return "DomainShift1"

    def get_model(self, cfg: str | dict | None = None, weights: str | Path | None = None, verbose: bool = True):
        """Instantiate CAMTL model."""
        model = CAMTL_YOLO(cfg=cfg or self.args.model, ch=self.data["channels"], nc=self.data["nc"], verbose=verbose and RANK in {-1, 0})
        # keep a copy of active task to drive dataloaders
        self.active_task = str(model.yaml.get("TASK", "DomainShift1"))
        return model

    def set_model_attributes(self):
        """Attach common attributes for logging."""
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args

    # ------------------------ Preprocess & Loss Routing ------------------------ #

    def preprocess_batch(self, batch: dict) -> dict:
        """Move to device, normalize, and set BN domain based on batch."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        batch["img"] = batch["img"].float()

        # Expect single-domain batches; set DualBN switch
        if "bn_domain" in batch and len(batch["bn_domain"]):
            # choose the first; enforce consistency
            dom = map_domain_name(batch["bn_domain"][0])
            set_bn_domain(dom)

        return batch

    # ------------------------ Validator & Loss Names ------------------------ #

    def get_validator(self):
        """Return CAMTL validator."""
        # losses: det, seg, cons, align, l2sp, total
        if self.model.task == "DomainShift1":
            self.loss_names = (
                "det", "seg", "cons", "align", "l2sp", "total",
                "p3_bce", "p4_bce", "p5_bce",
                "p3_dice", "p4_dice", "p5_dice",
            )
        else:
            self.loss_names = (
                "det", "seg", "cons", "align", "l2sp", "total",
                "box", "cls", "dfl",
                "p3_bce", "p4_bce", "p5_bce",
                "p3_dice", "p4_dice", "p5_dice",
            )
        return CAMTLValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def label_loss_items(self, loss_items: torch.Tensor | None = None, prefix: str = "train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            vals = [round(float(x), 6) for x in (loss_items.tolist() if torch.is_tensor(loss_items) else loss_items)]
            return dict(zip(keys, vals))
        return keys

    def final_eval(self):
        """Evaluate using our task-aware checkpoint (DomainShift1/CAMTL) so criteria exist."""

        # 1) Save a fresh task checkpoint to guarantee it exists
        task_ckpt_path: Path | None = None
        try:
            task_ckpt_path = unwrap_model(self.model).save_task_checkpoint(
                self.wdir, epoch=self.epoch, optimizer=self.optimizer
            )
            task_ckpt_path = Path(task_ckpt_path)
        except Exception as e:
            LOGGER.warning(f"[final_eval] save_task_checkpoint failed: {e}")

        # 2) Load task checkpoint as a MODULE and attach criteria
        val_model = None
        if task_ckpt_path and task_ckpt_path.exists():
            ckpt = torch.load(task_ckpt_path, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict) and "model" in ckpt and hasattr(ckpt["model"], "state_dict"):
                val_model = ckpt["model"]
            elif hasattr(ckpt, "state_dict"):
                val_model = ckpt  # already a nn.Module
            else:
                # Fallback: rebuild model and load state_dict from ckpt["model"]
                from camtl_yolo.model.model import CAMTL_YOLO
                val_model = CAMTL_YOLO(cfg=self.model.yaml, ch=self.data["channels"], nc=self.data["nc"], verbose=False)
                sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
                if isinstance(sd, dict):
                    val_model.load_state_dict(sd, strict=False)
            val_model = None
            return

        # 4) Prepare module for validation and ensure criteria are attached
        base = unwrap_model(val_model)
        try:
            base.nc = self.data["nc"]
            base.names = self.data["names"]
            base.args = self.args  # SimpleNamespace from get_cfg
        except Exception:
            pass
        try:
            if not all(hasattr(base, a) for a in ("segment_criterion", "detect_criterion", "consistency_criterion", "align_criterion")):
                base.init_criterion()
        except Exception as e:
            LOGGER.warning(f"[final_eval] init_criterion on task model failed: {e}")

        val_model.to(self.device)
        use_half = (self.device.type == "cuda") and bool(self.amp)
        val_model = val_model.half() if use_half else val_model.float()
        base.eval()

        # 5) Validate with the MODULE (not a Path)
        LOGGER.info(f"\nValidating task checkpoint {task_ckpt_path}...")
        self.validator.args.plots = self.args.plots
        self.validator.args.compile = False
        self.metrics = self.validator(model=val_model)
        self.metrics.pop("fitness", None)
        self.run_callbacks("on_fit_epoch_end")


    # ------------------------ Save Hook ------------------------ #

    def save_model(self):
        """Augment BaseTrainer saving with task-aware checkpoint."""
        super().save_model()
        try:
            unwrap_model(self.model).save_task_checkpoint(self.wdir, epoch=self.epoch, optimizer=self.optimizer)
        except Exception as e:
            LOGGER.warning(f"Task checkpoint save failed: {e}")

    # ------------------------ Metrics persistence & plots ------------------------ #

    def save_metrics(self, metrics: Mapping[str, float]) -> None:  # type: ignore[override]
        """
        Save training metrics with CAMTL extras (per-scale Dice and ratio) merged in.
        Delegates the actual CSV write to BaseTrainer.save_metrics.
        """
        try:
            enriched = dict(metrics)
            enriched.update(self._camtl_extra_metrics(metrics))
        except Exception as e:
            LOGGER.warning(f"[save_metrics] enriching metrics failed: {e}")
            enriched = dict(metrics)
        super().save_metrics(enriched)

    def _camtl_extra_metrics(self, metrics: Mapping[str, float]) -> dict[str, float]:
        """
        Collect CAMTL-specific extras not guaranteed to be present in Base metrics.
        Returns a dict with any of: val/dice_p3, val/dice_p4, val/dice_p5, val/dice_full,
        plus static training mix ratio keys camtl/ratio_det, camtl/ratio_seg when available.
        """
        extras: dict[str, float] = {}

        # 1) Attempt to source per-scale Dice from returned metrics or validator fields.
        def find_scalar(name: str) -> Optional[float]:
            # Preferred CSV key
            for k in (f"val/dice_{name}", f"dice_{name}", f"metrics/seg/dice_{name}", f"seg/dice_{name}"):
                if k in metrics and metrics[k] is not None:
                    try:
                        return float(metrics[k])  # type: ignore[arg-type]
                    except Exception:
                        pass
            # Common validator attributes
            cand_attrs = ("last_seg_dice", "seg_dice", "dice_scales", "last_dice", "dice")
            for attr in cand_attrs:
                obj = getattr(self.validator, attr, None)
                if isinstance(obj, dict):
                    for kk in (name, f"dice_{name}"):
                        if kk in obj and obj[kk] is not None:
                            try:
                                return float(obj[kk])
                            except Exception:
                                continue
            return None

        for nm in ("p3", "p4", "p5", "full"):
            v = find_scalar(nm)
            if v is not None:
                extras[f"val/dice_{nm}"] = v

        # 2) Persist det:seg scheduling ratio for reference.
        try:
            if hasattr(self, "train_loader") and hasattr(self.train_loader, "ratio"):
                rx, ry = getattr(self.train_loader, "ratio", (None, None))
                if rx is not None and ry is not None:
                    extras["camtl/ratio_det"] = float(rx)
                    extras["camtl/ratio_seg"] = float(ry)
        except Exception:
            pass

        return extras

    def plot_metrics(self) -> None:  # type: ignore[override]
        """
        Plot standard Ultralytics metrics, then CAMTL-specific figures if columns exist.
        """

        #super().plot_metrics()
        try:
           self._plot_camtl_metrics()
        except Exception as e:
            LOGGER.warning(f"[plot_metrics] CAMTL plotting failed: {e}")

    def _plot_camtl_metrics(self) -> None:
        """Delegate CAMTL plots to centralized plotting utilities."""
        try:
            plot_utils.plot_camtl_metrics(self)
        except Exception as e:
            LOGGER.warning(f"[plot_metrics] CAMTL plotting failed: {e}")

    # ------------------------ Progress String ------------------------ #

    def progress_string(self) -> str:  # type: ignore[override]
        names = tuple(self.loss_names) 
        return ("\n" + "%11s" * (4 + len(names))) % (
            "Epoch",
            "GPU_mem",
            *names,
            "Instances",
            "Size",
        )

    # ------------------------ Helpers ------------------------ #

    def _compute_total_iterations(self, loader, batch_size: int, nbs: int, epochs: int) -> int:
        """
        Return total optimizer iterations for schedulers/optimizers.

        If the loader exposes `.dataset`, use the canonical Ultralytics logic:
            ceil(len(dataset) / max(batch_size, nbs)) * epochs
        Else fall back to: len(loader) * epochs, where len(loader) is number of batches/epoch.
        """
        import math
        try:
            ds_len = len(loader.dataset)  # standard DataLoader path
            per_epoch = math.ceil(ds_len / max(batch_size, nbs))
        except Exception:
            per_epoch = len(loader)       # AlternatingLoader path
        return int(per_epoch * epochs)


    def _log_ratio_once(self, trainer: BaseTrainer):
        """Log det:seg ratio exactly once at the start of training."""
        if getattr(self, "_ratio_logged", False):
            return
        ratio_str = None
        if hasattr(self, "train_loader") and hasattr(self.train_loader, "ratio"):
            rx, ry = getattr(self.train_loader, "ratio", (1, 1))
            ratio_str = f"[det:seg={rx}:{ry}]"
        if ratio_str:
            LOGGER.info(f"Training mix ratio {ratio_str}")
        self._ratio_logged = True

    def _capture_epoch_losses(self, trainer: BaseTrainer):
        """Capture averaged losses at the end of each epoch for display next epoch."""
        tl = getattr(trainer, "tloss", None)
        if isinstance(tl, torch.Tensor):
            try:
                self._prev_epoch_losses = [float(x) for x in tl.detach().cpu().tolist()]
            except Exception:
                self._prev_epoch_losses = None
        else:
            self._prev_epoch_losses = None

    def _clear_cuda_cache(self, trainer: BaseTrainer):
        """Clear CUDA cache between epochs to mitigate fragmentation/OOM."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
