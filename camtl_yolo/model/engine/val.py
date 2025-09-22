# camtl_yolo/engine/val.py
from __future__ import annotations

from copy import copy
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional

from pathlib import Path
import yaml # type: ignore
import contextlib

import numpy as np
import os
import pandas as pd # type: ignore

import torch
import torch.nn as nn
import torch.nn.functional as F

from camtl_yolo.model.utils.normalization import set_bn_domain
from camtl_yolo.model.utils.samplers import map_domain_name
from camtl_yolo.model.utils.plotting import make_final_camtl_viz
from camtl_yolo.model.utils.metrics import dice
from camtl_yolo.external.ultralytics.ultralytics.engine.validator import BaseValidator
from camtl_yolo.external.ultralytics.ultralytics.utils import LOGGER, TQDM
from camtl_yolo.external.ultralytics.ultralytics.utils.ops import Profile
from camtl_yolo.external.ultralytics.ultralytics.utils.torch_utils import smart_inference_mode, unwrap_model
from camtl_yolo.external.ultralytics.ultralytics.utils.metrics import DetMetrics, box_iou
from camtl_yolo.external.ultralytics.ultralytics.utils import nms
from camtl_yolo.external.ultralytics.ultralytics.utils import ops as ULops


class CAMTLValidator(BaseValidator):
    """
    CAMTL (Cross-Attention Multi-Task Learning) validator.

    - Works in-training (mid-epoch) and standalone (final evaluation from weights).
    - Computes detection/segmentation loss components via the model's .loss() API.
    - Accumulates Dice for segmentation batches, including per-scale metrics {p3,p4,p5,full}.
    - Reuses BaseValidator's callback, profiling, and plotting flow.
    """

    # ---------------------- Construction ---------------------- #
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.metrics = SimpleNamespace(
            keys=[
                "val/det", "val/seg", "val/cons", "val/align", "val/l2sp", "val/total",
                # detection metrics to be persisted in results.csv:
                "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
            ]
        )
        # Removed Dice accumulators: no Dice metrics are computed during validation
        self._dice_sums: Dict[str, float] = {}
        self._dice_counts: Dict[str, int] = {}
        # ---- prediction & feature-map capture state ----
        self._save_fm_max: int = 4                       # save 4 times per training
        self._fm_layers: list[int] = []                  # auto-filled when hooking; leave empty to only grab Detect inputs
        self._fm_last: Dict[int, torch.Tensor] = {}      # layer_index -> tensor[B, C, H, W]
        self._fm_hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._logged_epoch_saves: set[int] = set()       # epochs already saved
        self._seg_cache: Dict[str, torch.Tensor] = {}    # latest seg outputs for current batch

        # Detection metrics
        self.det_metrics = DetMetrics()
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move tensors to device, set precision, and switch DualBN branch by batch domain tag.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)

        # standardize image dtype
        img = batch.get("img")
        if img is None:
            raise KeyError("Batch missing 'img' tensor.")
        batch["img"] = img.half() if getattr(self.args, "half", False) else img.float()

        # Set BN domain if provided
        bn_dom = batch.get("bn_domain")
        if isinstance(bn_dom, (list, tuple)) and len(bn_dom):
            try:
                set_bn_domain(map_domain_name(bn_dom[0]))
            except Exception as e:
                LOGGER.warning(f"[Validator] set_bn_domain failed: {e}")

        return batch

    def postprocess(self, preds: Any) -> Any:
        """
        CAMTL does not require NMS or class mapping for Dice computation.
        Pass-through for predictions.
        """
        return preds

    def _postprocess_det(self, det_raw: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Apply NMS to per-image predictions."""
        # args fallbacks
        conf = float(getattr(self.args, "conf", 0.25))
        iou = float(getattr(self.args, "iou", 0.5))
        max_det = int(getattr(self.args, "max_det", 300))
        agnostic = bool(getattr(self.args, "agnostic_nms", False))
        single_cls = bool(getattr(self.args, "single_cls", False))
        nc = 0 if single_cls else len(getattr(self, "names", [])) or 1

        outputs = nms.non_max_suppression(
            det_raw, conf, iou, nc=nc, multi_label=True, agnostic=agnostic, max_det=max_det, end2end=False, rotated=False
        )
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in outputs]


    # ---------------------- Core call ---------------------- #

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """
        Validate mid-epoch (trainer!=None) or standalone (weights or nn.Module).
        Returns a dict of floats. During training, merges stats with labeled loss items from BaseTrainer.
        """
        # Training vs standalone
        self.training = trainer is not None
        self.args.task = trainer.model.task
        augment = False  # no aug at val

        if self.training:
            # device/data from trainer
            self.device = trainer.device
            self.data = trainer.data
            # half precision matches trainer AMP
            self.args.half = self.device.type != "cpu" and trainer.amp
            # pick EMA if available for forward, but compute losses with the underlying unwrapped model
            fwd_model = trainer.model
            if trainer.args.compile and hasattr(fwd_model, "_orig_mod"):
                fwd_model = fwd_model._orig_mod
            fwd_model = fwd_model.half() if self.args.half else fwd_model.float()
            fwd_model.eval()

            # prepare loss accumulator vector like BaseValidator
            self.loss = torch.zeros_like(trainer.loss_items, device=self.device)

            # model whose .loss we call (unwrapped)
            self.base_model = unwrap_model(trainer.model)
        else:
            # Standalone evaluation path: accept a nn.Module or a checkpoint path
            if model is None:
                raise RuntimeError("Standalone validation requires 'model' to be an nn.Module or a checkpoint path.")

            # Load if given a path-like
            from pathlib import Path as _P

            if isinstance(model, (str, _P)):
                ckpt = torch.load(model, map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict):
                    model = ckpt.get("model") or ckpt
            if not isinstance(model, nn.Module):
                raise TypeError("Loaded 'model' is not an nn.Module.")

            # device/precision
            try:
                self.device = next(unwrap_model(model).parameters()).device
            except Exception:
                self.device = torch.device("cpu")
            self.args.half = self.device.type == "cuda"

            fwd_model = model.to(self.device)
            fwd_model = fwd_model.half() if self.args.half else fwd_model.float()
            unwrap_model(fwd_model).eval()
            self.base_model = unwrap_model(fwd_model)

            # Ensure criteria exist on the loaded model
            need = ("segment_criterion", "detect_criterion", "consistency_criterion", "align_criterion")
            if any(not hasattr(self.base_model, n) for n in need) and hasattr(self.base_model, "init_criterion"):
                try:
                    self.base_model.init_criterion()
                except Exception as e:
                    LOGGER.warning(f"[Validator] init_criterion() failed on loaded model: {e}")

        # Profilers and bar
        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(unwrap_model(fwd_model))
        self.jdict = []  # reserved for future JSON exports

        # Collect a small set of images and per-image seg outputs to render a final panel at the end
        viz_imgs: list[torch.Tensor] = []
        viz_segs: list[dict] = []

        # Iterate
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i

            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = fwd_model(batch["img"], augment=augment)

            # Collect up to 16 images and their segmentation outputs for a final visualization panel
            try:
                if len(viz_imgs) < 16:
                    imgs = batch.get("img")  # [B, C, H, W]
                    if torch.is_tensor(imgs) and imgs.ndim == 4:
                        take = int(min(imgs.size(0), 16 - len(viz_imgs)))
                        # Extract segmentation predictions from model output
                        seg_out = self._extract_seg_predictions(preds)
                        for i in range(take):
                            viz_imgs.append(imgs[i].detach().cpu())
                            # Build a dict with expected keys; handle dict or tensor outputs
                            if isinstance(seg_out, dict):
                                seg_dict = {
                                    "full": seg_out.get("full"),
                                    "p3": seg_out.get("p3"),
                                    "p4": seg_out.get("p4"),
                                    "p5": seg_out.get("p5"),
                                }
                            else:
                                # Single tensor -> treat as full-resolution logits
                                seg_dict = {"full": seg_out, "p3": None, "p4": None, "p5": None}
                            # Index per-image if batched tensors are present
                            for k, v in list(seg_dict.items()):
                                if torch.is_tensor(v):
                                    if v.ndim == 4:  # [B,1,h,w]
                                        seg_dict[k] = v[i].detach().cpu()
                                    else:
                                        seg_dict[k] = v.detach().cpu()
                                else:
                                    seg_dict[k] = v
                            viz_segs.append(seg_dict)
            except Exception:
                # Best-effort; do not interrupt validation on visualization issues
                pass

            # Loss accumulation vector for BaseTrainer CSV/print (training only)
            with dt[2]:
                if self.training:
                    try:
                        self.loss += unwrap_model(trainer.model).loss(batch, preds)[1]
                    except Exception as e:
                        LOGGER.warning(f"[Validator] accumulating self.loss failed: {e}")

            # Postprocess
            with dt[3]:
                preds_pp = self.postprocess(preds)
                
            # Make the model available to update_metrics for .loss(batch, preds) in both modes
            self.update_metrics(preds_pp, batch)

            # Optional sample plots
            if getattr(self.args, "plots", False) and batch_i < 3:
                try:
                    self.plot_val_samples(batch, batch_i)
                    self.plot_predictions(batch, preds_pp, batch_i)
                except Exception as e:
                    LOGGER.warning(f"[Validator] plotting failed: {e}")

            self.run_callbacks("on_val_batch_end")

        # Summaries
        stats = self.get_stats()
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")

        # Save the final visualization panel if we collected any samples
        try:
            if viz_imgs:
                imgs_b = torch.stack(viz_imgs, dim=0)  # [N, C, H, W]
                # Map generic task name if needed; default to DomainShift1 layout
                task = getattr(self.args, "task", "DomainShift1")
                # Robustly select next numeric suffix even when >= 10
                prefix = f"val_{str(task).lower()}_"
                saving_folder = os.path.join(self.save_dir, "visualizations")
                os.makedirs(saving_folder, exist_ok=True)
                existing_files = list(Path(saving_folder).glob(f"{prefix}*.png"))
                max_idx = -1
                for p in existing_files:
                    stem = p.stem  # e.g., val_task_12
                    # Split once from the right and parse numeric suffix
                    parts = stem.rsplit("_", 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        try:
                            idx = int(parts[1])
                            if idx > max_idx:
                                max_idx = idx
                        except Exception:
                            pass
                num = max_idx + 1
                out_file = Path(saving_folder) / f"{prefix}{num}.png"
                make_final_camtl_viz(task=str(task), imgs=imgs_b, seg_outputs=viz_segs,
                                     out_path=out_file, max_images=16, rows=4)
        except Exception as e:
            try:
                LOGGER.warning(f"[final_viz] skipped: {e}")
            except Exception:
                pass

        self._remove_fm_hooks()
        # Return dict consistent with BaseValidator
        if self.training:
            # Merge stats with trainer-labeled loss items (prefix val/)
            labeled = trainer.label_loss_items(self.loss.detach().cpu() / len(self.dataloader), prefix="val")
            merged = {**stats, **labeled}
            return {k: round(float(v), 5) for k, v in merged.items()}
        else:
            return stats

    # ---------------------- Metrics ---------------------- #

    def init_metrics(self, model: nn.Module) -> None:
        """
        Initialize counters and per-scale accumulators.
        """
        self.names = getattr(model, "names", {0: "object"})
        self.nc = len(self.names) if isinstance(self.names, (list, dict)) else 1

        # For BaseValidator compatibility
        self.seen = 0
        self.batch_i = 0

        # Per-component loss totals (averaged at the end)
        self._sum_det = 0.0
        self._sum_seg = 0.0
        self._sum_cons = 0.0
        self._sum_align = 0.0
        self._sum_l2sp = 0.0
        self._cnt_det = 0
        self._cnt_seg = 0

        # Dice metrics disabled
        self._dice_sums = {"p3": 0.0, "p4": 0.0, "p5": 0.0, "full": 0.0}
        self._dice_counts = {"p3": 0, "p4": 0, "p5": 0, "full": 0}

    def _update_det_metrics(self, det_preds: list[dict[str, torch.Tensor]], batch: Mapping[str, Any]) -> None:
        """Update DetMetrics with current batch predictions and GT, both in input-pixel space."""
        for si, pred in enumerate(det_preds):
            # select this image's GT
            idx = (batch["batch_idx"] == si)
            cls_t = batch["cls"][idx].squeeze(-1)  # [Nt]
            bxywh = batch["bboxes"][idx]           # normalized
            imgsz = batch["img"].shape[2:]         # (H, W)
            if bxywh.numel():
                scale = torch.tensor([imgsz[1], imgsz[0], imgsz[1], imgsz[0]], device=bxywh.device, dtype=bxywh.dtype)
                txyxy = ULops.xywh2xyxy(bxywh) * scale
            else:
                txyxy = torch.zeros((0, 4), device=batch["img"].device, dtype=torch.float32)

            # Ensure txyxy is a tensor, as it might be inferred as ndarray by some tools
            txyxy = torch.as_tensor(txyxy)

            # predictions already in input-pixel xyxy from Detect → NMS
            predn = {"bboxes": pred["bboxes"], "conf": pred["conf"], "cls": (pred["cls"] * 0 if getattr(self.args, "single_cls", False) else pred["cls"])}

            # build TP matrix at all IoU thresholds
            if cls_t.shape[0] == 0 or predn["cls"].shape[0] == 0:
                tp = np.zeros((predn["cls"].shape[0], self.iouv.numel()), dtype=bool)
            else:
                iou = box_iou(txyxy, predn["bboxes"])
                tp = self.match_predictions(predn["cls"], cls_t, iou).cpu().numpy()

            # update stats
            no_pred = predn["cls"].shape[0] == 0
            self.det_metrics.update_stats(
                {
                    "tp": tp,
                    "conf": np.zeros(0) if no_pred else predn["conf"].detach().cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].detach().cpu().numpy(),
                    "target_cls": cls_t.detach().cpu().numpy(),
                    "target_img": np.unique(cls_t.detach().cpu().numpy()),
                }
            )

    def update_metrics(self, preds: Any, batch: Mapping[str, Any]) -> None:
        """
        Accumulate losses and Dice. Save feature maps and panels at selected epochs.
        Fix: do not mark an epoch as logged before passing the gating checks.
        """
        super().update_metrics(preds, batch)

        # ----- loss accumulation -----
        if isinstance(self.base_model, nn.Module):
            with torch.no_grad():
                loss_fn = getattr(self.base_model, "loss", None)
                if callable(loss_fn):
                    _, items = loss_fn(batch, preds)
                else:
                    items = torch.zeros(15)

            det_val = float(items[0]); seg_val = float(items[1]); cons_val = float(items[2])
            align_val = float(items[3]); l2sp_val = float(items[4])

            # robust task flags
            is_seg_batch = bool(torch.as_tensor(batch.get("is_seg", False)).any().item())
            has_det = ("bboxes" in batch) and torch.is_tensor(batch["bboxes"]) and batch["bboxes"].numel() > 0
            is_det_batch = bool(has_det and not is_seg_batch)

            # accumulate with correct denominators
            if is_det_batch:
                self._sum_det += det_val
                self._cnt_det += 1
            if is_seg_batch:
                self._sum_seg += seg_val
                self._cnt_seg += 1
            # components defined for both streams aggregate over all batches
            self._sum_cons += cons_val
            self._sum_align += align_val
            self._sum_l2sp += l2sp_val

        # ----- detection metrics (NEW) -----
        try:
            # preds may be (det_out, seg_out)
            det_raw = preds[0] if isinstance(preds, (tuple, list)) and len(preds) >= 1 else preds
            if torch.is_tensor(det_raw):
                det_pp = self._postprocess_det(det_raw)  # NMS per image
                self._update_det_metrics(det_pp, batch)
        except Exception as e:
            LOGGER.warning(f"[CAMTLValidator] detection metrics update skipped: {e}")

        # ----- Dice accumulation (unchanged) -----
        is_seg_batch = bool(torch.as_tensor(batch.get("is_seg", False)).any().item())
        if is_seg_batch:
            seg_preds = self._extract_seg_predictions(preds)
            if isinstance(seg_preds, dict):
                for k in ("p3", "p4", "p5"):
                    if k in seg_preds and batch.get(f"mask_{k}") is not None:
                        d = dice(seg_preds[k], batch[f"mask_{k}"])
                        self._dice_sums[k] += float(d); self._dice_counts[k] += 1
                if "full" in seg_preds and batch.get("mask") is not None:
                    d = dice(seg_preds["full"], batch["mask"])
                    self._dice_sums["full"] += float(d); self._dice_counts["full"] += 1

        # ----- timepointed saving gate -----
        save_dir = Path(getattr(self, "save_dir", Path("runs/val/exp")))
        results_csv = save_dir / "results.csv"
        args_yaml = save_dir / "args.yaml"
        if not results_csv.exists() or not args_yaml.exists():
            return
        try:
            df = pd.read_csv(results_csv)
            current_epoch = int(df["epoch"].max()) if "epoch" in df.columns else 0
            with open(args_yaml, "r", encoding="utf-8") as f:
                args = yaml.safe_load(f) or {}
            total_epochs = int(args.get("epochs", 0))
        except Exception:
            return

        if int(getattr(self, "batch_i", 0)) != 0:
            return

        tps = set(int(x) for x in self._timepoints(total_epochs).tolist())
        epoch_1based = int(current_epoch + 1)

        # Correct order: check gates first, then mark as logged after saving
        if (epoch_1based not in tps) or (epoch_1based in self._logged_epoch_saves):
            return

        base_dir = save_dir / "val_preds_fm" / f"epoch_{epoch_1based}"
        fm_dir = base_dir / "feature_maps"; seg_dir = base_dir / "segment"; det_dir = base_dir / "detect"
        base_dir.mkdir(parents=True, exist_ok=True)

        self._capture_once_and_save(
            base_model=self.base_model,
            batch=dict(batch),
            fm_dir=fm_dir,
            seg_dir=seg_dir,
            det_dir=det_dir,
            is_seg_batch=is_seg_batch,
            det_preds=preds if isinstance(preds, (list, tuple)) and not is_seg_batch else None,
        )

        self._logged_epoch_saves.add(epoch_1based)
        LOGGER.info(f"[CAMTLValidator] Saved validation assets for epoch {epoch_1based} to {base_dir}")


    def finalize_metrics(self) -> None:
        """
        No-op hook for now. Reserved for future aggregation on distributed setups.
        """
        return

    def get_stats(self) -> Dict[str, float]:
        """
        Return validation statistics to be merged with loss items by BaseTrainer.
        """
        denom_all = max(1, self._cnt_det + self._cnt_seg)
        avg_det = self._sum_det / max(1, self._cnt_det)
        avg_seg = self._sum_seg / max(1, self._cnt_seg)
        avg_cons = self._sum_cons / denom_all
        avg_align = self._sum_align / denom_all
        avg_l2sp = self._sum_l2sp / denom_all
        avg_total = avg_det + avg_seg + avg_cons + avg_align + avg_l2sp

        # finalize detector metrics into a dict with keys like 'metrics/mAP50(B)'
        try:
            self.det_metrics.process(save_dir=self.save_dir, plot=False, on_plot=self.on_plot)
            det_res = self.det_metrics.results_dict
        except Exception:
            det_res = {
                "metrics/precision(B)": 0.0,
                "metrics/recall(B)": 0.0,
                "metrics/mAP50(B)": 0.0,
                "metrics/mAP50-95(B)": 0.0,
            }

        return {
            "val/det": avg_det, "val/seg": avg_seg, "val/cons": avg_cons, "val/align": avg_align, "val/l2sp": avg_l2sp,
            "val/total": avg_total,
            # these are included so you still see Dice on the console if you print it elsewhere; not persisted by default
            "val/dice_p3": float(self._dice_sums.get("p3", 0.0) / max(1, self._dice_counts.get("p3", 0))),
            "val/dice_p4": float(self._dice_sums.get("p4", 0.0) / max(1, self._dice_counts.get("p4", 0))),
            "val/dice_p5": float(self._dice_sums.get("p5", 0.0) / max(1, self._dice_counts.get("p5", 0))),
            "val/dice_full": float(self._dice_sums.get("full", 0.0) / max(1, self._dice_counts.get("full", 0))),
            # detection metrics will be added to results.csv because we advertised the keys in __init__
            **det_res,
        }

    def print_results(self) -> None:
        """
        Log compact per-epoch summary.
        """
        s = self.get_stats()
        
        LOGGER.info(f"Model task: {self.args.task}")
        if self.args.task == "CAMTL":
            if s['val/align'] != 0.0:
                LOGGER.info(
                    f"Loss: total {s['val/total']:.4f} | det {s['val/det']:.4f} | seg {s['val/seg']:.4f} | cons {s['val/cons']:.4f} | l2sp {s['val/l2sp']:.4f} | align {s['val/align']:.4f}  \n"
                    f"Segmentation: dice p3 {s['val/dice_p3']:.4f} p4 {s['val/dice_p4']:.4f} p5 {s['val/dice_p5']:.4f} full {s['val/dice_full']:.4f} | \n"
                    f"Detection: Precision {s['metrics/precision(B)']:.4f} | Recall {s['metrics/recall(B)']:.4f} | mAP@50 {s['metrics/mAP50(B)']:.4f} | mAP@50-95 {s['metrics/mAP50-95(B)']:.4f}"
                )
            else:
                LOGGER.info(
                    f"Loss: total {s['val/total']:.4f} | det {s['val/det']:.4f} | seg {s['val/seg']:.4f} | cons {s['val/cons']:.4f} | l2sp {s['val/l2sp']:.4f} | \n"
                    f"Segmentation: dice p3 {s['val/dice_p3']:.4f} p4 {s['val/dice_p4']:.4f} p5 {s['val/dice_p5']:.4f} full {s['val/dice_full']:.4f} | \n"
                    f"Detection: Precision {s['metrics/precision(B)']:.4f} | Recall {s['metrics/recall(B)']:.4f} | mAP@50 {s['metrics/mAP50(B)']:.4f} | mAP@50-95 {s['metrics/mAP50-95(B)']:.4f}"
                )
        else:
            if s['val/align'] != 0.0:
                LOGGER.info(
                    f"Loss: total {s['val/total']:.4f} | det {s['val/det']:.4f} | seg {s['val/seg']:.4f} | cons {s['val/cons']:.4f} | l2sp {s['val/l2sp']:.4f} | align {s['val/align']:.4f}  \n"
                    f"Segmentation: dice p3 {s['val/dice_p3']:.4f} p4 {s['val/dice_p4']:.4f} p5 {s['val/dice_p5']:.4f} full {s['val/dice_full']:.4f} | \n"
                )
            else:
                LOGGER.info(
                    f"Loss: total {s['val/total']:.4f} | det {s['val/det']:.4f} | seg {s['val/seg']:.4f} | cons {s['val/cons']:.4f} | l2sp {s['val/l2sp']:.4f} | \n"
                    f"Segmentation: dice p3 {s['val/dice_p3']:.4f} p4 {s['val/dice_p4']:.4f} p5 {s['val/dice_p5']:.4f} full {s['val/dice_full']:.4f} | \n"
                )
    def get_desc(self) -> str:
        """
        Description string for progress bars.
        """
        return "[Validator] Cross-Attention Multi-Task Learning YOLO (M.Pascual et al. 2025)"

    @property
    def metric_keys(self) -> list[str]:
        """
        Keys that this validator can emit in stats.
        """
        return list(self.metrics.keys)

    # ---------------------- Utilities ---------------------- #

    @staticmethod
    def _extract_seg_predictions(preds: Any) -> Any:
        """
        Extract segmentation logits/probabilities from model forward output.

        Supports:
          - tuple/list: (det_out, seg_out) -> returns seg_out
          - dict with keys among {'p3','p4','p5','full'} -> returns dict
          - tensor -> returns tensor as 'full'
        """
        if isinstance(preds, (tuple, list)):
            if len(preds) >= 2:
                return preds[1]
            # if single element, fall through
            preds = preds[0]
        return preds

    def _timepoints(self, total_epochs: int) -> np.ndarray:
        import numpy as np
        if total_epochs <= 0:
            return np.array([], dtype=int)
        q = max(total_epochs // self._save_fm_max, 1)
        return np.arange(q, total_epochs + q, step=q, dtype=int)



    # ---------------------- Hooks ---------------------- #
    def _register_temp_fm_hooks(self, base_model: nn.Module) -> None:
        """
        Register short-lived hooks to capture Detect inputs and optional extra layers.
        Hooks are removed right after a single forced forward.
        """
        if getattr(self, "_fm_hook_handles", None) is None:
            self._fm_hook_handles = []

        seq = getattr(base_model, "model", None)
        if seq is None:
            return

        idx2mod: Dict[int, nn.Module] = {getattr(m, "i", i): m for i, m in enumerate(seq) if hasattr(m, "forward")}

        # Detect PRE-hook → capture P3/P4/P5 features
        detect_idx = None
        for i in sorted(idx2mod):
            if type(idx2mod[i]).__name__ == "Detect":
                detect_idx = i
                break

        if detect_idx is not None:
            det_mod = idx2mod[detect_idx]

            def _detect_prehook(_m: nn.Module, _in: Any) -> None:
                xs = _in[0] if isinstance(_in, (tuple, list)) else _in
                if isinstance(xs, (list, tuple)):
                    for si, t in enumerate(xs):
                        if torch.is_tensor(t):
                            key = detect_idx * 10 + si  # e.g., 280/281/282
                            self._fm_last[key] = t.detach()

            self._fm_hook_handles.append(det_mod.register_forward_pre_hook(_detect_prehook))

        # Optional extra layer hooks (none by default)
        for li in getattr(self, "_fm_layers", []):
            mod = idx2mod.get(li)
            if mod is None:
                continue

            def _f_hook(_m: nn.Module, _in: Any, out: Any, idx: int = li) -> None:
                if isinstance(out, (list, tuple)):
                    out = out[0]
                if torch.is_tensor(out):
                    self._fm_last[idx] = out.detach()

            self._fm_hook_handles.append(mod.register_forward_hook(_f_hook))

    def _capture_once_and_save(
        self,
        base_model: nn.Module,
        batch: Dict[str, Any],
        fm_dir: "Path",
        seg_dir: "Path",
        det_dir: "Path",
        is_seg_batch: bool,
        det_preds: Any,
    ) -> None:
        """
        Register hooks, run one forward to populate FM and seg logits, then remove hooks and save.
        Ensures no hooks remain on the model after saving (pickle-safe).

        Mixed-precision safe: inputs are cast to the model's parameter dtype.
        """

        imgs = batch.get("img", None)
        if imgs is None or not torch.is_tensor(imgs):
            return

        # Reset caches
        self._fm_last.clear()
        self._seg_cache = {}

        # Temp hooks
        self._register_temp_fm_hooks(base_model)

        try:
            base_model.eval()

            # Cast inputs to match model parameter dtype to avoid half/float mismatch
            try:
                p = next(base_model.parameters())
                param_dtype = p.dtype
            except StopIteration:
                param_dtype = imgs.dtype
            imgs_cast = imgs.to(self.device, dtype=param_dtype, non_blocking=True)

            # Autocast only on CUDA; keep disabled otherwise
            use_cuda = (self.device.type == "cuda")
            use_amp = use_cuda and (param_dtype in (torch.float16, torch.bfloat16))
            autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp) if use_cuda else contextlib.nullcontext()

            with torch.no_grad(), autocast_ctx:
                out = base_model(imgs_cast)

            # Cache seg logits for panel if this is a segmentation batch
            seg_out = self._extract_seg_predictions(out)
            if isinstance(seg_out, dict):
                self._seg_cache = {k: v.detach() for k, v in seg_out.items() if torch.is_tensor(v)}
            elif torch.is_tensor(seg_out):
                self._seg_cache = {"full": seg_out.detach()}
        except Exception as e:
            LOGGER.warning(f"[CAMTLValidator] forced forward for FM/seg capture failed: {e}")
        finally:
            # Critical: remove hooks before any save() that might pickle the model
            self._remove_fm_hooks()

        # Save FM tensors (first image only)
        _ = self._save_feature_maps(Path(fm_dir), batch) if self._fm_last else 0

        # Save predictions
        if is_seg_batch and self._seg_cache:
            # Build the requested 1x4 panel
            img0 = batch["img"][0] if batch["img"].ndim == 4 else batch["img"]  # CHW
            panel_path = Path(seg_dir) / "panel_seg.png"
            self._save_camtl_panel(panel_path, img0, self._seg_cache)
        elif (not is_seg_batch) and isinstance(det_preds, (list, tuple)):
            _ = self._save_detect_preds(Path(det_dir), batch, det_preds)


    def _save_camtl_panel(self, out_path: Path, img_chw: torch.Tensor, seg_dict: Dict[str, torch.Tensor]) -> None:
        """
        Save a 1x4 panel: [original+full] | [p3] | [p4] | [p5] at native sizes.
        No upsampling. Uses nearest rendering to preserve pixel scale.
        Handles 4D tensors by selecting the first image in the batch.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure image is CHW
        if img_chw.ndim == 4:
            img_chw = img_chw[0]
        if img_chw.ndim != 3:
            raise ValueError(f"_save_camtl_panel expected CHW image, got shape={tuple(img_chw.shape)}")

        # Prepare base image (C,H,W) -> BGR HxWxC
        x = img_chw.detach().float().clamp(0, 1).cpu().numpy()  # CHW
        x = (x * 255.0).round().astype(np.uint8)
        x = np.transpose(x, (1, 2, 0))
        if x.shape[2] == 1:
            x = np.repeat(x, 3, axis=2)
        img_bgr = x[:, :, ::-1]

        # Single-image per-scale tensors
        single_seg: Dict[str, torch.Tensor] = {}
        for k in ("p3", "p4", "p5", "full"):
            t = seg_dict.get(k, None)
            if torch.is_tensor(t):
                if t.ndim == 4:
                    t = t[0]
                single_seg[k] = t

        def _as_2d(m: torch.Tensor) -> np.ndarray:
            a = m.detach().float().cpu()
            if a.ndim == 3 and a.shape[0] == 1:
                a = a.squeeze(0)
            if a.ndim == 3:
                a = a[0]
            return torch.sigmoid(a).numpy()

        p3 = single_seg.get("p3", None)
        p4 = single_seg.get("p4", None)
        p5 = single_seg.get("p5", None)
        full = single_seg.get("full", None)

        fig, axs = plt.subplots(1, 4, figsize=(12, 3), squeeze=True)

        # Panel 1: original + full overlay if available (no resize)
        axs[0].imshow(img_bgr[..., ::-1])  # display RGB
        axs[0].set_title("img + full")
        if isinstance(full, torch.Tensor):
            mf = _as_2d(full)
            axs[0].imshow(mf, alpha=0.35, interpolation="nearest")
        axs[0].axis("off")

        # Panels 2-4: P3/P4/P5 at native sizes, no interpolation to 512
        for j, (title, tt) in enumerate([("P3", p3), ("P4", p4), ("P5", p5)], start=1):
            axs[j].set_title(title)
            if isinstance(tt, torch.Tensor):
                mj = _as_2d(tt)
                axs[j].imshow(mj, interpolation="nearest")
            axs[j].axis("off")

        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)



    def _save_feature_maps(self, fm_dir: Path, batch: Dict[str, Any]) -> int:
        from pathlib import Path
        fm_dir = Path(fm_dir)
        fm_dir.mkdir(parents=True, exist_ok=True)
        img_tensor = batch.get("img", torch.empty(0))
        B = int(getattr(img_tensor, "shape", [0])[0]) if hasattr(img_tensor, "shape") else 0
        im_files = batch.get("im_file", [str(i) for i in range(B)])
        n_saved = 0
        for li, t in sorted(self._fm_last.items()):
            if not torch.is_tensor(t) or t.ndim < 3:
                continue
            bsz = t.shape[0]
            for bi in range(min(B, bsz, 1)):  # one image is enough
                stem = Path(im_files[bi]).stem if isinstance(im_files, (list, tuple)) and len(im_files) > bi else str(bi)
                out_path = fm_dir / f"{stem}_{li}.pt"
                torch.save({"layer_index": int(li), "shape": tuple(t[bi].shape), "tensor": t[bi].cpu()}, out_path)
                n_saved += 1
        return n_saved

    def _img_tensor_to_bgr(self, t: torch.Tensor) -> np.ndarray:
        import numpy as np
        if t.ndim != 3:
            raise ValueError(f"expected CHW, got shape={tuple(t.shape)}")
        x = t.detach().float().clamp(0, 1).cpu().numpy()       # CHW, 0..1
        x = (x * 255.0).round().astype(np.uint8)               # CHW, 0..255
        x = np.transpose(x, (1, 2, 0))                         # HWC
        if x.shape[2] == 1:
            x = np.repeat(x, 3, axis=2)
        return x[:, :, ::-1]                                   # RGB->BGR

    def _draw_dets(self, img_bgr: np.ndarray, boxes_xyxy: np.ndarray,
                cls: np.ndarray, conf: np.ndarray) -> np.ndarray:
        import cv2
        out = img_bgr.copy()
        n = boxes_xyxy.shape[0]
        for i in range(n):
            x1, y1, x2, y2 = boxes_xyxy[i].astype(int).tolist()
            label = f"{int(cls[i])}:{float(conf[i]):.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2, lineType=cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            y1t = max(y1, th + 3)
            cv2.rectangle(out, (x1, y1t - th - 4), (x1 + tw + 4, y1t), (0, 255, 0), -1)
            cv2.putText(out, label, (x1 + 2, y1t - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        return out

    def _save_detect_preds(self, det_dir: "Path", batch: Dict[str, Any], det_preds: Any) -> int:
        from pathlib import Path
        import numpy as np
        import cv2
        det_dir = Path(det_dir)
        det_dir.mkdir(parents=True, exist_ok=True)
        img_tensor = batch.get("img", torch.empty(0))
        B = int(getattr(img_tensor, "shape", [0])[0]) if hasattr(img_tensor, "shape") else 0
        im_files = batch.get("im_file", [str(i) for i in range(B)])
        n_overlays = 0
        if not isinstance(det_preds, (list, tuple)):
            return 0
        for bi in range(min(B, len(det_preds), 1)):  # save first image of the batch
            stem = Path(im_files[bi]).stem if len(im_files) > bi else str(bi)
            try:
                img_bgr = self._img_tensor_to_bgr(img_tensor[bi])
                p = det_preds[bi] if isinstance(det_preds[bi], dict) else {}
                boxes = np.asarray(p.get("bboxes", torch.zeros((0, 4))).detach().cpu().numpy())
                conf = np.asarray(p.get("conf", torch.zeros((0,))).detach().cpu().numpy())
                cls = np.asarray(p.get("cls", torch.zeros((0,))).detach().cpu().numpy())
                overlay = self._draw_dets(img_bgr, boxes, cls, conf)
                cv2.imwrite(str(det_dir / f"{stem}_pred.jpg"), overlay)
                n_overlays += 1
            except Exception as e:
                from camtl_yolo.external.ultralytics.ultralytics.utils import LOGGER
                LOGGER.debug(f"[CAMTLValidator] detect overlay save failed for {stem}: {e}")
        return n_overlays

    def _save_seg_preds(self, seg_dir: "Path", batch: Dict[str, Any]) -> int:
        from pathlib import Path
        import cv2
        seg_dir = Path(seg_dir)
        seg_dir.mkdir(parents=True, exist_ok=True)
        if not self._seg_cache:
            return 0
        img_tensor = batch.get("img", torch.empty(0))
        B = int(getattr(img_tensor, "shape", [0])[0]) if hasattr(img_tensor, "shape") else 0
        im_files = batch.get("im_file", [str(i) for i in range(B)])
        n_masks = 0
        for bi in range(min(B, 1)):  # save first image only
            stem = Path(im_files[bi]).stem if len(im_files) > bi else str(bi)
            for sk, t in self._seg_cache.items():
                try:
                    m_t = t[bi]
                    if m_t.ndim == 3 and m_t.shape[0] == 1:
                        m_t = m_t.squeeze(0)
                    m_t = torch.sigmoid(m_t).detach().cpu()
                    if m_t.ndim == 3:
                        m_t = m_t[0]
                    m = m_t.numpy()
                    m_img = (m * 255).astype("uint8")
                    cv2.imwrite(str(seg_dir / f"{stem}_{sk}.png"), m_img)
                    n_masks += 1
                except Exception:
                    pass
        return n_masks

    def _remove_fm_hooks(self) -> None:
        """Remove any registered hooks to keep the model pickle-safe."""
        for h in getattr(self, "_fm_hook_handles", []):
            try:
                h.remove()
            except Exception:
                pass
        self._fm_hook_handles = []