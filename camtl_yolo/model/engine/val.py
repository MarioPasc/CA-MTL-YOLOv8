# camtl_yolo/engine/val.py
from __future__ import annotations

from copy import copy
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from camtl_yolo.model.utils.normalization import set_bn_domain
from camtl_yolo.model.utils.samplers import map_domain_name
from camtl_yolo.external.ultralytics.ultralytics.engine.validator import BaseValidator
from camtl_yolo.external.ultralytics.ultralytics.utils import LOGGER, TQDM
from camtl_yolo.external.ultralytics.ultralytics.utils.ops import Profile
from camtl_yolo.external.ultralytics.ultralytics.utils.torch_utils import smart_inference_mode, unwrap_model


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
        # Task descriptor for logs; not used for Ultralytics dataset routing
        self.args.task = "detect+segment"
        # Optional descriptor of metric keys (for printers that read .metrics.keys)
        self.metrics = SimpleNamespace(
            keys=[
                "val/det",
                "val/seg",
                "val/cons",
                "val/align",
                "val/l2sp",
                "val/total",
                "val/dice",
                "val/dice_p3",
                "val/dice_p4",
                "val/dice_p5",
                "val/dice_full",
            ]
        )

        # Dice accumulators (initialized in init_metrics)
        self._dice_sums: Dict[str, float] = {}
        self._dice_counts: Dict[str, int] = {}

    # ---------------------- Hooks ---------------------- #

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

        # Dice accumulators
        self._dice_sums = {"p3": 0.0, "p4": 0.0, "p5": 0.0, "full": 0.0}
        self._dice_counts = {"p3": 0, "p4": 0, "p5": 0, "full": 0}

    def update_metrics(self, preds: Any, batch: Mapping[str, Any]) -> None:
        """
        Update running metrics for one batch:
          - Obtain component losses from model.loss(batch, preds)
          - If segmentation batch, compute Dice per available scale
        """
        # Get the model object that holds criteria
        # During training BaseValidator passes the (ema or raw) module; unwrap to access .loss
        model = getattr(self, "_current_model_for_loss", None)
        if model is None:
            LOGGER.warning("[Validator] _current_model_for_loss not set; skipping loss accumulation.")
            loss_items = None
        else:
            try:
                loss_total, loss_items = model.loss(batch, preds)  # type: ignore[assignment]
            except Exception as e:
                LOGGER.warning(f"[Validator] model.loss failed: {e}")
                loss_items = None

        # Parse component losses if available
        if torch.is_tensor(loss_items):
            li = [float(x) for x in loss_items.tolist()]
            # Expecting [det, seg, cons, align, l2sp, total]
            if len(li) >= 6:
                det, seg, cons, align, l2sp, _ = li[:6]
            else:
                # Fallback: try to map the first 5 and ignore remainder
                vals = li + [0.0] * (6 - len(li))
                det, seg, cons, align, l2sp, _ = vals[:6]
        elif isinstance(loss_items, Mapping):
            det = float(loss_items.get("det", 0.0))
            seg = float(loss_items.get("seg", 0.0))
            cons = float(loss_items.get("cons", 0.0))
            align = float(loss_items.get("align", 0.0))
            l2sp = float(loss_items.get("l2sp", 0.0))
        else:
            det = seg = cons = align = l2sp = 0.0

        # Update running sums and counters by batch type
        is_seg_batch = bool(torch.as_tensor(batch.get("is_seg", False)).any().item())
        if is_seg_batch:
            self._sum_seg += seg
            self._cnt_seg += 1
        else:
            self._sum_det += det
            self._cnt_det += 1

        self._sum_cons += cons
        self._sum_align += align
        self._sum_l2sp += l2sp

        # Dice calculation for segmentation batches
        if is_seg_batch:
            y = batch.get("mask")
            if not torch.is_tensor(y):
                LOGGER.warning("[Validator] segmentation batch without 'mask' tensor.")
            else:
                # access segmentation predictions from forward output
                seg_pred = self._extract_seg_predictions(preds)
                if isinstance(seg_pred, dict):
                    for k in ("p3", "p4", "p5", "full"):
                        if k in seg_pred and torch.is_tensor(seg_pred[k]):
                            d = float(self._dice(seg_pred[k], y))
                            self._dice_sums[k] += d
                            self._dice_counts[k] += 1
                elif torch.is_tensor(seg_pred):
                    # treat as full-res
                    d = float(self._dice(seg_pred, y))
                    self._dice_sums["full"] += d
                    self._dice_counts["full"] += 1

        bsz = int(batch["img"].shape[0]) if torch.is_tensor(batch.get("img")) else 0
        self.seen += bsz

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

        # Mean dice across segmentation batches
        mean_dice = (
            (self._dice_sums["p3"] + self._dice_sums["p4"] + self._dice_sums["p5"] + self._dice_sums["full"])
            / max(1, self._dice_counts["p3"] + self._dice_counts["p4"] + self._dice_counts["p5"] + self._dice_counts["full"])
        )

        return {
            "val/det": avg_det,
            "val/seg": avg_seg,
            "val/cons": avg_cons,
            "val/align": avg_align,
            "val/l2sp": avg_l2sp,
            "val/total": avg_total,
            "val/dice": mean_dice,
            "val/dice_p3": self._dice_sums["p3"] / max(1, self._dice_counts["p3"]),
            "val/dice_p4": self._dice_sums["p4"] / max(1, self._dice_counts["p4"]),
            "val/dice_p5": self._dice_sums["p5"] / max(1, self._dice_counts["p5"]),
            "val/dice_full": self._dice_sums["full"] / max(1, self._dice_counts["full"]),
        }

    def print_results(self) -> None:
        """
        Log compact per-epoch summary.
        """
        s = self.get_stats()
        LOGGER.info(
            f"val: det {s['val/det']:.4f} | seg {s['val/seg']:.4f} | cons {s['val/cons']:.4f} | "
            f"align {s['val/align']:.4f} | l2sp {s['val/l2sp']:.4f} | total {s['val/total']:.4f} | "
            f"dice p3 {s['val/dice_p3']:.4f} p4 {s['val/dice_p4']:.4f} p5 {s['val/dice_p5']:.4f} full {s['val/dice_full']:.4f}"
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

    # ---------------------- Core call ---------------------- #

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """
        Validate mid-epoch (trainer!=None) or standalone (weights or nn.Module).
        Returns a dict of floats. During training, merges stats with labeled loss items from BaseTrainer.
        """
        # Training vs standalone
        self.training = trainer is not None
        augment = False  # no aug at val

        if self.training:
            # device/data from trainer
            self.device = trainer.device
            self.data = trainer.data
            # half precision matches trainer AMP
            self.args.half = self.device.type != "cpu" and trainer.amp
            # pick EMA if available for forward, but compute losses with the underlying unwrapped model
            fwd_model = trainer.ema.ema or trainer.model
            if trainer.args.compile and hasattr(fwd_model, "_orig_mod"):
                fwd_model = fwd_model._orig_mod
            fwd_model = fwd_model.half() if self.args.half else fwd_model.float()
            fwd_model.eval()

            # prepare loss accumulator vector like BaseValidator
            self.loss = torch.zeros_like(trainer.loss_items, device=self.device)

            # model whose .loss we call (unwrapped)
            base_model = unwrap_model(trainer.model)
        else:
            # Standalone evaluation path: accept a nn.Module or a checkpoint path
            if model is None:
                raise RuntimeError("Standalone validation requires 'model' to be an nn.Module or a checkpoint path.")

            # Load if given a path-like
            from pathlib import Path as _P

            if isinstance(model, (str, _P)):
                ckpt = torch.load(model, map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict):
                    model = ckpt.get("ema") or ckpt.get("model") or ckpt
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
            base_model = unwrap_model(fwd_model)

            # Ensure criteria exist on the loaded model
            need = ("segment_criterion", "detect_criterion", "consistency_criterion", "align_criterion")
            if any(not hasattr(base_model, n) for n in need) and hasattr(base_model, "init_criterion"):
                try:
                    base_model.init_criterion()
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
            self._current_model_for_loss = base_model
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

        # Return dict consistent with BaseValidator
        if self.training:
            # Merge stats with trainer-labeled loss items (prefix val/)
            labeled = trainer.label_loss_items(self.loss.detach().cpu() / len(self.dataloader), prefix="val")
            merged = {**stats, **labeled}
            return {k: round(float(v), 5) for k, v in merged.items()}
        else:
            return stats

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

    @staticmethod
    def _dice(pred_logits: torch.Tensor, y: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        Soft Dice on probabilities; accepts logits or probabilities.

        Args:
            pred_logits: [B,1,h,w] or [B,1,H,W] logits or probabilities.
            y:          [B,1,H,W] binary mask in {0,1} or float in [0,1].

        Returns:
            Mean Dice over batch as a scalar tensor.
        """
        if pred_logits.ndim == 3:
            pred_logits = pred_logits.unsqueeze(1)
        if y.ndim == 3:
            y = y.unsqueeze(1)

        # convert logits -> probabilities
        p = torch.sigmoid(pred_logits) if pred_logits.dtype.is_floating_point else pred_logits

        # resize to GT spatial size if needed
        if p.shape[-2:] != y.shape[-2:]:
            p = F.interpolate(p, size=y.shape[-2:], mode="bilinear", align_corners=False)

        y = (y > 0.5).float()
        inter = (p * y).sum(dim=(1, 2, 3))
        denom = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
        dice = (2 * inter + eps) / (denom + eps)
        return dice.mean()
