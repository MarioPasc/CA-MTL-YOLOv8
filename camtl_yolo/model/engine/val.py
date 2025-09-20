# camtl_yolo/engine/val.py
from __future__ import annotations

from copy import copy
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional

from pathlib import Path
import yaml # type: ignore
import contextlib

import numpy as np
import pandas as pd # type: ignore

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
        self.args.task = "detect+segment"
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
        # Dice accumulators (initialized later)
        self._dice_sums: Dict[str, float] = {}
        self._dice_counts: Dict[str, int] = {}
        # ---- prediction & feature-map capture state ----
        self._save_fm_max: int = 4                       # save 4 times per training
        self._fm_layers: list[int] = []                  # auto-filled when hooking; leave empty to only grab Detect inputs
        self._fm_last: Dict[int, torch.Tensor] = {}      # layer_index -> tensor[B, C, H, W]
        self._fm_hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._logged_epoch_saves: set[int] = set()       # epochs already saved
        self._seg_cache: Dict[str, torch.Tensor] = {}    # latest seg outputs for current batch



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
        from pathlib import Path
        from camtl_yolo.external.ultralytics.ultralytics.utils import LOGGER

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
                    m = t[bi]
                    if m.ndim == 3 and m.shape[0] == 1:
                        m = m.squeeze(0)
                    m = torch.sigmoid(m).detach().cpu().numpy()
                    if m.ndim == 3:
                        m = m[0]
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
        Standard accumulation plus timepointed saving. Only rank-0 and only batch 0.
        Saves under:
            val_preds_fm/epoch_{k}/feature_maps/
            val_preds_fm/epoch_{k}/segment/   (seg panel)
            val_preds_fm/epoch_{k}/detect/    (det overlay)
        """
        import pandas as pd  # type: ignore
        import yaml  # type: ignore
        from pathlib import Path
        from camtl_yolo.external.ultralytics.ultralytics.utils import LOGGER

        super().update_metrics(preds, batch)

        if not self._is_main() or int(getattr(self, "batch_i", 0)) != 0:
            return

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
        except Exception as e:
            LOGGER.debug(f"[CAMTLValidator] reading results/args failed: {e}")
            return

        tps = set(int(x) for x in self._timepoints(total_epochs).tolist())
        epoch_1based = int(current_epoch + 1)
        if epoch_1based not in tps or epoch_1based in self._logged_epoch_saves:
            return

        base_dir = save_dir / "val_preds_fm" / f"epoch_{epoch_1based}"
        fm_dir = base_dir / "feature_maps"
        seg_dir = base_dir / "segment"
        det_dir = base_dir / "detect"
        base_dir.mkdir(parents=True, exist_ok=True)

        # Decide batch type
        is_seg_batch = bool(torch.as_tensor(batch.get("is_seg", False)).any().item())

        # Use the unwrapped training model for the temporary forward
        base_model = getattr(self, "_current_model_for_loss", None)
        if not isinstance(base_model, nn.Module):
            LOGGER.debug("[CAMTLValidator] base model not available for capture; skipping save.")
            return

        # Capture once and save. Hooks are removed before any checkpoint is written.
        self._capture_once_and_save(
            base_model=base_model,
            batch=dict(batch),
            fm_dir=fm_dir,
            seg_dir=seg_dir,
            det_dir=det_dir,
            is_seg_batch=is_seg_batch,
            det_preds=preds if isinstance(preds, (list, tuple)) and not is_seg_batch else None,
        )

        LOGGER.info(f"[CAMTLValidator] Saved validation assets for epoch {epoch_1based} to {base_dir}")
        self._logged_epoch_saves.add(epoch_1based)




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

        self._remove_fm_hooks()
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

    def _is_main(self) -> bool:
        return int(getattr(self, "rank", 0)) in (0, -1)

    def _timepoints(self, total_epochs: int) -> np.ndarray:
        import numpy as np
        if total_epochs <= 0:
            return np.array([], dtype=int)
        q = max(total_epochs // self._save_fm_max, 1)
        return np.arange(q, total_epochs + q, step=q, dtype=int)

