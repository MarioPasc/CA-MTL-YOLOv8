import torch
import torch.nn as nn
from pathlib import Path
import yaml

# Ultralytics modules
from camtl_yolo.external.ultralytics.ultralytics.nn.modules import (
    Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
    BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, DWConvTranspose2d, C3x, RepC3, Concat, Detect, Segment)
from camtl_yolo.external.ultralytics.ultralytics.nn.tasks import DetectionModel
from camtl_yolo.external.ultralytics.ultralytics.utils import LOGGER
from camtl_yolo.external.ultralytics.ultralytics.utils.ops import make_divisible

# Cross-Attention Multi-Task Learning modules
from camtl_yolo.model.nn import CSAM, CTAM, FPMA, SegHead
from camtl_yolo.model.losses import DetectionLoss, MultiScaleBCEDiceLoss, ConsistencyMaskFromBoxes, AttentionAlignmentLoss

def _find_idx(modules, cls):
    for i, m in enumerate(modules):
        if isinstance(m, cls):
            return i
    return None

class CAMTL_YOLO(DetectionModel):
    """
    CA-MTL-YOLOv8 model for combined object detection and segmentation.
    This class should be used with a custom 'ca_mtl' task trainer.
    """

    def __init__(self, cfg='camtl_yolov8.yaml', ch=3, nc=None, verbose=True):
        super().__init__()
        # Load YAML
        if isinstance(cfg, dict):
            self.yaml = cfg
            self.yaml_file = Path(__file__).resolve().parents[2] / "configs/models/camtl_yolov8.yaml"
        else:
            self.yaml_file = Path(cfg).name
            with open(cfg, 'r', encoding='utf-8') as f:
                self.yaml = yaml.safe_load(f)

        # Common knobs
        self.task = str(self.yaml.get("TASK", "DomainShift1"))
        self.scale = str(self.yaml.get("SCALE", "s"))
        self.pretrained_root = Path(self.yaml.get("PRETRAINED_MODELS_PATH", "."))

        # Default COCO pretrained paths (used in DomainShift1)
        self.seg_weights = self.pretrained_root / f"yolov8{self.scale}-seg.pt"
        self.det_weights = self.pretrained_root / f"yolov8{self.scale}.pt"

        # Build architecture
        self.model, self.save = self.parse_model(self.yaml, ch=[ch] if isinstance(ch, int) else ch, nc=nc, verbose=verbose)

        # Task-aware weight loading
        self._load_task_weights()

    def _forward_once(self, x):
        """
        Forward pass. Returns (det_out, seg_out) from the actual head modules,
        not from cached y indices.
        """
        y = []
        det_out, seg_out = None, None

        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            # capture head outputs directly and keep normal caching
            if isinstance(m, Detect):
                det_out = x
            elif isinstance(m, SegHead):
                seg_out = x
            y.append(x if m.i in self.save else None)

        return det_out, seg_out


    def init_criterion(self):
        """
        Initialize multi-task criteria.
        Reads optional weights from self.yaml['LOSS'].
        """
        cfg = self.yaml.get("LOSS", {}) if isinstance(self.yaml, dict) else {}
        # Detection loss from Ultralytics (uses model internals)
        self.detect_criterion = DetectionLoss(self, hyp=cfg.get("det_hyp"))
        # Segmentation loss: BCE+Dice at one or more scales
        seg_scales = cfg.get("seg_scale_weights")  # e.g., [0.25, 0.35, 0.40]
        self.segment_criterion = MultiScaleBCEDiceLoss(
            bce_weight=float(cfg.get("seg_bce", 1.0)),
            dice_weight=float(cfg.get("seg_dice", 1.0)),
            scale_weights=seg_scales,
            from_logits=bool(cfg.get("seg_from_logits", True)),
        )
        # Consistency: boxes → pseudo-mask on detection-domain images
        self.consistency_criterion = ConsistencyMaskFromBoxes(
            weight=float(cfg.get("consistency", 0.1)),
            loss=str(cfg.get("consistency_loss", "bce")).lower(),
        )
        # Attention alignment between domains using CTAM attention snapshots
        self.align_criterion = AttentionAlignmentLoss(
            model=self,
            source_name=str(cfg.get("source_domain", "retinography")),
            target_name=str(cfg.get("target_domain", "angiography")),
            weight=float(cfg.get("align", 0.1)),
        )
        # scalar weights for combining
        self._lambda_det = float(cfg.get("lambda_det", 1.0))
        self._lambda_seg = float(cfg.get("lambda_seg", 1.0))
        self._lambda_cons = float(cfg.get("lambda_cons", 0.1))
        self._lambda_align = float(cfg.get("lambda_align", 0.1))
        return True

    def loss(self, batch, preds):
        """
        Compute total multi-task loss for a mixed batch.
        - Detection loss on batches with detection labels.
        - Segmentation loss on batches flagged as segmentation.
        - Consistency loss on detection-domain images using GT boxes as pseudo masks.
        - Attention alignment loss across domains if CTAM attention is available.
        - Optional L2-SP regularization if tasks.configure_task attached self._l2sp.
        """
        det_preds, seg_preds = preds
        device = batch["img"].device

        # detection
        has_det = batch["bboxes"].numel() > 0
        loss_det = torch.zeros((), device=device)
        det_items = {}
        if has_det:
            loss_det, det_items = self.detect_criterion(det_preds, batch)

        # segmentation (only if any seg sample exists in batch)
        has_seg = bool(batch["is_seg"].any().item()) if isinstance(batch["is_seg"], torch.Tensor) else any(batch["is_seg"])
        loss_seg = torch.zeros((), device=device)
        seg_items = {}
        if has_seg:
            loss_seg, seg_items = self.segment_criterion(seg_preds, batch)

        # consistency on detection-domain images
        cons_loss, cons_items = self.consistency_criterion(seg_preds, batch)

        # attention alignment between domains
        align_loss, align_items = self.align_criterion(batch)

        # L2-SP if configured by task
        l2sp_term = torch.zeros((), device=device)
        if hasattr(self, "_l2sp") and self._l2sp is not None:
            l2sp_term = self._l2sp(self)

        total = (
            self._lambda_det * loss_det
            + self._lambda_seg * loss_seg
            + self._lambda_cons * cons_loss
            + self._lambda_align * align_loss
            + l2sp_term
        )

        items = {}
        items.update({f"det_{k}": v for k, v in det_items.items()})
        items.update(seg_items)
        items.update(cons_items)
        items.update(align_items)
        items["l2sp"] = l2sp_term.detach()
        items["loss"] = total.detach()
        return total, items
    
    def _load_pretrained_weights(self):
        """Load COCO weights: backbone+neck from *seg* ckpt, Detect head from *det* ckpt.
        Skip classification branches when nc!=COCO_nc to avoid shape errors.
        """
        if not self.seg_weights.exists():
            raise FileNotFoundError(f"Segmentation weights not found at {self.seg_weights}")
        if not self.det_weights.exists():
            raise FileNotFoundError(f"Detection weights not found at {self.det_weights}")

        LOGGER.info(f"Loading backbone and neck from {self.seg_weights}...")
        ckpt_seg = torch.load(self.seg_weights, map_location="cpu", weights_only=False)
        csd_seg = ckpt_seg["model"].float().state_dict()

        LOGGER.info(f"Loading detection head from {self.det_weights}...")
        ckpt_det = torch.load(self.det_weights, map_location="cpu", weights_only=False)
        csd_det = ckpt_det["model"].float().state_dict()

        new_sd = self.state_dict()
        det_idx = _find_idx(self.model, Detect)

        loaded, skipped, mismatched = 0, 0, 0
        det_loaded, det_total_eligible = 0, 0

        # 1) load backbone+neck from yolov8*-seg.pt, exclude its seg head block ('model.22.')
        for k, v in csd_seg.items():
            if k.startswith("model.22."):  # seg head in Ultralytics seg ckpt
                skipped += 1
                continue
            if k in new_sd and new_sd[k].shape == v.shape:
                new_sd[k] = v
                loaded += 1
            else:
                mismatched += 1

        # 2) load detect head from yolov8*.pt into our Detect idx, but ignore cls branch if nc differs
        if det_idx is None:
            LOGGER.warning("Detect head not found; skipping detect-head remap")
        else:
            # determine if classification tensors are shape-compatible
            # In Ultralytics Detect, 'cv3' corresponds to classification convs and depends on nc.
            model_nc = getattr(self.model[det_idx], "nc", None)
            coco_nc = ckpt_det["model"].nc if hasattr(ckpt_det["model"], "nc") else 80
            skip_cls = (model_nc is not None) and (int(model_nc) != int(coco_nc))

            for k_det, v_det in csd_det.items():
                if not k_det.startswith("model.22."):
                    continue
                sub = k_det.split("model.22.", 1)[1]
                if skip_cls and sub.startswith("cv3"):
                    # classification tensors depend on nc -> skip when nc!=COCO
                    skipped += 1
                    continue

                new_k = f"model.{det_idx}.{sub}"
                det_total_eligible += 1
                if new_k in new_sd and new_sd[new_k].shape == v_det.shape:
                    new_sd[new_k] = v_det
                    loaded += 1
                    det_loaded += 1
                else:
                    mismatched += 1
                    LOGGER.warning(f"Could not map {k_det} -> {new_k} (missing or shape mismatch)")

        # commit
        self.load_state_dict(new_sd, strict=False)
        LOGGER.info(
            f"Pretrained load summary: loaded={loaded}, skipped={skipped}, "
            f"shape_mismatch={mismatched}, det_mapped={det_loaded}/{det_total_eligible}"
        )

    # --- Task-aware weight loading and saving ---------------------------------

    def _load_task_weights(self) -> None:
        """
        Load weights based on YAML TASK:
          - DomainShift1: COCO seg + COCO det (mapped) [strict mapping]
          - CAMTL: fine-tuned checkpoint from DomainShift1 (single .pt)
        """
        if self.task == "DomainShift1":
            self._load_coco_pretrained_weights()
            return

        if self.task == "CAMTL":
            ckpt_path = self.pretrained_root / f"yolov8{self.scale}-domainshift1.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"[CAMTL] DomainShift1 checkpoint not found: {ckpt_path}")

            # Ensure seg-stream normalization is GN before loading if your DomainShift1 used GN
            try:
                from camtl_yolo.model.utils.normalization import replace_seg_stream_bn_with_groupnorm
                replaced = replace_seg_stream_bn_with_groupnorm(self, max_groups=int(self.yaml.get("GN_GROUPS", 32)))
                if replaced:
                    LOGGER.info(f"[CAMTL] Converted {replaced} BN layers to GroupNorm in segmentation stream before loading.")
            except Exception as e:
                LOGGER.warning(f"[CAMTL] GN conversion skipped: {e}")

            self._load_finetuned_weights(ckpt_path)
            return

        LOGGER.warning(f"[TASK={self.task}] Unknown task. Skipping pretrained loading.")

    def _load_coco_pretrained_weights(self):
        """DomainShift1: backbone+neck from *seg* ckpt, Detect head from *det* ckpt with safe remap."""
        if not self.seg_weights.exists():
            raise FileNotFoundError(f"Segmentation weights not found at {self.seg_weights}")
        if not self.det_weights.exists():
            raise FileNotFoundError(f"Detection weights not found at {self.det_weights}")

        LOGGER.info(f"Loading backbone and neck from {self.seg_weights}...")
        ckpt_seg = torch.load(self.seg_weights, map_location="cpu", weights_only=False)
        csd_seg = ckpt_seg["model"].float().state_dict()

        LOGGER.info(f"Loading detection head from {self.det_weights}...")
        ckpt_det = torch.load(self.det_weights, map_location="cpu", weights_only=False)
        csd_det = ckpt_det["model"].float().state_dict()

        new_sd = self.state_dict()
        det_idx = _find_idx(self.model, Detect)

        loaded, skipped, mismatched = 0, 0, 0
        det_loaded, det_total_eligible = 0, 0

        # Backbone+neck from yolov8*-seg.pt (exclude seg head 'model.22.')
        for k, v in csd_seg.items():
            if k.startswith("model.22."):
                skipped += 1
                continue
            if k in new_sd and new_sd[k].shape == v.shape:
                new_sd[k] = v
                loaded += 1
            else:
                mismatched += 1

        # Detect head from yolov8*.pt → our Detect index; ignore classification if nc differs
        if det_idx is None:
            LOGGER.warning("Detect head not found; skipping detect-head remap")
        else:
            model_nc = getattr(self.model[det_idx], "nc", None)
            coco_nc = ckpt_det["model"].nc if hasattr(ckpt_det["model"], "nc") else 80
            skip_cls = (model_nc is not None) and (int(model_nc) != int(coco_nc))

            for k_det, v_det in csd_det.items():
                if not k_det.startswith("model.22."):
                    continue
                sub = k_det.split("model.22.", 1)[1]
                if skip_cls and sub.startswith("cv3"):
                    skipped += 1
                    continue
                new_k = f"model.{det_idx}.{sub}"
                det_total_eligible += 1
                if new_k in new_sd and new_sd[new_k].shape == v_det.shape:
                    new_sd[new_k] = v_det
                    loaded += 1
                    det_loaded += 1
                else:
                    mismatched += 1
                    LOGGER.warning(f"Could not map {k_det} -> {new_k} (missing or shape mismatch)")

        self.load_state_dict(new_sd, strict=False)
        LOGGER.info(
            f"Pretrained load summary: loaded={loaded}, skipped={skipped}, "
            f"shape_mismatch={mismatched}, det_mapped={det_loaded}/{det_total_eligible}"
        )

    def _load_finetuned_weights(self, ckpt_path: Path) -> None:
        """
        CAMTL: load a single fine-tuned checkpoint produced after DomainShift1.
        Accepts Ultralytics-style {'model': nn.Module, ...} or raw state_dict under 'model'.
        """
        LOGGER.info(f"[CAMTL] Loading fine-tuned checkpoint from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict) and "model" in ckpt:
            src = ckpt["model"]
            state_src = src.state_dict() if hasattr(src, "state_dict") else dict(ckpt["model"])
        elif isinstance(ckpt, dict):
            # assume already a state_dict
            state_src = ckpt
        else:
            # loaded a bare nn.Module
            state_src = ckpt.state_dict()  # type: ignore[attr-defined]

        state_dst = self.state_dict()
        loaded, skipped, mismatched = 0, 0, 0

        for k, v in state_src.items():
            if k in state_dst and state_dst[k].shape == v.shape:
                state_dst[k] = v
                loaded += 1
            else:
                mismatched += 1

        self.load_state_dict(state_dst, strict=False)
        LOGGER.info(f"[CAMTL] Load summary: loaded={loaded}, mismatched_or_missing={mismatched}, skipped={skipped}")

    @torch.no_grad()
    def save_task_checkpoint(self, save_dir: str | Path, filename: str | None = None,
                             epoch: int | None = None, optimizer=None, extra: dict | None = None) -> Path:
        """
        Save a task-aware checkpoint:
          - DomainShift1 -> yolov8{SCALE}-domainshift1.pt
          - CAMTL       -> yolov8{SCALE}-camtl.pt
        Stores an Ultralytics-compatible dict with 'model' = this Module.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            suffix = "domainshift1" if self.task == "DomainShift1" else "camtl" if self.task == "CAMTL" else "custom"
            filename = f"yolov8{self.scale}-{suffix}.pt"

        ckpt = {
            "model": self.float(),  # store in FP32 for portability
            "epoch": int(epoch or 0),
            "optimizer": (optimizer.state_dict() if optimizer is not None else None),
            "yaml": self.yaml,
            "args": {"task": self.task, "scale": self.scale},
            "extra": extra or {},
        }
        out = save_dir / filename
        torch.save(ckpt, out)
        LOGGER.info(f"Saved checkpoint: {out}")
        return out

    def parse_model(self, d: dict, ch, nc: int | None = None, verbose: bool = True):
        """
        Parse a CAMTL-YOLO model.yaml into a torch.nn.Sequential.
        Supports sections: backbone, neck, seg_decoder, attn_fusion, seg_head, detect_head (and legacy 'head').
        Applies scales[SCALE] for depth/width/max_channels.
        """
        import ast, contextlib, copy, torch
        import torch.nn as nn

        # --- scaling knobs ---
        act = d.get("activation")
        scales = d.get("scales")
        scale = d.get("SCALE", d.get("scale"))
        depth, width, max_channels = (d.get("depth_multiple", 1.0), d.get("width_multiple", 1.0), float("inf"))
        if scales:
            if not scale:
                scale = next(iter(scales.keys()))
                LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
            with contextlib.suppress(Exception):
                depth, width, max_channels = scales[scale]

        if nc is None:
            nc = int(d.get("nc", 80))

        if act:
            Conv.default_act = eval(act)

        if verbose:
            LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<42}{'arguments':<30}")

        # --- make an isolated, deep-copied spec so we never mutate the YAML dict ---
        section_order = ("backbone", "neck", "seg_decoder", "attn_fusion", "seg_head", "det_head")
        spec = []
        for k in section_order:
            if k in d and isinstance(d[k], list):
                spec += copy.deepcopy(d[k])

        # --- bookkeeping ---
        ch = [ch] if isinstance(ch, int) else list(ch)
        layers, save = [], []
        c2 = ch[-1]
        legacy = True

        base_modules = frozenset({
            Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
            BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3
        })
        repeat_modules = frozenset({BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3})

        for i, (f, n, m, args) in enumerate(spec):
            # resolve symbol -> class
            if isinstance(m, str):
                if m.startswith("nn."):
                    m = getattr(nn, m[3:])
                elif m.startswith("torchvision.ops."):
                    m = getattr(__import__("torchvision").ops, m[16:])
                elif hasattr(nn, m):
                    m = getattr(nn, m)
                else:
                    m = globals()[m]

            # local, non-shared args list
            args = list(args)
            for j, a in enumerate(args):
                if isinstance(a, str):
                    with contextlib.suppress(ValueError, SyntaxError):
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

            # depth scaling
            n_ = n
            n = max(round(n * depth), 1) if n > 1 else n

            # channel flow
            if m in base_modules:
                c1, c2 = ch[f], args[0]
                if c2 != nc:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                args = [c1, c2, *args[1:]]
                if m in repeat_modules:
                    args.insert(2, n)
                    n = 1
            elif m is Concat:
                f_list = f if isinstance(f, list) else [f]
                c2 = sum(ch[x] if not isinstance(ch[x], (list, tuple)) else ch[x][-1] for x in f_list)
            elif m is Detect:
                # Compute input channel list robustly
                if isinstance(f, list):
                    in_list = [ch[x] if not isinstance(ch[x], (list, tuple)) else list(ch[x]) for x in f]
                    in_list = [int(c) for c in in_list]
                else:
                    c = ch[f]
                    in_list = list(c) if isinstance(c, (list, tuple)) else [int(c)]
                args.append(in_list)
                m.legacy = legacy
                c2 = in_list[-1]
            elif m is Segment:
                if isinstance(f, list):
                    in_list = [ch[x] if not isinstance(ch[x], (list, tuple)) else list(ch[x]) for x in f]
                    in_list = [int(c) for c in in_list]
                else:
                    c = ch[f]
                    in_list = list(c) if isinstance(c, (list, tuple)) else [int(c)]
                args.append(in_list)
                if len(args) >= 3 and isinstance(args[2], (int, float)):
                    args[2] = make_divisible(min(args[2], max_channels) * width, 8)
                m.legacy = legacy
                c2 = in_list[-1]
            elif m is nn.Identity:
                c2 = ch[f] if isinstance(f, int) else ch[f[-1]]
                # Optional: attach attributes if tests expect them
                # (noop if not accessed)
            elif m in {CTAM, CSAM, FPMA}:
                c_in = [ch[x] for x in f] if isinstance(f, list) else ch[f]
                args = [c_in, *args]

                if m is FPMA:
                    # output keeps fine channels: first input if list
                    c2 = c_in[0] if isinstance(c_in, list) else c_in
                elif m is CTAM:
                    # output keeps target channels: first input if list
                    c2 = c_in[0] if isinstance(c_in, list) else c_in
                else:  # CSAM returns a list of per-scale tensors; keep list of channels
                    c2 = list(c_in) if isinstance(c_in, list) else [c_in]

                LOGGER.debug(f"Custom {m.__name__}: in={c_in}, args={args}, out_ch={c2}")
            else:
                c2 = ch[f] if isinstance(f, int) else ch[f[-1]]

            # instantiate
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            if m is Detect:
                # recover the input channels we computed earlier for Detect
                # args layout at call time: [nc, ..., in_list]
                in_list = args[-1] if isinstance(args[-1], (list, tuple)) else [args[-1]]
                if not hasattr(m_, "ch"):
                    m_.ch = list(map(int, in_list))
            t = str(m)[8:-2].replace("__main__.", "")
            m_.np = sum(x.numel() for x in m_.parameters())
            m_.i, m_.f, m_.type = i, f, t

            if verbose:
                LOGGER.info(f"{i:>3}{str(f):>18}{n_:>3}{m_.np:10.0f}  {t:<42}{str(args):<30}")

            f_list = [f] if isinstance(f, int) else list(f)
            save.extend(x % i for x in f_list if x != -1)

            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)

        return nn.Sequential(*layers), sorted(set(save))


