import torch
import torch.nn as nn
from pathlib import Path
import yaml # type: ignore
import weakref

# Ultralytics modules
from camtl_yolo.external.ultralytics.ultralytics.nn.modules import (
    Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
    BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, DWConvTranspose2d, C3x, RepC3, Concat, Detect, Segment)
from camtl_yolo.external.ultralytics.ultralytics.nn.tasks import DetectionModel
from camtl_yolo.external.ultralytics.ultralytics.utils import LOGGER
from camtl_yolo.external.ultralytics.ultralytics.utils.ops import make_divisible
from camtl_yolo.external.ultralytics.ultralytics.utils import LOGGER

# Cross-Attention Multi-Task Learning modules
from camtl_yolo.model.nn import CSAM, CTAM, FPMA, SegHead
from camtl_yolo.model.losses import DetectionLoss, MultiScaleBCEDiceLoss, ConsistencyMaskFromBoxes, AttentionAlignmentLoss
from camtl_yolo.model.utils.normalization import replace_seg_stream_bn_with_groupnorm
from camtl_yolo.model.utils.debug_cycles import assert_no_module_cycles

def _find_idx(modules, cls):
    for i, m in enumerate(modules):
        if isinstance(m, cls):
            return i
    return None

def _dump_children(mod: nn.Module, max_depth: int = 3, prefix: str = "root", depth: int = 0):
    if depth > max_depth:
        return
    for k, v in mod._modules.items():
        print(f"{prefix}.{k}: {v.__class__.__name__}")
        _dump_children(v, max_depth=max_depth, prefix=f"{prefix}.{k}", depth=depth+1)


class CAMTL_YOLO(DetectionModel):
    """
    CA-MTL-YOLOv8 model for combined object detection and segmentation.
    This class should be used with a custom 'ca_mtl' task trainer.
    """

    def __init__(self, cfg='camtl_yolov8.yaml', ch=3, nc=None, verbose=True):
        """
        Custom init that avoids Ultralytics DetectionModel.__init__ stride-probing,
        which assumes forward() returns a single Detect output. We:
        1) nn.Module init
        2) load YAML and build via parse_model
        3) compute stride from det outputs using _forward_once()
        4) init weights
        5) task-aware weight loading (COCO or fine-tuned)
        """
        import torch
        import torch.nn as nn
        from pathlib import Path
        import yaml as _yaml  # type: ignore
        from camtl_yolo.external.ultralytics.ultralytics.utils.torch_utils import initialize_weights

        nn.Module.__init__(self)  # avoid DetectionModel.__init__()

        # --- Load YAML ---
        if isinstance(cfg, dict):
            self.yaml = cfg
            self.yaml_file = Path(__file__).resolve().parents[2] / "configs/models/camtl_yolov8.yaml"
        else:
            self.yaml_file = Path(cfg)
            with open(cfg, 'r', encoding='utf-8') as f:
                self.yaml = _yaml.safe_load(f)

        # Common knobs
        self.task = str(self.yaml.get("TASK", "DomainShift1"))
        self.scale = str(self.yaml.get("SCALE", "s"))
        self.pretrained_root = Path(self.yaml.get("PRETRAINED_MODELS_PATH", "."))

        # Default COCO pretrained paths (used in DomainShift1)
        self.seg_weights = self.pretrained_root / f"yolov8{self.scale}-seg.pt"
        self.det_weights = self.pretrained_root / f"yolov8{self.scale}.pt"

        # --- Build architecture ---
        # Note: pass nc override if provided
        self.model, self.save = self.parse_model(self.yaml, ch=[ch] if isinstance(ch, int) else ch,
                                                nc=(int(nc) if nc is not None else None),
                                                verbose=verbose)
        self.inplace = bool(self.yaml.get("inplace", True))
        # names dict like Ultralytics
        self.names = {i: f"{i}" for i in range(int(self.yaml.get("nc", nc if nc is not None else 80)))}
        self.end2end = getattr(self.model[-1], "end2end", False)

        # --- Compute stride using our forward_once → detect feature maps ---
        det_idx = _find_idx(self.model, Detect)
        if det_idx is not None:
            s = 256  # probe size
            x_dummy = torch.zeros(1, ch if isinstance(ch, int) else int(ch[0]), s, s)
            self.model.eval()
            m_det = self.model[det_idx]
            if hasattr(m_det, "inplace"):
                m_det.inplace = self.inplace

            det_out, _ = self._forward_once(x_dummy)  # may be list/tuple and can be nested

            # Normalize to a flat list of feature maps [P3, P4, P5]
            fms = None
            if isinstance(det_out, (list, tuple)):
                # Case A: nested like (preds, [P3,P4,P5]) or ([P3,P4,P5], preds)
                for elem in det_out:
                    if isinstance(elem, (list, tuple)) and all(torch.is_tensor(t) for t in elem):
                        fms = list(elem)
                        break
                # Case B: flat list/tuple of tensors already
                if fms is None and all(torch.is_tensor(t) for t in det_out):
                    fms = list(det_out)
            elif torch.is_tensor(det_out):
                # Rare case: single fm
                fms = [det_out]

            if not fms or not all(torch.is_tensor(t) and t.ndim >= 3 for t in fms):
                raise RuntimeError(f"Could not extract Detect feature maps for stride probing. Got type={type(det_out)}")

            m_det.stride = torch.tensor([s / fm.shape[-2] for fm in fms], dtype=torch.float32)
            self.stride = m_det.stride
            self.model.train()
            if hasattr(m_det, "bias_init"):
                m_det.bias_init()
        else:
            self.stride = torch.tensor([32.0], dtype=torch.float32)

        # --- Init weights like Ultralytics ---
        initialize_weights(self)
        if verbose:
            self.info = lambda detailed=False, verbose=True, imgsz=640: None  # optional no-op info
            LOGGER.info(f"YOLOv8{self.scale} summary: {sum(p.numel() for p in self.parameters())} parameters")

        # --- Task-aware weight loading (COCO or fine-tuned) ---
        self._load_task_weights()

    def forward(self, x, *args, **kwargs):
        """
        Return a 2-tuple (det_preds, seg_preds).
        det_preds is the Detect head raw outputs (list of 3 tensors).
        seg_preds is a tensor or a list of tensors, depending on SegHead.
        """
        return self._forward_once(x)

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
        Also ensures Ultralytics v8DetectionLoss finds hyperparams under self.args.
        """
        from types import SimpleNamespace

        cfg = self.yaml.get("LOSS", {}) if isinstance(self.yaml, dict) else {}
        det_hyp = cfg.get("det_hyp") or {}
        # Defaults match Ultralytics-style gains needed by v8DetectionLoss
        det_defaults = {"box": 7.5, "cls": 0.5, "dfl": 1.5}
        # Create/patch self.args so external v8DetectionLoss can read model.args
        if not hasattr(self, "args") or self.args is None:
            self.args = SimpleNamespace(**{**det_defaults, **det_hyp})
        else:
            # add missing fields without overwriting existing
            for k, v in {**det_defaults, **det_hyp}.items():
                if not hasattr(self.args, k):
                    setattr(self.args, k, v)

        # Detection loss from Ultralytics (uses model.args)
        self.detect_criterion = DetectionLoss(self, hyp=det_hyp)

        # Segmentation loss: BCE+Dice at one or more scales
        seg_scales = cfg.get("seg_scale_weights")
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

        # Attention alignment
        # IMPORTANT: pass a weak proxy to avoid creating a Module cycle (child→parent back-reference).
        self.align_criterion = AttentionAlignmentLoss(
            model=self,
            source_name=str(cfg.get("source_domain", "retinography")),
            target_name=str(cfg.get("target_domain", "angiography")),
            weight=float(cfg.get("align", 0.1)),
        )

        # Scalar weights
        self._lambda_det = float(cfg.get("lambda_det", 1.0))
        self._lambda_seg = float(cfg.get("lambda_seg", 1.0))
        self._lambda_cons = float(cfg.get("lambda_cons", 0.1))
        self._lambda_align = float(cfg.get("lambda_align", 0.1))

        _dump_children(self, max_depth=2)
        # Fail fast if any other accidental cycles exist
        try:
            assert_no_module_cycles(self)
        except Exception as e:
            # Turn into a hard error in dev; WARN if you prefer soft behavior
            raise RuntimeError(f"Module cycle detected after criterion init: {e}")
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
        def _shape_str(obj):
            if obj is None:
                return "None"
            if torch.is_tensor(obj):
                return str(tuple(obj.shape))
            if isinstance(obj, (list, tuple)):
                return "[" + ", ".join(_shape_str(t) for t in obj) + "]"
            return type(obj).__name__

        # Unpack robustly: expect a 2-tuple; if not, assume det-only and set seg=None
        if isinstance(preds, (list, tuple)) and len(preds) == 2:
            det_preds, seg_preds = preds
        else:
            det_preds, seg_preds = preds, None

        device = batch["img"].device

        # detection
        has_det = batch["bboxes"].numel() > 0

        loss_det = torch.zeros((), device=device)
        det_items = {}
        if has_det:
            loss_det, det_items = self.detect_criterion(det_preds, batch)
            if isinstance(loss_det, torch.Tensor) and loss_det.ndim > 0:
                loss_det = loss_det.sum()

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

        def _to_scalar(t: torch.Tensor) -> torch.Tensor:
            # Sum if vector per level, keep dtype/device
            return t if t.ndim == 0 else t.sum()

        loss_det   = _to_scalar(loss_det)
        loss_seg   = _to_scalar(loss_seg)
        cons_loss  = _to_scalar(cons_loss)
        align_loss = _to_scalar(align_loss)

        total = (
            self._lambda_det * loss_det
            + self._lambda_seg * loss_seg
            + self._lambda_cons * cons_loss
            + self._lambda_align * align_loss
            + l2sp_term
        )

        loss_items = torch.stack([
            loss_det.detach(),
            loss_seg.detach(),
            cons_loss.detach(),
            align_loss.detach(),
            l2sp_term.detach(),
            total.detach(),
        ])
        return total, loss_items

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
                replaced = replace_seg_stream_bn_with_groupnorm(self, max_groups=int(self.yaml.get("GN_GROUPS", 32)))
                assert_no_module_cycles(self)
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

