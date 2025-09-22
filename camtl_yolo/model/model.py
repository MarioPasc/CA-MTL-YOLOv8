import torch
import torch.nn as nn
from pathlib import Path
import yaml # type: ignore
from types import SimpleNamespace
from torch.amp import autocast

# Ultralytics modules
from camtl_yolo.external.ultralytics.ultralytics.nn.modules import (
    Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
    BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, DWConvTranspose2d, C3x, RepC3, Concat, Detect, Segment)
from camtl_yolo.external.ultralytics.ultralytics.nn.tasks import DetectionModel
from camtl_yolo.external.ultralytics.ultralytics.utils import LOGGER
from camtl_yolo.external.ultralytics.ultralytics.utils.ops import make_divisible
from camtl_yolo.external.ultralytics.ultralytics.utils import LOGGER

# Cross-Attention Multi-Task Learning modules
from camtl_yolo.model.nn import CSAM, CTAM, FPMA, SegHeadMulti
from camtl_yolo.model.losses import (
    DetectionLoss, 
    DeepSupervisionConfigurableLoss, 
    build_alignment_loss,
    L2SPRegularizer,
    AttentionGuidanceKLLoss
    )
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
        
def _model_info(self, detailed: bool = False, verbose: bool = True, imgsz: int = 512):
    """Pickle-safe model summary. Ultralytics-compatible."""
    from camtl_yolo.external.ultralytics.ultralytics.utils import LOGGER
    nparams = sum(p.numel() for p in self.parameters())
    if verbose:
        LOGGER.info(
            f"CA-MTL-YOLO: params={nparams:,}, stride={getattr(self, 'stride', None)}, imgsz={int(imgsz)}"
        )
    return {
        "params": int(nparams),
        "stride": getattr(self, "stride", None),
        "imgsz": int(imgsz),
        "detailed": bool(detailed),
    }
    

def info(self, detailed: bool = False, verbose: bool = True, imgsz: int = 640):
    """Public entry used by Ultralytics notebooks."""
    return self._model_info(detailed=detailed, verbose=verbose, imgsz=imgsz)

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
        # --- Task-aware weight loading (COCO or fine-tuned) ---
        self._load_task_weights()
        

    def forward(self, x, *args, **kwargs):
        """
        Ultralytics-compatible forward.

        If `x` is a dict (training/eval step), compute loss and return (total_loss, loss_items).
        If `x` is a tensor (inference), return a tuple (det_out, seg_out).

        This avoids storing any non-picklable callable on the model (no local closures).
        """
        # In training/eval the dataloader passes a batch dict
        if isinstance(x, dict):
            # Ensure criteria are initialized
            if getattr(self, "detect_criterion", None) is None:
                self.init_criterion()
            preds = self._forward_once(x["img"])
            total, items = self.loss(x, preds)
            return total, items

        # Inference path: tensor input → tuple(det_out, seg_out)
        return self._forward_once(x)


    def predict(self, x, profile: bool = False, visualize: bool = False,
                augment: bool = False, embed=None):
        """
        Ultralytics-compatible predict that returns (det_out, seg_out).
        No augmented prediction for CAMTL; falls back to single-scale.
        """
        if augment:
            LOGGER.warning("CAMTL_YOLO does not support 'augment=True' prediction. Falling back to single-scale.")
        # profile/visualize/embed can be integrated later; kept for signature parity
        return self._forward_once(x)

    def _predict_once(self, x, profile: bool = False, visualize: bool = False, embed=None):
        """
        Bridge for BaseModel.predict() to ensure we always emit (det_out, seg_out).
        """
        return self._forward_once(x)


    def _forward_once(self, x: torch.Tensor):
        """
        Forward pass that returns a tuple (det_out, seg_out).

        Notes:
        - seg_out is typically a dict with keys {'p3','p4','p5','full'} from SegHeadMulti.
        - If SegHeadMulti is absent or emits a single tensor, return that tensor in seg_out.
        """
        y = []
        det_out, seg_out = None, None
        input_hw = x.shape[-2:] if torch.is_tensor(x) else None  # (H, W)

        for i, m in enumerate(self.model):
            # Build module input(s) from cache
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            # Run
            if isinstance(m, SegHeadMulti):
                seg_out = m(x, input_hw=input_hw)   # x is List[P3,P4,P5]; returns dict or tensor
                out = seg_out
            else:
                out = m(x)

            # Capture detect output explicitly
            if isinstance(m, Detect):
                det_out = out

            # Cache if requested
            y.append(out if m.i in self.save else None)
            x = out

        return det_out, seg_out

    def init_criterion(self) -> None:
        """
        Initialize CAMTL loss components on the model.

        Side effects:
            - Sets `self.detect_criterion`, `self.segment_criterion`, `self.align_criterion`
            - Sets scalar weights `_lambda_*`
            - Attaches picklable `_l2sp` object if enabled
        """
        cfg = self.yaml.get("LOSS", {}) if isinstance(self.yaml, dict) else {}
        det_hyp = cfg.get("det_hyp") or {}
        det_defaults = {"box": 7.5, "cls": 0.5, "dfl": 1.5}
        for k, v in det_defaults.items():
            det_hyp.setdefault(k, v)

        # Expose Ultralytics-style args if available
        if det_hyp is None:
            if not hasattr(self, "args"):
                self.args = SimpleNamespace()
            if isinstance(det_hyp, dict):
                for k, v in det_hyp.items():
                    if not hasattr(self.args, k):
                        setattr(self.args, k, v)

        # Sub-criteria
        self.detect_criterion = DetectionLoss(self, hyp=det_hyp)
        self.segment_criterion = DeepSupervisionConfigurableLoss(  
            config={
                "p3": cfg.get("p3", ["BCE","DICE"]),
                "p4": cfg.get("p4", ["BCE","DICE"]),
                "p5": cfg.get("p5", ["BCE","DICE"]),
            },
            init_weights={
                "p3": float(cfg.get("w_p3", 1.0)),
                "p4": float(cfg.get("w_p4", 1.0)),
                "p5": float(cfg.get("w_p5", 1.0)),
            },
        )

        self.align_criterion = build_alignment_loss(cfg, model=self)

        self.consistency_criterion = AttentionGuidanceKLLoss(
            model=self,
            weight=float(cfg.get("consistency", 1e-3)),
            include_csam=bool(cfg.get("consistency_include_csam", True)),
            eps=float(cfg.get("consistency_eps", 1e-8)),
        )

        # Scalars
        self._lambda_det = float(cfg.get("lambda_det", 1.0))
        self._lambda_seg = float(cfg.get("lambda_seg", 1.0))
        self._lambda_attn = float(cfg.get("lambda_attn", 0.1))
        self._lambda_align = float(cfg.get("lambda_align", 0.1))
        l2sp_lambda = float(cfg.get("lambda_l2sp", 0.0))

        # ---- L2-SP anchors (picklable) ----
        def _include(name: str) -> bool:
            skip_tags = (".cv2.", ".cv3.", ".dfl.", "SegHeadMulti", "CTAM", "CSAM", "FPMA", "Detect")
            return (not any(t in name for t in skip_tags))

        if l2sp_lambda > 0.0:
            anchors: dict[str, torch.Tensor] = {}
            for n, p in self.named_parameters():
                if p.requires_grad and _include(n):
                    anchors[n] = p.detach().clone()
            # top-level, picklable object
            self._l2sp = L2SPRegularizer(anchors=anchors, lam=l2sp_lambda)
        else:
            self._l2sp = None  # explicit


        # no return

    def _ensure_loss_reporting(self) -> None:
        """
        Ensure the model exposes a stable loss header aligned with the fixed-length loss vector.
        Length and order must match what `loss()` returns every step.
        """
        names = (
            # high-level scalars
            "det", "seg", "attn_kl", "align", "l2sp", "total",
            # detection breakdown
            "box", "cls", "dfl",
            # segmentation breakdown (deep supervision)
            "p3_bce", "p4_bce", "p5_bce",
            "p3_dice", "p4_dice", "p5_dice",
        )
        # Set once. Ultralytics trainer will read `self.loss_names`.
        if not hasattr(self, "loss_names"):
            self.loss_names = names  # tuple is fine


    def loss(self, batch, preds):
        """
        CAMTL multi-task loss with fixed-length reporting.

        Returns:
            total: scalar tensor
            loss_items: 1D tensor of length 15 with the following order:

            ("det","seg","attn_kl","align","l2sp","total",
            "box","cls","dfl",
            "p3_bce","p4_bce","p5_bce","p3_dice","p4_dice","p5_dice")

        Absent components are zero. Shapes and header names are stable across steps.
        """

        # Ensure header exists before trainer reads it
        self._ensure_loss_reporting()

        # Unpack predictions
        if isinstance(preds, (list, tuple)) and len(preds) == 2:
            det_preds, seg_preds = preds
        else:
            det_preds, seg_preds = preds, None

        device = batch["img"].device
        dtype = torch.float32  # reporting dtype; computation dtype is carried by individual criteria

        # ---- Task flags (robust) ----
        # Accept several conventions to avoid key errors
        def _truthy(x):
            if torch.is_tensor(x):
                return bool(x.detach().sum().item() > 0)
            if isinstance(x, (list, tuple)):
                return any(bool(v) for v in x)
            return bool(x)

        has_det = False
        if "bboxes" in batch:
            bx = batch["bboxes"]
            has_det = torch.is_tensor(bx) and bx.numel() > 0
        elif "is_det" in batch:
            has_det = _truthy(batch["is_det"])
        elif batch.get("task", None) == "det":
            has_det = True

        has_seg = False
        if "is_seg" in batch:
            has_seg = _truthy(batch["is_seg"])
        elif batch.get("task", None) == "seg":
            has_seg = True
        elif "masks" in batch:
            mk = batch["masks"]
            has_seg = torch.is_tensor(mk) and mk.numel() > 0

        # ---- Defaults to avoid UnboundLocalError ----
        det_box = det_cls = det_dfl = 0.0
        p3_bce = p4_bce = p5_bce = 0.0
        p3_dice = p4_dice = p5_dice = 0.0

        # ---- Detection loss ----
        loss_det = torch.zeros((), device=device, dtype=dtype)
        if has_det:
            try:
                l_det, det_items = self.detect_criterion(det_preds, batch)  # Ultralytics-style
                # det_items may be dict with 'box','cls','dfl'
                if isinstance(det_items, dict):
                    det_box = float(det_items.get("box", 0.0))
                    det_cls = float(det_items.get("cls", 0.0))
                    det_dfl = float(det_items.get("dfl", 0.0))
                loss_det = l_det if l_det.ndim == 0 else l_det.sum()
            except Exception as e:
                LOGGER.warning(f"[loss] detection loss failed: {e}")

        # ---- Segmentation loss ----
        loss_seg = torch.zeros((), device=device, dtype=dtype)
        if has_seg:
            try:
                # Compute segmentation loss in FP32 to avoid AMP half-precision overflows
                with autocast("cuda", enabled=False):
                    seg_inputs = seg_preds
                    if isinstance(seg_preds, dict):
                        seg_inputs = {k: (v.float() if torch.is_tensor(v) else v) for k, v in seg_preds.items()}
                    elif torch.is_tensor(seg_preds):
                        seg_inputs = seg_preds.float()
                    l_seg, seg_items = self.segment_criterion(seg_inputs, batch)
                if isinstance(seg_items, dict):
                    p3_bce = float(seg_items.get("p3_bce", 0.0))
                    p4_bce = float(seg_items.get("p4_bce", 0.0))
                    p5_bce = float(seg_items.get("p5_bce", 0.0))
                    p3_dice = float(seg_items.get("p3_dice", 0.0))
                    p4_dice = float(seg_items.get("p4_dice", 0.0))
                    p5_dice = float(seg_items.get("p5_dice", 0.0))
                loss_seg = l_seg if l_seg.ndim == 0 else l_seg.sum()
            except Exception as e:
                LOGGER.warning(f"[loss] segmentation loss failed: {e}")

        align_loss = torch.zeros((), device=device, dtype=dtype)
        if self.args.task != "DomainShift1":
            # ---- alignment (containment or CTAM), only meaningful if enabled in YAML ----
            try:
                a_l, _ = self.align_criterion(seg_preds, det_preds, batch)  # new signature
                align_loss = a_l if a_l.ndim == 0 else a_l.sum()
            except Exception as e:
                LOGGER.warning(f"[loss] alignment loss failed: {e}")

        # ---- AttnKL (attention guidance or fallback) ----
        attn_kl = torch.zeros((), device=device, dtype=dtype)
        try:
            c_l, _ = self.consistency_criterion(seg_preds, batch)
            attn_kl = c_l if c_l.ndim == 0 else c_l.sum()
        except Exception as e:
            LOGGER.warning(f"[loss] consistency loss failed: {e}")

        # ---- L2-SP ----
        l2sp_term = torch.zeros((), device=device, dtype=dtype)
        if getattr(self, "_l2sp", None) is not None:
            try:
                l2sp_term = self._l2sp(self)  # type: ignore
            except Exception as e:
                LOGGER.warning(f"[loss] L2-SP failed: {e}")


        # ---- Total ----
        total = (
            self._lambda_det * loss_det
            + self._lambda_seg * loss_seg
            + self._lambda_attn * attn_kl
            + self._lambda_align * align_loss
            + l2sp_term
        )

        # ---- Reporting tensor (task-aware ordering/length) ----
        task = str(getattr(self, "task", "DomainShift1"))
        if task == "DomainShift1":
            # Expected by CAMTLTrainer for DomainShift1 (12 items, no det breakdown)
            vals = [
                loss_det.detach(),
                loss_seg.detach(),
                attn_kl.detach(),
                align_loss.detach(),
                l2sp_term.detach(),
                total.detach(),
                torch.tensor(p3_bce, device=device, dtype=dtype),
                torch.tensor(p4_bce, device=device, dtype=dtype),
                torch.tensor(p5_bce, device=device, dtype=dtype),
                torch.tensor(p3_dice, device=device, dtype=dtype),
                torch.tensor(p4_dice, device=device, dtype=dtype),
                torch.tensor(p5_dice, device=device, dtype=dtype),
            ]
        else:
            # Full 15-length vector used in CAMTL (includes det breakdown)
            vals = [
                loss_det.detach(), loss_seg.detach(), attn_kl.detach(), align_loss.detach(), l2sp_term.detach(), total.detach(),
                torch.tensor(det_box, device=device, dtype=dtype),
                torch.tensor(det_cls, device=device, dtype=dtype),
                torch.tensor(det_dfl, device=device, dtype=dtype),
                torch.tensor(p3_bce, device=device, dtype=dtype),
                torch.tensor(p4_bce, device=device, dtype=dtype),
                torch.tensor(p5_bce, device=device, dtype=dtype),
                torch.tensor(p3_dice, device=device, dtype=dtype),
                torch.tensor(p4_dice, device=device, dtype=dtype),
                torch.tensor(p5_dice, device=device, dtype=dtype),
            ]
        loss_items = torch.stack(vals, dim=0)
        return total, loss_items


    @torch.no_grad()
    def save_task_checkpoint(self, save_dir: str | Path, filename: str | None = None,
                             epoch: int | None = None, optimizer=None, extra: dict | None = None) -> Path:
        """
        Save a task-aware checkpoint:
          - DomainShift1 -> yolov8{SCALE}-domainshift1.pt
          - CAMTL       -> yolov8{SCALE}-camtl.pt

        Implementation notes:
          - Avoid pickling the live Module (which may hold non-picklable refs like weakref in criteria).
          - Save only a state_dict and lightweight metadata instead.
          - Do not mutate this model's dtype/device during save.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            suffix = "domainshift1" if self.task == "DomainShift1" else "camtl" if self.task == "CAMTL" else "custom"
            filename = f"yolov8{self.scale}-{suffix}.pt"

        # Collect a CPU-FP32 copy of the state_dict without mutating the live model
        sd = {k: v.detach().to(dtype=torch.float32, device="cpu") for k, v in self.state_dict().items()}
        ckpt = {
            "model": sd,  # state_dict only
            "epoch": int(epoch or 0),
            "optimizer": None,  # keep ckpt lean; training resumes via Ultralytics flow
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
            elif m is SegHeadMulti:
                # compute actual input channels from f
                if isinstance(f, list):
                    in_list = [ch[x] if not isinstance(ch[x], (list, tuple)) else ch[x][-1] for x in f]
                else:
                    c = ch[f]
                    in_list = [c] if not isinstance(c, (list, tuple)) else list(c)
                in_list = [int(c) for c in in_list]
                LOGGER.debug(f"Custom {m.__name__}: in={in_list}, args={args}")

                # normalize args to exactly: [in_channels, out_channels, fuse]
                out_ch = 1
                fuse = True

                # Case A: YAML provided [in_channels, out_ch, fuse] → replace in_channels with computed to ensure match
                if len(args) >= 1 and isinstance(args[0], (list, tuple)) and len(args[0]) == 3:
                    # optional overrides
                    if len(args) >= 2 and isinstance(args[1], (int, float)):
                        out_ch = int(args[1])
                    if len(args) >= 3 and isinstance(args[2], (bool,)):
                        fuse = bool(args[2])
                    args = [in_list, out_ch, fuse]
                else:
                    # Case B: YAML did not provide in_channels → use computed and keep optional overrides
                    if len(args) >= 1 and isinstance(args[0], (int, float)):
                        out_ch = int(args[0])
                    if len(args) >= 2 and isinstance(args[1], (bool,)):
                        fuse = bool(args[1])
                    args = [in_list, out_ch, fuse]

                m.legacy = legacy
                c2 = in_list[-1]
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


    # --- Weight loading and saving ---------------------------------

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

    def _load_task_weights(self) -> None:
        """
        Load pretrained or fine-tuned weights based on YAML 'TASK'.

        - DomainShift1: COCO seg + COCO det (safe remap of Detect head)
        - CAMTL: single fine-tuned checkpoint 'yolov8{SCALE}-domainshift1.pt'
        """
        task = str(self.task)
        if task == "DomainShift1":
            self._load_coco_pretrained_weights()
            return
        if task == "CAMTL":
            ckpt_path = self.pretrained_root / f"yolov8{self.scale}-domainshift1.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"[CAMTL] DomainShift1 checkpoint not found: {ckpt_path}")
            # Ensure seg-stream GN if that is your phase-1 norm policy
            try:
                replaced = replace_seg_stream_bn_with_groupnorm(self, max_groups=int(self.yaml.get("GN_GROUPS", 32)))
                assert_no_module_cycles(self)
                if replaced:
                    LOGGER.info(f"[CAMTL] Converted {replaced} BN layers to GroupNorm in segmentation stream before loading.")
            except Exception as e:
                LOGGER.warning(f"[CAMTL] GN conversion skipped: {e}")
            self._load_finetuned_weights(ckpt_path)
            return
        LOGGER.warning(f"[TASK={task}] Unknown task. Skipping pretrained loading.")

    def _load_coco_pretrained_weights(self) -> None:
        """
        DomainShift1 boot: load backbone+neck from *seg* checkpoint and map Detect head from *det* checkpoint.
        Safely skip classification branch tensors when 'nc' differs.
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

        # 1) backbone+neck from seg ckpt, exclude its seg head block ('model.22.')
        for k, v in csd_seg.items():
            if k.startswith("model.22."):
                skipped += 1
                continue
            if k in new_sd and new_sd[k].shape == v.shape:
                new_sd[k] = v
                loaded += 1
            else:
                mismatched += 1

        # 2) Detect head remap
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

        # Commit
        self.load_state_dict(new_sd, strict=False)
        LOGGER.info(
            f"Pretrained load summary: loaded={loaded}, skipped={skipped}, "
            f"shape_mismatch={mismatched}, det_mapped={det_loaded}/{det_total_eligible}"
        )

    def _load_finetuned_weights(self, ckpt_path: Path) -> None:
        """
        CAMTL phase: load a single fine-tuned checkpoint produced after DomainShift1.
        Accepts Ultralytics-style {'model': nn.Module, ...} or a raw state_dict under 'model'.
        """
        LOGGER.info(f"[CAMTL] Loading fine-tuned checkpoint from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict) and "model" in ckpt:
            src = ckpt["model"]
            state_src = src.state_dict() if hasattr(src, "state_dict") else dict(ckpt["model"])
        elif isinstance(ckpt, dict):
            state_src = ckpt  # already a state_dict
        else:
            state_src = ckpt.state_dict()  # bare nn.Module

        state_dst = self.state_dict()
        loaded, mismatched = 0, 0

        for k, v in state_src.items():
            if k in state_dst and state_dst[k].shape == v.shape:
                state_dst[k] = v
                loaded += 1
            else:
                mismatched += 1

        self.load_state_dict(state_dst, strict=False)
        LOGGER.info(f"[CAMTL] Load summary: loaded={loaded}, mismatched_or_missing={mismatched}")
