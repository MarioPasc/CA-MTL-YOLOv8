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
from camtl_yolo.external.ultralytics.ultralytics.utils.loss import v8DetectionLoss
from camtl_yolo.external.ultralytics.ultralytics.utils.ops import make_divisible

# Cross-Attention Multi-Task Learning modules
from camtl_yolo.model.nn import CSAM, CTAM, FPMA

class CAMTL_YOLO(DetectionModel):
    """
    CA-MTL-YOLOv8 model for combined object detection and segmentation.
    This class should be used with a custom 'ca_mtl' task trainer.
    """

    def __init__(self, cfg='ca_mtl_yolov8.yaml', ch=3, nc=None, verbose=True):
        """Initializes the model, parses the YAML config, and builds the network."""
        super().__init__()
        self.yaml_file = Path(cfg).name
        
        # Load YAML and get model configuration
        with open(cfg, 'r', encoding='utf-8') as f:
            self.yaml = yaml.safe_load(f)
        
        # Get pretrained model paths from hyperparameters
        pretrained_path = self.yaml.get('PRETRAINED_MODELS_PATH')
        scale = self.yaml.get('SCALE', 's')
        
        # Define paths for pretrained weights
        self.seg_weights = Path(pretrained_path) / f"yolov8{scale}-seg.pt"
        self.det_weights = Path(pretrained_path) / f"yolov8{scale}.pt"

        # Parse the model structure from YAML
        self.model, self.save = self.parse_model(self.yaml, ch=[ch] if isinstance(ch, int) else ch, nc=nc)
        
        # Load pretrained weights
        self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        """Loads pretrained weights from detection and segmentation checkpoints."""
        if not self.seg_weights.exists():
            raise FileNotFoundError(f"Segmentation weights not found at {self.seg_weights}")
        if not self.det_weights.exists():
            raise FileNotFoundError(f"Detection weights not found at {self.det_weights}")

        LOGGER.info(f"Loading backbone and neck from {self.seg_weights}...")
        ckpt_seg = torch.load(self.seg_weights, map_location=torch.device('cpu'), weights_only=False)
        csd_seg = ckpt_seg['model'].float().state_dict()

        LOGGER.info(f"Loading detection head from {self.det_weights}...")
        ckpt_det = torch.load(self.det_weights, map_location=torch.device('cpu'), weights_only=False)
        csd_det = ckpt_det['model'].float().state_dict()

        # Create a new state dict for our model
        new_state_dict = self.state_dict()
        
        # Load backbone and neck weights from segmentation model
        for k, v in csd_seg.items():
            if k in new_state_dict and new_state_dict[k].shape == v.shape:
                # In yolov8s-seg.pt, the segmentation head is layer 22. We don't want to load it.
                if 'model.22.' not in k:
                    new_state_dict[k] = v
                    LOGGER.info(f"Loaded {k} from segmentation model.")

        # Load detection head weights from detection model
        # In yolov8s.pt, the detection head is layer 22. In our custom model, it's layer 33.
        # This mapping needs to be adjusted based on the final YAML structure.
        # Assuming detect_head is at index 33
        for k_det, v_det in csd_det.items():
            if 'model.22.' in k_det:
                # Map layer 22 from detection checkpoint to layer 33 in our model
                new_k = k_det.replace('model.22.', 'model.33.')
                if new_k in new_state_dict and new_state_dict[new_k].shape == v_det.shape:
                    new_state_dict[new_k] = v_det
                    LOGGER.info(f"Loaded {k_det} as {new_k} from detection model.")
                else:
                    LOGGER.warning(f"Could not load {k_det} as {new_k}. Shape mismatch or key not found.")

        # Load the combined state dict
        self.load_state_dict(new_state_dict, strict=False)
        LOGGER.info("Pretrained weights loaded successfully. Unmatched weights will be initialized randomly.")

    def _forward_once(self, x):
        """
        Executes the forward pass, returning raw outputs from detection and segmentation heads.
        """
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        
        # Assuming detect_head is at index 33 and seg_head is at 34 from the YAML
        # These indices must match your ca_mtl_yolov8.yaml
        detect_head_idx = -2 # Typically the second to last module
        seg_head_idx = -1 # Typically the last module
        
        det_out = y[self.model[detect_head_idx].i]
        seg_out = y[self.model[seg_head_idx].i]
        
        return (det_out, seg_out)

    def init_criterion(self):
        """Initializes the loss functions for multi-task training."""
        # This would be called by your custom CAMTLTrainer
        # self.detect_criterion = v8DetectionLoss(...)
        # self.segment_criterion = SegmentationLoss(...)
        # self.consistency_criterion = ConsistencyLoss(...)
        LOGGER.info("Custom multi-task criterion initialized.")
        # For now, we can just return a placeholder
        return nn.CrossEntropyLoss() # Placeholder

    def loss(self, batch, preds):
        """
        Computes the combined loss for detection, segmentation, and auxiliary tasks.
        """
        # This method will be called by your CAMTLTrainer during the training loop.
        
        # 1. Unpack predictions
        det_preds, seg_preds = preds
        
        # 2. Calculate detection loss
        # loss_det, loss_det_items = self.detect_criterion(det_preds, batch)
        
        # 3. Calculate segmentation loss
        # loss_seg, loss_seg_items = self.segment_criterion(seg_preds, batch)
        
        # 4. Calculate auxiliary losses (consistency, etc.)
        # loss_aux, loss_aux_items = self.consistency_criterion(...)
        
        # 5. Combine losses
        # total_loss = loss_det + loss_seg + loss_aux
        # loss_items = {**loss_det_items, **loss_seg_items, **loss_aux_items}
        
        # Placeholder implementation
        LOGGER.info("Calculating combined loss (placeholder).")
        total_loss = torch.tensor(0.0, device=batch['img'].device, requires_grad=True)
        loss_items = {'loss': total_loss}

        return total_loss, loss_items

    def parse_model(self, d: dict, ch, nc: int | None = None, verbose: bool = True):
        """
        Parse a CAMTL-YOLO model.yaml into a torch.nn.Sequential.
        Supports sections: backbone, neck, seg_decoder, attn_fusion, seg_head, detect_head (and legacy 'head').
        Applies scales[SCALE] for depth/width/max_channels.
        """
        import ast
        import contextlib
        import torch
        import torch.nn as nn

        # --- scaling knobs ---
        act = d.get("activation")
        scales = d.get("scales")
        scale = d.get("SCALE", d.get("scale"))
        depth, width, max_channels = (
            d.get("depth_multiple", 1.0),
            d.get("width_multiple", 1.0),
            float("inf"),
        )
        if scales:
            if not scale:
                scale = next(iter(scales.keys()))
                LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
            try:
                depth, width, max_channels = scales[scale]
            except Exception:
                LOGGER.warning(f"scale '{scale}' not found in 'scales'. Using depth/width defaults.")

        if nc is None:
            nc = int(d.get("nc", 80))

        if act:
            Conv.default_act = eval(act)

        if verbose:
            LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<42}{'arguments':<30}")

        # --- section order (must match YAML indices) ---
        section_order = ("backbone", "neck", "seg_decoder", "attn_fusion", "seg_head", "detect_head", "head")
        layers_spec = []
        for k in section_order:
            if k in d and isinstance(d[k], list):
                layers_spec += d[k]

        # --- bookkeeping ---
        ch = [ch] if isinstance(ch, int) else list(ch)
        layers, save = [], []
        c2 = ch[-1]
        legacy = True  # keep YOLO detect/segment legacy flag behavior

        base_modules = frozenset({
            Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
            BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3
        })
        repeat_modules = frozenset({BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3})

        # --- build ---
        for i, (f, n, m, args) in enumerate(layers_spec):
            # resolve module symbol -> class
            if isinstance(m, str):
                if m.startswith("nn."):
                    m = getattr(nn, m[3:])
                elif m.startswith("torchvision.ops."):
                    m = getattr(__import__("torchvision").ops, m[16:])
                elif hasattr(nn, m):  # allow bare names like "Identity"
                    m = getattr(nn, m)
                else:
                    m = globals()[m]

            # safe literal-eval of string args (e.g., 'heads=4')
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
                if c2 != nc:  # not a classifier output
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                args = [c1, c2, *args[1:]]
                if m in repeat_modules:
                    args.insert(2, n)
                    n = 1
            elif m is Concat:
                f_list = f if isinstance(f, list) else [f]
                c2 = sum(ch[x] for x in f_list)
            elif m is Detect:
                # args: [nc, ...] -> append list of in-channels
                args.append([ch[x] for x in (f if isinstance(f, list) else [f])])
                m.legacy = legacy
            elif m is Segment:
                args.append([ch[x] for x in (f if isinstance(f, list) else [f])])
                # prototype channels at args[2]
                if len(args) >= 3 and isinstance(args[2], (int, float)):
                    args[2] = make_divisible(min(args[2], max_channels) * width, 8)
                m.legacy = legacy
            elif m is nn.Identity:
                # passthrough, no args, output ch equals input ch
                c2 = ch[f] if isinstance(f, int) else ch[f[-1]]
            elif m in {CTAM, CSAM, FPMA}:
                # custom blocks take input channel list first
                c_in = [ch[x] for x in f] if isinstance(f, list) else ch[f]
                args = [c_in, *args]
                # infer output channels: second arg if int, else pass-through last in
                LOGGER.debug(f"Custom block {m.__name__} with input channels {c_in} and args {args}")
                LOGGER.debug(f"Arguments for {m.__name__}: {args}")
                c2 = args[1] if len(args) > 1 and isinstance(args[1], int) else (c_in[-1] if isinstance(c_in, list) else c_in)
            else:
                # default: keep last input channel size
                c2 = ch[f] if isinstance(f, int) else ch[f[-1]]

            # instantiate
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)[8:-2].replace("__main__.", "")
            m_.np = sum(x.numel() for x in m_.parameters())
            m_.i, m_.f, m_.type = i, f, t

            if verbose:
                LOGGER.info(f"{i:>3}{str(f):>18}{n_:>3}{m_.np:10.0f}  {t:<42}{str(args):<30}")

            # record savelist
            f_list = [f] if isinstance(f, int) else list(f)
            save.extend(x % i for x in f_list if x != -1)

            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)

        return nn.Sequential(*layers), sorted(set(save))

