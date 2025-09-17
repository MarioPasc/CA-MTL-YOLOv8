import torch
import yaml
from pathlib import Path
import pytest
import sys

import copy 
# Ensure the project root is in the Python path
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from camtl_yolo.model.model import CAMTL_YOLO
from camtl_yolo.model.nn import SegHead, CSAM, CTAM, FPMA
from camtl_yolo.external.ultralytics.ultralytics.nn.modules import Detect
from camtl_yolo.external.ultralytics.ultralytics.utils import LOGGER

# --- Fixtures ---

@pytest.fixture(scope="module")
def model_config_path():
    """Provide the path to the base model configuration file."""
    return Path(PROJECT_ROOT) / 'camtl_yolo/model/configs/models/camtl_yolov8.yaml'

@pytest.fixture(scope="module")
def hyperparams_path():
    """Provide the path to the hyperparameters configuration file."""
    return Path(PROJECT_ROOT) / 'camtl_yolo/model/configs/hyperparams/defaults.yaml'

@pytest.fixture(scope="module")
def hyperparams(hyperparams_path):
    """Load hyperparameters from the YAML file."""
    with open(hyperparams_path, 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="module")
def base_model_config(model_config_path):
    """Load the base model configuration from the YAML file."""
    with open(model_config_path, 'r') as f:
        return yaml.safe_load(f)

# --- Helper Functions ---

def find_modules(model, module_type):
    """Find all modules of a specific type in the model."""
    return [m for m in model.modules() if isinstance(m, module_type)]

# --- Tests ---
from camtl_yolo.external.ultralytics.ultralytics.utils.ops import make_divisible


def expected_detect_ch(scales_dict, scale, base=(256, 512, 1024)):
    depth, width, maxc = scales_dict[scale]
    return tuple(int(make_divisible(min(c, maxc) * width, 8)) for c in base)

@pytest.mark.parametrize("scale", ["n", "s", "m", "l", "x"])
def _is_ultra_conv_wrapper(m):
    # ultralytics Conv wrapper has attribute `.conv` which is nn.Conv2d
    return hasattr(m, "conv") and isinstance(getattr(m, "conv"), torch.nn.Conv2d)

@pytest.mark.parametrize("scale", ["n", "s", "m", "l", "x"])
def test_model_creation_and_structure(base_model_config, hyperparams, scale):
    cfg = copy.deepcopy(base_model_config)
    cfg["SCALE"] = scale

    model = CAMTL_YOLO(cfg=cfg, ch=3, nc=cfg["nc"])
    model.eval()

    # --- basic ---
    assert model is not None
    assert model.yaml.get("nc") == cfg["nc"]

    # --- heads present ---
    det_heads = find_modules(model, Detect)
    assert len(det_heads) == 1, f"Expected 1 Detect head, found {len(det_heads)}."
    seg_heads = [m for m in model.modules() if getattr(m, "is_seg_head", False) or isinstance(m, SegHead)]
    assert len(seg_heads) == 1, f"Expected 1 Segment head, found {len(seg_heads)}."
    seg_head = seg_heads[0]

    # --- detect input channels (width scaling + cap) ---
    det = det_heads[0]
    exp_ch = expected_detect_ch(cfg["scales"], scale)
    assert tuple(det.ch) == exp_ch, f"Detect.ch mismatch for scale '{scale}': got {tuple(det.ch)}, expected {exp_ch}."

    # --- seg logits: scan backward from SegHead for nearest top-level Conv with out_channels==1 ---
    # NOTE: Detect's 1-ch convs are nested inside Detect block at a later index, so won't interfere.
    seq = model.model  # nn.Sequential
    seg_idx = None
    for i, m in enumerate(seq):
        if m is seg_head:
            seg_idx = i
            break
    assert seg_idx is not None, "SegHead not found in top-level Sequential."

    found_out_c = None
    found_idx = None
    for i in range(seg_idx - 1, -1, -1):
        m = seq[i]
        if _is_ultra_conv_wrapper(m):
            out_c = m.conv.out_channels
            if out_c == 1:
                found_out_c = out_c
                found_idx = i
                break
    assert found_out_c == 1, f"Seg logits channels mismatch: got {found_out_c}, expected 1."
    # (Optional) sanity: ensure we are not accidentally reading a conv that belongs to Detect
    assert not isinstance(seq[found_idx], Detect), "Picked Detect conv instead of seg logits."

    # --- quick structure sanity (presence of custom blocks) ---
    assert find_modules(model, FPMA), "FPMA not instantiated."
    assert find_modules(model, CTAM), "CTAM not instantiated."
    assert find_modules(model, CSAM), "CSAM not instantiated."


@pytest.mark.parametrize("scale", ["n", "s", "m", "l", "x"])
def test_model_forward_pass(base_model_config, scale):
    """
    Tests the model's forward pass with a dummy input tensor to ensure all
    layers are connected correctly and output shapes are as expected.
    """
    # 1. Prepare configuration and instantiate the model
    model_config = base_model_config.copy()
    model_config['SCALE'] = scale
    
    model = CAMTL_YOLO(cfg=model_config, ch=3, nc=model_config['nc'])
    model.eval()

    # 2. Create a dummy input tensor
    input_size = model_config['scales'][scale][2]
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, input_size, input_size)

    # 3. Perform forward pass
    try:
        outputs = model(dummy_input)
    except Exception as e:
        pytest.fail(f"Model forward pass failed for scale '{scale}' with error: {e}")

    # 4. Validate output structure and shapes
    assert isinstance(outputs, list), "Model output should be a list."
    assert len(outputs) == 2, f"Expected 2 outputs (detection, segmentation), but got {len(outputs)}."

    det_out, seg_out = outputs
    
    # Detection output validation
    assert isinstance(det_out, torch.Tensor), "Detection output should be a tensor."
    # Shape: (batch_size, num_classes + 4_bbox_coords + num_masks, num_predictions)
    # The number of predictions depends on the input size and model strides.
    assert det_out.shape[0] == batch_size
    assert det_out.shape[1] == model.nc + 4 + 32 # nc + box + nm

    # Segmentation output validation
    assert isinstance(seg_out, list), "Segmentation output should be a list of tensors."
    assert len(seg_out) == 2, "Expected 2 segmentation outputs (P4, P3 logits)."
    
    # Check shapes of segmentation logits (P4 and P3)
    # P4 logits shape: (batch_size, 1, H/16, W/16)
    p4_logits = seg_out[0]
    expected_p4_shape = (batch_size, 1, input_size // 16, input_size // 16)
    assert p4_logits.shape == expected_p4_shape, f"P4 seg logits shape mismatch. Got {p4_logits.shape}, expected {expected_p4_shape}."

    # P3 logits shape: (batch_size, 1, H/8, W/8)
    p3_logits = seg_out[1]
    expected_p3_shape = (batch_size, 1, input_size // 8, input_size // 8)
    assert p3_logits.shape == expected_p3_shape, f"P3 seg logits shape mismatch. Got {p3_logits.shape}, expected {expected_p3_shape}."

    LOGGER.info(f"Forward pass test passed for scale '{scale}'.")
