# tests/test_model_forward.py
import torch, yaml, pytest
from pathlib import Path
import copy


# --- Fixtures ---
import sys
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
print(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from camtl_yolo.model.model import CAMTL_YOLO

@pytest.fixture(scope="module")
def model_config_path():
    """Provide the path to the base model configuration file."""
    return Path(PROJECT_ROOT) / 'camtl_yolo/model/configs/models/camtl_yolov8.yaml'

@pytest.fixture(scope="module")
def base_model_config(model_config_path):
    """Load the base model configuration from the YAML file."""
    with open(model_config_path, 'r') as f:
        return yaml.safe_load(f)

# --- Helper Functions ---

def _first_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (list, tuple)):
        for v in x:
            t = _first_tensor(v)
            if t is not None:
                return t
    if isinstance(x, dict):
        for v in x.values():
            t = _first_tensor(v)
            if t is not None:
                return t
    return None

# --- Tests ---

@pytest.mark.parametrize("scale", ["n", "s", "m", "l", "x"])
def test_model_forward_pass(base_model_config, monkeypatch, scale):
    """
    Tests the model's forward pass for each scale with a dummy input tensor.
    Ensures all layers are connected and output shapes are correct.
    """
    # 1. Prepare configuration for the specific scale
    cfg = copy.deepcopy(base_model_config)
    cfg["SCALE"] = scale
    input_size = cfg["scales"][scale][2]

    # 2. Avoid loading external files during tests
    monkeypatch.setattr(CAMTL_YOLO, "_load_pretrained_weights", lambda self: None, raising=True)

    # 3. Instantiate the model
    model = CAMTL_YOLO(cfg=cfg, ch=3, nc=cfg["nc"])
    model.eval()

    # 4. Capture SegHead output robustly using a forward hook
    seg_out = {}
    try:
        seg_mod = next(m for m in model.modules() if getattr(m, "is_seg_head", False))
        hook = seg_mod.register_forward_hook(lambda m, i, o: seg_out.setdefault("x", o))
    except StopIteration:
        pytest.fail("Could not find a module with 'is_seg_head=True' to hook into.")

    # 5. Run the sequential graph end-to-end
    x = torch.randn(1, 3, input_size, input_size)
    det_raw = model(x)

    hook.remove()

    # 6. Validate Detection Output
    det_tensor = _first_tensor(det_raw)
    assert det_tensor is not None, "Forward pass did not produce a tensor for detection."
    assert det_tensor.shape[0] == 1, "Detection output has incorrect batch size."
    assert torch.isfinite(det_tensor).all(), "Detection output contains non-finite values."

    # 7. Validate Segmentation Output
    assert "x" in seg_out and torch.is_tensor(seg_out["x"]), "Hook did not capture segmentation output."
    seg = seg_out["x"]
    
    # Expect 1-channel logits at the model's minimum stride resolution (typically P3, stride 8)
    min_stride = int(model.stride.min().item())
    expect_hw = input_size // min_stride
    
    expected_shape = (1, 1, expect_hw, expect_hw)
    assert seg.shape == expected_shape, \
        f"Seg output shape mismatch for scale '{scale}'. Got {seg.shape}, expected {expected_shape}."
    assert torch.isfinite(seg).all(), "Segmentation output contains non-finite values."

