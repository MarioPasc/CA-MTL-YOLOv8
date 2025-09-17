import torch
import yaml
from pathlib import Path
import pytest

import sys
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from camtl_yolo.model.model import CAMTL_YOLO

@pytest.fixture
def model_config_path():
    """Provide the path to the model configuration file."""
    return Path(__file__).parent.parent / 'model/configs/models/camtl_yolov8.yaml'

@pytest.fixture
def hyperparams_path():
    """Provide the path to the hyperparameters configuration file."""
    return Path(__file__).parent.parent / 'model/configs/hyperparams/defaults.yaml'

@pytest.mark.parametrize("scale", ["n", "s", "m", "l"])
def test_model_creation_and_weight_loading(model_config_path, hyperparams_path, scale):
    """
    Tests if the CAMTL_YOLO model can be created and if weights are loaded correctly for a given scale.
    """
    # Load hyperparameters to get pretrained model paths
    with open(hyperparams_path, 'r') as f:
        hyperparams = yaml.safe_load(f)

    # Create a temporary model config for the test, injecting the absolute path and scale
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    model_config['PRETRAINED_MODELS_PATH'] = hyperparams['PRETRAINED_MODELS_PATH']
    model_config['SCALE'] = scale
    
    # Create a temporary yaml file for the model to load
    temp_config_path = Path(f'./temp_test_config_{scale}.yaml')
    with open(temp_config_path, 'w') as f:
        yaml.dump(model_config, f)

    # 1. Instantiate the model
    try:
        model = CAMTL_YOLO(cfg=str(temp_config_path), ch=3, nc=model_config['nc'])
        model.eval()
    finally:
        # Clean up the temporary file
        if temp_config_path.exists():
            temp_config_path.unlink()

    # 2. Load original checkpoints to verify against
    pretrained_path = Path(hyperparams['PRETRAINED_MODELS_PATH'])
    seg_weights_path = pretrained_path / f"yolov8{scale}-seg.pt"
    det_weights_path = pretrained_path / f"yolov8{scale}.pt"

    ckpt_seg = torch.load(seg_weights_path, map_location='cpu', weights_only=False)
    csd_seg = ckpt_seg['model'].float().state_dict()

    ckpt_det = torch.load(det_weights_path, map_location='cpu', weights_only=False)
    csd_det = ckpt_det['model'].float().state_dict()

    # 3. Verify that weights have been loaded correctly
    
    # Check a backbone layer (e.g., model.0.conv.weight)
    torch.testing.assert_close(
        model.model[0].conv.weight.data,
        csd_seg['model.0.conv.weight']
    )
    
    # Check a neck layer (e.g., model.12.m.0.cv1.conv.weight)
    torch.testing.assert_close(
        model.model[12].m[0].cv1.conv.weight.data,
        csd_seg['model.12.m.0.cv1.conv.weight']
    )

    # Check the detection head (model.33 in our case, model.22 in original)
    # The index of the detection head can change based on the model structure.
    # We need to find the 'Detect' module in our model.
    detect_head_idx = -1
    for i, module in enumerate(model.model):
        if module.type == 'camtl_yolo.external.ultralytics.ultralytics.nn.modules.head.Detect':
            detect_head_idx = i
            break
    
    assert detect_head_idx != -1, "Detection head not found in the model."

    torch.testing.assert_close(
        model.model[detect_head_idx].cv2[0][0].conv.weight.data,
        csd_det['model.22.cv2.0.0.conv.weight']
    )

    print(f"Model weight loading test passed successfully for scale '{scale}'!")
