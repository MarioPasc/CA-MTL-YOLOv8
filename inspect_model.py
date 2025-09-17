import torch
import sys
from pathlib import Path


# Load the checkpoint without enforcing GPU
from ultralytics.nn.tasks import SegmentationModel as SegModel
torch.serialization.add_safe_globals([SegModel])
ckpt_det = torch.load("/media/mpascual/PortableSSD/Coronariografías/CAMTL_YOLO/pretrained_models/yolov8s.pt", map_location="cpu", weights_only=False)
ckpt_seg = torch.load("/media/mpascual/PortableSSD/Coronariografías/CAMTL_YOLO/pretrained_models/yolov8s-seg.pt", map_location="cpu", weights_only=False)

# List all top-level keys
print("Detection model checkpoint keys:")
print(ckpt_det.keys())

print("\nSegmentation model checkpoint keys:")
print(ckpt_seg.keys())

print("\nInspecting detection model state_dict:")
# Inspect the model sub-dictionary
model_state = ckpt_det['model']  # This is an nn.Module or its state_dict
print(type(model_state))

# If it's an nn.Module object, you can access its state_dict
if hasattr(model_state, 'state_dict'):
    state = model_state.state_dict()
else:
    state = model_state

# List layer names
for name, tensor in state.items():
    print(name, tensor.shape)
    
print("\nInspecting segmentation model state_dict:")
model_state_seg = ckpt_seg['model']  # This is an nn.Module or its state_dict
print(type(model_state_seg))

# If it's an nn.Module object, you can access its state_dict
if hasattr(model_state_seg, 'state_dict'):
    state_seg = model_state_seg.state_dict()
else:
    state_seg = model_state_seg

# List layer names
for name, tensor in state_seg.items():
    print(name, tensor.shape)
