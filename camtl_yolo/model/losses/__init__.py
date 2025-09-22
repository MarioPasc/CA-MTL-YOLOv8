from camtl_yolo.model.losses.regularizers import L2SPRegularizer, snapshot_reference
from camtl_yolo.model.losses.detection import DetectionLoss
from camtl_yolo.model.losses.consistency import ConsistencyMaskFromBoxes
from camtl_yolo.model.losses.segmentation import DeepSupervisionBCEDiceLoss
from camtl_yolo.model.losses.align import AttentionAlignmentLoss, build_alignment_loss

__all__ = [
    "L2SPRegularizer", 
    "snapshot_reference", 
    "DetectionLoss", 
    "ConsistencyMaskFromBoxes", 
    "DeepSupervisionBCEDiceLoss",
    "AttentionAlignmentLoss",
    "build_alignment_loss"
]

