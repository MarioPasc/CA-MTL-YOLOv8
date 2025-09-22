from camtl_yolo.model.losses.regularizers import L2SPRegularizer, snapshot_reference
from camtl_yolo.model.losses.detection import DetectionLoss
from camtl_yolo.model.losses.segmentation import DeepSupervisionConfigurableLoss
from camtl_yolo.model.losses.align import CompositeAlignmentLoss, build_alignment_loss
from camtl_yolo.model.losses.kl import AttentionGuidanceKLLoss
__all__ = [
    "L2SPRegularizer", 
    "snapshot_reference", 
    "DetectionLoss", 
    "DeepSupervisionConfigurableLoss",
    "CompositeAlignmentLoss",
    "build_alignment_loss",
    "AttentionGuidanceKLLoss"
]

