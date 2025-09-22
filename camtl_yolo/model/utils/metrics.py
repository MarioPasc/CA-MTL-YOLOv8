
import torch
import torch.nn.functional as F

def dice(pred_logits: torch.Tensor, y: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Soft Dice on probabilities; accepts logits or probabilities.

    Args:
        pred_logits: [B,1,h,w] or [B,1,H,W] logits or probabilities.
        y:          [B,1,H,W] binary mask in {0,1} or float in [0,1].

    Returns:
        Mean Dice over batch as a scalar tensor.
    """
    if pred_logits.ndim == 3:
        pred_logits = pred_logits.unsqueeze(1)
    if y.ndim == 3:
        y = y.unsqueeze(1)

    # sanitize logits and convert to probabilities in float32
    if pred_logits.dtype.is_floating_point:
        pl = torch.nan_to_num(pred_logits.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp_(-20.0, 20.0)
        p = torch.sigmoid(pl)
    else:
        p = pred_logits.float().clamp(0.0, 1.0)

    # resize to GT spatial size if needed
    if p.shape[-2:] != y.shape[-2:]:
        p = F.interpolate(p, size=y.shape[-2:], mode="bilinear", align_corners=False)

    y = torch.nan_to_num(y.float(), nan=0.0, posinf=1.0, neginf=0.0)
    y = (y > 0.5).float()
    inter = (p * y).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    dice = torch.nan_to_num(dice, nan=0.0)
    return dice.mean()