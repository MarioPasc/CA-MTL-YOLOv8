import torch
from torch import nn
from typing import Optional

class CTAM(nn.Module):
    """
    Cross-Task Attention Module (seg -> det):
    Refines detection features using segmentation features as K/V.
    """
    def __init__(self, in_channels_seg: int, in_channels_det: int,
                 num_heads: int, embed_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        embed_dim = embed_dim or in_channels_det  # output dim matches det channels
        # Q from detection, K/V from segmentation
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True,
                                          kdim=in_channels_seg, vdim=in_channels_seg)
        self.proj = nn.Conv2d(embed_dim, in_channels_det, kernel_size=1)
        self.norm = nn.GroupNorm(1, in_channels_det)
        self.act = nn.ReLU(inplace=True)

    def forward(self, det_feat: torch.Tensor, seg_feat: torch.Tensor) -> torch.Tensor:
        """
        det_feat: [B, C_det, H, W]  (queries)
        seg_feat: [B, C_seg, H, W]  (keys/values)
        returns:  [B, C_det, H, W]  (refined detection features)
        """
        B, C_d, H, W = det_feat.shape
        q = det_feat.flatten(2).permute(0, 2, 1)  # [B, HW, C_det]
        k = seg_feat.flatten(2).permute(0, 2, 1)  # [B, HW, C_seg]
        v = k
        out, _ = self.attn(q, k, v)               # [B, HW, embed_dim]
        out = out.permute(0, 2, 1).reshape(B, -1, H, W)
        out = self.act(self.norm(self.proj(out)))
        return det_feat + out
