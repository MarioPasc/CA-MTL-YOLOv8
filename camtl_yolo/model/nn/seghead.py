# seghead.py
from __future__ import annotations
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class _HalvingBlock(nn.Module):
    """Conv-BN-Act that halves channels. Keeps HxW."""
    def __init__(self, c_in: int, act: str = "silu") -> None:
        super().__init__()
        c_out = max(1, c_in // 2)
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True) if act == "silu" else nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)

class SegHeadMulti(nn.Module):
    """
    Deep-supervised segmentation head.
    Inputs: [P3_feat, P4_feat, P5_feat] with strides [8, 16, 32].
    Per-scale path: 3×(Conv-BN-Act halving) → 1×1 logits.
    Fusion: upsample logits to P3, compute per-pixel softmax gates over {P3,P4,P5},
            fuse as weighted sum, optional 3×3 refinement, then upsample to input size.
    Outputs:
      {
        'p3': logits at P3, 'p4': logits at P4, 'p5': logits at P5,
        'full': fused logits at input H×W
      }
    """
    is_seg_head: bool = True

    def __init__(self, in_channels: Tuple[int, int, int], out_channels: int = 1, fuse: bool = True) -> None:
        super().__init__()
        c3, c4, c5 = in_channels

        def make_path(c: int) -> nn.Sequential:
            b1 = _HalvingBlock(c)
            b2 = _HalvingBlock(max(1, c // 2))
            b3 = _HalvingBlock(max(1, c // 4))
            head = nn.Conv2d(max(1, c // 8), out_channels, kernel_size=1, bias=True)
            return nn.Sequential(b1, b2, b3, head)

        self.p3_path = make_path(c3)
        self.p4_path = make_path(c4)
        self.p5_path = make_path(c5)

        self.fuse = fuse
        if fuse:
            # Produce 3 gating maps, one per scale, then softmax across channel dim
            self.fuse_attn = nn.Conv2d(3 * out_channels, 3, kernel_size=1, bias=True)
            self.fuse_refine = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, feats: List[torch.Tensor], input_hw: Tuple[int, int] | None = None) -> Dict[str, torch.Tensor]:
        assert len(feats) == 3, "SegHeadMulti expects [P3, P4, P5]"
        p3, p4, p5 = feats

        l3 = self.p3_path(p3)  # [B,1,h3,w3]
        l4 = self.p4_path(p4)  # [B,1,h4,w4]
        l5 = self.p5_path(p5)  # [B,1,h5,w5]

        out: Dict[str, torch.Tensor] = {"p3": l3, "p4": l4, "p5": l5}

        if self.fuse:
            h3, w3 = l3.shape[-2:]
            l4u = F.interpolate(l4, size=(h3, w3), mode="bilinear", align_corners=False)
            l5u = F.interpolate(l5, size=(h3, w3), mode="bilinear", align_corners=False)

            cat = torch.cat([l3, l4u, l5u], dim=1)                    # [B,3,h3,w3]
            gates = torch.softmax(self.fuse_attn(cat), dim=1)         # [B,3,h3,w3], sum=1 per-pixel

            fused_p3 = gates[:, 0:1] * l3 + gates[:, 1:2] * l4u + gates[:, 2:3] * l5u
            fused_p3 = self.fuse_refine(fused_p3)

            if input_hw is not None:
                H, W = input_hw
                full = F.interpolate(fused_p3, size=(H, W), mode="bilinear", align_corners=False)
            else:
                full = F.interpolate(fused_p3, scale_factor=8.0, mode="bilinear", align_corners=False)
            out["full"] = full

        return out

