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
    For each feat: 3 halving blocks → 1x1 logits at its own HxW.
    Fusion: upsample logits to P3, concat, 1x1 fuse, upsample to input size.
    Outputs:
      {
        'p3': logits at P3, 'p4': logits at P4, 'p5': logits at P5,
        'full': fused logits at input HxW
      }
    """
    is_seg_head: bool = True  # used by model to detect seg head

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
            # concat 3 logits at P3 → 1 channel
            self.fuse_conv = nn.Conv2d(3 * out_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, feats: List[torch.Tensor], input_hw: Tuple[int, int] | None = None) -> Dict[str, torch.Tensor]:
        assert len(feats) == 3, "SegHeadMulti expects [P3, P4, P5]"
        p3, p4, p5 = feats  # shapes: [B,C, H/8, W/8], [B,C,H/16,W/16], [B,C,H/32,W/32]

        l3 = self.p3_path(p3)  # [B,1,h3,w3]
        l4 = self.p4_path(p4)  # [B,1,h4,w4]
        l5 = self.p5_path(p5)  # [B,1,h5,w5]

        out: Dict[str, torch.Tensor] = {"p3": l3, "p4": l4, "p5": l5}

        if self.fuse:
            # upsample l4,l5 to P3 spatial size then fuse
            h3, w3 = l3.shape[-2:]
            l4u = F.interpolate(l4, size=(h3, w3), mode="bilinear", align_corners=False)
            l5u = F.interpolate(l5, size=(h3, w3), mode="bilinear", align_corners=False)
            fused_p3 = self.fuse_conv(torch.cat([l3, l4u, l5u], dim=1))  # [B,1,h3,w3]
            if input_hw is not None:
                H, W = input_hw
                full = F.interpolate(fused_p3, size=(H, W), mode="bilinear", align_corners=False)
            else:
                # assume 8× stride to input if not given
                full = F.interpolate(fused_p3, scale_factor=8.0, mode="bilinear", align_corners=False)
            out["full"] = full

        return out
