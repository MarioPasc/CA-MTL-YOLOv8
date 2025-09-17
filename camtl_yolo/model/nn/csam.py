import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Sequence


class CSAM(nn.Module):
    """
    Cross-Scale Attention Module (CSAM), paper-faithful minimal version.

    Implements sequential coarse→fine aggregation within a task:
      For target scale k, aggregate attention outputs from all coarser scales {j > k}
      that are upsampled to the spatial size of k, concatenate channel-wise, fuse via 1x1 conv,
      and residual-add to the target feature.

    Expected order: features = [P3, P4, P5] with P3 the finest (largest HxW) and P5 the coarsest.

    Args:
        in_channels_list: channels per scale, in the order [P3, P4, P5].
        embed_dim: common attention dimension after 1x1 projections.
        num_heads: number of attention heads.
        dropout: dropout in multi-head attention.
        upsample_mode: interpolation mode for coarse→fine upsampling.
    Notes:
        Reference: Kim et al., “Sequential Cross Attention Based Multi-task Learning,” ICIP 2022 (arXiv:2209.02518).
    """
    def __init__(
        self,
        in_ch_list: Sequence[int],
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        upsample_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self.in_channels_list = [int(c) for c in in_ch_list]
        self.embed_dim = int(embed_dim)
        self.upsample_mode = upsample_mode
        L = len(self.in_channels_list)

        self.q_proj = nn.ModuleList([
            nn.Conv2d(c, self.embed_dim, kernel_size=1, bias=False) if c != self.embed_dim else nn.Identity()
            for c in self.in_channels_list
        ])
        self.kv_proj = nn.ModuleList([
            nn.Conv2d(c, self.embed_dim, kernel_size=1, bias=False) if c != self.embed_dim else nn.Identity()
            for c in self.in_channels_list
        ])

        self.attn = nn.MultiheadAttention(self.embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(max(1, (L - 1 - k)) * self.embed_dim, self.in_channels_list[k], kernel_size=1, bias=False),
                nn.BatchNorm2d(self.in_channels_list[k]),
                nn.SiLU(inplace=True),
            )
            for k in range(L)
        ])

    def _mha(self, q_map: torch.Tensor, kv_map: torch.Tensor) -> torch.Tensor:
        """Helper: run MHA with q from target and k/v from source maps, return [B, embed, H, W]."""
        B, _, H, W = q_map.shape
        q = q_map.flatten(2).permute(0, 2, 1)  # [B, HW, E]
        k = kv_map.flatten(2).permute(0, 2, 1)  # [B, HW, E]
        v = k
        out, _ = self.attn(q, k, v)            # [B, HW, E]
        return out.permute(0, 2, 1).reshape(B, self.embed_dim, H, W)

    def forward(self, x) -> List[torch.Tensor]:
        """
        Args:
            features: [P3, P4, P5] where P3 has highest resolution and P5 the lowest.
        Returns:
            List of refined feature maps in the same order.
        """
        assert isinstance(x, (list, tuple)) and len(x) >= 1, "CSAM expects a list of scale features"
        features: List[torch.Tensor] = list(x)
        L = len(features)
        outputs = list(features)

        for k in range(L - 2, -1, -1):  # from next-to-coarsest down to finest
            B, _, Hk, Wk = outputs[k].shape
            q_map = self.q_proj[k](outputs[k])

            attn_maps = []
            for j in range(k + 1, L):
                src = outputs[j]
                if src.shape[-2:] != (Hk, Wk):
                    src = F.interpolate(src, size=(Hk, Wk), mode=self.upsample_mode,
                                        align_corners=False if self.upsample_mode in ("bilinear", "bicubic") else None)
                kv_map = self.kv_proj[j](src)
                attn_maps.append(self._mha(q_map, kv_map))

            if not attn_maps:
                attn_maps = [q_map]

            fused = torch.cat(attn_maps, dim=1) if len(attn_maps) > 1 else attn_maps[0]
            outputs[k] = outputs[k] + self.fuse[k](fused)

        return outputs
