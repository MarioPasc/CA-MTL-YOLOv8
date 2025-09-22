import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Sequence, Optional, Tuple


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

    def _cache_spatial_maps(self, attn_list: Sequence[Optional[torch.Tensor]], key_hw_list: Sequence[Tuple[int,int]], scale_tags: Sequence[str]) -> None:
        """
        attn_list[i]: [B, heads, Q, K_i] across scales; produce per-scale [B,1,Hi,Wi].
        Stores dict self.last_attn_maps[scale_tag] = map.
        """
        self.last_attn_maps = {}
        for attn, (Hi, Wi), tag in zip(attn_list, key_hw_list, scale_tags):
            if attn is None:
                continue
            try:
                B, Hh, Q, K = attn.shape
                m = attn.mean(dim=(1, 2)).reshape(B, 1, Hi, Wi)
                self.last_attn_maps[str(tag)] = m
            except Exception:
                continue

    def _mha(self, q_map: torch.Tensor, kv_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Helper: run MHA with q from target and k/v from source maps.
        Returns:
            - out_map: [B, embed, H, W]
            - attn: [B, heads, Q, K] per-head attention weights
        """
        B, _, H, W = q_map.shape
        q = q_map.flatten(2).permute(0, 2, 1)  # [B, HW, E]
        k = kv_map.flatten(2).permute(0, 2, 1)  # [B, HW, E]
        v = k
        out, attn = self.attn(q, k, v, need_weights=True, average_attn_weights=False)  # [B, HW, E], [B, heads, Q, K]
        return out.permute(0, 2, 1).reshape(B, self.embed_dim, H, W), attn

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
        # Track per-target-scale attention weights averaged over contributing coarser sources
        collected_attn: List[Optional[torch.Tensor]] = [None] * L
        scale_sizes: List[Tuple[int, int]] = [(int(f.shape[-2]), int(f.shape[-1])) for f in features]

        for k in range(L - 2, -1, -1):  # from next-to-coarsest down to finest
            B, _, Hk, Wk = outputs[k].shape
            q_map = self.q_proj[k](outputs[k])

            attn_maps = []
            attn_weights_for_k = []
            for j in range(k + 1, L):
                src = outputs[j]
                if src.shape[-2:] != (Hk, Wk):
                    src = F.interpolate(src, size=(Hk, Wk), mode=self.upsample_mode,
                                        align_corners=False if self.upsample_mode in ("bilinear", "bicubic") else None)
                kv_map = self.kv_proj[j](src)
                out_map, attn_w = self._mha(q_map, kv_map)
                attn_maps.append(out_map)
                attn_weights_for_k.append(attn_w)

            if not attn_maps:
                attn_maps = [q_map]
            else:
                # Average attention weights over all contributing coarser sources for this target scale
                if len(attn_weights_for_k) == 1:
                    collected_attn[k] = attn_weights_for_k[0]
                else:
                    collected_attn[k] = torch.stack(attn_weights_for_k, dim=0).mean(dim=0)

            fused = torch.cat(attn_maps, dim=1) if len(attn_maps) > 1 else attn_maps[0]
            outputs[k] = outputs[k] + self.fuse[k](fused)

        # Cache spatial attention maps per scale (tags: p3, p4, p5 ... in order)
        scale_tags = [f"p{i+3}" for i in range(L)]
        self._cache_spatial_maps(collected_attn, scale_sizes, scale_tags)

        return outputs
