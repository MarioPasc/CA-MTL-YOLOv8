import torch
from torch import nn
from typing import Optional, Sequence


class CTAM(nn.Module):
    """
    Cross-Task Attention Module (CTAM).
    Refines a target task's feature (e.g., detection) using one or many source task features
    (e.g., segmentation) as K/V. For M>2, it computes CAM(target, source_j) for each j,
    concatenates the attention outputs along channels, fuses via 1x1 conv, and residual-adds
    to the target. This matches the paper's multi-source recipe.

    Args:
        in_channels_tgt: channels of target feature (queries).
        in_channels_src: int or list[int] for source feature(s) (keys/values).
        num_heads: multi-head attention heads.
        embed_dim: common attention dimension after 1x1 projections. Defaults to in_channels_tgt.
        dropout: attention dropout.

    Notes:
        Reference: Kim et al., “Sequential Cross Attention Based Multi-task Learning,” ICIP 2022 (arXiv:2209.02518).
    """
    def __init__(
        self,
        in_ch: Sequence[int],
        num_heads: int,
        embed_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert len(in_ch) >= 2, "CTAM expects at least [target, source]"
        in_channels_tgt = int(in_ch[0])
        src_list = [int(c) for c in in_ch[1:]]

        self.embed_dim = int(embed_dim or in_channels_tgt)
        self.num_sources = len(src_list)
        self.in_channels_tgt = in_channels_tgt

        self.q_proj = (
            nn.Conv2d(in_channels_tgt, self.embed_dim, kernel_size=1, bias=False)
            if in_channels_tgt != self.embed_dim else nn.Identity()
        )
        self.kv_proj = nn.ModuleList([
            nn.Conv2d(c, self.embed_dim, kernel_size=1, bias=False) if c != self.embed_dim else nn.Identity()
            for c in src_list
        ])

        self.attn = nn.MultiheadAttention(self.embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Conv2d(self.num_sources * self.embed_dim, in_channels_tgt, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels_tgt),
            nn.SiLU(inplace=True),
        )

    def _cache_spatial_key_attention(self, attn_weights: torch.Tensor, key_hw: tuple[int, int]) -> None:
        """
        attn_weights: [B, heads, Q, K] attention over K=Hk*Wk key positions.
        Produces a single-channel spatial map over key positions: mean over heads and queries.
        Stores to self.last_attn_map as [B,1,Hk,Wk] on the same device.
        """
        try:
            B, Hh, Q, K = attn_weights.shape
            Hk, Wk = int(key_hw[0]), int(key_hw[1])
            m = attn_weights.mean(dim=(1, 2))  # [B, K]
            m = m.reshape(B, 1, Hk, Wk)        # [B,1,Hk,Wk]
            # keep as raw, non-normalized saliency; KL routine will normalize
            self.last_attn_map = m
        except Exception:
            # on any mismatch, clear the cache to avoid stale tensors
            self.last_attn_map = None

    def _attend(self, q_map: torch.Tensor, kv_map: torch.Tensor) -> torch.Tensor:
        """Run MHA(q, k=kv_map, v=kv_map) and return map [B, E, H, W]."""
        B, E, H, W = q_map.shape
        q = q_map.flatten(2).permute(0, 2, 1)  # [B, HW, E]
        k = kv_map.flatten(2).permute(0, 2, 1)  # [B, HW, E]
        v = k
        # Retrieve per-head attention weights to cache a spatial saliency map over key positions
        Hk, Wk = kv_map.shape[-2], kv_map.shape[-1]
        out, attn = self.attn(q, k, v, need_weights=True, average_attn_weights=False)  # [B, HW, E], [B, heads, Q, K]
        # Cache mean attention over heads/queries reshaped to key spatial size
        self._cache_spatial_key_attention(attn, key_hw=(Hk, Wk))
        
        # Return attended features in map form
        return out.permute(0, 2, 1).reshape(B, E, H, W)

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            tgt_feat: [B, C_tgt, H, W] target feature (queries).
            src_feats: either a single tensor [B, C_src, H, W]
                       or a list of tensors with same spatial size and possibly different channels.
        Returns:
            Refined target feature [B, C_tgt, H, W].
        """
        assert isinstance(x, (list, tuple)) and len(x) >= 2, "CTAM forward expects [tgt, src...]"
        tgt = x[0]
        srcs = list(x[1:])

        # Ensure spatial alignment
        H, W = tgt.shape[-2:]
        aligned = []
        for s in srcs:
            if s.shape[-2:] != (H, W):
                s = torch.nn.functional.interpolate(s, size=(H, W), mode="bilinear", align_corners=False)
            aligned.append(s)

        q_map = self.q_proj(tgt)
        attn_maps = []
        for s, proj in zip(aligned, self.kv_proj):
            kv_map = proj(s)
            attn_maps.append(self._attend(q_map, kv_map))

        fused = torch.cat(attn_maps, dim=1) if len(attn_maps) > 1 else attn_maps[0]
        return tgt + self.fuse(fused)
