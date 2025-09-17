from torch import nn
from typing import Optional, Sequence

class FPMA(nn.Module):
    """
    Feature Propagation Module with Attention (FPMA).

    Enhances fine-scale features using coarse features via cross-attention:
      queries = fine, keys/values = coarse. The attention output is projected,
      normalized, activated, and residual-added to the fine feature.

    Optional: a lightweight local self-attention on the fine feature after fusion
    (kept off by default to keep changes minimal and cost bounded).

    Args:
        in_channels_coarse: channels of the coarse (lower-resolution) feature.
        in_channels_fine: channels of the fine (higher-resolution) feature.
        heads: number of attention heads.
        embed: attention dimension; defaults to in_channels_fine.
        dropout: attention dropout.
        use_self_attn: if True, apply an extra MHA on the fused fine feature.
    Notes:
        Reference: Kim et al., “Sequential Cross Attention Based Multi-task Learning,” ICIP 2022 (arXiv:2209.02518).
    """
    def __init__(
        self,
        in_ch: Sequence[int],
        heads: int,
        embed: Optional[int] = None,
        dropout: float = 0.0,
        use_self_attn: bool = False,
    ) -> None:
        super().__init__()
        assert len(in_ch) == 2, "FPMA expects two inputs: [fine, coarse]"
        in_channels_fine, in_channels_coarse = int(in_ch[0]), int(in_ch[1])

        embed_dim = int(embed or in_channels_fine)
        self.embed_dim = embed_dim
        self.in_channels_fine = in_channels_fine
        self.use_self_attn = use_self_attn

        # Projections to common attention dim
        self.q_proj = (
            nn.Conv2d(in_channels_fine, embed_dim, kernel_size=1, bias=False)
            if in_channels_fine != embed_dim else nn.Identity()
        )
        self.kv_proj = (
            nn.Conv2d(in_channels_coarse, embed_dim, kernel_size=1, bias=False)
            if in_channels_coarse != embed_dim else nn.Identity()
        )

        # Cross-attention: Q from fine, K/V from coarse
        self.attn = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)

        # Fuse back to fine channels
        self.conv = nn.Conv2d(embed_dim, in_channels_fine, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(in_channels_fine)
        self.act = nn.SiLU(inplace=True)

        # Optional self-attention refinement on the fused fine feature
        if use_self_attn:
            self.self_attn = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
            self.sa_qkv = (
                nn.Conv2d(in_channels_fine, embed_dim, kernel_size=1, bias=False)
                if in_channels_fine != embed_dim else nn.Identity()
            )
            self.sa_proj = nn.Sequential(
                nn.Conv2d(embed_dim, in_channels_fine, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels_fine),
                nn.SiLU(inplace=True),
            )

    def forward(self, x):        
        """
        Args:
            fine_feat:   [B, C_f, H_f, W_f] high-resolution feature map (queries).
            coarse_feat: [B, C_c, H_c, W_c] low-resolution feature map (keys/values),
                         expected to be pre-upsampled to (H_f, W_f) by the caller.
        Returns:
            [B, C_f, H_f, W_f] refined fine feature map.
        """
        assert isinstance(x, (list, tuple)) and len(x) == 2, "FPMA forward expects [fine, coarse]"
        fine_feat, coarse_feat = x[0], x[1]

        # Spatial align coarse to fine if needed
        if coarse_feat.shape[-2:] != fine_feat.shape[-2:]:
            coarse_feat = nn.functional.interpolate(coarse_feat, size=fine_feat.shape[-2:], mode="bilinear", align_corners=False)

        # Project
        q_map = self.q_proj(fine_feat)
        kv_map = self.kv_proj(coarse_feat)

        # Cross-attention
        B, E, H, W = q_map.shape
        q = q_map.flatten(2).permute(0, 2, 1)  # [B, HW, E]
        k = kv_map.flatten(2).permute(0, 2, 1)  # [B, HW, E]
        v = k
        attn_out, _ = self.attn(q, k, v)
        attn_map = attn_out.permute(0, 2, 1).reshape(B, E, H, W)

        # Fuse to fine channels + residual
        refined = self.act(self.norm(self.conv(attn_map)))
        out = fine_feat + refined

        if self.use_self_attn:
            sa_map = self.sa_qkv(out)
            q2 = k2 = v2 = sa_map.flatten(2).permute(0, 2, 1)
            sa_out, _ = self.self_attn(q2, k2, v2)
            sa_map = sa_out.permute(0, 2, 1).reshape(B, E, H, W)
            out = out + self.sa_proj(sa_map)

        return out
