import torch
from torch import nn


class CSAM(nn.Module):
    """Cross-Scale Attention Module: self-attention across multi-scale features (e.g. P3, P4, P5)."""
    def __init__(self, in_channels_list: list, embed_dim: int, num_heads: int, dropout: float=0.0):
        super(CSAM, self).__init__()
        self.embed_dim = embed_dim
        # Projections for each scale into common embed_dim
        self.proj_in = nn.ModuleList([
            nn.Conv2d(c, embed_dim, kernel_size=1) if c != embed_dim else nn.Identity() 
            for c in in_channels_list
        ])
        # Projections back to original channels for each scale
        self.proj_out = nn.ModuleList([
            nn.Conv2d(embed_dim, c, kernel_size=1) if c != embed_dim else nn.Identity()
            for c in in_channels_list
        ])
        # Multi-head self-attention (batch_first for [B,L,C] inputs)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
    def forward(self, features: list) -> list:
        """
        features: list of Tensors [B, C_i, H_i, W_i] for each scale (e.g., [P3, P4, P5]).
        Returns: list of Tensors of the same shapes, after cross-scale attention.
        """
        B = features[0].size(0)
        token_list = []
        lengths = []
        # Project and flatten each scale
        for i, x in enumerate(features):
            x_proj = self.proj_in[i](x)            # [B, embed_dim, H_i, W_i]
            tokens = x_proj.flatten(2).permute(0, 2, 1)  # [B, L_i, embed_dim]
            token_list.append(tokens)
            lengths.append(tokens.shape[1])
        # Concatenate tokens from all scales
        combined = torch.cat(token_list, dim=1)         # [B, L_total, embed_dim]
        # Global self-attention
        attn_out, _ = self.attn(combined, combined, combined)
        # Split attended tokens and reshape back to feature maps
        outputs = []
        offset = 0
        for i, L in enumerate(lengths):
            out_tokens = attn_out[:, offset:offset+L, :]     # [B, L_i, embed_dim] for scale i
            offset += L
            H_i, W_i = features[i].shape[2], features[i].shape[3]
            out_map = out_tokens.permute(0, 2, 1).reshape(B, self.embed_dim, H_i, W_i)
            out_map = self.proj_out[i](out_map)              # [B, C_i, H_i, W_i]
            # Residual add original feature
            out_map = out_map + features[i]
            outputs.append(out_map)
        return outputs
