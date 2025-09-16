import torch
from torch import nn
from typing import Optional

class FPMA(nn.Module):
    """Feature Propagation Module with Attention: enhances coarse-to-fine feature upsampling."""
    def __init__(self, in_channels_coarse: int, in_channels_fine: int, num_heads: int, 
                 embed_dim: Optional[int] = None, dropout: float=0.0):
        super(FPMA, self).__init__()
        embed_dim = embed_dim or in_channels_fine
        # Multi-head cross-attention: queries=fine, keys/values=coarse
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True,
                                         kdim=in_channels_coarse, vdim=in_channels_coarse)
        self.conv = nn.Conv2d(embed_dim, in_channels_fine, kernel_size=1)
        self.norm = nn.GroupNorm(1, num_channels=in_channels_fine)
        self.act = nn.ReLU(inplace=True)
    def forward(self, fine_feat: torch.Tensor, coarse_feat: torch.Tensor) -> torch.Tensor:
        """
        fine_feat: [B, C_f, H_f, W_f] (higher-resolution feature map)
        coarse_feat: [B, C_c, H_c, W_c] (lower-resolution feature map)
        Returns: [B, C_f, H_f, W_f] refined fine feature map.
        """
        B, C_f, H_f, W_f = fine_feat.shape
        # Flatten spatial dims
        query = fine_feat.flatten(2).permute(0, 2, 1)      # [B, H_f*W_f, C_f]
        key   = coarse_feat.flatten(2).permute(0, 2, 1)     # [B, H_c*W_c, C_c]
        value = key
        attn_out, _ = self.attn(query, key, value)         # [B, H_f*W_f, embed_dim]
        attn_map = attn_out.permute(0, 2, 1).reshape(B, -1, H_f, W_f)  # [B, embed_dim, H_f, W_f]
        attn_map = self.norm(self.conv(attn_map))
        attn_map = self.act(attn_map)
        return fine_feat + attn_map
