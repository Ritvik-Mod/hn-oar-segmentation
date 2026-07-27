"""
Swin-UNet: Unet-like Pure Transformer for Medical Image Segmentation
Faithful implementation based on Cao et al. 2021 (arXiv:2105.05537)

Architecture: Pure Transformer U-shaped encoder-decoder. NO convolutions in backbone.
- Patch Partition + Linear Embedding
- Encoder: Swin Transformer blocks + Patch Merging (downsampling)
- Bottleneck: Swin Transformer blocks
- Decoder: Swin Transformer blocks + Patch Expanding (upsampling)
- Skip connections between encoder and decoder stages

Key design from paper:
- Window-based Multi-Head Self-Attention (W-MSA) + Shifted Window (SW-MSA)
- Patch Merging for downsampling (concatenate 2x2 neighbors, linear project)
- Patch Expanding for upsampling (linear project to 2x channels, rearrange)
- Relative position bias in attention
- No CNN components (pure transformer)

Adapted for:
- 512x512 single-channel CT input (paper uses 224x224)
- 2-channel output (left/right parotid, independent sigmoid)
- Training from scratch (no ImageNet pretraining)
- Final upsample via bilinear interpolation (memory efficient)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Patch Embedding and Patch Operations
# ===========================================================================

class PatchEmbedding(nn.Module):
    """
    Partition image into non-overlapping 4x4 patches and project to embedding dim.
    512x512 -> 128x128 patches, each with dim C.
    """
    def __init__(self, in_channels=1, embed_dim=96, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim,
                                    kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.projection(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    """
    Downsample by 2x: group 2x2 neighboring patches, concatenate, linear project.
    (B, H*W, C) -> (B, H/2*W/2, 2C)
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, H // 2, W // 2


class PatchExpanding(nn.Module):
    """
    Upsample by 2x: linear project to 2*C, then rearrange spatially.
    (B, H*W, C) -> (B, 2H*2W, C/2)
    """
    def __init__(self, dim):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = self.expand(x)
        x = x.view(B, H, W, 2 * C)
        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, 2 * H, 2 * W, C // 2)
        x = x.view(B, -1, C // 2)
        x = self.norm(x)
        return x, 2 * H, 2 * W


# ===========================================================================
# Window Attention
# ===========================================================================

def window_partition(x, window_size):
    """Partition into non-overlapping windows. (B, H, W, C) -> (num_windows*B, ws, ws, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partition. (num_windows*B, ws, ws, C) -> (B, H, W, C)"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based Multi-Head Self-Attention with relative position bias."""
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


# ===========================================================================
# Swin Transformer Block
# ===========================================================================

class SwinTransformerBlock(nn.Module):
    """
    Single Swin Transformer block with W-MSA or SW-MSA.
    LN -> W-MSA/SW-MSA -> residual -> LN -> MLP -> residual
    """
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4.0, input_resolution=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.input_resolution = input_resolution

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

        if self.shift_size > 0 and input_resolution is not None:
            H, W = input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -window_size),
                        slice(-window_size, -shift_size),
                        slice(-shift_size, None))
            w_slices = (slice(0, -window_size),
                        slice(-window_size, -shift_size),
                        slice(-shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, window_size)
            mask_windows = mask_windows.view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class SwinTransformerStage(nn.Module):
    """Two consecutive Swin Transformer blocks (W-MSA then SW-MSA)."""
    def __init__(self, dim, num_heads, window_size=7, input_resolution=None):
        super().__init__()
        self.block1 = SwinTransformerBlock(
            dim=dim, num_heads=num_heads, window_size=window_size,
            shift_size=0, input_resolution=input_resolution
        )
        self.block2 = SwinTransformerBlock(
            dim=dim, num_heads=num_heads, window_size=window_size,
            shift_size=window_size // 2, input_resolution=input_resolution
        )

    def forward(self, x, H, W):
        x = self.block1(x, H, W)
        x = self.block2(x, H, W)
        return x


# ===========================================================================
# Swin-UNet
# ===========================================================================

class SwinUNet(nn.Module):
    """
    Swin-UNet (Cao et al. 2021) for 512x512 single-channel CT segmentation.

    Pure Transformer U-shaped architecture. No CNN components.

    Encoder:
        Stage 0: Patch Embed -> (B, 128*128, 96)   -> Swin blocks -> Patch Merge
        Stage 1: (B, 64*64, 192)                    -> Swin blocks -> Patch Merge
        Stage 2: (B, 32*32, 384)                    -> Swin blocks -> Patch Merge

    Bottleneck:
        (B, 16*16, 768)                             -> Swin blocks

    Decoder:
        Stage 2: Patch Expand -> concat skip -> (B, 32*32, 384)  -> Swin blocks
        Stage 1: Patch Expand -> concat skip -> (B, 64*64, 192)  -> Swin blocks
        Stage 0: Patch Expand -> concat skip -> (B, 128*128, 96) -> Swin blocks

    Head: Linear -> reshape -> bilinear upsample 4x -> (B, 2, 512, 512)
    """
    def __init__(self, in_channels=1, out_channels=2, embed_dim=96,
                 window_size=8, num_heads=(3, 6, 12, 24)):
        super().__init__()
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size=4)

        # Encoder
        self.encoder0 = SwinTransformerStage(
            dim=embed_dim, num_heads=num_heads[0],
            window_size=window_size, input_resolution=(128, 128))
        self.merge0 = PatchMerging(embed_dim)

        self.encoder1 = SwinTransformerStage(
            dim=embed_dim * 2, num_heads=num_heads[1],
            window_size=window_size, input_resolution=(64, 64))
        self.merge1 = PatchMerging(embed_dim * 2)

        self.encoder2 = SwinTransformerStage(
            dim=embed_dim * 4, num_heads=num_heads[2],
            window_size=window_size, input_resolution=(32, 32))
        self.merge2 = PatchMerging(embed_dim * 4)

        # Bottleneck
        self.bottleneck = SwinTransformerStage(
            dim=embed_dim * 8, num_heads=num_heads[3],
            window_size=window_size, input_resolution=(16, 16))

        # Decoder
        self.expand2 = PatchExpanding(embed_dim * 8)
        self.concat_proj2 = nn.Linear(embed_dim * 8, embed_dim * 4)
        self.decoder2 = SwinTransformerStage(
            dim=embed_dim * 4, num_heads=num_heads[2],
            window_size=window_size, input_resolution=(32, 32))

        self.expand1 = PatchExpanding(embed_dim * 4)
        self.concat_proj1 = nn.Linear(embed_dim * 4, embed_dim * 2)
        self.decoder1 = SwinTransformerStage(
            dim=embed_dim * 2, num_heads=num_heads[1],
            window_size=window_size, input_resolution=(64, 64))

        self.expand0 = PatchExpanding(embed_dim * 2)
        self.concat_proj0 = nn.Linear(embed_dim * 2, embed_dim)
        self.decoder0 = SwinTransformerStage(
            dim=embed_dim, num_heads=num_heads[0],
            window_size=window_size, input_resolution=(128, 128))

        # Segmentation head
        self.seg_head = nn.Linear(embed_dim, out_channels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]

        # Patch Embedding
        x, H, W = self.patch_embed(x)

        # Encoder
        x = self.encoder0(x, H, W)
        skip0 = x
        x, H, W = self.merge0(x, H, W)

        x = self.encoder1(x, H, W)
        skip1 = x
        x, H, W = self.merge1(x, H, W)

        x = self.encoder2(x, H, W)
        skip2 = x
        x, H, W = self.merge2(x, H, W)

        # Bottleneck
        x = self.bottleneck(x, H, W)

        # Decoder
        x, H, W = self.expand2(x, H, W)
        x = torch.cat([x, skip2], dim=-1)
        x = self.concat_proj2(x)
        x = self.decoder2(x, H, W)

        x, H, W = self.expand1(x, H, W)
        x = torch.cat([x, skip1], dim=-1)
        x = self.concat_proj1(x)
        x = self.decoder1(x, H, W)

        x, H, W = self.expand0(x, H, W)
        x = torch.cat([x, skip0], dim=-1)
        x = self.concat_proj0(x)
        x = self.decoder0(x, H, W)

        # Segmentation head
        x = self.seg_head(x)                              # (B, 16384, 2)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)      # (B, 2, 128, 128)
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)

        return x


if __name__ == '__main__':
    print("Initializing Swin-UNet (Cao et al. 2021)...")
    model = SwinUNet(in_channels=1, out_channels=2)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params/1e6:.2f}M")

    dummy = torch.randn(1, 1, 512, 512)
    with torch.no_grad():
        out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (1, 2, 512, 512), "Shape mismatch!"
    print("Architecture verified. Ready for training.")
