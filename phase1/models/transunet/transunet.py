"""
TransUNet — Faithful to Chen et al. 2021
"TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation"

Architecture: R50 (first 3 stages) -> Patch Embedding (1x1) -> 12 Transformer Layers -> CUP Decoder with skip connections

Adapted for:
- 512x512 single-channel CT input (no ImageNet pretraining)
- 2-channel output (left/right parotid, independent sigmoid)

Key design from paper:
- ResNet-50 first 3 stages as CNN encoder (produces 1/4, 1/8, 1/16 features)
- 1x1 patch embedding from CNN feature map (NOT raw pixels)
- 12 Transformer encoder layers, 12 heads, hidden_size=768, MLP=3072
- Cascading Upsampler (CUP) decoder: upsample -> 3x3 conv -> ReLU, with skip connections
- Learnable position embeddings
"""

import torch
import torch.nn as nn


# ==============================================================================
# ResNet-50 Encoder (First 3 Stages)
# ==============================================================================

class Bottleneck(nn.Module):
    """Standard ResNet bottleneck block with expansion=4."""
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super().__init__()
        out_channels = mid_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet50Encoder(nn.Module):
    """
    ResNet-50 first 3 stages.
    Input: (B, 1, 512, 512)

    Produces:
        skip1: (B, 256, 128, 128)   -- after stage 1 (1/4 resolution)
        skip2: (B, 512, 64, 64)     -- after stage 2 (1/8 resolution)
        out:   (B, 1024, 32, 32)    -- after stage 3 (1/16 resolution)
    """

    def __init__(self, in_channels=1):
        super().__init__()
        # Stem: conv7x7 -> BN -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 1: 3 bottleneck blocks, 64->256, stride=1 (resolution stays 128x128)
        self.layer1 = self._make_stage(64, 64, num_blocks=3, stride=1)
        # Stage 2: 4 bottleneck blocks, 256->512, stride=2 (128->64)
        self.layer2 = self._make_stage(256, 128, num_blocks=4, stride=2)
        # Stage 3: 6 bottleneck blocks, 512->1024, stride=2 (64->32)
        self.layer3 = self._make_stage(512, 256, num_blocks=6, stride=2)

    def _make_stage(self, in_channels, mid_channels, num_blocks, stride):
        out_channels = mid_channels * Bottleneck.expansion
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = [Bottleneck(in_channels, mid_channels, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(out_channels, mid_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem: (B,1,512,512) -> (B,64,128,128)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # Stage 1: (B,64,128,128) -> (B,256,128,128)
        skip1 = self.layer1(x)
        # Stage 2: (B,256,128,128) -> (B,512,64,64)
        skip2 = self.layer2(skip1)
        # Stage 3: (B,512,64,64) -> (B,1024,32,32)
        out = self.layer3(skip2)

        return out, skip1, skip2


# ==============================================================================
# Transformer Encoder
# ==============================================================================

class PatchEmbedding(nn.Module):
    """
    1x1 patch embedding from CNN feature map.
    Projects CNN channels to transformer hidden dimension.
    (B, 1024, 32, 32) -> (B, 1024, 768) with learnable position embeddings.
    """

    def __init__(self, in_channels=1024, hidden_size=768, spatial_size=32):
        super().__init__()
        self.seq_len = spatial_size * spatial_size  # 1024
        self.projection = nn.Conv2d(in_channels, hidden_size, kernel_size=1)
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.seq_len, hidden_size) * 0.02
        )

    def forward(self, x):
        # x: (B, 1024, 32, 32)
        x = self.projection(x)            # (B, 768, 32, 32)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, 1024, 768)
        x = x + self.position_embeddings  # Add positional encoding
        return x, H, W


class TransformerBlock(nn.Module):
    """
    Standard ViT Transformer Encoder.
    12 layers, 12 heads, hidden_size=768, MLP dim=3072.
    """

    def __init__(self, hidden_size=768, num_heads=12, num_layers=12,
                 mlp_dim=3072, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm (matches ViT)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.norm(x)
        return x


# ==============================================================================
# Cascading Upsampler (CUP) Decoder
# ==============================================================================

class DecoderCUPBlock(nn.Module):
    """
    Single CUP decoder block: Upsample 2x -> Concat skip -> 3x3 Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ==============================================================================
# TransUNet
# ==============================================================================

class TransUNet(nn.Module):
    """
    TransUNet (Chen et al. 2021) adapted for 512x512 CT segmentation.

    Encoder:
        ResNet-50 stages 1-3: (B,1,512,512) -> (B,1024,32,32)
        Skip connections from stage 1 (256ch, 128x128) and stage 2 (512ch, 64x64)

    Bottleneck:
        1x1 Patch Embedding: (B,1024,32,32) -> (B,1024,768)
        12-layer Transformer Encoder
        Reshape back: (B,1024,768) -> (B,768,32,32)

    Decoder (CUP):
        Level 1: (512, 32,32) -> upsample -> concat skip2(512) -> (256, 64,64)
        Level 2: (256, 64,64) -> upsample -> concat skip1(256) -> (128, 128,128)
        Level 3: (128, 128,128) -> upsample -> (64, 256,256)
        Level 4: (64, 256,256) -> upsample -> (32, 512,512)

    Head: 1x1 conv -> 2 channels
    """

    def __init__(self, in_channels=1, out_channels=2,
                 hidden_size=768, num_heads=12, num_layers=12, mlp_dim=3072):
        super().__init__()

        # CNN Encoder
        self.resnet = ResNet50Encoder(in_channels=in_channels)

        # Transformer Bottleneck
        self.patch_embedding = PatchEmbedding(
            in_channels=1024, hidden_size=hidden_size, spatial_size=32
        )
        self.transformer = TransformerBlock(
            hidden_size=hidden_size, num_heads=num_heads,
            num_layers=num_layers, mlp_dim=mlp_dim
        )

        # Project transformer output back to spatial
        self.conv_after_transformer = nn.Sequential(
            nn.Conv2d(hidden_size, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # CUP Decoder
        # Level 1: 512 + skip2(512) -> 256, resolution 32->64
        self.decoder1 = DecoderCUPBlock(512, 512, 256)
        # Level 2: 256 + skip1(256) -> 128, resolution 64->128
        self.decoder2 = DecoderCUPBlock(256, 256, 128)
        # Level 3: 128 + no skip -> 64, resolution 128->256
        self.decoder3 = DecoderCUPBlock(128, 0, 64)
        # Level 4: 64 + no skip -> 32, resolution 256->512
        self.decoder4 = DecoderCUPBlock(64, 0, 32)

        # Segmentation Head
        self.seg_head = nn.Conv2d(32, out_channels, kernel_size=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # -- CNN Encoder --
        # x: (B, 1, 512, 512)
        cnn_out, skip1, skip2 = self.resnet(x)
        # cnn_out: (B, 1024, 32, 32)
        # skip1:   (B, 256, 128, 128)
        # skip2:   (B, 512, 64, 64)

        # -- Transformer Bottleneck --
        tokens, H, W = self.patch_embedding(cnn_out)  # (B, 1024, 768)
        tokens = self.transformer(tokens)              # (B, 1024, 768)

        # Reshape back to spatial
        B = tokens.shape[0]
        x = tokens.transpose(1, 2).view(B, -1, H, W)  # (B, 768, 32, 32)
        x = self.conv_after_transformer(x)              # (B, 512, 32, 32)

        # -- CUP Decoder --
        x = self.decoder1(x, skip2)   # (B, 256, 64, 64)
        x = self.decoder2(x, skip1)   # (B, 128, 128, 128)
        x = self.decoder3(x)          # (B, 64, 256, 256)
        x = self.decoder4(x)          # (B, 32, 512, 512)

        # -- Segmentation Head --
        out = self.seg_head(x)         # (B, 2, 512, 512)
        return out


if __name__ == '__main__':
    print("Initializing TransUNet (Chen et al. 2021)...")
    model = TransUNet(in_channels=1, out_channels=2)

    # Parameter count
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.2f}M")

    # Shape check
    dummy = torch.randn(1, 1, 512, 512)
    with torch.no_grad():
        out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (1, 2, 512, 512), "Shape mismatch!"
    print("Architecture verified. Ready for training.")
