import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two consecutive Conv2d -> BatchNorm -> ReLU blocks.
    This is the fundamental building block of U-Net.
    Original paper did not use BatchNorm but it is standard practice now
    and improves training stability significantly.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """
    One encoder step: DoubleConv followed by MaxPool2d.
    Returns both the pooled output (passed to next level)
    and the pre-pool feature map (stored for skip connection).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)   # stored for skip connection
        pooled = self.pool(skip)
        return pooled, skip


class Decoder(nn.Module):
    """
    One decoder step: TransposedConv upsampling, concatenate skip connection,
    then DoubleConv.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        # Concatenate skip connection along channel dimension
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for parotid gland segmentation.

    Input:  (B, 1, 512, 512)  — single channel CT slice
    Output: (B, 2, 512, 512)  — two channel logits (PAROTID_R, PAROTID_L)

    Architecture follows Ronneberger et al. 2015 with two modifications:
    1. Same padding (padding=1) to preserve spatial dimensions — required
       since input and output must both be 512x512
    2. BatchNorm added after each convolution — improves training stability
       and is standard practice in all modern U-Net implementations
    """
    def __init__(self, in_channels=1, out_channels=2, features=[64, 128, 256, 512]):
        super().__init__()

        # Encoder path
        self.encoders = nn.ModuleList()
        current_channels = in_channels
        for f in features:
            self.encoders.append(Encoder(current_channels, f))
            current_channels = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder path
        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.decoders.append(Decoder(f * 2, f))

        # Final 1x1 convolution to produce output channels
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder — collect skip connections
        skip_connections = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skip_connections.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder — apply skip connections in reverse order
        skip_connections = skip_connections[::-1]
        for decoder, skip in zip(self.decoders, skip_connections):
            x = decoder(x, skip)

        return self.final_conv(x)


if __name__ == '__main__':
    model = UNet(in_channels=1, out_channels=2)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")

    # Forward pass sanity check
    x = torch.randn(2, 1, 512, 512)
    out = model(x)
    print(f"\nInput shape  : {x.shape}")
    print(f"Output shape : {out.shape}")
    assert out.shape == (2, 2, 512, 512), "Output shape mismatch"
    print("Shape check passed")

    # Check output is raw logits (no sigmoid applied)
    print(f"Output range : {out.min().item():.3f} to {out.max().item():.3f}")
    print("Output is raw logits — sigmoid applied in loss function")