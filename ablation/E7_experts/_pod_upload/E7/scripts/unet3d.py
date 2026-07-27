"""
E4 — a plain, hand-built 3D U-Net (Q4: architecture vs pipeline).

Deliberately a *different implementation* from nnU-Net's PlainConvUNet, trained on
the SAME nnU-Net-style preprocessed data (target spacing [3.0,0.977,0.977],
CTNormalization, patch [48,224,192]) so the ONLY thing that differs from the
nnU-Net 3d_fullres single-fold baseline (0.8187) is the code, not the pipeline.

Design (standard 3D U-Net, extends the project's 2D unet.py to 3D):
  - DoubleConv3d = (Conv3d 3x3x3 -> InstanceNorm3d -> LeakyReLU) x2
  - 5 encoder levels, features [32,64,128,256,320]; MaxPool3d(2) between them
  - ConvTranspose3d(2) decoder with skip concatenation
  - final 1x1x1 conv -> 2 channels (PAROTID_R, PAROTID_L), raw logits
    (independent sigmoid in the loss, matching the project's 2-channel convention)

Patch [48,224,192] is divisible by 16, so 4 isotropic poolings are exact
(48->3, 224->14, 192->12). (nnU-Net uses anisotropic strides; using plain
isotropic pooling is intentional — it is a genuinely different, vanilla
implementation. Noted in RESULT.md.)
"""
import torch
import torch.nn as nn


class DoubleConv3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, features=(32, 64, 128, 256, 320)):
        super().__init__()
        self.pool = nn.MaxPool3d(2, 2)
        # encoder
        self.enc = nn.ModuleList()
        c = in_channels
        for f in features:
            self.enc.append(DoubleConv3d(c, f))
            c = f
        # decoder (upsample from deepest feature; skips are the shallower encoder outputs)
        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        rev = list(reversed(features))          # [320,256,128,64,32]
        for i in range(len(features) - 1):
            self.ups.append(nn.ConvTranspose3d(rev[i], rev[i + 1], 2, 2))
            self.dec.append(DoubleConv3d(rev[i + 1] * 2, rev[i + 1]))
        self.head = nn.Conv3d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = self.pool(x)
        skips = skips[::-1]
        for i, (up, dec) in enumerate(zip(self.ups, self.dec)):
            x = up(x)
            s = skips[i]
            # guard against odd sizes (defensive; patch is chosen divisible)
            if x.shape[2:] != s.shape[2:]:
                x = nn.functional.interpolate(x, size=s.shape[2:], mode="trilinear",
                                              align_corners=False)
            x = dec(torch.cat([s, x], dim=1))
        return self.head(x)


if __name__ == "__main__":
    m = UNet3D()
    n = sum(p.numel() for p in m.parameters())
    print(f"UNet3D params: {n/1e6:.2f}M")
    x = torch.randn(1, 1, 48, 224, 192)
    with torch.no_grad():
        y = m(x)
    print("in", tuple(x.shape), "out", tuple(y.shape))
    assert y.shape == (1, 2, 48, 224, 192)
    print("shape OK")
