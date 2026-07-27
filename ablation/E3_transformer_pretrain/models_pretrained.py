"""
E3 — ImageNet-pretrained variants of the two Phase-1 transformers.

The point of E3 is to isolate the "trained from random init" handicap (Q3): the
Phase-1 TransUNet and Swin-UNet were trained from scratch, whereas both are
*designed* to start from ImageNet-pretrained backbones. Here we re-create their
intended pretrained configuration, holding the Phase-1 2D protocol otherwise
identical (Dice+BCE, Adam 1e-4, batch 8, AMP, same dataloader/weighting/aug).

Two arms:

  TransUNetPretrained  — the FROZEN hand-built TransUNet (transunet.py) with its
    ResNet-50 encoder initialised from torchvision IMAGENET1K_V2 weights. The
    hand-built `ResNet50Encoder` module names match torchvision resnet50 exactly
    (verified: 258/258 encoder tensors), so this is a clean, same-architecture,
    only-initialisation-changed swap. The 7x7 conv1 stem is adapted 3->1 channel
    by averaging the pretrained RGB filters. The 12-layer transformer + CUP
    decoder stay from-scratch (ImageNet has no matching ViT for this exact stack;
    the plan's minimum requirement is a pretrained ResNet-50, which this meets).

  SwinUNetPretrained   — a Swin-UNet whose ENCODER is timm's ImageNet-pretrained
    `swin_tiny_patch4_window7_224` (features_only, in_chans=1 -> timm adapts the
    patch-embed stem), feeding the SAME from-scratch Swin decoder blocks used by
    the frozen Swin-UNet (PatchExpanding + SwinTransformerStage + seg head +
    bilinear-to-512).
    *** CAVEAT (flagged for the owner / logged in RESULT.md, golden rule #6): ***
    a true "same-architecture, only-init-changed" Swin is NOT possible with an
    off-the-shelf ImageNet Swin-T. The frozen Swin-UNet uses window=8 / 512-input
    (relative_position_bias_table 225, depths [2,2,2]); ImageNet Swin-T uses
    window=7 / 224-input (bias table 169, depths [2,2,6,2]). The window/depth
    mismatch makes the pretrained attention weights un-loadable into the frozen
    architecture. So the Swin arm changes the encoder to timm's Swin-T, which
    means the Swin pretrained-vs-scratch comparison carries a mild architecture
    confound (encoder depth/window differ). The TransUNet arm has no such caveat.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_DT = os.path.join(_ROOT, "dicom test")
for _sub in ("transunet_codes_from_hpc", "swinunet_codes_from_hpc"):
    p = os.path.join(_DT, _sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from transunet import TransUNet                                   # noqa: E402
from swin_unet import (PatchExpanding, SwinTransformerStage)      # noqa: E402


# ---------------------------------------------------------------------------
# TransUNet with ImageNet-pretrained ResNet-50 encoder
# ---------------------------------------------------------------------------
def build_transunet_pretrained(pretrained=True, verbose=True):
    """Frozen TransUNet + torchvision ResNet-50 ImageNet weights in the encoder."""
    model = TransUNet(in_channels=1, out_channels=2)
    if not pretrained:
        return model, {"loaded": 0, "averaged_stem": False, "note": "random init"}

    from torchvision.models import resnet50, ResNet50_Weights
    tv = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    tv_sd = tv.state_dict()

    enc = model.resnet                      # ResNet50Encoder (conv1,bn1,layer1..3)
    enc_sd = enc.state_dict()
    new_sd = {}
    n_loaded = 0
    for k, v in enc_sd.items():
        if k == "conv1.weight":
            # 3->1 channel: average the pretrained RGB filters
            new_sd[k] = tv_sd[k].mean(dim=1, keepdim=True)
            n_loaded += 1
        elif k in tv_sd and tv_sd[k].shape == v.shape:
            new_sd[k] = tv_sd[k]
            n_loaded += 1
        else:
            new_sd[k] = v                   # keep (should not happen; verified 258/258)
    missing, unexpected = enc.load_state_dict(new_sd, strict=True)
    info = {"loaded": n_loaded, "total_encoder_tensors": len(enc_sd),
            "averaged_stem": True}
    if verbose:
        print(f"[TransUNetPretrained] loaded {n_loaded}/{len(enc_sd)} encoder "
              f"tensors from torchvision resnet50 IMAGENET1K_V2 (conv1 stem 3->1 averaged)")
    return model, info


# ---------------------------------------------------------------------------
# Swin-UNet with ImageNet-pretrained Swin-T encoder (timm) + from-scratch decoder
# ---------------------------------------------------------------------------
class SwinUNetPretrained(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, embed_dim=96,
                 window_size=8, num_heads=(3, 6, 12, 24), pretrained=True):
        super().__init__()
        import timm
        self.encoder = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=pretrained,
            features_only=True, in_chans=in_channels, img_size=512,
            out_indices=(0, 1, 2, 3))
        # encoder stage outputs: [ (B,128,128,96), (B,64,64,192),
        #                          (B,32,32,384), (B,16,16,768) ]

        # from-scratch bottleneck + decoder (same blocks as the frozen Swin-UNet)
        self.bottleneck = SwinTransformerStage(
            dim=embed_dim * 8, num_heads=num_heads[3],
            window_size=window_size, input_resolution=(16, 16))

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

        self.seg_head = nn.Linear(embed_dim, out_channels)
        self._init_decoder()

    def _init_decoder(self):
        for name, m in self.named_modules():
            if name.startswith("encoder"):
                continue
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _tokens(feat):
        # (B,H,W,C) -> (B, H*W, C), H, W
        B, H, W, C = feat.shape
        return feat.reshape(B, H * W, C), H, W

    def forward(self, x):
        B = x.shape[0]
        f0, f1, f2, f3 = self.encoder(x)
        skip0, _, _ = self._tokens(f0)     # (B,16384,96)
        skip1, _, _ = self._tokens(f1)     # (B,4096,192)
        skip2, _, _ = self._tokens(f2)     # (B,1024,384)
        x, H, W = self._tokens(f3)         # (B,256,768) @16x16

        x = self.bottleneck(x, H, W)

        x, H, W = self.expand2(x, H, W)
        x = self.concat_proj2(torch.cat([x, skip2], dim=-1))
        x = self.decoder2(x, H, W)

        x, H, W = self.expand1(x, H, W)
        x = self.concat_proj1(torch.cat([x, skip1], dim=-1))
        x = self.decoder1(x, H, W)

        x, H, W = self.expand0(x, H, W)
        x = self.concat_proj0(torch.cat([x, skip0], dim=-1))
        x = self.decoder0(x, H, W)

        x = self.seg_head(x)               # (B,16384,2)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)     # (B,2,128,128)
        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        return x


def build_swin_pretrained(pretrained=True, verbose=True):
    model = SwinUNetPretrained(in_channels=1, out_channels=2, pretrained=pretrained)
    if verbose:
        n = sum(p.numel() for p in model.parameters())
        print(f"[SwinUNetPretrained] timm swin_tiny encoder "
              f"(pretrained={pretrained}) + from-scratch decoder; {n/1e6:.1f}M params")
    return model, {"encoder": "timm swin_tiny_patch4_window7_224",
                   "pretrained": pretrained}


BUILDERS = {
    "transunet_pretrained": build_transunet_pretrained,
    "swin_pretrained": build_swin_pretrained,
}
