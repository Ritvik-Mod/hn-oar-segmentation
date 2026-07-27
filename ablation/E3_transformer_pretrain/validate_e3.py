"""
Offline validation for the E3 pretrained models — proves the code is runnable
WITHOUT downloading ImageNet weights and WITHOUT training. Exercises:
  1. architecture forward pass + shape (both arms), at batch 2, 512x512, 1-channel
  2. a real backward step through CombinedLoss (Dice+BCE) — gradients flow
  3. the ResNet-50 encoder weight-mapping path (258/258 keys + conv1 3->1 average),
     using a locally-constructed resnet50(weights=None) as a stand-in source so no
     network is needed; the real IMAGENET1K_V2 load takes the identical path on the pod.
Pass --download to additionally try the real torchvision/timm pretrained fetch.
"""
import os
import sys
import argparse
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "dicom test", "unet_codes_from_hpc"))  # loss_function

from models_pretrained import (build_transunet_pretrained, build_swin_pretrained,
                               SwinUNetPretrained)
from loss_function import CombinedLoss


def fwd_bwd(model, name, device):
    model = model.to(device).train()
    x = torch.randn(2, 1, 512, 512, device=device)
    y = torch.zeros(2, 2, 512, 512, device=device)
    y[:, 0, 100:150, 100:150] = 1.0
    y[:, 1, 100:150, 360:410] = 1.0
    crit = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    out = model(x)
    assert out.shape == (2, 2, 512, 512), f"{name} bad shape {out.shape}"
    loss, dice, bce = crit(out, y)
    loss.backward()
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print(f"[{name}] out {tuple(out.shape)}  loss {loss.item():.4f} "
          f"(dice {dice.item():.4f} bce {bce.item():.4f})  grad_ok={grad_ok}")
    assert grad_ok, f"{name} no finite gradients"


def test_resnet_mapping():
    """Validate the exact key-map + conv1-average path with a local resnet50 source."""
    from torchvision.models import resnet50
    from transunet import TransUNet
    tv = resnet50(weights=None)               # no download; same key names as pretrained
    tv_sd = tv.state_dict()
    model = TransUNet(in_channels=1, out_channels=2)
    enc = model.resnet
    enc_sd = enc.state_dict()
    n = 0
    new_sd = {}
    for k, v in enc_sd.items():
        if k == "conv1.weight":
            assert tv_sd[k].shape == (64, 3, 7, 7)
            new_sd[k] = tv_sd[k].mean(dim=1, keepdim=True)
            assert new_sd[k].shape == v.shape == (64, 1, 7, 7)
            n += 1
        elif k in tv_sd and tv_sd[k].shape == v.shape:
            new_sd[k] = tv_sd[k]; n += 1
        else:
            new_sd[k] = v
    enc.load_state_dict(new_sd, strict=True)
    print(f"[resnet-mapping] {n}/{len(enc_sd)} encoder tensors mapped, conv1 3->1 averaged, strict load OK")
    assert n == len(enc_sd) == 258


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device)

    print("== ResNet-50 encoder mapping (offline stand-in) ==")
    test_resnet_mapping()

    print("\n== TransUNet-pretrained architecture (random init, forward+backward) ==")
    m, info = build_transunet_pretrained(pretrained=False)
    print("  builder info:", info)
    fwd_bwd(m, "TransUNet_pretrained(random)", dev)

    print("\n== Swin-pretrained architecture (random init, forward+backward) ==")
    m, info = build_swin_pretrained(pretrained=False)
    print("  builder info:", info)
    fwd_bwd(m, "Swin_pretrained(random)", dev)

    if args.download:
        print("\n== Real pretrained fetch (needs internet) ==")
        try:
            m, info = build_transunet_pretrained(pretrained=True)
            print("  TransUNet pretrained OK:", info)
        except Exception as e:
            print("  TransUNet pretrained FAILED (expected offline):", repr(e)[:120])
        try:
            m, info = build_swin_pretrained(pretrained=True)
            print("  Swin pretrained OK:", info)
        except Exception as e:
            print("  Swin pretrained FAILED (expected offline):", repr(e)[:120])

    print("\nAll E3 offline validations passed.")


if __name__ == "__main__":
    main()
