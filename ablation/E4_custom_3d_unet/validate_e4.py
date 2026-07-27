"""
Offline validation for E4 — proves preprocessing, dataloader, model, loss and the
sliding-window inference wiring all run, WITHOUT a full-size 3D forward on CPU
(the heavy real-patch forward is left for the GPU). Uses small tensors / a small
patch so it finishes in seconds.
"""
import os
import sys
import shutil
import tempfile
import numpy as np
import nibabel as nib
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from unet3d import UNet3D
from loss3d import DiceBCELoss3D
from preprocess_e4 import ct_norm_constants, ct_normalize
from dataloader_e4 import Patch3DDataset, nnunet_fold0_split, _crop_pad
import predict_e4

DS = os.path.join(ROOT, "Parotid-Project", "Datasets", "Dataset002_Parotid")


def main():
    tmp = tempfile.mkdtemp(prefix="e4val_")
    train_dir = os.path.join(tmp, "train"); os.makedirs(train_dir)

    # 1. CTNormalization constants + normalize 3 real train cases
    c = ct_norm_constants()
    print("1. CTNorm constants:", c)
    cases = sorted(f[:-len("_0000.nii.gz")] for f in os.listdir(os.path.join(DS, "imagesTr"))
                   if f.endswith("_0000.nii.gz"))[:3]
    for par in cases:
        img = np.transpose(np.asarray(nib.load(f"{DS}/imagesTr/{par}_0000.nii.gz").dataobj), (2, 0, 1))
        lab = np.transpose(np.asarray(nib.load(f"{DS}/labelsTr/{par}.nii.gz").dataobj), (2, 0, 1))
        data = ct_normalize(img, c)
        np.savez_compressed(f"{train_dir}/{par}.npz", data=data.astype(np.float16), seg=lab.astype(np.uint8))
        print(f"   {par}: shape={data.shape} norm mean={data.mean():.3f} std={data.std():.3f} "
              f"fg_vox={int((lab>0).sum())}")

    # 2. fold split reproduction (on the full 208 case-id list)
    all_ids = sorted(f[:-len("_0000.nii.gz")] for f in os.listdir(os.path.join(DS, "imagesTr"))
                     if f.endswith("_0000.nii.gz"))
    tr, val = nnunet_fold0_split(all_ids)
    print(f"2. fold-0 split: {len(all_ids)} cases -> train {len(tr)} / val {len(val)} "
          f"(expected ~166/42)")
    assert len(tr) + len(val) == len(all_ids)

    # 3. dataloader patch shapes + foreground oversampling
    ds = Patch3DDataset(train_dir, cases, augment=True, samples_per_epoch=40, seed=0)
    fg_hits = 0
    for i in range(20):
        x, y = ds[i]
        assert x.shape == (1, 48, 224, 192), x.shape
        assert y.shape == (2, 48, 224, 192), y.shape
        if y.sum() > 0:
            fg_hits += 1
    print(f"3. dataloader: 20 samples OK, {fg_hits}/20 contained foreground "
          f"(oversample_fg=0.33 -> expect several)")
    assert fg_hits >= 3, "foreground oversampling not producing fg patches"

    # 4. model + loss forward/backward on a SMALL patch (cheap)
    m = UNet3D(1, 2)
    xb = torch.randn(2, 1, 16, 64, 64)
    yb = torch.zeros(2, 2, 16, 64, 64); yb[:, 0, 4:8, 20:30, 20:30] = 1
    crit = DiceBCELoss3D()
    out = m(xb); loss, d, b = crit(out, yb); loss.backward()
    print(f"4. model+loss small forward/backward OK: out {tuple(out.shape)} "
          f"loss {loss.item():.4f} (dice {d.item():.4f} bce {b.item():.4f})")

    # 5. sliding-window tiling + labelmap orientation, small patch on a small volume
    small_patch = (16, 64, 64)
    vol = np.load(f"{train_dir}/{cases[0]}.npz")["data"].astype(np.float32)[:32, :128, :128]
    prob = predict_e4.sliding_window(m, vol, torch.device("cpu"), patch=small_patch,
                                     overlap=0.5, use_amp=False)
    assert prob.shape == (2, 32, 128, 128), prob.shape
    lm = predict_e4.to_labelmap(prob)
    lm_yxz = np.transpose(lm, (1, 2, 0))
    print(f"5. sliding-window OK: prob {prob.shape} -> labelmap {lm.shape} "
          f"-> saved-orient {lm_yxz.shape} (Y,X,Z); labels {sorted(np.unique(lm).tolist())}")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nAll E4 offline validations passed (full-size 3D forward deferred to GPU).")


if __name__ == "__main__":
    main()
