"""
E4 dataloader — random-patch sampling over the nnU-Net-preprocessed Dataset002
volumes, with nnU-Net-style foreground oversampling and the nnU-Net default fold-0
split, so training conditions match the 3d_fullres single-fold baseline.

- Patch [48,224,192] (from plans.json), constant-padded if a volume is smaller.
- oversample_foreground_percent=0.33 (nnU-Net default): a third of patches are
  centred on a random foreground (parotid) voxel; the rest are uniformly random.
- Fold split reproduces nnU-Net's do_split(): KFold(5, shuffle=True,
  random_state=12345) over sorted case ids; fold 0 -> (train, val).
  On the pod, PREFER the real splits_final.json if nnU-Net generated one; this
  reproduction matches nnU-Net's default when no split file exists.
- 2-channel target (ch0=PAROTID_R from seg==1, ch1=PAROTID_L from seg==2).

Volumes are held in RAM as float16 (208 cases ~ a few GB); patches are cast to
float32 on the fly.
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

PATCH = (48, 224, 192)


def nnunet_fold0_split(case_ids, n_splits=5, seed=12345, fold=0):
    from sklearn.model_selection import KFold
    keys = np.sort(np.array(case_ids))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(kf.split(keys))
    tr_idx, val_idx = splits[fold]
    return list(keys[tr_idx]), list(keys[val_idx])


def _crop_pad(vol, center, patch):
    """Extract a patch of size `patch` centred at `center` from vol (Z,Y,X),
    constant-padding (0) where the patch runs off the volume."""
    out = np.zeros(patch, dtype=vol.dtype)
    src_lo, src_hi, dst_lo, dst_hi = [], [], [], []
    for d in range(3):
        lo = center[d] - patch[d] // 2
        hi = lo + patch[d]
        s_lo, s_hi = max(0, lo), min(vol.shape[d], hi)
        src_lo.append(s_lo); src_hi.append(s_hi)
        dst_lo.append(s_lo - lo); dst_hi.append(s_hi - lo)
    out[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]] = \
        vol[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
    return out


class Patch3DDataset(Dataset):
    def __init__(self, preproc_dir, case_ids, patch=PATCH, augment=True,
                 oversample_fg=0.33, samples_per_epoch=None, max_fg_coords=20000,
                 seed=42):
        self.patch = patch
        self.augment = augment
        self.oversample_fg = oversample_fg
        self.rng = np.random.RandomState(seed)
        self.cases = []
        self.data = {}
        self.seg = {}
        self.fg_coords = {}
        for cid in case_ids:
            f = os.path.join(preproc_dir, f"{cid}.npz")
            if not os.path.exists(f):
                continue
            d = np.load(f)
            self.data[cid] = d["data"]           # float16 (Z,Y,X)
            self.seg[cid] = d["seg"]             # uint8 (Z,Y,X)
            fg = np.argwhere(d["seg"] > 0)
            if len(fg) > max_fg_coords:
                fg = fg[self.rng.choice(len(fg), max_fg_coords, replace=False)]
            self.fg_coords[cid] = fg
            self.cases.append(cid)
        self.samples_per_epoch = samples_per_epoch or (len(self.cases) * 50)
        print(f"Patch3DDataset: {len(self.cases)} cases, "
              f"{self.samples_per_epoch} samples/epoch, patch {patch}")

    def __len__(self):
        return self.samples_per_epoch

    def _random_center(self, cid, force_fg):
        shp = self.data[cid].shape
        fg = self.fg_coords[cid]
        if force_fg and len(fg) > 0:
            return tuple(fg[self.rng.randint(len(fg))])
        return tuple(self.rng.randint(0, max(1, shp[d]) ) for d in range(3))

    def __getitem__(self, idx):
        cid = self.cases[self.rng.randint(len(self.cases))]
        force_fg = self.rng.rand() < self.oversample_fg
        center = self._random_center(cid, force_fg)
        img = _crop_pad(self.data[cid], center, self.patch).astype(np.float32)
        seg = _crop_pad(self.seg[cid], center, self.patch)

        if self.augment and self.rng.rand() > 0.5:
            # horizontal (L/R) flip along the X axis; swap R/L labels to stay anatomical
            img = img[:, :, ::-1].copy()
            seg = seg[:, :, ::-1].copy()
            swapped = np.zeros_like(seg)
            swapped[seg == 1] = 2
            swapped[seg == 2] = 1
            seg = swapped

        target = np.zeros((2, *self.patch), np.float32)
        target[0] = (seg == 1)
        target[1] = (seg == 2)
        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(target)


class ValPatchDataset(Patch3DDataset):
    """Deterministic-ish validation sampling (foreground-centred, no aug)."""
    def __init__(self, *a, **k):
        k.setdefault("augment", False)
        k.setdefault("oversample_fg", 1.0)
        super().__init__(*a, **k)


if __name__ == "__main__":
    import sys
    pd = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "preprocessed", "train")
    ids = [os.path.basename(f)[:-4] for f in glob.glob(os.path.join(pd, "*.npz"))]
    tr, val = nnunet_fold0_split(ids) if len(ids) >= 5 else (ids, ids)
    print(f"cases={len(ids)} fold0 train={len(tr)} val={len(val)}")
    ds = Patch3DDataset(pd, ids, samples_per_epoch=4)
    x, y = ds[0]
    print("patch img", tuple(x.shape), "target", tuple(y.shape),
          "fg_vox", int(y.sum().item()))
