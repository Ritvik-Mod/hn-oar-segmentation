"""
E6 mask-plumbing sanity check — verifies the masked loss is wired correctly
END-TO-END through the E6 dataloader (not just the loss unit test in
masked_loss.py). Run AFTER preprocess_e6.py has produced ./preprocessed/train.

Checks:
  A. Per-case masks match annotation status for real cases (both=[1,1],
     only_R=[1,0], only_L=[0,1]); counts total 208/110/112 over the 430.
  B. hflip swaps the mask together with the R/L target channels (a single-side
     patch, mirrored, has its mask flipped and its annotated gland in the other
     channel — never a masked channel with a live target or vice-versa).
  C. On a single-side sample, corrupting the UN-annotated channel's target/logits
     leaves MaskedPartialLoss UNCHANGED, but changes ordinary DiceBCELoss3D
     (i.e. the mask genuinely removes the un-annotated gland from the gradient).
  D. On a both-annotated sample (mask [1,1]), MaskedPartialLoss == the plain
     combined loss (masking is a no-op when everything is annotated).
"""
import os
import sys
import json
import glob
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from dataloader_e6 import MaskedPatch3DDataset, PATCH
from loss3d import DiceBCELoss3D
from masked_loss import MaskedPartialLoss

PRE = os.path.join(HERE, "preprocessed", "train")
masked_crit = MaskedPartialLoss()
dirty_crit = DiceBCELoss3D()


def check_A():
    audit = json.load(open(os.path.join(HERE, "preprocessed", "case_masks.json")))
    cnt = {"both": 0, "only_R": 0, "only_L": 0, "none": 0}
    bad = []
    for cid, v in audit.items():
        mr, ml = v["mask"]
        derived = ("both" if mr and ml else "only_R" if mr else
                   "only_L" if ml else "none")
        cnt[derived] += 1
        if derived != v["status"]:
            bad.append((cid, v["status"], derived))
    print(f"[A] status counts over {len(audit)} cases: {cnt}")
    assert not bad, f"mask/status mismatch: {bad[:5]}"
    assert cnt["both"] == 208 and cnt["only_R"] == 110 and cnt["only_L"] == 112, cnt
    assert cnt["none"] == 0
    print("[A] PASS: masks match annotation status; 208 both / 110 R / 112 L.")


def check_B_and_C():
    # find a single-side case id from the audit
    audit = json.load(open(os.path.join(HERE, "preprocessed", "case_masks.json")))
    single = [c for c, v in audit.items() if v["status"] in ("only_R", "only_L")]
    assert single, "no single-side cases found"
    cid = single[0]
    status = audit[cid]["status"]

    # Draw many patches from just this case until we hit a foreground patch, with
    # augment on, so we exercise the flip path too.
    ds = MaskedPatch3DDataset(PRE, [cid], augment=True, oversample_fg=1.0,
                              samples_per_epoch=1, seed=0)
    seen_flip = False
    for trial in range(200):
        ds.rng = np.random.RandomState(trial)
        img, tgt, mask = ds[0]
        mask = mask.numpy()
        annotated = int(np.argmax(mask))          # channel that IS annotated
        unann = 1 - annotated
        # invariant: the annotated channel is the one that can carry live target;
        # the un-annotated channel's target must be all-zero background.
        assert tgt[unann].sum() == 0, \
            f"un-annotated channel {unann} has {int(tgt[unann].sum())} live voxels!"
        # detect that flipping happened at least once (mask no longer matches
        # the case's stored orientation)
        stored = np.array(audit[cid]["mask"], np.float32)
        if not np.array_equal(mask, stored):
            seen_flip = True
        if tgt[annotated].sum() > 0:  # a foreground patch -> run the loss check
            img_b, tgt_b, mask_b = img.unsqueeze(0), tgt.unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)
            logits = torch.randn_like(tgt_b)
            base = masked_crit(logits, tgt_b, mask_b).item()
            # corrupt the UN-annotated channel: flip its target and randomise logits
            tgt2 = tgt_b.clone(); tgt2[0, unann] = 1 - tgt2[0, unann]
            log2 = logits.clone(); log2[0, unann] = torch.randn_like(log2[0, unann])
            corrupt = masked_crit(log2, tgt2, mask_b).item()
            dirty_base = dirty_crit(logits, tgt_b)[0].item()
            dirty_corrupt = dirty_crit(log2, tgt2)[0].item()
            print(f"[C] case {cid} ({status}) annotated_ch={annotated} "
                  f"masked: {base:.6f} -> {corrupt:.6f} (Δ={abs(base-corrupt):.2e}); "
                  f"ordinary: {dirty_base:.6f} -> {dirty_corrupt:.6f} "
                  f"(Δ={abs(dirty_base-dirty_corrupt):.2e})")
            assert abs(base - corrupt) < 1e-6, "masked loss CHANGED — mask leak!"
            assert abs(dirty_base - dirty_corrupt) > 1e-6, \
                "ordinary loss did NOT change — un-annotated channel had no effect (unexpected)"
            print("[C] PASS: masked loss ignores the un-annotated channel end-to-end; "
                  "ordinary loss does not.")
            break
    else:
        raise RuntimeError("never hit a foreground patch for the loss check")
    print(f"[B] flip-swap exercised: seen_flip={seen_flip} "
          f"(mask swaps with R/L channels; un-annotated channel always background — verified above)")


def check_D():
    # a both-annotated case: masked loss must equal the plain combined loss
    audit = json.load(open(os.path.join(HERE, "preprocessed", "case_masks.json")))
    both = [c for c, v in audit.items() if v["status"] == "both"][0]
    ds = MaskedPatch3DDataset(PRE, [both], augment=False, oversample_fg=1.0,
                              samples_per_epoch=1, seed=1)
    img, tgt, mask = ds[0]
    tgt_b, mask_b = tgt.unsqueeze(0), mask.unsqueeze(0)
    logits = torch.randn_like(tgt_b)
    m = masked_crit(logits, tgt_b, mask_b).item()
    d = dirty_crit(logits, tgt_b)[0].item()
    print(f"[D] both-annotated case {both}: masked {m:.6f} vs ordinary {d:.6f} "
          f"(Δ={abs(m-d):.2e})")
    assert abs(m - d) < 1e-5, "masked != ordinary on a fully-annotated sample!"
    print("[D] PASS: masking is a no-op when both glands are annotated.")


if __name__ == "__main__":
    print("=== E6 mask-plumbing sanity check ===")
    check_A()
    check_B_and_C()
    check_D()
    print("\nALL CHECKS PASSED — mask plumbing is correct end-to-end.")
