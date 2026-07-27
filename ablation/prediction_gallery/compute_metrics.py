"""Compute per-case Dice for every nii.gz model vs GT; save metrics + pick reps."""
import os, csv, json
import numpy as np
import galcommon as g

cases = g.test_cases()
print(f"{len(cases)} test cases")

# per-case mean dice: rows[case][model] = mean dice (present sides only)
percase = {c: {} for c in cases}
percase_sides = {c: {} for c in cases}
agg = {}

for key in g.MODELS:
    m = g.MODELS[key]
    mdir = g.P(m["dir"])
    dvals = []
    for c in cases:
        p = os.path.join(mdir, f"{c}.nii.gz")
        gtp = os.path.join(g.GT_DIR, f"{c}.nii.gz")
        if not os.path.exists(p):
            continue
        pred = g.load_vol(p)[0].astype(np.uint8)
        gt = g.load_vol(gtp)[0].astype(np.uint8)
        psides = g.norm_pred(pred, m["remap"])
        gt_sd = {"R": gt == 1, "L": gt == 2}
        ds = {}
        for side in psides:  # only sides this model predicts
            ds[side] = g.dice(psides[side], gt_sd[side])
        mean_d = float(np.mean(list(ds.values()))) if ds else float("nan")
        percase[c][key] = mean_d
        percase_sides[c][key] = ds
        dvals.append(mean_d)
    agg[key] = float(np.nanmean(dvals)) if dvals else float("nan")
    print(f"  {key:16s} n={len(dvals):2d}  mean-of-per-case Dice = {agg[key]:.4f}"
          f"   (table {m['dice']})")

# write full per-case CSV (both-parotid models only, i.e. those predicting R&L)
both_models = [k for k in g.ORDER if set(g.MODELS[k]["remap"].values()) == {"R", "L"}]
with open(os.path.join(g.OUT, "metrics_percase.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case"] + both_models)
    for c in cases:
        w.writerow([c] + [f"{percase[c].get(k, float('nan')):.4f}" for k in both_models])

# pick representative cases by the reference model (nnunet_1fold)
ref = "nnunet_1fold"
ranked = sorted(cases, key=lambda c: percase[c][ref])
worst = ranked[0]
best = ranked[-1]
median = ranked[len(ranked) // 2]

# disagreement case: largest spread across the both-parotid comparison models
def spread(c):
    vals = [percase[c][k] for k in g.COMPARE_ORDER if k in percase[c]]
    return max(vals) - min(vals)
disagree = max(cases, key=spread)

reps = []
for tag, c in [("best", best), ("median", median), ("worst", worst),
               ("disagreement", disagree)]:
    if c not in [r[1] for r in reps]:
        reps.append((tag, c))
# add two more mid cases for variety
for c in [ranked[len(ranked)//4], ranked[3*len(ranked)//4]]:
    if c not in [r[1] for r in reps]:
        reps.append(("quartile", c))

sel = {
    "agg": agg,
    "reps": reps,
    "ref_ranked": ranked,
    "percase": percase,
}
with open(os.path.join(g.OUT, "_selection.json"), "w") as f:
    json.dump(sel, f, indent=1)

print("\nRepresentative cases (by nnU-Net 1-fold per-case Dice):")
for tag, c in reps:
    print(f"  {tag:12s} {c}  ref-Dice={percase[c][ref]:.3f}  spread={spread(c):.3f}")
print("\nWrote metrics_percase.csv + _selection.json")
