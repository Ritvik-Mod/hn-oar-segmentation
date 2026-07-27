"""Render per-case 2D montages + 3D for every nii.gz model. Resumable."""
import os, sys, json, time
import numpy as np
import galcommon as g

DO_3D = "--no3d" not in sys.argv
ONLY = None
for a in sys.argv[1:]:
    if a.startswith("--only="):
        ONLY = a.split("=", 1)[1].split(",")

cases = g.test_cases()
sel = json.load(open(os.path.join(g.OUT, "_selection.json")))
percase = sel["percase"]

manifest = {}
t0 = time.time()
for key in g.ORDER:
    if ONLY and key not in ONLY:
        continue
    m = g.MODELS[key]
    mdir = g.P(m["dir"])
    odir = os.path.join(g.OUT, key)
    os.makedirs(odir, exist_ok=True)
    rows = []
    for c in cases:
        pp = os.path.join(mdir, f"{c}.nii.gz")
        if not os.path.exists(pp):
            continue
        img, spacing = g.load_vol(os.path.join(g.CT_DIR, f"{c}_0000.nii.gz"))
        pred = g.load_vol(pp)[0].astype(np.uint8)
        ps = g.norm_pred(pred, m["remap"])
        gt = g.load_gt(c)
        # restrict GT overlay to the sides this model predicts (single-side experts)
        gt_use = {s: gt[s] for s in ps} if gt else None
        d = percase[c][key]
        sides = "/".join(f"{s} {g.dice(ps[s], gt[s]):.3f}" for s in ps) if gt else "?"
        title = f"{m['label']}  —  {c}   Dice {d:.3f}  ({sides})   [agg {m['dice']}]" \
            if m["dice"] else f"{m['label']}  —  {c}   Dice {d:.3f}  ({sides})"
        mont = os.path.join(odir, f"{c}_slices.png")
        if not os.path.exists(mont):
            g.render_montage(img, ps, gt_use, mont, title)
        r3d = os.path.join(odir, f"{c}_3d.png")
        if DO_3D and not os.path.exists(r3d):
            g.render_3d(ps, spacing, r3d, f"{m['label']} — {c} (red=R, blue=L)")
        rows.append({"case": c, "dice": d})
    manifest[key] = {"label": m["label"], "dice": m["dice"], "note": m["note"],
                     "cases": rows}
    print(f"[{time.time()-t0:6.1f}s] {key:16s} {len(rows)} cases done")

# merge into manifest.json (don't clobber P0 section if present)
mpath = os.path.join(g.OUT, "manifest.json")
existing = json.load(open(mpath)) if os.path.exists(mpath) else {}
existing.update(manifest)
json.dump(existing, open(mpath, "w"), indent=1)
print(f"\nDone in {time.time()-t0:.1f}s. Wrote manifest.json")
