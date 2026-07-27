"""
E5 — ensembling delta: nnU-Net 3d_fullres single fold (fold 0) vs 5-fold ensemble.

Both models are already trained and already scored on the SAME locked 43-case test set
(Dataset002_Parotid/labelsTs). This script does no training and no inference: it performs a
PAIRED per-case comparison of the two existing prediction sets, because the headline delta
(+0.0015 Dice) is far too small to interpret from two aggregate numbers alone.

Inputs (read-only):
  Parotid-Project/Visualizations/parotid_viz_testset/metrics.csv   -> single fold (fold 0)
  Parotid-Project/Visualizations/parotid_viz_ensemble/metrics.csv  -> 5-fold ensemble
  Parotid-Project/Results/results_tracker_TEST.csv                 -> headline single-fold row
  Parotid-Project/Results/results_5fold_TEST.csv                   -> headline ensemble row

Output: ablation_study/E5_ensembling/paired_per_case.csv + printed summary.
"""

import csv
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

SINGLE = ROOT / "Parotid-Project/Visualizations/parotid_viz_testset/metrics.csv"
ENSEMB = ROOT / "Parotid-Project/Visualizations/parotid_viz_ensemble/metrics.csv"


def load(path):
    with open(path) as f:
        return {r["case"]: r for r in csv.DictReader(f)}


def fnum(x):
    """Parse a metric, mapping blanks/nan to None (HD95 is NaN when one side is empty)."""
    if x is None or x.strip() == "" or x.strip().lower() == "nan":
        return None
    return float(x)


def wilcoxon_signed_rank(deltas):
    """
    Two-sided Wilcoxon signed-rank test on paired deltas, via scipy if available.
    Returns (statistic, p_value) or (None, None) if scipy is unavailable or all deltas are 0.
    """
    nz = [d for d in deltas if d != 0]
    if not nz:
        return None, None
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None, None
    stat, p = wilcoxon(nz, alternative="two-sided")
    return float(stat), float(p)


def main():
    single, ensemble = load(SINGLE), load(ENSEMB)
    assert set(single) == set(ensemble), "test case sets differ — not a valid paired comparison"
    cases = sorted(single)

    rows = []
    for c in cases:
        s, e = single[c], ensemble[c]
        s_dice, e_dice = fnum(s["dice_mean"]), fnum(e["dice_mean"])
        # HD95 per case = mean of the two sides, skipping NaN sides (matches eval_testset nanmean)
        def hd95(r):
            vals = [v for v in (fnum(r["hd95_R_mm"]), fnum(r["hd95_L_mm"])) if v is not None]
            return st.mean(vals) if vals else None
        s_hd, e_hd = hd95(s), hd95(e)
        rows.append({
            "case": c,
            "single_dice": s_dice,
            "ensemble_dice": e_dice,
            "delta_dice": e_dice - s_dice,
            "single_hd95_mm": s_hd,
            "ensemble_hd95_mm": e_hd,
            "delta_hd95_mm": (e_hd - s_hd) if (s_hd is not None and e_hd is not None) else None,
        })

    with open(OUT / "paired_per_case.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    d_dice = [r["delta_dice"] for r in rows]
    n = len(d_dice)
    wins = sum(1 for d in d_dice if d > 0)
    losses = sum(1 for d in d_dice if d < 0)
    ties = n - wins - losses

    print(f"Paired comparison over {n} locked test cases (ensemble - single fold)\n")
    print(f"  single-fold mean Dice : {st.mean([r['single_dice'] for r in rows]):.4f}")
    print(f"  ensemble   mean Dice : {st.mean([r['ensemble_dice'] for r in rows]):.4f}")
    print(f"  mean delta           : {st.mean(d_dice):+.4f}")
    print(f"  median delta         : {st.median(d_dice):+.4f}")
    print(f"  std of delta         : {st.stdev(d_dice):.4f}")
    print(f"  range of delta       : {min(d_dice):+.4f} .. {max(d_dice):+.4f}")
    print(f"  ensemble better/worse/tied : {wins}/{losses}/{ties}")

    stat, p = wilcoxon_signed_rank(d_dice)
    if p is None:
        print("  Wilcoxon signed-rank : scipy unavailable")
    else:
        print(f"  Wilcoxon signed-rank : W={stat:.1f}, p={p:.3f}")

    # Per-case delta is dwarfed by per-case spread: contextualise it.
    spread = st.stdev([r["single_dice"] for r in rows])
    print(f"\n  per-case Dice std (single fold) : {spread:.4f}")
    print(f"  |mean delta| / per-case std     : {abs(st.mean(d_dice)) / spread:.3f}")

    hd_pairs = [r for r in rows if r["delta_hd95_mm"] is not None]
    if hd_pairs:
        dh = [r["delta_hd95_mm"] for r in hd_pairs]
        print(f"\n  HD95 pairs available : {len(hd_pairs)}/{n}")
        print(f"  single-fold mean HD95 : {st.mean([r['single_hd95_mm'] for r in hd_pairs]):.2f} mm")
        print(f"  ensemble   mean HD95 : {st.mean([r['ensemble_hd95_mm'] for r in hd_pairs]):.2f} mm")
        print(f"  mean delta HD95      : {st.mean(dh):+.2f} mm")
        hstat, hp = wilcoxon_signed_rank(dh)
        if hp is not None:
            print(f"  Wilcoxon (HD95)      : W={hstat:.1f}, p={hp:.3f}")

    biggest = sorted(rows, key=lambda r: abs(r["delta_dice"]), reverse=True)[:5]
    print("\n  Largest per-case swings (either direction):")
    for r in biggest:
        print(f"    {r['case']}: {r['single_dice']:.4f} -> {r['ensemble_dice']:.4f} ({r['delta_dice']:+.4f})")


if __name__ == "__main__":
    main()
