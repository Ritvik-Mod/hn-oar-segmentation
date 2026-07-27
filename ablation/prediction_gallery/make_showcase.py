"""Build a curated 'best-of' showcase: each model's top cases by Dice + an overall
leaderboard. Copies the chosen montages/3D into showcase/ so the folder is
self-contained. Run AFTER render_all.py + render_p0.py have (re)rendered montages."""
import os, json, shutil
import galcommon as g

TOPN_PER_MODEL = 8      # each model's best N cases
TOPN_OVERALL = 15       # overall leaderboard size

man = json.load(open(os.path.join(g.OUT, "manifest.json")))
p0p = os.path.join(g.OUT, "manifest_p0.json")
if os.path.exists(p0p):
    man.update(json.load(open(p0p)))

SHOW = os.path.join(g.OUT, "showcase")
if os.path.exists(SHOW):
    shutil.rmtree(SHOW)
os.makedirs(SHOW)

NII_ORDER = list(reversed(g.ORDER))          # best -> worst
P0_ORDER = ["P0_attention", "P0_unet", "P0_transunet", "P0_swin"]
DISPLAY = NII_ORDER + [k for k in P0_ORDER if k in man]

# ---- per-model top-N (copy files) ----
per_model = {}
leaderboard = []      # (dice, model_key, case)
for key in DISPLAY:
    m = man[key]
    ranked = sorted(m["cases"], key=lambda r: r["dice"], reverse=True)[:TOPN_PER_MODEL]
    mdir = os.path.join(SHOW, key)
    os.makedirs(mdir, exist_ok=True)
    rows = []
    for i, r in enumerate(ranked, 1):
        c = r["case"]
        src_m = os.path.join(g.OUT, key, f"{c}_slices.png")
        src_3d = os.path.join(g.OUT, key, f"{c}_3d.png")
        dst_m = os.path.join(mdir, f"rank{i:02d}_{c}_slices.png")
        if os.path.exists(src_m):
            shutil.copy2(src_m, dst_m)
        if os.path.exists(src_3d):
            shutil.copy2(src_3d, os.path.join(mdir, f"rank{i:02d}_{c}_3d.png"))
        rows.append({"rank": i, "case": c, "dice": r["dice"]})
        # single-side experts predict one gland only -> keep them out of the overall
        # both-parotid leaderboard so it's an apples-to-apples "best full segmentation"
        if set(g.MODELS.get(key, {}).get("remap", {1: "R", 2: "L"}).values()) == {"R", "L"} \
                or key.startswith("P0"):
            leaderboard.append((r["dice"], key, c))
    per_model[key] = {"label": m["label"], "dice": m["dice"], "note": m["note"],
                      "rows": rows}

leaderboard.sort(reverse=True)
top = leaderboard[:TOPN_OVERALL]
hof = os.path.join(SHOW, "_hall_of_fame")
os.makedirs(hof, exist_ok=True)
hof_rows = []
for i, (d, key, c) in enumerate(top, 1):
    src = os.path.join(g.OUT, key, f"{c}_slices.png")
    dst = os.path.join(hof, f"top{i:02d}_{key}_{c}_slices.png")
    if os.path.exists(src):
        shutil.copy2(src, dst)
    hof_rows.append({"rank": i, "key": key, "label": per_model[key]["label"],
                     "case": c, "dice": d, "file": f"_hall_of_fame/top{i:02d}_{key}_{c}_slices.png"})

# ---- showcase.html ----
CSS = """body{font-family:-apple-system,Segoe UI,sans-serif;max-width:1300px;margin:auto;
padding:24px;background:#0d0d0f;color:#e8e8ea}h1{margin-bottom:2px}h2{border-bottom:1px solid #333;
padding-bottom:6px;margin-top:40px}.sub{color:#9a9aa2;font-size:14px}
a{color:#5aa0ff;text-decoration:none}
.model{background:#161619;border:1px solid #26262b;border-radius:10px;padding:14px 16px;margin:16px 0}
.model .d{color:#7dd3fc;font-weight:700}
.row{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.cell{width:200px}.cell img{width:100%;border:1px solid #26262b;border-radius:6px}
.cell .cap{font-size:11px;color:#b8b8be;text-align:center;margin-top:2px}
.rank{display:inline-block;background:#2a2a30;border-radius:4px;padding:0 6px;color:#ffd479;font-weight:700}
nav{position:sticky;top:0;background:#0d0d0feb;padding:10px 0;border-bottom:1px solid #333;margin-bottom:16px;font-size:14px}
nav a{margin-right:12px}"""


def cells(key, rows, prefix):
    out = ""
    for r in rows:
        c = r["case"]; i = r["rank"]
        f = f"{key}/rank{i:02d}_{c}_slices.png"
        t = f"{key}/rank{i:02d}_{c}_3d.png"
        out += (f'<div class="cell"><a href="{f}"><img src="{f}" loading="lazy"></a>'
                f'<div class="cap"><span class="rank">#{i}</span> {c} · '
                f'Dice {r["dice"]:.3f} · <a href="{t}">3D</a></div></div>')
    return out


nav = ('<nav><a href="#hof">Hall of fame</a>'
       + "".join(f'<a href="#{k}">{per_model[k]["label"].split(" ")[0]}·{k.split("_")[-1]}</a>'
                 for k in DISPLAY) + '</nav>')

hof_html = ('<h2 id="hof">Hall of fame — top predictions across the study</h2>'
            '<p class="sub">Highest per-case Dice among all full both-parotid segmentations '
            '(nnU-Net + ablation models + P0 baselines). One row, best first.</p><div class="row">')
for r in hof_rows:
    hof_html += (f'<div class="cell"><a href="{r["file"]}"><img src="{r["file"]}" loading="lazy">'
                 f'</a><div class="cap"><span class="rank">#{r["rank"]}</span> '
                 f'{r["label"]}<br>{r["case"]} · Dice {r["dice"]:.3f}</div></div>')
hof_html += "</div>"

sections = ""
for key in DISPLAY:
    pm = per_model[key]
    dtxt = f'agg <span class="d">{pm["dice"]}</span>' if pm["dice"] else "single-side"
    sections += (f'<div class="model" id="{key}"><h3>{pm["label"]} &nbsp;'
                 f'<span class="sub">{dtxt} — top {len(pm["rows"])} cases — {pm["note"]}</span></h3>'
                 f'<div class="row">{cells(key, pm["rows"], key)}</div></div>')

html = (f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Parotid gallery — best-of showcase</title><style>{CSS}</style></head><body>"
        f"<h1>Best-of showcase</h1>"
        f"<p class='sub'>Each model's strongest test-set predictions, plus an overall leaderboard. "
        f"R=red, L=blue, GT=cream dashed. Full gallery: "
        f"<a href='../index.html'>../index.html</a>.</p>{nav}{hof_html}"
        f"<h2>Each model's best {TOPN_PER_MODEL} cases</h2>{sections}</body></html>")
with open(os.path.join(SHOW, "showcase.html"), "w") as f:
    f.write(html)

json.dump({"per_model": per_model, "hall_of_fame": hof_rows},
          open(os.path.join(SHOW, "_showcase.json"), "w"), indent=1)
n_copied = sum(len(os.listdir(os.path.join(SHOW, d))) for d in os.listdir(SHOW)
               if os.path.isdir(os.path.join(SHOW, d)))
print(f"showcase: {len(DISPLAY)} models x top {TOPN_PER_MODEL} + HoF {len(hof_rows)}; "
      f"{n_copied} files copied. Wrote showcase.html")
print("HoF top 5:")
for r in hof_rows[:5]:
    print(f"  #{r['rank']} {r['label']:30s} {r['case']}  Dice {r['dice']:.3f}")
