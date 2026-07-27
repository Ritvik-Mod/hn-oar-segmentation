"""Build index.html for the E8 prediction gallery from the manifests."""
import os, json
import galcommon as g

OUT = g.OUT
man = json.load(open(os.path.join(OUT, "manifest.json")))
p0man_path = os.path.join(OUT, "manifest_p0.json")
if os.path.exists(p0man_path):
    man.update(json.load(open(p0man_path)))
comp = json.load(open(os.path.join(OUT, "_comparison.json")))

NII_ORDER = list(reversed(g.ORDER))  # best -> worst for display
P0_ORDER = ["P0_attention", "P0_unet", "P0_transunet", "P0_swin"]

CSS = """
body{font-family:-apple-system,Segoe UI,sans-serif;max-width:1300px;margin:auto;
 padding:24px;color:#1c1c1e;background:#fafafa}
h1{margin-bottom:4px} h2{border-bottom:2px solid #ddd;padding-bottom:6px;margin-top:40px}
h3{margin:6px 0} .sub{color:#666;font-size:14px}
.legend{background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:12px 16px;
 display:inline-block;margin:8px 0;font-size:14px}
.chip{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:middle;margin-right:5px}
nav{position:sticky;top:0;background:#fafafaee;padding:10px 0;backdrop-filter:blur(4px);
 border-bottom:1px solid #ddd;margin-bottom:20px;font-size:14px}
nav a{margin-right:14px;color:#0a63c9;text-decoration:none}
.model{background:#fff;border:1px solid #e6e6e6;border-radius:10px;padding:16px;margin:18px 0}
.model .dice{font-weight:700;color:#0a63c9}
.grid{display:flex;flex-wrap:wrap;gap:4px}
.cell{width:180px} .cell img{width:100%;border:1px solid #ddd;border-radius:5px}
.cell .cap{font-size:11px;color:#444;text-align:center}
details>summary{cursor:pointer;font-weight:600;margin:8px 0;color:#0a63c9}
.big img{width:100%;border:1px solid #ccc;border-radius:8px;margin:6px 0}
.tag{font-size:11px;background:#eef;border-radius:4px;padding:1px 6px;color:#334}
table{border-collapse:collapse;font-size:13px} td,th{border:1px solid #ddd;padding:4px 9px}
"""


def dice_sorted(cases):
    return sorted(cases, key=lambda r: r["dice"], reverse=True)


def model_section(key):
    m = man[key]
    d = f"agg Dice <span class='dice'>{m['dice']}</span>" if m["dice"] else "single-side"
    rows = dice_sorted(m["cases"])
    cells = "".join(
        f'<div class="cell"><a href="{key}/{r["case"]}_slices.png">'
        f'<img src="{key}/{r["case"]}_slices.png" loading="lazy"></a>'
        f'<div class="cap">{r["case"]} · Dice {r["dice"]:.3f} · '
        f'<a href="{key}/{r["case"]}_3d.png">3D</a></div></div>'
        for r in rows)
    return (f'<div class="model" id="{key}"><h3>{m["label"]} &nbsp;<span class="sub">'
            f'{d} — {m["note"]}</span></h3>'
            f'<details open><summary>{len(rows)} cases (sorted best→worst; '
            f'click a montage to enlarge, "3D" for the surface render)</summary>'
            f'<div class="grid">{cells}</div></details></div>')


# comparison section
comp_html = '<h2 id="comparison">Same-case-across-models comparison grids</h2>'
comp_html += ('<p class="sub">The most useful view: one axial slice per case, every '
              'model side by side, GT dashed in each. Filled = prediction, dashed '
              'contour = ground truth.</p>')
for r in comp:
    comp_html += (f'<div class="big"><h3>{r["case"]} '
                  f'<span class="tag">{r["tag"]}</span> '
                  f'<span class="sub">nnU-Net Dice {r["ref_dice"]:.3f}, axial z={r["z"]}'
                  f'</span></h3>'
                  f'<a href="comparison/case_{r["case"]}_allmodels.png">'
                  f'<img src="comparison/case_{r["case"]}_allmodels.png"></a>')
    p0grid = f'comparison/case_{r["case"]}_p0baselines.png'
    if os.path.exists(os.path.join(OUT, p0grid)):
        comp_html += f'<a href="{p0grid}"><img src="{p0grid}"></a>'
    comp_html += '</div>'

# nav + legend
present_nii = [k for k in NII_ORDER if k in man]
present_p0 = [k for k in P0_ORDER if k in man]
nav = ('<nav><a href="showcase/showcase.html"><b>★ Best-of showcase</b></a>'
       '<a href="#comparison">Comparison grids</a>'
       '<a href="#primary">Primary models (.nii.gz)</a>'
       + ("".join(f'<a href="#{k}">{man[k]["label"].split(" ")[0]}</a>'
                  for k in present_nii))
       + ('<a href="#baselines">P0 baselines</a>' if present_p0 else '')
       + '</nav>')

legend = ('<div class="legend"><span class="chip" style="background:#ff3b30"></span>'
          'Right parotid (pred fill + solid contour) &nbsp;&nbsp;'
          '<span class="chip" style="background:#0a84ff"></span>Left parotid &nbsp;&nbsp;'
          '<span style="border:1px dashed #333;padding:0 6px">– –</span> Ground truth'
          ' &nbsp;·&nbsp; soft-tissue window (WL 40 / WW 400) · dpi 200+</div>')

# results mini-table
trows = "".join(
    f'<tr><td><a href="#{k}">{man[k]["label"]}</a></td><td>{man[k]["dice"] or "—"}</td>'
    f'<td>{man[k]["note"]}</td></tr>' for k in present_nii + present_p0)
table = (f'<table><tr><th>Model</th><th>Agg 3D Dice</th><th>Note</th></tr>{trows}</table>')

primary = ('<h2 id="primary">Primary models — nnU-Net-style .nii.gz (shared geometry)</h2>'
           + "".join(model_section(k) for k in present_nii))
baselines = ''
if present_p0:
    baselines = ('<h2 id="baselines">Secondary — P0 from-scratch 2D baselines</h2>'
                 '<p class="sub">Verified identical geometry to Dataset002 imagesTs '
                 '(npz GT matches labelsTs exactly, Dice 1.000), so overlaid on the '
                 'same CT.</p>'
                 + "".join(model_section(k) for k in present_p0))

html = (f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>E8 — Parotid prediction gallery</title><style>{CSS}</style></head><body>"
        f"<h1>E8 — Test-set prediction gallery</h1>"
        f"<p class='sub'>Parotid-gland segmentation · locked 43-case both-parotid test set · "
        f"pure rendering of existing predictions. R=red, L=blue, GT=dashed.</p>"
        f"{legend}{nav}{table}{comp_html}{primary}{baselines}"
        f"<p class='sub' style='margin-top:40px'>Generated by "
        f"ablation_study/prediction_gallery/build_index.py · see README.md.</p>"
        f"</body></html>")

with open(os.path.join(OUT, "index.html"), "w") as f:
    f.write(html)
print(f"Wrote index.html ({len(present_nii)} primary + {len(present_p0)} P0 models, "
      f"{len(comp)} comparison grids)")
