#!/usr/bin/env python
"""One DockQ-vs-CDR-H3-RMSD panel per interface, for a slideshow.

Same layout as the pooled figure (CAPRI bands at 0.23/0.49/0.80, the 1 A line), but one panel per
interface so the per-target structure is visible: whether the <1 A column is populated at low DockQ
(good loop, wrong dock) and where the ipTM-selected pose lands.

Points are the 250 poses of that interface. The star marks the pose ipTM actually selects.
Small canvas on purpose -- 161 panels get embedded as data URIs in one HTML page.
"""
import csv, glob, json, re, collections, os, base64
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "artifacts/h3_panels"; os.makedirs(OUT, exist_ok=True)
KEEP = set(l.strip() for l in open("artifacts/complexes_104.txt") if l.strip())
PRED = "/home/ebrahim/Protenix/foldbench_abag/predictions_template"
ifm = collections.defaultdict(list)
for r in csv.DictReader(open("/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv")):
    ifm[r["pdb_id"].split("-")[0]].append((r["antibody_chain"], r["antigen_chain"]))
h3 = {}
for r in csv.DictReader(open("artifacts/cdr_h3_rmsd_ost.csv")):
    s = int(r["seed"])
    if r["pdb"] in KEEP and 101 <= s <= 150:
        h3[(r["pdb"], s, int(r["sample"]))] = (float(r["h3_local"]), float(r["h3_global"]))

data = collections.defaultdict(list)     # interface -> (h3_local, h3_global, dockq, iptm)
for f in glob.glob("artifacts/abag_all_ost/*.json"):
    m = re.match(r"([0-9a-z]{4})_(\d+)_(\d+)\.json$", f.split("/")[-1])
    if not m: continue
    p, s, k = m.group(1), int(m.group(2)), int(m.group(3))
    if p not in KEEP or not (101 <= s <= 150): continue
    hv = h3.get((p, s, k))
    if hv is None: continue
    try:
        j = json.load(open(f))
        M = json.load(open(f"{PRED}/{p}/seed_{s}/predictions/{p}_summary_confidence_sample_{k}.json"))["chain_pair_iptm"]
    except Exception:
        continue
    q, ifs = j.get("dockq") or [], j.get("dockq_interfaces") or []
    for c1, c2 in ifm.get(p, []):
        for i, t in enumerate(ifs):
            if i < len(q) and {t[0].split("-")[0], t[1].split("-")[0]} == {c1, c2}:
                a, b = ord(t[2]) - 65, ord(t[3]) - 65
                ip = M[a][b] if (0 <= a < len(M) and 0 <= b < len(M)) else -1
                data[f"{p}_{c1}{c2}"].append((hv[0], hv[1], q[i], ip))
                break

meta = []
for key in sorted(data):
    v = np.array(data[key])
    if len(v) < 100: continue
    sel = v[np.argmax(v[:, 3])]
    orc = float(v[:, 2].max())
    status = "solved" if sel[2] >= 0.23 else ("confident decoy" if orc >= 0.23 else "no acceptable pose")
    xmax = float(min(11.0, max(2.0, np.percentile(v[:, 0], 99) * 1.25)))
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.axvspan(0, 1.0, color="#f7c9de", alpha=.5, zorder=0)
    ax.add_patch(plt.Rectangle((1.0, 0.80), xmax, .25, color="#bfe3d0", alpha=.5, zorder=0))
    ax.scatter(v[:, 0], v[:, 2], s=11, alpha=.5, c="#3b5f9e", edgecolors="none", zorder=2)
    ax.scatter([sel[0]], [sel[2]], marker="*", s=190, c="#e8b21a",
               edgecolors="#333", linewidths=.7, zorder=5)
    for t in (0.23, 0.49, 0.80):
        ax.axhline(t, ls="--", lw=.8, color="#333", zorder=3)
    ax.axvline(1.0, lw=.8, color="#333", zorder=3)
    ax.set_xlim(0, xmax); ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel("CDR H3 local RMSD (Å)", fontsize=8)
    ax.set_ylabel("DockQ", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f"{key}   sel {sel[2]:.2f} / oracle {orc:.2f}", fontsize=8.5)
    fig.tight_layout(pad=0.4)
    png = f"{OUT}/{key}.png"
    fig.savefig(png, dpi=95); plt.close(fig)
    frac1 = float((v[:, 0] <= 1.0).mean())
    meta.append(dict(k=key, n=len(v), sel=round(float(sel[2]), 3), orc=round(orc, 3),
                     status=status, frac_h3_sub1=round(frac1, 3),
                     med_h3=round(float(np.median(v[:, 0])), 2),
                     b64=base64.b64encode(open(png, "rb").read()).decode()))
json.dump(meta, open("artifacts/h3_panels/meta.json", "w"))
tot = sum(len(m["b64"]) for m in meta) / 1e6
print(f"{len(meta)} panels, embedded payload {tot:.1f} MB")
