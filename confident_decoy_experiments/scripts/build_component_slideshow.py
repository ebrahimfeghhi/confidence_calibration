#!/usr/bin/env python
"""Per-interface DockQ-vs-<component> slideshows for the three CAPRI components of DockQ.

fnat / iRMSD / LRMSD are COMPONENTS of DockQ, so an association with DockQ is partly definitional --
unlike CDR H3 RMSD and epitope RMSD, which are independent structural measurements. These pages are
therefore diagnostics of WHICH component carries an interface's variation, not evidence that the
component causes good docking. The informative comparison is between components: fnat explains 87% of
the variance in scaling gain versus 47% (iRMSD) and 36% (LRMSD).

Usage: build_component_slideshow.py fnat|irmsd|lrmsd
"""
import base64, collections, csv, glob, html, json, os, re, sys
import numpy as np
from scipy import stats as sps
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator, FuncFormatter

M = sys.argv[1]
CFG = {
    "fnat":  dict(idx=1, label="fnat (fraction of native contacts)", good_is_low=False,
                  thr=0.30, xmax=1.02, colour="#3b5f9e", hist="#7d95c9",
                  band="fnat ≥ 0.30", title="fnat",
                  yscale="linear", yticks=[0, 0.2, 0.4, 0.6, 0.8], ylim=(-0.03, 0.95)),
    "irmsd": dict(idx=2, label="interface RMSD (Å)", good_is_low=True,
                  thr=2.0, xmax=None, colour="#b06a2c", hist="#d9a771",
                  band="iRMSD ≤ 2 Å", title="iRMSD",
                  yscale="log", yticks=[0.05, 0.1, 0.5, 1, 2, 5, 10, 20], ylim=(0.025, 30)),
    "lrmsd": dict(idx=3, label="ligand RMSD (Å)", good_is_low=True,
                  thr=5.0, xmax=None, colour="#8a4f86", hist="#bd93ba",
                  band="LRMSD ≤ 5 Å", title="LRMSD",
                  yscale="log", yticks=[0.2, 0.5, 1, 2, 5, 10, 20, 50, 100], ylim=(0.15, 130)),
}[M]
OUT = f"artifacts/{M}_panels"; os.makedirs(OUT, exist_ok=True)
KEEP = set(l.strip() for l in open("artifacts/complexes_104.txt") if l.strip())
PRED = "/home/ebrahim/Protenix/foldbench_abag/predictions_template"
ifm = collections.defaultdict(list)
for r in csv.DictReader(open("/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv")):
    ifm[r["pdb_id"].split("-")[0]].append((r["antibody_chain"], r["antigen_chain"]))

data = collections.defaultdict(list)
for f in glob.glob("artifacts/abag_all_ost/*.json"):
    m = re.match(r"([0-9a-z]{4})_(\d+)_(\d+)\.json$", f.split("/")[-1])
    if not m: continue
    p, s, k = m.group(1), int(m.group(2)), int(m.group(3))
    if p not in KEEP or not (101 <= s <= 150): continue
    try:
        j = json.load(open(f))
        MM = json.load(open(f"{PRED}/{p}/seed_{s}/predictions/{p}_summary_confidence_sample_{k}.json"))["chain_pair_iptm"]
    except Exception: continue
    q = j.get("dockq") or []; comp = j.get(M) or []; ifs = j.get("dockq_interfaces") or []
    for c1, c2 in ifm.get(p, []):
        for i, t in enumerate(ifs):
            if i < len(q) and i < len(comp) and {t[0].split("-")[0], t[1].split("-")[0]} == {c1, c2}:
                a, b = ord(t[2]) - 65, ord(t[3]) - 65
                data[f"{p}_{c1}{c2}"].append((comp[i], q[i],
                                              MM[a][b] if (0 <= a < len(MM) and 0 <= b < len(MM)) else -1))
                break

meta, corr = [], []
for key in sorted(data):
    v = np.array(data[key])
    if len(v) < 100: continue
    sel = v[np.argmax(v[:, 2])]; orc = float(v[:, 1].max())
    status = "solved" if sel[1] >= 0.23 else ("confident decoy" if orc >= 0.23 else "no acceptable pose")
    xmax = CFG["xmax"] or float(min(30.0, max(3.0, np.percentile(v[:, 0], 98) * 1.25)))
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    if CFG["good_is_low"]:
        ax.axvspan(0, CFG["thr"], color="#f7c9de", alpha=.5, zorder=0)
        ax.add_patch(plt.Rectangle((CFG["thr"], 0.80), xmax, .25, color="#bfe3d0", alpha=.5, zorder=0))
    else:
        ax.axvspan(CFG["thr"], xmax, color="#f7c9de", alpha=.35, zorder=0)
    ax.scatter(v[:, 0], v[:, 1], s=11, alpha=.5, c=CFG["colour"], edgecolors="none", zorder=2)
    ax.scatter([sel[0]], [sel[1]], marker="*", s=190, c="#e8b21a", edgecolors="#333",
               linewidths=.7, zorder=5)
    for t in (0.23, 0.49, 0.80):
        ax.axhline(t, ls="--", lw=.8, color="#333", zorder=3)
    ax.axvline(CFG["thr"], lw=.8, color="#333", zorder=3)
    ax.set_xlim(0, xmax); ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel(CFG["label"], fontsize=8); ax.set_ylabel("DockQ", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f"{key}   sel {sel[1]:.2f} / oracle {orc:.2f}", fontsize=8.5)
    fig.tight_layout(pad=0.4)
    png = f"{OUT}/{key}.png"; fig.savefig(png, dpi=95); plt.close(fig)
    x, y = v[:, 0], v[:, 1]
    r = (sps.pearsonr(x, y)[0] if len(set(np.round(x, 4))) > 2 and len(set(np.round(y, 4))) > 2 else np.nan)
    frac = float((x >= CFG["thr"]).mean()) if not CFG["good_is_low"] else float((x <= CFG["thr"]).mean())
    meta.append(dict(k=key, n=len(v), sel=round(float(sel[1]), 3), orc=round(orc, 3), status=status,
                     frac_h3_sub1=round(frac, 3), med_h3=round(float(np.median(x)), 2),
                     b64=base64.b64encode(open(png, "rb").read()).decode()))
    if not np.isnan(r):
        corr.append((key, r, float(np.subtract(*np.percentile(x, [75, 25]))), status))

# ---- summary scatter ----
COL = {"solved": "#1d6b45", "confident decoy": "#d09a2c", "no acceptable pose": "#b3543f"}
fig = plt.figure(figsize=(8.6, 6.6))
gs = GridSpec(2, 1, height_ratios=[1, 4.3], hspace=0.30)
axt = fig.add_subplot(gs[0]); ax = fig.add_subplot(gs[1], sharex=axt)
rs = [c[1] for c in corr]
axt.hist(rs, bins=34, color=CFG["hist"], edgecolor="white", linewidth=.5)
axt.axvline(np.median(rs), color="#b3543f", lw=1.6)
axt.set_yticks([]); axt.tick_params(labelbottom=False)
for sp in axt.spines.values(): sp.set_visible(False)
axt.set_title(f"Within-interface correlation of {CFG['title']} with DockQ\n"
              f"n = {len(corr)} interfaces, 250 poses each   median r = {np.median(rs):+.2f}", fontsize=12)
for st in ("solved", "confident decoy", "no acceptable pose"):
    sub = [c for c in corr if c[3] == st]
    if sub:
        ys = [c[2] if CFG["yscale"] == "linear" else max(c[2], CFG["ylim"][0] * 1.15) for c in sub]
        ax.scatter([c[1] for c in sub], ys, s=38, alpha=.85,
                   c=COL[st], edgecolors="#2b2b2b", linewidths=.4, label=f"{st} ({len(sub)})")
ax.axvline(0, color="#333", lw=1.1)
ax.set_yscale(CFG["yscale"])
ax.yaxis.set_major_locator(FixedLocator(CFG["yticks"]))
ax.yaxis.set_minor_locator(FixedLocator([]))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_ylim(*CFG["ylim"])
ax.legend(fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.005), ncol=3, frameon=False,
          handletextpad=.4, columnspacing=1.6, borderpad=0)
ax.set_xlim(-1.02, 1.02)
ax.set_xlabel(f"Pearson r between {CFG['title']} and DockQ, within interface", fontsize=11)
ax.set_ylabel(f"{CFG['title']} spread (IQR)", fontsize=11)
fig.text(0.5, 0.030, f"{CFG['title']} is a COMPONENT of DockQ, so this association is partly definitional.",
         ha="center", va="bottom", fontsize=8.5, color="#5a5a5a")
fig.text(0.5, 0.008, "Status is computed from these 250 poses only, not from deeper rescoring.",
         ha="center", va="bottom", fontsize=8.5, color="#5a5a5a")
fig.subplots_adjust(left=0.095, right=0.985, top=0.885, bottom=0.155)
fig.savefig(f"artifacts/fig_{M}_corr_summary.png", dpi=165); plt.close(fig)
summ = dict(b64=base64.b64encode(open(f"artifacts/fig_{M}_corr_summary.png", "rb").read()).decode(),
            n=len(corr), median=round(float(np.median(rs)), 3),
            neg=sum(1 for r in rs if r < 0), strong=sum(1 for r in rs if abs(r) > 0.5))
json.dump(meta, open(f"{OUT}/meta.json", "w")); json.dump(summ, open(f"{OUT}/summary.json", "w"))
print(f"{M}: {len(meta)} panels | median r {summ['median']:+.2f} | "
      f"negative {summ['neg']}/{len(corr)} | |r|>0.5 {summ['strong']}/{len(corr)}")
