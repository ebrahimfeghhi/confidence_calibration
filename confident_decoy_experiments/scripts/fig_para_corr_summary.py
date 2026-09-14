#!/usr/bin/env python
"""Summary scatter: within-interface correlation between paratope local RMSD and DockQ.

Antigen-side counterpart of fig_h3_corr_summary.py. One point per interface; x = Pearson r across
that interface's 250 poses; y = the interface's epitope-RMSD spread, because r is only interpretable
where the quantity varies.
"""
import csv, glob, json, re, collections, base64
import numpy as np
from scipy import stats as sps
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator, FuncFormatter

KEEP = set(l.strip() for l in open("artifacts/complexes_104.txt") if l.strip())
PRED = "/home/ebrahim/Protenix/foldbench_abag/predictions_template"
ifm = collections.defaultdict(list)
for r in csv.DictReader(open("/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv")):
    ifm[r["pdb_id"].split("-")[0]].append((r["antibody_chain"], r["antigen_chain"]))
epi = {}
for r in csv.DictReader(open("artifacts/paratope_rmsd.csv")):
    s = int(r["seed"])
    if 101 <= s <= 150:
        epi[(r["pdb"], r["ab_chain"], r["ag_chain"], s, int(r["sample"]))] = float(r["para_local"])
data = collections.defaultdict(list)
for f in glob.glob("artifacts/abag_all_ost/*.json"):
    m = re.match(r"([0-9a-z]{4})_(\d+)_(\d+)\.json$", f.split("/")[-1])
    if not m: continue
    p, s, k = m.group(1), int(m.group(2)), int(m.group(3))
    if p not in KEEP or not (101 <= s <= 150): continue
    try:
        j = json.load(open(f))
        M = json.load(open(f"{PRED}/{p}/seed_{s}/predictions/{p}_summary_confidence_sample_{k}.json"))["chain_pair_iptm"]
    except Exception: continue
    q, ifs = j.get("dockq") or [], j.get("dockq_interfaces") or []
    for c1, c2 in ifm.get(p, []):
        ev = epi.get((p, c1, c2, s, k))
        if ev is None: continue
        for i, t in enumerate(ifs):
            if i < len(q) and {t[0].split("-")[0], t[1].split("-")[0]} == {c1, c2}:
                a, b = ord(t[2]) - 65, ord(t[3]) - 65
                data[f"{p}_{c1}{c2}"].append((ev, q[i], M[a][b] if (0<=a<len(M) and 0<=b<len(M)) else -1))
                break
rows = []
for key, v in data.items():
    v = np.array(v)
    if len(v) < 100: continue
    x, y = v[:, 0], v[:, 1]
    r = (sps.pearsonr(x, y)[0] if len(set(np.round(x, 4))) > 2 and len(set(np.round(y, 4))) > 2
         else np.nan)
    sel = float(v[np.argmax(v[:, 2]), 1]); orc = float(y.max())
    st = "solved" if sel >= 0.23 else ("confident decoy" if orc >= 0.23 else "no acceptable pose")
    rows.append((key, r, float(np.subtract(*np.percentile(x, [75, 25]))), st))
ok = [q for q in rows if not np.isnan(q[1])]
COL = {"solved": "#1d6b45", "confident decoy": "#d09a2c", "no acceptable pose": "#b3543f"}
fig = plt.figure(figsize=(8.6, 6.6))
gs = GridSpec(2, 1, height_ratios=[1, 4.3], hspace=0.30)
axt = fig.add_subplot(gs[0]); ax = fig.add_subplot(gs[1], sharex=axt)
rs = [q[1] for q in ok]
axt.hist(rs, bins=34, color="#bd93ba", edgecolor="white", linewidth=.5)
axt.axvline(np.median(rs), color="#b3543f", lw=1.6)
axt.set_yticks([]); axt.tick_params(labelbottom=False)
for sp in axt.spines.values(): sp.set_visible(False)
axt.set_title(f"Within-interface correlation of paratope local RMSD with DockQ\n"
              f"n = {len(ok)} interfaces, 250 poses each   median r = {np.median(rs):+.2f}", fontsize=12)
for st in ("solved", "confident decoy", "no acceptable pose"):
    sub = [q for q in ok if q[3] == st]
    ax.scatter([q[1] for q in sub], [max(q[2], 0.004) for q in sub], s=38, alpha=.85,
               c=COL[st], edgecolors="#2b2b2b", linewidths=.4, label=f"{st} ({len(sub)})")
ax.axvline(0, color="#333", lw=1.1)
ax.axhline(0.10, color="#888", ls=":", lw=1.1)
ax.set_yscale("log")
ticks = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_locator(FixedLocator([]))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.legend(fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.005), ncol=3,
          frameon=False, handletextpad=.4, columnspacing=1.6, borderpad=0)
ax.set_xlim(-1.02, 1.02)
ax.set_xlabel("Pearson r between paratope local RMSD and DockQ, within interface", fontsize=11)
ax.set_ylabel("paratope local RMSD spread (IQR, Å)", fontsize=11)
fig.text(0.5, 0.030, "Dotted line: below it the paratope barely varies across poses, so r there is mostly noise.",
         ha="center", va="bottom", fontsize=8.5, color="#5a5a5a")
fig.text(0.5, 0.008, "Status is computed from these 250 poses only, not from deeper rescoring.",
         ha="center", va="bottom", fontsize=8.5, color="#5a5a5a")
fig.subplots_adjust(left=0.095, right=0.985, top=0.885, bottom=0.155)
fig.savefig("artifacts/fig_para_corr_summary.png", dpi=165)
neg = sum(1 for q in ok if q[1] < 0); strong = sum(1 for q in ok if abs(q[1]) > 0.5)
print(f"n={len(ok)} median r={np.median(rs):+.3f} negative {neg}/{len(ok)} |r|>0.5 {strong}/{len(ok)}")
json.dump({"b64": base64.b64encode(open("artifacts/fig_para_corr_summary.png","rb").read()).decode(),
           "n": len(ok), "median": round(float(np.median(rs)),3), "neg": neg, "strong": strong},
          open("artifacts/para_panels/summary.json","w"))
