#!/usr/bin/env python
"""Per-interface panels + summary: fnat vs CDR H3 local RMSD, across the 250 poses of each interface.

WHY THIS PAIR: fnat and DockQ are not independent (fnat is a DockQ component), so correlating them
says little. fnat vs CDR H3 IS a fair comparison -- fnat is a contact measure scored against the
native, H3 local RMSD is a fold measure computed from coordinates alone. Neither enters the other.

The question: within one target, do the poses that build a better CDR H3 loop also recover more of
the true epitope? If yes, loop quality and contact recovery are coupled; if not, the model can place
the antibody anywhere regardless of how well the loop is built.

x = CDR H3 local RMSD (A, lower better), y = fnat (0-1, higher better). A negative correlation means
better loop goes with better contacts. Star = the pose ipTM selects.
"""
import base64, collections, csv, glob, json, os, re
import numpy as np
from scipy import stats as sps
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator, FuncFormatter

OUT = "artifacts/fnat_h3_panels"; os.makedirs(OUT, exist_ok=True)
KEEP = set(l.strip() for l in open("artifacts/complexes_104.txt") if l.strip())
PRED = "/home/ebrahim/Protenix/foldbench_abag/predictions_template"
ifm = collections.defaultdict(list)
for r in csv.DictReader(open("/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv")):
    ifm[r["pdb_id"].split("-")[0]].append((r["antibody_chain"], r["antigen_chain"]))
h3 = {}
for r in csv.DictReader(open("artifacts/cdr_h3_rmsd_ost.csv")):
    s = int(r["seed"])
    if r["pdb"] in KEEP and 101 <= s <= 150:
        h3[(r["pdb"], s, int(r["sample"]))] = float(r["h3_local"])

data = collections.defaultdict(list)
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
    except Exception: continue
    q, fn, ifs = j.get("dockq") or [], j.get("fnat") or [], j.get("dockq_interfaces") or []
    for c1, c2 in ifm.get(p, []):
        for i, t in enumerate(ifs):
            if i < len(q) and i < len(fn) and {t[0].split("-")[0], t[1].split("-")[0]} == {c1, c2}:
                a, b = ord(t[2]) - 65, ord(t[3]) - 65
                data[f"{p}_{c1}{c2}"].append((hv, fn[i], q[i],
                                              M[a][b] if (0 <= a < len(M) and 0 <= b < len(M)) else -1))
                break

meta, corr = [], []
for key in sorted(data):
    v = np.array(data[key])
    if len(v) < 100: continue
    sel = v[np.argmax(v[:, 3])]
    orc = float(v[:, 2].max())
    status = "solved" if sel[2] >= 0.23 else ("confident decoy" if orc >= 0.23 else "no acceptable pose")
    xmax = float(min(11.0, max(1.5, np.percentile(v[:, 0], 99) * 1.25)))
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.axvspan(0, 1.0, color="#f7c9de", alpha=.5, zorder=0)
    ax.axhspan(0.30, 1.02, color="#bfe3d0", alpha=.35, zorder=0)
    ax.scatter(v[:, 0], v[:, 1], s=11, alpha=.5, c="#7a4fa3", edgecolors="none", zorder=2)
    ax.scatter([sel[0]], [sel[1]], marker="*", s=190, c="#e8b21a", edgecolors="#333",
               linewidths=.7, zorder=5)
    ax.axhline(0.30, ls="--", lw=.8, color="#333", zorder=3)
    ax.axvline(1.0, lw=.8, color="#333", zorder=3)
    ax.set_xlim(0, xmax); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("CDR H3 local RMSD (Å)", fontsize=8); ax.set_ylabel("fnat", fontsize=8)
    ax.tick_params(labelsize=7)
    x, y = v[:, 0], v[:, 1]
    r = (sps.pearsonr(x, y)[0] if len(set(np.round(x, 4))) > 2 and len(set(np.round(y, 4))) > 2 else np.nan)
    ax.set_title(f"{key}   r={r:+.2f}" if not np.isnan(r) else key, fontsize=8.5)
    fig.tight_layout(pad=0.4)
    png = f"{OUT}/{key}.png"; fig.savefig(png, dpi=95); plt.close(fig)
    meta.append(dict(k=key, n=len(v), sel=round(float(sel[2]), 3), orc=round(orc, 3), status=status,
                     frac_h3_sub1=round(float((x <= 1.0).mean()), 3),
                     med_h3=round(float(np.median(x)), 2),
                     b64=base64.b64encode(open(png, "rb").read()).decode()))
    if not np.isnan(r):
        corr.append((key, r, float(np.subtract(*np.percentile(x, [75, 25]))), status))

COL = {"solved": "#1d6b45", "confident decoy": "#d09a2c", "no acceptable pose": "#b3543f"}
fig = plt.figure(figsize=(8.6, 6.6))
gs = GridSpec(2, 1, height_ratios=[1, 4.3], hspace=0.30)
axt = fig.add_subplot(gs[0]); ax = fig.add_subplot(gs[1], sharex=axt)
rs = [c[1] for c in corr]
axt.hist(rs, bins=34, color="#b49bd0", edgecolor="white", linewidth=.5)
axt.axvline(np.median(rs), color="#b3543f", lw=1.6)
axt.set_yticks([]); axt.tick_params(labelbottom=False)
for sp in axt.spines.values(): sp.set_visible(False)
axt.set_title("Within-interface correlation of CDR H3 local RMSD with fnat\n"
              f"n = {len(corr)} interfaces, 250 poses each   median r = {np.median(rs):+.2f}", fontsize=12)
for st_ in ("solved", "confident decoy", "no acceptable pose"):
    sub = [c for c in corr if c[3] == st_]
    if sub:
        ax.scatter([c[1] for c in sub], [max(c[2], 0.012) for c in sub], s=38, alpha=.85,
                   c=COL[st_], edgecolors="#2b2b2b", linewidths=.4, label=f"{st_} ({len(sub)})")
ax.axvline(0, color="#333", lw=1.1); ax.axhline(0.10, color="#888", ls=":", lw=1.1)
ax.set_yscale("log")
tk = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
ax.yaxis.set_major_locator(FixedLocator(tk)); ax.yaxis.set_minor_locator(FixedLocator([]))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.legend(fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.005), ncol=3, frameon=False,
          handletextpad=.4, columnspacing=1.6, borderpad=0)
ax.set_xlim(-1.02, 1.02); ax.set_ylim(0.012, 3.4)
ax.set_xlabel("Pearson r between CDR H3 local RMSD and fnat, within interface", fontsize=11)
ax.set_ylabel("H3 local RMSD spread (IQR, Å)", fontsize=11)
fig.text(0.5, 0.030, "Negative r = better loop goes with more native contacts. Dotted line: below it the loop barely varies, so r is mostly noise.",
         ha="center", va="bottom", fontsize=8.5, color="#5a5a5a")
fig.text(0.5, 0.008, "fnat and H3 are independent measurements — neither is used to compute the other.",
         ha="center", va="bottom", fontsize=8.5, color="#5a5a5a")
fig.subplots_adjust(left=0.095, right=0.985, top=0.885, bottom=0.155)
fig.savefig("artifacts/fig_fnat_h3_corr_summary.png", dpi=165); plt.close(fig)
json.dump(meta, open(f"{OUT}/meta.json", "w"))
json.dump(dict(b64=base64.b64encode(open("artifacts/fig_fnat_h3_corr_summary.png","rb").read()).decode(),
               n=len(corr), median=round(float(np.median(rs)),3),
               neg=sum(1 for r in rs if r<0), strong=sum(1 for r in rs if abs(r)>0.5)),
          open(f"{OUT}/summary.json","w"))
print(f"{len(meta)} panels | median r {np.median(rs):+.3f} | negative {sum(1 for r in rs if r<0)}/{len(rs)} | |r|>0.5 {sum(1 for r in rs if abs(r)>0.5)}/{len(rs)}")
