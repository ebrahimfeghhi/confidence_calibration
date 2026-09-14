#!/usr/bin/env python
"""One DockQ-vs-epitope-RMSD panel per interface, mirroring the CDR H3 slideshow.

Epitope RMSD is the antigen-side counterpart of CDR H3 RMSD: the native epitope (antigen residues
within 5 A of the antibody IN THE GROUND TRUTH -- a fixed residue set per interface) superposed onto
itself, so it measures how well that surface patch is BUILT, independent of where the antibody went.

Unlike H3, epitope RMSD is interface-specific rather than complex-specific, so there is no duplication
across the two interfaces of a complex.
"""
import csv, glob, json, re, collections, os, base64
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "artifacts/para_panels"; os.makedirs(OUT, exist_ok=True)
KEEP = set(l.strip() for l in open("artifacts/complexes_104.txt") if l.strip())
PRED = "/home/ebrahim/Protenix/foldbench_abag/predictions_template"
ifm = collections.defaultdict(list)
for r in csv.DictReader(open("/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv")):
    ifm[r["pdb_id"].split("-")[0]].append((r["antibody_chain"], r["antigen_chain"]))

epi = {}
for r in csv.DictReader(open("artifacts/paratope_rmsd.csv")):
    s = int(r["seed"])
    if 101 <= s <= 150:
        epi[(r["pdb"], r["ab_chain"], r["ag_chain"], s, int(r["sample"]))] = \
            (float(r["para_local"]), float(r["para_global"]))

data = collections.defaultdict(list)
for f in glob.glob("artifacts/abag_all_ost/*.json"):
    m = re.match(r"([0-9a-z]{4})_(\d+)_(\d+)\.json$", f.split("/")[-1])
    if not m: continue
    p, s, k = m.group(1), int(m.group(2)), int(m.group(3))
    if p not in KEEP or not (101 <= s <= 150): continue
    try:
        j = json.load(open(f))
        M = json.load(open(f"{PRED}/{p}/seed_{s}/predictions/{p}_summary_confidence_sample_{k}.json"))["chain_pair_iptm"]
    except Exception:
        continue
    q, ifs = j.get("dockq") or [], j.get("dockq_interfaces") or []
    for c1, c2 in ifm.get(p, []):
        ev = epi.get((p, c1, c2, s, k))
        if ev is None: continue
        for i, t in enumerate(ifs):
            if i < len(q) and {t[0].split("-")[0], t[1].split("-")[0]} == {c1, c2}:
                a, b = ord(t[2]) - 65, ord(t[3]) - 65
                data[f"{p}_{c1}{c2}"].append((ev[0], ev[1], q[i],
                                              M[a][b] if (0 <= a < len(M) and 0 <= b < len(M)) else -1))
                break

meta = []
for key in sorted(data):
    v = np.array(data[key])
    if len(v) < 100: continue
    sel = v[np.argmax(v[:, 3])]; orc = float(v[:, 2].max())
    status = "solved" if sel[2] >= 0.23 else ("confident decoy" if orc >= 0.23 else "no acceptable pose")
    xmax = float(min(11.0, max(1.5, np.percentile(v[:, 0], 99) * 1.3)))
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.axvspan(0, 1.0, color="#f7c9de", alpha=.5, zorder=0)
    ax.add_patch(plt.Rectangle((1.0, 0.80), xmax, .25, color="#bfe3d0", alpha=.5, zorder=0))
    ax.scatter(v[:, 0], v[:, 2], s=11, alpha=.5, c="#8a4f86", edgecolors="none", zorder=2)
    ax.scatter([sel[0]], [sel[2]], marker="*", s=190, c="#e8b21a",
               edgecolors="#333", linewidths=.7, zorder=5)
    for t in (0.23, 0.49, 0.80):
        ax.axhline(t, ls="--", lw=.8, color="#333", zorder=3)
    ax.axvline(1.0, lw=.8, color="#333", zorder=3)
    ax.set_xlim(0, xmax); ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel("paratope local RMSD (Å)", fontsize=8)
    ax.set_ylabel("DockQ", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f"{key}   sel {sel[2]:.2f} / oracle {orc:.2f}", fontsize=8.5)
    fig.tight_layout(pad=0.4)
    png = f"{OUT}/{key}.png"; fig.savefig(png, dpi=95); plt.close(fig)
    meta.append(dict(k=key, n=len(v), sel=round(float(sel[2]), 3), orc=round(orc, 3),
                     status=status, frac_h3_sub1=round(float((v[:, 0] <= 1.0).mean()), 3),
                     med_h3=round(float(np.median(v[:, 0])), 2),
                     b64=base64.b64encode(open(png, "rb").read()).decode()))
json.dump(meta, open(f"{OUT}/meta.json", "w"))
print(f"{len(meta)} panels, payload {sum(len(m['b64']) for m in meta)/1e6:.1f} MB")
