#!/usr/bin/env python
"""iptm_calibration.py — reproduce the paper's S11 iPTM-vs-DockQ calibration on our benchmark.

Tests: is iPTM a poor DockQ predictor in the middle (0.4-0.8) but good at the extremes (<0.4, >0.8)?
Uses the full benchmark master (all 97 complexes x ~50 seeds) so DockQ spans the whole range.
  /home/ebrahim/esm/.venv/bin/python steer/experiments/iptm_calibration.py
"""
import csv, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

MASTER="results/csv_files/dockq_ost_esmfold2fast_tts_v2_all.csv"
INK="#22252b"; MUTE="#8a9099"; GRID="#e6e8ec"; A="#2e8b6f"
BINS=[(0.0,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.01)]; LAB=["0–0.4","0.4–0.6","0.6–0.8","0.8+"]
plt.rcParams.update({"font.family":"DejaVu Sans","axes.edgecolor":MUTE,"text.color":INK,
    "axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,"figure.dpi":150})

def load(col):
    dq=[]; ip=[]
    for r in csv.DictReader(open(MASTER)):
        if r.get("dockq_status")=="chain_unmapped": continue
        try: d=float(r["dockq"]); v=float(r[col])
        except (ValueError,KeyError): continue
        if math.isnan(d) or math.isnan(v): continue
        dq.append(d); ip.append(v)
    return np.array(dq), np.array(ip)

def report(col):
    dq,ip=load(col)
    print(f"\n=== {col}  (n={len(dq)}, range {ip.min():.2f}-{ip.max():.2f}) ===")
    print(f"overall Spearman({col},DockQ) = {spearmanr(ip,dq).correlation:+.2f}")
    groups=[]
    print(f"{'iptm bin':>10} {'n':>6} {'DockQ med':>10} {'IQR':>14} {'%acceptable':>12}")
    for (lo,hi),lab in zip(BINS,LAB):
        m=(ip>=lo)&(ip<hi); d=dq[m]; groups.append(d)
        if len(d)==0: print(f"{lab:>10} {0:>6}"); continue
        q1,med,q3=np.percentile(d,[25,50,75])
        print(f"{lab:>10} {len(d):>6} {med:>10.2f} {q1:>6.2f}-{q3:<6.2f} {100*np.mean(d>=0.23):>11.0f}%")
    # discrimination inside the middle band vs the extremes
    mid=(ip>=0.4)&(ip<0.8); ext=~mid
    if mid.sum()>5: print(f"Spearman within 0.4-0.8 band = {spearmanr(ip[mid],dq[mid]).correlation:+.2f} (n={mid.sum()})")
    return groups,col

def main():
    for col in ("pair_iptm","global_iptm"):
        groups,c=report(col)
        fig,ax=plt.subplots(figsize=(6.2,4.6))
        bp=ax.boxplot([g for g in groups],labels=LAB,widths=0.6,patch_artist=True,
                      medianprops=dict(color=INK,lw=1.8),whiskerprops=dict(color=MUTE),
                      capprops=dict(color=MUTE),flierprops=dict(marker='.',ms=2,mfc=MUTE,mec='none',alpha=.3))
        for b in bp['boxes']: b.set(facecolor=A,alpha=.55,edgecolor=MUTE)
        ax.axhline(0.23,color=MUTE,lw=1,ls=(0,(4,3)))
        ax.text(0.5,0.245,"DockQ 0.23 (acceptable)",fontsize=8.5,color=MUTE)
        ns=[len(g) for g in groups]
        for i,n in enumerate(ns): ax.text(i+1,1.02,f"n={n}",ha="center",fontsize=8,color=MUTE)
        ax.set_ylim(-0.02,1.08); ax.set_xlabel(f"{c} bin"); ax.set_ylabel("DockQ")
        ax.grid(axis="y",color=GRID,lw=0.8); ax.set_axisbelow(True)
        for s in ("top","right"): ax.spines[s].set_visible(False)
        ax.set_title(f"DockQ by {c} bin  (our benchmark, 97 complexes)",fontsize=12.5,fontweight="bold",loc="left")
        fig.tight_layout(); fig.savefig(f"results/iptm_calibration_{c}.png",bbox_inches="tight")
        print(f"wrote results/iptm_calibration_{c}.png")

if __name__=="__main__":
    main()
