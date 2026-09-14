#!/usr/bin/env python
"""plot_dockq_vs_iptm.py — DockQ vs ipTM per confident-decoy interface (fold-error cases excluded).

Two figures:
  results/dockq_vs_iptm_grid.png   — small-multiples: one DockQ(x) vs pair_iptm(y) scatter per interface,
      max-ipTM point (what selection picks) in red, max-DockQ point (oracle) in green, per-panel Spearman ρ.
  results/dockq_vs_iptm_pooled.png — all structures pooled into one 2D density (the global pattern).

Excludes the antigen-misfold interfaces (decoy_fold_quality.csv, ag_lddt<0.70) so only clean
pure-docking cases are shown.  python steer/experiments/plot_dockq_vs_iptm.py
"""
import csv, math, collections, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

DEEP="results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"
IFACES="results/csv_files/decoy_structural_analysis.csv"
FOLD="results/csv_files/decoy_fold_quality.csv"
INK="#22252b"; MUTE="#8a9099"; GRID="#e6e8ec"; SEL="#c0392b"; ORC="#2e8b6f"; PT="#9aa3ad"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.edgecolor":MUTE,"text.color":INK,
    "axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,"figure.dpi":140})

VECTOR = bool(os.environ.get("VECTOR"))

def keep_ifaces():
    want=[(r["cx"][:4],r["c1"],r["c2"]) for r in csv.DictReader(open(IFACES))]
    bad=set()
    for r in csv.DictReader(open(FOLD)):
        try: ag=float(r["ag_lddt"])
        except: ag=float("nan")
        if not (ag==ag) or ag<0.70: bad.add((r["cx"][:4],r["c1"],r["c2"]))
    return [k for k in want if k not in bad], set(want)

def load_pool(keepset):
    pool=collections.defaultdict(list)
    for r in csv.DictReader(open(DEEP)):
        k=(r["assembly_id"][:4],r["chain1"],r["chain2"])
        if k not in keepset:
            k=(r["assembly_id"][:4],r["chain2"],r["chain1"])
            if k not in keepset: continue
        try: dq=float(r["dockq"]); ip=float(r["pair_iptm"])
        except: continue
        if math.isnan(ip): continue
        pool[k].append((dq,ip))
    return pool

def main():
    keep,allset=keep_ifaces()
    pool=load_pool(set(keep))
    ifaces=[k for k in keep if k in pool and len(pool[k])>=5]
    # per-interface spearman
    rho={}
    for k in ifaces:
        dq=[d for d,_ in pool[k]]; ip=[i for _,i in pool[k]]
        rho[k]=spearmanr(dq,ip).correlation if len(set(dq))>1 and len(set(ip))>1 else float("nan")
    order=sorted(ifaces,key=lambda k:(-(rho[k] if rho[k]==rho[k] else -9)))

    # ---------- Fig 1: small-multiples grid ----------
    n=len(order); ncol=7; nrow=math.ceil(n/ncol)
    fig,axes=plt.subplots(nrow,ncol,figsize=(ncol*2.05,nrow*2.05),sharex=True,sharey=True)
    axes=np.array(axes).reshape(-1)
    for ax in axes[n:]: ax.axis("off")
    for ax,k in zip(axes,order):
        pts=pool[k]; dq=np.array([d for d,_ in pts]); ip=np.array([i for _,i in pts])
        cap=1600 if VECTOR else 700; st=max(1,len(dq)//cap)
        ax.scatter(dq[::st],ip[::st],s=6,color=PT,alpha=0.40,edgecolors="none",rasterized=not VECTOR,zorder=1)
        o=int(np.argmax(dq)); ax.scatter([dq[o]],[ip[o]],s=48,marker="o",color=ORC,edgecolor="white",lw=0.8,zorder=3)  # oracle
        j=int(np.argmax(ip))
        ax.scatter([dq[j]],[ip[j]],s=270,marker="o",facecolors="none",edgecolors=SEL,lw=1.7,zorder=5)  # halo ring
        ax.scatter([dq[j]],[ip[j]],s=150,marker="*",color=SEL,edgecolor="white",lw=1.2,zorder=6)       # selected star
        ax.axvline(0.23,color=MUTE,lw=0.6,ls=(0,(3,3)),zorder=0)
        ax.set_xlim(-0.03,1.03); ax.set_ylim(0,max(0.2,ip.max()*1.1))
        r=rho[k]; ax.set_title(f"{k[0]} {k[1]}-{k[2]}  ρ={r:+.2f}" if r==r else f"{k[0]} {k[1]}-{k[2]}",
                               fontsize=8,pad=2)
        ax.tick_params(labelsize=6); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
        for s in ("top","right"): ax.spines[s].set_visible(False)
    fig.suptitle("DockQ vs ipTM per interface — red ★ = ipTM-selected, green ● = oracle (best DockQ), "
                 "dashed = DockQ 0.23",fontsize=12,fontweight="bold",x=0.02,ha="left")
    fig.supxlabel("DockQ",fontsize=11); fig.supylabel("pair_iptm",fontsize=11)
    fig.tight_layout(rect=(0.01,0.01,1,0.97))
    fig.savefig(os.environ.get("GRID_OUT","results/dockq_vs_iptm_grid.png"),
                bbox_inches="tight",dpi=int(os.environ.get("GRID_DPI","140")))
    print("wrote", os.environ.get("GRID_OUT","results/dockq_vs_iptm_grid.png"))

    # ---------- Fig 2: pooled density ----------
    allp=[p for k in ifaces for p in pool[k]]
    dq=np.array([d for d,_ in allp]); ip=np.array([i for _,i in allp])
    fig,ax=plt.subplots(figsize=(6.6,5.2))
    hb=ax.hexbin(dq,ip,gridsize=45,bins="log",cmap="magma_r",mincnt=1)
    ax.axvline(0.23,color="white",lw=1.3,ls=(0,(4,3)))
    ax.text(0.235,ax.get_ylim()[1]*0.97,"DockQ 0.23",color="white",fontsize=9,va="top")
    cb=fig.colorbar(hb,ax=ax); cb.set_label("structures (log)")
    ax.set_xlabel("DockQ"); ax.set_ylabel("pair_iptm")
    ax.set_title(f"Pooled DockQ vs ipTM  ({len(ifaces)} interfaces, {len(allp):,} structures)",
                 fontsize=12,fontweight="bold",loc="left",pad=24)
    ax.text(0,1.015,"confident decoys = mass at high ipTM, DockQ<0.23 (left of the line)",
            transform=ax.transAxes,fontsize=9.5,color=MUTE,va="bottom")
    fig.tight_layout(); fig.savefig("results/dockq_vs_iptm_pooled.png",bbox_inches="tight")
    print("wrote results/dockq_vs_iptm_pooled.png")

    rr=[rho[k] for k in ifaces if rho[k]==rho[k]]
    print(f"\ninterfaces plotted: {len(ifaces)}  |  within-interface Spearman ρ(DockQ,ipTM): "
          f"median={np.median(rr):+.2f}  mean={np.mean(rr):+.2f}  "
          f"| ρ<0.2: {sum(r<0.2 for r in rr)}/{len(rr)}  ρ<0: {sum(r<0 for r in rr)}/{len(rr)}")

if __name__=="__main__":
    main()
