#!/usr/bin/env python
"""plot_confident_decoy.py — two figures for the confident-decoy story.

  results/selection_vs_depth.png  — ipTM-selected DockQ stays flat as seeds scale (selection wall)
  results/decoy_error_breakdown.png — decoy error types (wrong antigen surface, correct CDR paratope)

  python steer/experiments/plot_confident_decoy.py
"""
import csv, collections, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEEP = "results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"
IFACES = "results/csv_files/decoy_structural_analysis.csv"
DEPTHS = [50, 100, 250, 500, 1024]

# palette
INK="#22252b"; MUTE="#8a9099"; GRID="#e6e8ec"
DECOY="#c0392b"; GOOD="#2e8b6f"; NEUTRAL="#b9bec6"; ACCENT="#c0392b"

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":11,"axes.edgecolor":MUTE,
    "axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,
    "axes.linewidth":0.9,"figure.dpi":150,
})

def load_pool():
    want=set((r["cx"][:4],r["c1"],r["c2"]) for r in csv.DictReader(open(IFACES)))
    pool=collections.defaultdict(list)
    for r in csv.DictReader(open(DEEP)):
        k=(r["assembly_id"][:4],r["chain1"],r["chain2"])
        if k not in want:
            k=(r["assembly_id"][:4],r["chain2"],r["chain1"])
            if k not in want: continue
        if not r["seed"].lstrip("-").isdigit(): continue
        try: dq=float(r["dockq"])
        except: continue
        try: ip=float(r["pair_iptm"])
        except: ip=float("nan")
        pool[k].append((int(r["seed"]),dq,ip))
    return pool

def sel_at(rows, depth):
    sub=[(dq,ip) for s,dq,ip in rows if s<depth and not math.isnan(ip)]
    if not sub: return None
    return max(sub,key=lambda x:x[1])[0]

# ---------- Figure 1: selection vs depth ----------
def fig_depth(pool):
    x=np.arange(len(DEPTHS))
    per=[]; oracles=[]
    for k,rows in pool.items():
        row=[sel_at(rows,d) for d in DEPTHS]
        if all(v is None for v in row): continue
        per.append(row); oracles.append(max(dq for _,dq,_ in rows))
    means=[np.nanmean([r[i] if r[i] is not None else np.nan for r in per]) for i in range(len(DEPTHS))]
    orc=float(np.mean(oracles))

    fig,ax=plt.subplots(figsize=(7.2,4.6))
    # faint per-interface traces
    for r in per:
        y=[v if v is not None else np.nan for v in r]
        ax.plot(x,y,color=NEUTRAL,lw=0.7,alpha=0.35,zorder=1)
    # oracle ceiling
    ax.axhline(orc,color=GOOD,lw=1.6,ls=(0,(5,2)),zorder=3)
    ax.text(len(DEPTHS)-1.02,orc+0.02,f"oracle (good dock exists) — mean {orc:.2f}",
            ha="right",va="bottom",color=GOOD,fontsize=10,fontweight="bold")
    # shaded locked gain
    ax.fill_between(x,means,[orc]*len(x),color=GOOD,alpha=0.06,zorder=0)
    # mean selected
    ax.plot(x,means,color=DECOY,lw=2.6,marker="o",ms=6,zorder=4)
    ax.text(x[-1],means[-1]-0.035,f"ipTM-selected — mean {means[-1]:.2f}",
            ha="right",va="top",color=DECOY,fontsize=10,fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(DEPTHS)
    ax.set_xlabel("seeds in the pool (depth)")
    ax.set_ylabel("DockQ")
    ax.set_ylim(-0.02,0.85); ax.set_xlim(-0.15,len(DEPTHS)-0.85)
    ax.grid(axis="y",color=GRID,lw=0.8); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.set_title("Scaling seeds does not rescue selection",fontsize=13,fontweight="bold",loc="left",pad=26)
    ax.text(0,1.015,"51 of 55 confident-decoy interfaces stay stuck on a decoy even at 1024 seeds",
            transform=ax.transAxes,fontsize=10,color=MUTE,va="bottom")
    fig.tight_layout()
    fig.savefig("results/selection_vs_depth.png",bbox_inches="tight")
    print("wrote results/selection_vs_depth.png")

# ---------- Figure 2: error breakdown ----------
def fig_errors():
    rows=list(csv.DictReader(open(IFACES)))
    def f(r,k):
        try:
            v=float(r[k]); return None if math.isnan(v) else v
        except: return None
    valid=[r for r in rows if f(r,"epitope_overlap") is not None]
    ov=[f(r,"epitope_overlap") for r in valid]
    dq=[f(r,"decoy_dq") for r in valid]
    cdr=[f(r,"decoy_paratope_cdr_frac") for r in valid]
    bands=[("Wrong surface\n(different epitope)",sum(o<0.1 for o in ov),ACCENT),
           ("Right region,\nwrong register",sum(0.1<=o<0.5 for o in ov),NEUTRAL),
           ("Right surface,\nbad pose",sum(o>=0.5 for o in ov),NEUTRAL)]
    n=len(valid)

    fig,(axL,axR)=plt.subplots(1,2,figsize=(11,4.6),gridspec_kw={"width_ratios":[1,1.05]})

    # LEFT: horizontal bars of antigen-side error
    labels=[b[0] for b in bands][::-1]; vals=[b[1] for b in bands][::-1]; cols=[b[2] for b in bands][::-1]
    y=np.arange(len(labels))
    axL.barh(y,vals,color=cols,height=0.62,zorder=3)
    for yi,v in zip(y,vals):
        axL.text(v+0.6,yi,f"{v}  ({100*v/n:.0f}%)",va="center",ha="left",fontsize=10.5,fontweight="bold",color=INK)
    axL.set_yticks(y); axL.set_yticklabels(labels,fontsize=10.5)
    axL.set_xlim(0,max(vals)*1.28); axL.set_xlabel(f"interfaces (n={n})")
    axL.grid(axis="x",color=GRID,lw=0.8); axL.set_axisbelow(True)
    for s in ("top","right","left"): axL.spines[s].set_visible(False)
    axL.tick_params(axis="y",length=0)
    axL.set_title("Where the decoy docks on the antigen",fontsize=12.5,fontweight="bold",loc="left",pad=8)

    # RIGHT: scatter overlap vs decoy DockQ, colored by paratope machinery
    isc=[c is not None and c>=0.5 for c in cdr]
    axR.scatter([o for o,k in zip(ov,isc) if k],[d for d,k in zip(dq,isc) if k],
                s=46,color=GOOD,alpha=0.75,edgecolor="white",lw=0.6,zorder=3,label="CDR paratope (94%)")
    axR.scatter([o for o,k in zip(ov,isc) if not k],[d for d,k in zip(dq,isc) if not k],
                s=52,color=DECOY,alpha=0.85,marker="s",edgecolor="white",lw=0.6,zorder=4,label="framework/off-Fv (6%)")
    axR.axhline(0.23,color=MUTE,lw=1.0,ls=(0,(4,3)),zorder=1)
    axR.text(0.98,0.235,"DockQ 0.23 (acceptable)",ha="right",va="bottom",fontsize=8.5,color=MUTE)
    axR.set_xlabel("decoy–GT epitope overlap  (0 = different surface)")
    axR.set_ylabel("decoy DockQ")
    axR.set_xlim(-0.03,1.03); axR.set_ylim(-0.02,0.62)
    axR.grid(color=GRID,lw=0.8); axR.set_axisbelow(True)
    for s in ("top","right"): axR.spines[s].set_visible(False)
    axR.legend(frameon=False,fontsize=9,loc="upper left")
    axR.set_title("Correct paratope, wrong antigen surface",fontsize=12.5,fontweight="bold",loc="left",pad=8)

    fig.suptitle("Confident-decoy error anatomy — the paratope is right, the antigen surface is wrong",
                 fontsize=13.5,fontweight="bold",x=0.02,ha="left")
    fig.tight_layout(rect=(0,0,1,0.95))
    fig.savefig("results/decoy_error_breakdown.png",bbox_inches="tight")
    print("wrote results/decoy_error_breakdown.png")

if __name__=="__main__":
    fig_depth(load_pool())
    fig_errors()
