#!/usr/bin/env python
"""plot_dockq_iptm_slides.py — DockQ vs ipTM as 3x3 slides (per-panel y-scaling, big markers).

Paginates the 46 clean confident-decoy interfaces (fold-error cases excluded) into 3x3 slides,
each panel autoscaled to its own data so the ipTM-selected marker is never clipped. Renders PNGs
into scratchpad/slides/. A companion builder wraps them in a flip-through HTML slideshow.
"""
import csv, math, collections, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

DEEP="results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"
IFACES="results/csv_files/decoy_structural_analysis.csv"
FOLD="results/csv_files/decoy_fold_quality.csv"
OUTDIR=os.environ.get("SLIDES_DIR",
    "/tmp/claude-1004/-home-ebrahim-tts-abag/ea5db07e-452e-45f8-a6d6-98121d6d1264/scratchpad/slides")
INK="#22252b"; MUTE="#8a9099"; GRID="#e6e8ec"; SEL="#c0392b"; ORC="#2e8b6f"; PT="#9aa3ad"
PER=9; NC=3; NR=3
plt.rcParams.update({"font.family":"DejaVu Sans","axes.edgecolor":MUTE,"text.color":INK,
    "axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,"figure.dpi":150})

def load():
    want=[(r["cx"][:4],r["c1"],r["c2"]) for r in csv.DictReader(open(IFACES))]
    bad=set()
    for r in csv.DictReader(open(FOLD)):
        try: ag=float(r["ag_lddt"])
        except: ag=float("nan")
        if not (ag==ag) or ag<0.70: bad.add((r["cx"][:4],r["c1"],r["c2"]))
    keep=set(k for k in want if k not in bad)
    pool=collections.defaultdict(list)
    for r in csv.DictReader(open(DEEP)):
        k=(r["assembly_id"][:4],r["chain1"],r["chain2"])
        if k not in keep:
            k=(r["assembly_id"][:4],r["chain2"],r["chain1"])
            if k not in keep: continue
        try: dq=float(r["dockq"]); ip=float(r["pair_iptm"])
        except: continue
        if math.isnan(ip): continue
        pool[k].append((dq,ip))
    ifaces=[k for k in want if k in pool and len(pool[k])>=5]
    rho={}
    for k in ifaces:
        dq=[d for d,_ in pool[k]]; ip=[i for _,i in pool[k]]
        rho[k]=spearmanr(dq,ip).correlation if len(set(dq))>1 and len(set(ip))>1 else float("nan")
    ifaces.sort(key=lambda k:-(rho[k] if rho[k]==rho[k] else -9))
    return ifaces,pool,rho

def draw(ax,key,pts,r,show_x,show_y):
    dq=np.array([d for d,_ in pts]); ip=np.array([i for _,i in pts])
    st=max(1,len(dq)//1600)
    ax.scatter(dq[::st],ip[::st],s=13,color=PT,alpha=0.42,edgecolors="none",zorder=1)
    o=int(np.argmax(dq)); ax.scatter([dq[o]],[ip[o]],s=110,marker="o",color=ORC,edgecolor="white",lw=1.1,zorder=3)
    j=int(np.argmax(ip))
    ax.scatter([dq[j]],[ip[j]],s=620,marker="o",facecolors="none",edgecolors=SEL,lw=2.2,zorder=5)
    ax.scatter([dq[j]],[ip[j]],s=340,marker="*",color=SEL,edgecolor="white",lw=1.4,zorder=6)
    ax.axvline(0.23,color=MUTE,lw=0.9,ls=(0,(3,3)),zorder=0)
    ax.set_xlim(-0.03,1.03); ax.set_ylim(0,max(ip.max(),0.15)*1.16)
    ttl=f"{key[0]} {key[1]}-{key[2]}   " + (f"ρ={r:+.2f}" if r==r else "ρ=n/a")
    ax.set_title(ttl,fontsize=12.5,fontweight="600",pad=6)
    ax.grid(color=GRID,lw=0.7); ax.set_axisbelow(True); ax.tick_params(labelsize=9)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    if show_x: ax.set_xlabel("DockQ",fontsize=10.5)
    if show_y: ax.set_ylabel("pair_iptm",fontsize=10.5)

def main():
    os.makedirs(OUTDIR,exist_ok=True)
    for f in os.listdir(OUTDIR):
        if f.startswith("slide_") and f.endswith(".png"): os.remove(os.path.join(OUTDIR,f))
    ifaces,pool,rho=load()
    nslide=math.ceil(len(ifaces)/PER)
    for s in range(nslide):
        chunk=ifaces[s*PER:(s+1)*PER]
        fig,axes=plt.subplots(NR,NC,figsize=(13.2,12.4))
        axes=np.array(axes).reshape(-1)
        for idx,ax in enumerate(axes):
            if idx>=len(chunk): ax.axis("off"); continue
            row,col=divmod(idx,NC)
            draw(ax,chunk[idx],pool[chunk[idx]],rho[chunk[idx]],
                 show_x=(row==NR-1 or idx+NC>=len(chunk)), show_y=(col==0))
        fig.suptitle(f"DockQ vs ipTM  —  slide {s+1} of {nslide}   "
                     f"(red ★ = ipTM-selected, green ● = oracle, dashed = DockQ 0.23)",
                     fontsize=13,fontweight="bold",x=0.02,ha="left")
        fig.tight_layout(rect=(0,0,1,0.97))
        p=os.path.join(OUTDIR,f"slide_{s:02d}.png"); fig.savefig(p,bbox_inches="tight"); plt.close(fig)
        print("wrote",p)
    print(f"{nslide} slides, {len(ifaces)} interfaces")

if __name__=="__main__":
    main()
