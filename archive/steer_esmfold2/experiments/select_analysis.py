"""select_analysis.py — the decisive confident-decoy test: for the interfaces where ipTM SELECTS a decoy
(good dock exists but ipTM's top pose is wrong), would selecting by the TRUE iTM instead pick the good dock?
If yes -> the failure is fixable by a better confidence head (metric suffices). If no -> the metric itself
can't select it.

Uses the curated ab/ag chain labels (decoy_structural_analysis.csv) -> correct interface, no multimer
artifact. Per interface, computes true iTM over a pose set that includes ALL good-dock candidates
(DockQ>=0.2) + the top-ipTM poses + a DockQ-stratified spread, then compares selection by ipTM vs true iTM
vs oracle. CPU-parallel over complexes.
"""
import csv, sys, numpy as np
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0,"steer/experiments")
import true_iptm_all as T   # parse, frames, true_iTM, ident, sq, match, PREDROOT, GTDIR, DEEP
META="results/csv_files/decoy_structural_analysis.csv"

# metadata: failure interfaces
IF=defaultdict(list)
for r in csv.DictReader(open(META)):
    try:
        if r['good_dq'] and r['decoy_dq'] and float(r['good_dq'])>0.5 and float(r['decoy_dq'])<0.23 \
           and r['pred_ab_chain'] and r['pred_ag_chain']:
            IF[r['cx']].append((r['c1'],r['c2'],r['pred_ab_chain'],r['pred_ag_chain'],
                                r['gt_ab_label'],r['gt_ag_label'],float(r['good_dq']),float(r['decoy_dq'])))
    except: pass

# deep CSV: per (cx,c1,c2) -> {(seed,sample):(dockq,pair_iptm)}
DEEPD=defaultdict(dict)
for r in csv.DictReader(open(T.DEEP)):
    cx=r['assembly_id'][:4]
    if cx not in IF: continue
    try: DEEPD[(cx,r['chain1'],r['chain2'])][(int(r['seed']),int(r['sample']))]=(float(r['dockq']),float(r['pair_iptm']))
    except: pass

def process(cx):
    out=[]
    gt=T.parse(f"{T.GTDIR}/{cx}-assembly1.cif")
    for c1,c2,pab,pag,gab_meta,gag_meta,good_dq,decoy_dq in IF[cx]:
        d=DEEPD.get((cx,c1,c2),{})
        if len(d)<20: continue
        # metadata GT labels don't match assembly1.cif chains -> sequence-match model chains pab/pag to GT
        refp=None
        for (s,k) in d:
            f=T.PREDROOT/cx/f"seed_{s}"/"predictions"/f"{cx}_sample_{k}.cif"
            if f.exists(): refp=T.parse(str(f)); break
        if not refp or pab not in refp or pag not in refp: continue
        gab=max(gt,key=lambda gc:T.ident(T.sq(refp[pab]),T.sq(gt[gc])))
        gag=max([gc for gc in gt if gc!=gab],key=lambda gc:T.ident(T.sq(refp[pag]),T.sq(gt[gc])),default=None)
        if gag is None or T.ident(T.sq(refp[pab]),T.sq(gt[gab]))<0.5 or T.ident(T.sq(refp[pag]),T.sq(gt[gag]))<0.5: continue
        items=[(s,k,dq,pv) for (s,k),(dq,pv) in d.items()]
        # pose set: all good-dock candidates + top-40 ipTM + stratified ~600 by DockQ
        goods=[(s,k) for s,k,dq,pv in items if dq>=0.2]
        iptop=[(s,k) for s,k,dq,pv in sorted(items,key=lambda x:-x[3])[:40]]
        by_dq=sorted(items,key=lambda x:x[2]); spread=[(by_dq[i][0],by_dq[i][1]) for i in range(0,len(by_dq),max(1,len(by_dq)//600))]
        pick=set(goods)|set(iptop)|set(spread)
        rows=[]
        for s,k in pick:
            f=T.PREDROOT/cx/f"seed_{s}"/"predictions"/f"{cx}_sample_{k}.cif"
            if not f.exists(): continue
            try:
                p=T.parse(str(f))
                if pab not in p or pag not in p: continue
                P1,G1=T.match(p[pab],gt[gab]); P2,G2=T.match(p[pag],gt[gag])
                if len(P1)<5 or len(P2)<5: continue
                t=T.true_iTM(P1,P2,G1,G2); dq,pv=d[(s,k)]; rows.append((dq,pv,t))
            except Exception: continue
        if len(rows)<20: continue
        a=np.array(rows)   # cols: dockq, pred_iptm, true_itm
        iptm_sel=a[np.argmax(a[:,1]),0]      # DockQ of the max-ipTM pose
        true_sel=a[np.argmax(a[:,2]),0]      # DockQ of the max-true-iTM pose
        oracle  =a[:,0].max()
        out.append((cx,f"{c1}-{c2}",good_dq,decoy_dq,iptm_sel,true_sel,oracle,len(rows)))
    return out

def main():
    cxs=sorted(IF.keys())
    print(f"confident-decoy complexes: {len(cxs)} ({sum(len(v) for v in IF.values())} interfaces)",flush=True)
    with Pool(16) as p: res=p.map(process, cxs)
    flat=[x for r in res for x in r]
    print(f"\n{'cx':>6} {'iface':>6} {'good_dq':>7} {'ipTM_sel':>8} {'trueiTM_sel':>11} {'oracle':>7} {'n':>5}  {'rescue?'}")
    resc23=resc50=0
    for cx,iface,gd,dd,ips,ts,orc,n in sorted(flat,key=lambda x:-x[5]):
        r=("GOOD" if ts>0.5 else "acc" if ts>0.23 else "no")
        if ts>0.23: resc23+=1
        if ts>0.5: resc50+=1
        print(f"{cx:>6} {iface:>6} {gd:>7.2f} {ips:>8.2f} {ts:>11.2f} {orc:>7.2f} {n:>5}  {r}")
    N=len(flat)
    print(f"\n=== {N} confident-decoy interfaces (ipTM picks a decoy) ===")
    print(f"true-iTM selection lands ACCEPTABLE (DockQ>0.23): {resc23}/{N} = {100*resc23/N:.0f}%")
    print(f"true-iTM selection lands GOOD (DockQ>0.5):        {resc50}/{N} = {100*resc50/N:.0f}%")
    print(f"(for reference: ipTM selection is a decoy by construction; oracle is the good dock)")

if __name__=="__main__":
    main()
