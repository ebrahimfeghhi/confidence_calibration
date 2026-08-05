#!/usr/bin/env python
"""Pool leader20 + dockqdonor + NATIVE templates, select by ipTM. Does the exact fold win the pool?

Adding dockqdonor's single (oracle) template to leader20's 20 cost 2 rescues: decoy-fold templates
produced poses at ipTM 0.71 / 0.64 with DockQ 0.02 / 0.01, out-confidencing correct poses. This asks
whether the EXACT NATIVE fold -- the strongest possible template, 14/15 alone -- is immune to that,
or whether decoy-template poses out-score even it.

If native's poses win the pool everywhere, the problem is purely template quality and pooling is safe
given a good enough template. If they do not, pooling is unsafe at any template quality.

None of this is deployable: native and dockqdonor donors both require the answer.

Usage: python scripts/pool_with_native.py
Writes: artifacts/pool_with_native.json
"""
import json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, SEEDS, SAMPLES, THRESH
from score_native_arm import locate

L_OUT=f"{EXP}/artifacts/sweep_out/leader20/leader20"; L_CACHE=f"{EXP}/artifacts/leader20_ost"
D_OUT=f"{EXP}/artifacts/sweep_out/dockqdonor/dockqdonor"; D_CACHE=f"{EXP}/artifacts/dockqdonor_ost"
N_CACHE=f"{EXP}/artifacts/native_arm_ost"
ljobs=json.load(open(f"{EXP}/artifacts/leader20_jobs.json"))
djobs={j["interface"]:j for j in json.load(open(f"{EXP}/artifacts/dockq_arm_jobs.json"))}
ld=json.load(open(f"{EXP}/artifacts/leader20_scores.json"))
dq=json.load(open(f"{EXP}/artifacts/dockqdonor_scores.json"))
nat=json.load(open(f"{EXP}/artifacts/native_arm_scores.json"))

def ch(lab):
    p=lab.split("_",1)[1]
    return (p[0],p[1:]) if len(p)>2 else (p[0],p[1])

def grab(cache_tag, out, name, lab, seeds, src, dir_has_name=False):
    """dir_has_name: score_native_arm.locate() returns the directory WITH the run name already
    appended, unlike the sweep arms where out + name must be joined."""
    c1,c2=ch(lab); out_=[]
    for s in seeds:
        for k in SAMPLES:
            o=f"{cache_tag}_{s}_{k}.json"
            base=out if dir_has_name else f"{out}/{name}"
            cf=f"{base}/seed_{s}/predictions/{name}_summary_confidence_sample_{k}.json"
            if not (os.path.exists(o) and os.path.exists(cf)): continue
            try: d=json.load(open(o)); M=json.load(open(cf))["chain_pair_iptm"]
            except Exception: continue
            qs,ifs=d.get("dockq") or [],d.get("dockq_interfaces") or []
            for i,t in enumerate(ifs):
                if {t[0],t[1]}=={c1,c2} and i<len(qs):
                    x,y=ord(t[2])-65,ord(t[3])-65
                    if 0<=x<len(M) and 0<=y<len(M): out_.append((src,M[x][y],qs[i]))
                    break
    return out_

rows=[]
print(f"{'interface':11} {'native':>8} {'dockq':>7} {'leader':>7} | {'POOLED 22':>10} {'src':>11} {'':8}")
for lab in sorted(set(ld)&set(dq)&set(nat)):
    L=[]
    for j in [x for x in ljobs if x["interface"]==lab]:
        L+=grab(f"{L_CACHE}/{lab}_L{j['leader_rank']:02d}", L_OUT, j["name"], lab, j["seeds"],
                f"leader{j['leader_rank']}")
    D=grab(f"{D_CACHE}/{lab}", D_OUT, djobs[lab]["name"], lab, SEEDS, "dockqdonor")
    nd,nn=locate(nat[lab]["pdb"])
    N=grab(f"{N_CACHE}/{nat[lab]['pdb']}", nd, nn, lab, SEEDS, "native", dir_has_name=True) if nd else []
    if not (L and D and N): continue
    pool=L+D+N
    w=max(pool,key=lambda x:x[1])
    nw=max(N,key=lambda x:x[1])
    rows.append({"interface":lab,"native_alone":nat[lab]["rescued"],
                 "dockqdonor_alone":dq[lab]["rescued"],"leader20_alone":ld[lab]["method_rescued"],
                 "pooled_dockq":round(w[2],3),"pooled_iptm":round(w[1],3),"pooled_src":w[0],
                 "pooled_rescued":w[2]>=THRESH,
                 "native_best_iptm":round(nw[1],3),"native_best_iptm_dockq":round(nw[2],3)})
    r=rows[-1]
    print(f"{lab:11} {'R' if r['native_alone'] else '.':>8} {'R' if r['dockqdonor_alone'] else '.':>7} "
          f"{'R' if r['leader20_alone'] else '.':>7} | {r['pooled_dockq']:10.3f} {r['pooled_src']:>11} "
          f"{'RESCUED' if r['pooled_rescued'] else ''}")
n=len(rows)
for nm,key in (("native alone","native_alone"),("dockqdonor alone","dockqdonor_alone"),
               ("leader20 alone","leader20_alone"),("POOLED (22 templates)","pooled_rescued")):
    print(f"  {nm:24} {sum(1 for r in rows if r[key])}/{n}")
c=collections.Counter(r["pooled_src"] for r in rows)
print(f"\n  winner came from: {dict(c.most_common(6))}")
lost=[r["interface"] for r in rows if r["native_alone"] and not r["pooled_rescued"]]
print(f"  native rescued but POOLING lost it: {lost}")
json.dump(rows,open(f"{EXP}/artifacts/pool_with_native.json","w"),indent=1)
