#!/usr/bin/env python
"""Does templating raise ipTM for the SAME quality of pose, or only produce better poses?

On the 9 interfaces dockqdonor rescues, both things changed: more correct poses AND higher ipTM on
them. Two explanations:
  (a) RECALIBRATION -- templating makes the model more confident in a pose of a given quality
  (b) BETTER POSES  -- templating produces higher-DockQ poses, and ipTM follows DockQ as it always did

They are separated by comparing ipTM at MATCHED DockQ: bin poses by DockQ and compare the free run
against the templated run inside each bin. If (b) alone, the curves coincide.

Usage: python scripts/iptm_at_matched_dockq.py
Writes: artifacts/iptm_at_matched_dockq.json
"""
import json, os, sys, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, PRED, SEEDS, SAMPLES, THRESH

F_CACHE=f"{EXP}/artifacts/abag_all_ost"
D_OUT=f"{EXP}/artifacts/sweep_out/dockqdonor/dockqdonor"; D_CACHE=f"{EXP}/artifacts/dockqdonor_ost"
djobs={j["interface"]:j for j in json.load(open(f"{EXP}/artifacts/dockq_arm_jobs.json"))}
dq=json.load(open(f"{EXP}/artifacts/dockqdonor_scores.json"))
sc=json.load(open(f"{EXP}/artifacts/native_arm_scores.json"))

def chains(lab):
    parts=lab.split("_",1)[1]
    return (parts[0],parts[1:]) if len(parts)>2 else (parts[0],parts[1])

def free_pool(lab):
    pdb=sc[lab]["pdb"]; c1,c2=chains(lab); out=[]
    for s in SEEDS:
        for k in SAMPLES:
            o=f"{F_CACHE}/{pdb}_{s}_{k}.json"
            cf=f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_summary_confidence_sample_{k}.json"
            if not (os.path.exists(o) and os.path.exists(cf)): continue
            try: d=json.load(open(o)); M=json.load(open(cf))["chain_pair_iptm"]
            except Exception: continue
            qs,ifs=d.get("dockq") or [],d.get("dockq_interfaces") or []
            for i,t in enumerate(ifs):
                if {t[0],t[1]}=={c1,c2} and i<len(qs):
                    x,y=ord(t[2])-65,ord(t[3])-65
                    if 0<=x<len(M) and 0<=y<len(M): out.append((M[x][y],qs[i]))
                    break
    return out

def tmpl_pool(lab):
    j=djobs[lab]; name=j["name"]; c1,c2=chains(lab); out=[]
    for s in SEEDS:
        for k in SAMPLES:
            o=f"{D_CACHE}/{lab}_{s}_{k}.json"
            cf=f"{D_OUT}/{name}/seed_{s}/predictions/{name}_summary_confidence_sample_{k}.json"
            if not (os.path.exists(o) and os.path.exists(cf)): continue
            try: d=json.load(open(o)); M=json.load(open(cf))["chain_pair_iptm"]
            except Exception: continue
            qs,ifs=d.get("dockq") or [],d.get("dockq_interfaces") or []
            for i,t in enumerate(ifs):
                if {t[0],t[1]}=={c1,c2} and i<len(qs):
                    x,y=ord(t[2])-65,ord(t[3])-65
                    if 0<=x<len(M) and 0<=y<len(M): out.append((M[x][y],qs[i]))
                    break
    return out

BINS=[(0.0,0.10),(0.10,0.23),(0.23,0.40),(0.40,0.60),(0.60,1.01)]
resc=[k for k in sorted(dq) if dq[k]["rescued"]]
agg={b:{"free":[], "tmpl":[]} for b in BINS}
rows=[]
print(f"{'interface':11} {'DockQ bin':>12} {'free ipTM':>10} {'n':>4} | {'tmpl ipTM':>10} {'n':>4} {'delta':>7}")
for lab in resc:
    f=free_pool(lab); t=tmpl_pool(lab)
    if not f or not t: continue
    for b in BINS:
        fv=[x[0] for x in f if b[0]<=x[1]<b[1]]; tv=[x[0] for x in t if b[0]<=x[1]<b[1]]
        agg[b]["free"]+=fv; agg[b]["tmpl"]+=tv
        if len(fv)>=3 and len(tv)>=3:
            d=st.median(tv)-st.median(fv)
            rows.append({"interface":lab,"bin":b,"free_med":round(st.median(fv),3),
                         "tmpl_med":round(st.median(tv),3),"delta":round(d,3),
                         "n_free":len(fv),"n_tmpl":len(tv)})
            print(f"{lab:11} {f'{b[0]:.2f}-{b[1]:.2f}':>12} {st.median(fv):10.3f} {len(fv):4d} | "
                  f"{st.median(tv):10.3f} {len(tv):4d} {d:+7.3f}")
print(f"\nPOOLED over the {len(resc)} rescued interfaces:")
print(f"{'DockQ bin':>12} {'free ipTM':>10} {'n':>5} | {'templated ipTM':>15} {'n':>5} {'delta':>8}")
for b in BINS:
    fv,tv=agg[b]["free"],agg[b]["tmpl"]
    if len(fv)>=5 and len(tv)>=5:
        print(f"{f'{b[0]:.2f}-{b[1]:.2f}':>12} {st.median(fv):10.3f} {len(fv):5d} | "
              f"{st.median(tv):15.3f} {len(tv):5d} {st.median(tv)-st.median(fv):+8.3f}")
json.dump(rows,open(f"{EXP}/artifacts/iptm_at_matched_dockq.json","w"),indent=1)
