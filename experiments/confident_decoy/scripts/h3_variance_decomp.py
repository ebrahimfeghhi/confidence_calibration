#!/usr/bin/env python
"""DIAGNOSTIC 1 — is CDR-H3 error a DIFFUSION-side or TRUNK-side failure?

Architecture fact (protenix/model/protenix.py): the trunk runs ONCE per seed (recycling loop,
N_cycle) producing s,z; then N_sample=5 diffusion samples are drawn from THAT SAME s,z.
Therefore, for a fixed chain:
  - spread across the 5 samples WITHIN a seed  = pure DIFFUSION stochasticity (trunk held fixed)
  - spread of the per-seed MEANS across seeds  = TRUNK variation (different recycling/MSA sampling)

Metric: framework-aligned CDR-H3 backbone (CA) RMSD to native (same as scripts/h3_rmsd.py).
Decomposition (classic one-way ANOVA / variance components over seeds):
  within_sd  = sqrt(mean over seeds of the within-seed variance)   -> DIFFUSION
  between_sd = sd of the per-seed means                            -> TRUNK
  ratio = between_sd / within_sd.  ratio << 1 => diffusion dominates; >> 1 => trunk dominates.
Also reports the same split for the H3 conformation SPREAD independent of native
(pairwise H3 RMSD among samples within a seed vs across seeds) so the conclusion does not
depend on native at all.
Run with the Protenix venv python (gemmi). No OST, no GPU.
"""
import gemmi, os, json, glob, statistics, difflib, re
from concurrent.futures import ThreadPoolExecutor

EXP  = "/home/ebrahim/confident_decoy_experiments"
PRED = "/home/ebrahim/Protenix/foldbench_abag/predictions_template"
JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
GT   = "/home/ebrahim/FoldBench/examples/ground_truths"
NATD = "/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl/native_chains"
OUT  = f"{EXP}/artifacts/h3_variance"; os.makedirs(OUT, exist_ok=True)
chainmap = json.load(open(f"{EXP}/artifacts/oracle_chainmap.json"))
SEEDS = list(range(101, 151))

TLC = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
       'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def std(ch): return [r for r in ch if r.name in TLC]
def seqof(rl): return "".join(TLC[r.name] for r in rl)
def ca(r):
    a = r.find_atom("CA", "*"); return a.pos if a else None
def kind(s):
    hasW=re.search(r'WG.G[TS]',s);hasF=re.search(r'FG.G[TS]',s);w=s[:24]
    ntL=re.search(r'([DE]I[QV][ML]TQ|[DE]VVMTQ|EIVLTQ|QSVLTQ|DIQLTQ|SYELTQ|DIVMTQ)',w)
    ntH=re.search(r'([QE][VL][QK]L[VLEQ][EQ]S|QVKLEES|VESGG[GA]|VQSGAE|VETGGG|QESGGG)',w)
    if not (95<=len(s)<=275): return 'ANTIGEN'
    light=bool(ntL) or (bool(hasF) and not hasW); heavy=bool(ntH) or bool(hasW)
    if not(light or heavy): return 'ANTIGEN'
    if light and not ntH: return 'LIGHT'
    return 'HEAVY'
def find_h3(seq):
    for m in re.finditer(r'WG.G', seq):
        w=m.start(); c=seq.rfind('C',0,w)
        if c==-1: continue
        if 1<=w-(c+1)<=35: return c+1, w
    return None
def best_match_chain(st, target):
    best,br=None,-1
    for ch in st[0]:
        s=seqof(std(ch))
        if len(s)<4: continue
        r=difflib.SequenceMatcher(None,s,target,autojunk=False).ratio()
        if r>br: br,best=r,ch
    return best
def native_reslist(pdb, letter):
    p=f"{NATD}/{pdb}_{letter}.pdb"
    if os.path.exists(p):
        st=gemmi.read_structure(p); st.setup_entities(); return std(st[0][0])
    st=gemmi.read_structure(f"{GT}/{pdb}-assembly1.cif"); st.setup_entities()
    return std(next(c for c in st[0] if c.name==letter))

def pose_h3(args):
    """Return (seed, sample, h3_rmsd_to_native, h3_CA_coords_in_framework_frame)."""
    cif, seed, samp, hseq, nat_res, nat_seq = args
    try: st=gemmi.read_structure(cif); st.setup_entities()
    except Exception: return None
    ch=best_match_chain(st, hseq)
    if ch is None: return None
    mres=std(ch); mseq=seqof(mres); h3=find_h3(mseq)
    if h3 is None: return None
    s,e=h3; vh_end=min(len(mseq), e+15)
    h3_idx=set(range(s,e)); fw_idx=set(range(0,vh_end))-h3_idx
    sm=difflib.SequenceMatcher(None,mseq,nat_seq,autojunk=False)
    fw_m,fw_n,h3_m,h3_n=[],[],[],[]
    for a,b,n in sm.get_matching_blocks():
        for k in range(n):
            mi,ni=a+k,b+k
            cm=ca(mres[mi]); cn=ca(nat_res[ni])
            if cm is None or cn is None: continue
            if mi in fw_idx: fw_m.append(cm); fw_n.append(cn)
            elif mi in h3_idx: h3_m.append(cm); h3_n.append(cn)
    if len(fw_m)<3 or len(h3_m)<1: return None
    sup=gemmi.superpose_positions(fw_n, fw_m); T=sup.transform
    coords=[]; sq=0.0
    for cm,cn in zip(h3_m,h3_n):
        p=T.apply(cm); coords.append((p.x,p.y,p.z))
        sq += (p.x-cn.x)**2+(p.y-cn.y)**2+(p.z-cn.z)**2
    return (seed, samp, (sq/len(h3_m))**0.5, coords)

def pair_rmsd(a,b):
    n=min(len(a),len(b))
    if n==0: return None
    s=sum((a[i][0]-b[i][0])**2+(a[i][1]-b[i][1])**2+(a[i][2]-b[i][2])**2 for i in range(n))
    return (s/n)**0.5

rows=[]
RAW={}
meta=json.load(open(f"{EXP}/artifacts/meta.json"))
for pdb in sorted(set(meta[k]["pdb"] for k in meta)):
    base=json.load(open(f"{JDIR}/{pdb}-final-updated.json")); o=base[0] if isinstance(base,list) else base
    qseqs=[e["proteinChain"]["sequence"] for e in o["sequences"] if "proteinChain" in e]
    for hq,hseq in enumerate(qseqs):
        if kind(hseq)!='HEAVY' or find_h3(hseq) is None: continue
        letter=chainmap[pdb].get(str(hq))
        if letter is None: continue
        nat_res=native_reslist(pdb, letter); nat_seq=seqof(nat_res)
        args=[]
        for s in SEEDS:
            for k in range(5):
                c=f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif"
                if os.path.exists(c): args.append((c,s,k,hseq,nat_res,nat_seq))
        res=[x for x in ThreadPoolExecutor(8).map(pose_h3,args) if x]
        if not res: continue
        byseed={}
        for seed,samp,rm,co in res: byseed.setdefault(seed,[]).append((rm,co))
        byseed={k:v for k,v in byseed.items() if len(v)>=2}
        if len(byseed)<2: continue
        # --- vs-native decomposition ---
        within_vars=[statistics.variance([r for r,_ in v]) for v in byseed.values() if len(v)>=2]
        seed_means=[statistics.mean([r for r,_ in v]) for v in byseed.values()]
        within_sd=(statistics.mean(within_vars))**0.5
        between_sd=statistics.stdev(seed_means)
        # --- native-free conformational spread ---
        win_pairs=[]
        for v in byseed.values():
            for i in range(len(v)):
                for j in range(i+1,len(v)):
                    d=pair_rmsd(v[i][1],v[j][1])
                    if d is not None: win_pairs.append(d)
        seeds_l=sorted(byseed)
        btw_pairs=[]
        for i in range(len(seeds_l)):
            for j in range(i+1,len(seeds_l)):
                d=pair_rmsd(byseed[seeds_l[i]][0][1], byseed[seeds_l[j]][0][1])
                if d is not None: btw_pairs.append(d)
        # native H3 coords in the SAME framework frame (native superposed on itself = identity)
        _s,_e=find_h3(seqof(nat_res)) if find_h3(seqof(nat_res)) else (None,None)
        nat_h3=None
        if _s is not None:
            nat_h3=[(ca(r).x,ca(r).y,ca(r).z) for r in nat_res[_s:_e] if ca(r)]
        if nat_h3:
            RAW[f"{pdb}_H{hq}"]={"native":nat_h3,"n_poses":len(res),
                                 "byseed":{str(k):[c for _,c in v] for k,v in byseed.items()}}
        h3len=find_h3(hseq); h3len=h3len[1]-h3len[0]
        row={"pdb":pdb,"heavy_qidx":hq,"h3_len":h3len,"n_seeds":len(byseed),"n_poses":len(res),
             "within_seed_sd_DIFFUSION":round(within_sd,3),
             "between_seed_sd_TRUNK":round(between_sd,3),
             "ratio_trunk_over_diffusion":round(between_sd/within_sd,2) if within_sd>0 else None,
             "mean_h3_rmsd":round(statistics.mean([r for r,_ in sum(byseed.values(),[])]),2),
             "pairwise_within_seed_A":round(statistics.mean(win_pairs),2) if win_pairs else None,
             "pairwise_between_seed_A":round(statistics.mean(btw_pairs),2) if btw_pairs else None}
        rows.append(row)
        print(f"{pdb}_H{hq} H3len={h3len:2d} seeds={len(byseed):2d} | meanRMSD={row['mean_h3_rmsd']:5.2f}A | "
              f"WITHIN(diffusion)={within_sd:.3f} BETWEEN(trunk)={between_sd:.3f} ratio={row['ratio_trunk_over_diffusion']} | "
              f"pairwise within={row['pairwise_within_seed_A']}A between={row['pairwise_between_seed_A']}A", flush=True)

json.dump(rows, open(f"{OUT}/h3_variance_results.json","w"), indent=1)
json.dump(RAW, open(f"{OUT}/raw_coords.json","w"))
import statistics as st
if rows:
    rr=[r["ratio_trunk_over_diffusion"] for r in rows if r["ratio_trunk_over_diffusion"]]
    pw=[(r["pairwise_within_seed_A"],r["pairwise_between_seed_A"]) for r in rows if r["pairwise_within_seed_A"] and r["pairwise_between_seed_A"]]
    print(f"\nMEDIAN ratio trunk/diffusion = {st.median(rr):.2f}  (>1 trunk-dominated, <1 diffusion-dominated)")
    print(f"MEDIAN pairwise H3 spread: within-seed(diffusion) {st.median([a for a,_ in pw]):.2f} A vs between-seed(trunk) {st.median([b for _,b in pw]):.2f} A")
print("WROTE", f"{OUT}/h3_variance_results.json")
