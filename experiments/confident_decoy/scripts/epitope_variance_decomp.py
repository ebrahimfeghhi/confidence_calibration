#!/usr/bin/env python
"""DIAGNOSTIC 1 for the EPITOPE — diffusion-side or trunk-side? (companion to h3_variance_decomp.py)

Same logic: trunk runs ONCE per seed; 5 diffusion samples share that s,z. So
  within-seed spread  = DIFFUSION stochasticity (trunk fixed)
  between-seed spread = TRUNK variation
Metric: antigen epitope backbone (CA) RMSD to native after superposing on the antigen's
NON-epitope residues (the analog of framework-alignment for CDR-H3). If a chain has <10
non-epitope residues (short peptides) we fall back to whole-chain superposition and flag it.
Epitope residues are copy-aware (reuses resolve_unit from binding_dependence.py).
Run with Protenix venv python.
"""
import gemmi, os, json, glob, statistics, difflib, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/home/ebrahim/confident_decoy_experiments/scripts")
import importlib.util
spec = importlib.util.spec_from_file_location("bd", "/home/ebrahim/confident_decoy_experiments/scripts/binding_dependence.py")
src = open("/home/ebrahim/confident_decoy_experiments/scripts/binding_dependence.py").read().split("rows = []")[0]
bd = {}
exec(compile(src, "bd", "exec"), bd)
resolve_unit = bd["resolve_unit"]; std = bd["std"]; seqof = bd["seqof"]

EXP  = "/home/ebrahim/confident_decoy_experiments"
PRED = "/home/ebrahim/Protenix/foldbench_abag/predictions_template"
JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
GT   = "/home/ebrahim/FoldBench/examples/ground_truths"
NATD = "/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl/native_chains"
OUT  = f"{EXP}/artifacts/epitope_variance"; os.makedirs(OUT, exist_ok=True)
agidx = json.load(open(f"{EXP}/artifacts/agonly_antigen_idx.json"))
SEEDS = list(range(101, 151))
TLC = bd["TLC"]
def ca(r):
    a = r.find_atom("CA","*"); return a.pos if a else None
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

def pose_epi(args):
    cif, seed, samp, qseq, nat_res, nat_seq, epi_nat = args
    try: st=gemmi.read_structure(cif); st.setup_entities()
    except Exception: return None
    ch=best_match_chain(st, qseq)
    if ch is None: return None
    mres=std(ch); mseq=seqof(mres)
    sm=difflib.SequenceMatcher(None,mseq,nat_seq,autojunk=False)
    natnums=[r.seqid.num for r in nat_res]
    epi_m,epi_n,fr_m,fr_n=[],[],[],[]
    for a,b,n in sm.get_matching_blocks():
        for k in range(n):
            mi,ni=a+k,b+k
            cm=ca(mres[mi]); cn=ca(nat_res[ni])
            if cm is None or cn is None: continue
            if natnums[ni] in epi_nat: epi_m.append(cm); epi_n.append(cn)
            else: fr_m.append(cm); fr_n.append(cn)
    if len(epi_m)<1: return None
    if len(fr_m)>=10: sup=gemmi.superpose_positions(fr_n,fr_m); frame="non-epitope"
    else:
        allm=epi_m+fr_m; alln=epi_n+fr_n
        if len(allm)<3: return None
        sup=gemmi.superpose_positions(alln,allm); frame="whole-chain"
    T=sup.transform; coords=[]; sq=0.0
    for cm,cn in zip(epi_m,epi_n):
        p=T.apply(cm); coords.append((p.x,p.y,p.z))
        sq += (p.x-cn.x)**2+(p.y-cn.y)**2+(p.z-cn.z)**2
    return (seed,samp,(sq/len(epi_m))**0.5,coords,frame,len(fr_m))
def pair_rmsd(a,b):
    n=min(len(a),len(b))
    if n==0: return None
    s=sum((a[i][0]-b[i][0])**2+(a[i][1]-b[i][1])**2+(a[i][2]-b[i][2])**2 for i in range(n))
    return (s/n)**0.5

rows=[]
RAW={}
for pdb in sorted(agidx):
    aidx=[int(x) for x in agidx[pdb]]
    base=json.load(open(f"{JDIR}/{pdb}-final-updated.json")); o=base[0] if isinstance(base,list) else base
    qseqs=[e["proteinChain"]["sequence"] for e in o["sequences"] if "proteinChain" in e]
    letters,ifaces=resolve_unit(pdb,qseqs,aidx)
    for qi in aidx:
        letter=letters.get(qi); epi=ifaces.get(qi,set())
        if letter is None or not epi: continue
        nat_res=native_reslist(pdb,letter); nat_seq=seqof(nat_res)
        args=[]
        for s in SEEDS:
            for k in range(5):
                c=f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif"
                if os.path.exists(c): args.append((c,s,k,qseqs[qi],nat_res,nat_seq,epi))
        res=[x for x in ThreadPoolExecutor(8).map(pose_epi,args) if x]
        if not res: continue
        frame=res[0][4]; nfr=res[0][5]
        byseed={}
        for seed,samp,rm,co,_,_ in res: byseed.setdefault(seed,[]).append((rm,co))
        byseed={k:v for k,v in byseed.items() if len(v)>=2}
        if len(byseed)<2: continue
        within_sd=(statistics.mean([statistics.variance([r for r,_ in v]) for v in byseed.values()]))**0.5
        seed_means=[statistics.mean([r for r,_ in v]) for v in byseed.values()]
        between_sd=statistics.stdev(seed_means)
        win=[];btw=[]
        for v in byseed.values():
            for i in range(len(v)):
                for j in range(i+1,len(v)):
                    d=pair_rmsd(v[i][1],v[j][1])
                    if d: win.append(d)
        sl=sorted(byseed)
        for i in range(len(sl)):
            for j in range(i+1,len(sl)):
                d=pair_rmsd(byseed[sl[i]][0][1],byseed[sl[j]][0][1])
                if d: btw.append(d)
        natnums=[r.seqid.num for r in nat_res]
        nat_epi=[(ca(r).x,ca(r).y,ca(r).z) for r,nn in zip(nat_res,natnums) if nn in epi and ca(r)]
        if nat_epi and len(nat_epi)==len(list(byseed.values())[0][0][1]):
            RAW[f"{pdb}_q{qi}"]={"native":nat_epi,"n_poses":len(res),
                                 "byseed":{str(k):[c for _,c in v] for k,v in byseed.items()}}
        mean_rm=statistics.mean([r for r,_ in sum(byseed.values(),[])])
        tot_sd=(within_sd**2+between_sd**2)**0.5
        row={"pdb":pdb,"antigen_qidx":qi,"chain_len":len(qseqs[qi]),"n_epitope_res":len(epi),
             "align_frame":frame,"n_frame_res":nfr,"n_seeds":len(byseed),
             "mean_epitope_rmsd":round(mean_rm,2),
             "within_seed_sd_DIFFUSION":round(within_sd,3),"between_seed_sd_TRUNK":round(between_sd,3),
             "bias_over_total_sd":round(mean_rm/tot_sd,1) if tot_sd>0 else None,
             "ratio_trunk_over_diffusion":round(between_sd/within_sd,2) if within_sd>0 else None,
             "pairwise_within_seed_A":round(statistics.mean(win),2) if win else None,
             "pairwise_between_seed_A":round(statistics.mean(btw),2) if btw else None}
        rows.append(row)
        print(f"{pdb} q{qi} len={len(qseqs[qi]):4d} epi={len(epi):3d} frame={frame:12}| meanRMSD={mean_rm:6.2f}A "
              f"bias/SD={row['bias_over_total_sd']} | WITHIN(diff)={within_sd:.3f} BETWEEN(trunk)={between_sd:.3f} "
              f"ratio={row['ratio_trunk_over_diffusion']} | pair within={row['pairwise_within_seed_A']} between={row['pairwise_between_seed_A']}", flush=True)
json.dump(rows,open(f"{OUT}/epitope_variance_results.json","w"),indent=1)
json.dump(RAW,open(f"{OUT}/raw_coords.json","w"))
if rows:
    rr=[r["ratio_trunk_over_diffusion"] for r in rows if r["ratio_trunk_over_diffusion"]]
    print(f"\nMEDIAN ratio trunk/diffusion = {statistics.median(rr):.2f}")
    pw=[(r["pairwise_within_seed_A"],r["pairwise_between_seed_A"]) for r in rows if r["pairwise_within_seed_A"] and r["pairwise_between_seed_A"]]
    print(f"MEDIAN pairwise epitope spread: within(diffusion) {statistics.median([a for a,_ in pw]):.2f} A vs between(trunk) {statistics.median([b for _,b in pw]):.2f} A")
print("WROTE",f"{OUT}/epitope_variance_results.json")
