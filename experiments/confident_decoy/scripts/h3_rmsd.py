#!/usr/bin/env python
"""Framework-aligned CDR-H3 backbone RMSD (Å) — the binding-relevant loop metric, to pressure-test
the lDDT-based generation-vs-selection verdict (README 2c).

Per antibody heavy chain, for every pose (seeds 101-150, 250):
  1. superpose model heavy chain onto native on the VH FRAMEWORK CA atoms (all VH residues EXCEPT H3;
     CH1 excluded via VH-end cutoff), 2. measure H3 backbone (CA) RMSD after that superposition.
Reports best-of-250 (MIN rmsd = closest H3 the model ever draws), median, and the RMSD of the poses
picked by global pLDDT and by H3-region pLDDT. Framework alignment RMSD reported as a sanity check.
Rough antibody conventions: <=1.5 A near-native, 1.5-3 A approximate, >3-4 A wrong loop.
Pure gemmi geometry (no OST). Run with Protenix venv python.
"""
import gemmi, os, json, glob, statistics, difflib, re
from concurrent.futures import ThreadPoolExecutor

EXP  = "/home/ebrahim/confident_decoy_experiments"
PRED = "/home/ebrahim/Protenix/foldbench_abag/predictions_template"
JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
GT   = "/home/ebrahim/FoldBench/examples/ground_truths"
NATD = "/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl/native_chains"
OUT  = f"{EXP}/artifacts/h3_inpool"; os.makedirs(OUT, exist_ok=True)
chainmap = json.load(open(f"{EXP}/artifacts/oracle_chainmap.json"))

TLC = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
       'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def std_res(ch): return [r for r in ch if r.name in TLC]
def seq_of(reslist): return "".join(TLC[r.name] for r in reslist)
def ca(r):
    a = r.find_atom("CA", "*")
    return a.pos if a else None
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
        s=seq_of(std_res(ch))
        if len(s)<4: continue
        r=difflib.SequenceMatcher(None,s,target,autojunk=False).ratio()
        if r>br: br,best=r,ch
    return best

def native_reslist(pdb, letter):
    p=f"{NATD}/{pdb}_{letter}.pdb"
    st=gemmi.read_structure(p) if os.path.exists(p) else gemmi.read_structure(f"{GT}/{pdb}-assembly1.cif")
    st.setup_entities()
    ch = st[0][0] if os.path.exists(p) else next(c for c in st[0] if c.name==letter)
    return std_res(ch)

def poses(pdb):
    out=[]
    for s in range(101,151): out += sorted(glob.glob(f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_*.cif"))
    return out

def score_pose(args):
    cif, hseq, nat_res, nat_seq = args
    try: st=gemmi.read_structure(cif); st.setup_entities()
    except Exception: return None
    ch=best_match_chain(st, hseq)
    if ch is None: return None
    m_res=std_res(ch); m_seq=seq_of(m_res)
    h3=find_h3(m_seq)
    if h3 is None: return None
    s,e=h3; vh_end=min(len(m_seq), e+15)
    h3_idx=set(range(s,e)); fw_idx=set(range(0,vh_end))-h3_idx
    # pLDDT (b_iso/100)
    chain_pl=statistics.mean([at.b_iso for r in m_res for at in r])/100.0
    h3_atoms=[at.b_iso for i,r in enumerate(m_res) if i in h3_idx for at in r]
    h3_pl=statistics.mean(h3_atoms)/100.0 if h3_atoms else 0.0
    # pair model<->native by sequence
    sm=difflib.SequenceMatcher(None, m_seq, nat_seq, autojunk=False)
    fw_m=[]; fw_n=[]; h3_m=[]; h3_n=[]
    for a,b,n in sm.get_matching_blocks():
        for k in range(n):
            mi=a+k; ni=b+k
            cm=ca(m_res[mi]); cn=ca(nat_res[ni])
            if cm is None or cn is None: continue
            if mi in fw_idx: fw_m.append(cm); fw_n.append(cn)
            elif mi in h3_idx: h3_m.append(cm); h3_n.append(cn)
    if len(fw_m)<3 or len(h3_m)<1: return None
    sup=gemmi.superpose_positions(fw_n, fw_m)   # map model(moving) onto native(fixed)
    T=sup.transform
    sq=0.0
    for cm,cn in zip(h3_m,h3_n):
        p=T.apply(cm); sq += (p.x-cn.x)**2 + (p.y-cn.y)**2 + (p.z-cn.z)**2
    h3_rmsd=(sq/len(h3_m))**0.5
    return {"h3_rmsd":round(h3_rmsd,3), "fw_rmsd":round(sup.rmsd,3), "n_h3":len(h3_m),
            "chain_plddt":chain_pl, "h3_plddt":h3_pl, "cif":os.path.relpath(cif,PRED)}

results=[]
meta=json.load(open(f"{EXP}/artifacts/meta.json"))
for pdb in sorted(set(meta[k]["pdb"] for k in meta)):
    base=json.load(open(f"{JDIR}/{pdb}-final-updated.json")); o=base[0] if isinstance(base,list) else base
    qseqs=[e["proteinChain"]["sequence"] for e in o["sequences"] if "proteinChain" in e]
    for hq,hseq in enumerate(qseqs):
        if kind(hseq)!='HEAVY': continue
        if find_h3(hseq) is None: continue
        natlet=chainmap[pdb].get(str(hq))
        if natlet is None: continue
        nat_res=native_reslist(pdb, natlet); nat_seq=seq_of(nat_res)
        args=[(c,hseq,nat_res,nat_seq) for c in poses(pdb)]
        scored=[x for x in ThreadPoolExecutor(16).map(score_pose,args) if x]
        if not scored: print(f"{pdb}_H{hq}: none", flush=True); continue
        rmsds=[x["h3_rmsd"] for x in scored]
        best=min(scored,key=lambda x:x["h3_rmsd"])          # closest H3 in pool
        gpick=max(scored,key=lambda x:x["chain_plddt"])       # global pLDDT pick
        hpick=max(scored,key=lambda x:x["h3_plddt"])          # H3 pLDDT pick
        h3len=find_h3(hseq); h3len=h3len[1]-h3len[0]
        row={"pdb":pdb,"heavy_qidx":hq,"native_chain":natlet,"h3_len":h3len,"n_poses":len(scored),
             "best_h3rmsd":best["h3_rmsd"],"median_h3rmsd":round(statistics.median(rmsds),3),
             "worst_h3rmsd":max(rmsds),"median_fw_rmsd":round(statistics.median([x["fw_rmsd"] for x in scored]),3),
             "globalplddt_pick_h3rmsd":gpick["h3_rmsd"],"h3plddt_pick_h3rmsd":hpick["h3_rmsd"],
             "n_h3_res":best["n_h3"]}
        results.append(row)
        print(f"{pdb}_H{hq:<2} H3len={h3len:2d} n={len(scored):3d} nH3res={best['n_h3']:2d} fwRMSD~{row['median_fw_rmsd']:.2f} | "
              f"best(min)={row['best_h3rmsd']:.2f}A median={row['median_h3rmsd']:.2f}A | "
              f"globalPLDDTpick={row['globalplddt_pick_h3rmsd']:.2f}A H3PLDDTpick={row['h3plddt_pick_h3rmsd']:.2f}A", flush=True)

json.dump(results, open(f"{OUT}/h3_rmsd_results.json","w"), indent=1)
print("\nWROTE", f"{OUT}/h3_rmsd_results.json")
