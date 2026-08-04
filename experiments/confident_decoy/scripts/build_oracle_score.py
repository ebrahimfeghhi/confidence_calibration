import json,glob,os,re,subprocess,statistics as st
from concurrent.futures import ThreadPoolExecutor
import gemmi
SC="/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl"
PRED="/home/ebrahim/Protenix/foldbench_abag/predictions_template"
OST="/home/ebrahim/miniconda3/envs/foldbench/bin/ost"
ENV=dict(os.environ); ENV["LD_LIBRARY_PATH"]="/home/ebrahim/miniconda3/envs/foldbench/lib"
TMP=f"{SC}/oracle_tmp"; os.makedirs(TMP,exist_ok=True); os.makedirs(f"{SC}/ost_oracle",exist_ok=True)
import sys
TLC={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def seqof(ch): return "".join(TLC[r.name] for r in ch if r.name in TLC)
# argv: [1]=chainmap json  [2]=native_chains dir  [3]=out folds json   (defaults = decoys)
CHAINMAP=sys.argv[1] if len(sys.argv)>1 else f"{SC}/oracle_chainmap.json"
NATDIR=sys.argv[2] if len(sys.argv)>2 else f"{SC}/native_chains"
OUTJSON=sys.argv[3] if len(sys.argv)>3 else f"{SC}/oracle_folds.json"
chainmap=json.load(open(CHAINMAP))   # {pdb:{qidx:native_letter}}
# native seqs
natseq={}
for pdb,d in chainmap.items():
    for qi,letter in d.items():
        stn=gemmi.read_structure(f"{NATDIR}/{pdb}_{letter}.pdb"); stn.setup_entities()
        natseq[(pdb,qi)]=("".join(seqof(c) for c in stn[0]))
def kmers(s,k=8): return set(s[i:i+k] for i in range(len(s)-k+1)) if len(s)>=k else ({s} if s else set())
def matchscore(s,target):
    if not s or not target: return 0.0
    if s==target: return 1.0
    if target in s or s in target: return 0.99
    kt=kmers(target)
    if not kt: return 0.0
    return len(kt & kmers(s))/len(kt)   # containment of native k-mers in model chain (offset-robust)
def extract(cif,target,outp):
    stm=gemmi.read_structure(cif); stm.setup_entities()
    best=None;bs=-1
    for ch in stm[0]:
        s=seqof(ch)
        if len(s)<5: continue
        sc=matchscore(s,target)
        if sc>bs: bs=sc;best=ch
    if best is None or bs<0.5: return None,None
    ns=gemmi.Structure(); ns.add_model(gemmi.Model("1")); nc=gemmi.Chain("A")
    for r in best: nc.add_residue(r)
    ns[0].add_chain(nc); ns.setup_entities(); ns.write_pdb(outp)
    return outp,best.name
def lddt_job(a):
    pdb,qi,cif,seed,samp=a
    o=f"{SC}/ost_oracle/{pdb}_{qi}_{seed}_{samp}.json"
    if os.path.exists(o) and os.path.getsize(o)>0:
        try: return (pdb,qi,seed,samp,cif,json.load(open(o)).get("lddt"),json.load(open(o)).get("_mc"))
        except: pass
    tmpc=f"{TMP}/{pdb}_{qi}_{seed}_{samp}.pdb"
    mc,cn=extract(cif,natseq[(pdb,qi)],tmpc)
    if mc is None: return (pdb,qi,seed,samp,cif,None,None)
    ref=f"{NATDIR}/{pdb}_{chainmap[pdb][qi]}.pdb"
    subprocess.run(["taskset","-c","0-39",OST,"compare-structures","-m",mc,"-r",ref,"-o",o,"--lddt","--lddt-no-stereochecks","--fault-tolerant"],env=ENV,capture_output=True,timeout=120)
    v=None
    try:
        d=json.load(open(o)); v=d.get("lddt"); d["_mc"]=cn; json.dump(d,open(o,"w"))
    except: pass
    if os.path.exists(tmpc): os.remove(tmpc)
    return (pdb,qi,seed,samp,cif,v,cn)
SEEDS=set(range(101,151))   # canonical 250-pose pool (50 seeds x 5 samples), seed-matched to free/native/agonly
jobs=[]
for pdb,d in chainmap.items():
    poses=sorted(glob.glob(f"{PRED}/{pdb}/seed_*/predictions/{pdb}_sample_*.cif"))
    for cif in poses:
        seed=int(re.search(r'seed_(\d+)',cif).group(1)); samp=int(re.search(r'sample_(\d+)',cif).group(1))
        if seed not in SEEDS: continue
        for qi in d: jobs.append((pdb,qi,cif,seed,samp))
print("oracle lddt jobs:",len(jobs),flush=True)
best={}   # (pdb,qi) -> dict
done=[0]
def run(a):
    r=lddt_job(a); done[0]+=1
    if done[0]%1500==0: print(f"  {done[0]}/{len(jobs)}",flush=True)
    return r
with ThreadPoolExecutor(max_workers=20) as ex:
    for pdb,qi,seed,samp,cif,v,cn in ex.map(run,jobs):
        if v is None: continue
        k=(pdb,qi)
        if k not in best or v>best[k]["lddt"]:
            best[k]={"lddt":v,"cif":cif,"model_chain":cn,"seed":seed,"sample":samp}
out={}
for (pdb,qi),info in best.items(): out.setdefault(pdb,{})[qi]=info
json.dump(out,open(OUTJSON,"w"),indent=1)
print("ORACLE_SCORE_DONE",flush=True)
for pdb in out:
    print(pdb, {qi:(round(v["lddt"],3),f"seed{v['seed']}/s{v['sample']}") for qi,v in out[pdb].items()})
