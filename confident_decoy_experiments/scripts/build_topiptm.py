import json,os,sys
sys.path.insert(0,"/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad")
from cifconv import convert
import gemmi
SC="/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl"
picks=json.load(open(f"{SC}/topiptm_picks.json"))
cache=json.load(open("/home/ebrahim/common/release_date_cache.json"))
TMPLDIR=f"{SC}/topiptm_tmpl"; A3M=f"{SC}/topiptm_a3m"; os.makedirs(TMPLDIR,exist_ok=True); os.makedirs(A3M,exist_ok=True)
TLC={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def seqof(ch): return "".join(TLC[r.name] for r in ch if r.name in TLC)
def mut(a): return "A" if a!="A" else "G"
tc=[1400]; built=[]
for pdb,pk in picks.items():
    cif=pk["cif"]
    base=json.load(open(f"/home/ebrahim/Protenix/foldbench_abag/jsons/{pdb}-final-updated.json"))
    obj=json.loads(json.dumps(base[0]))
    qseqs=[e["proteinChain"]["sequence"] for e in obj["sequences"] if "proteinChain" in e]
    st=gemmi.read_structure(cif); cs={ch.name:seqof(ch) for ch in st[0]}
    # one tid for the whole pose (all chains from same cif)
    tid=f"t{tc[0]:03d}"; tc[0]+=1
    if tid not in cache: cache[tid]={"release_date":"2000-01-01"}
    open(f"/home/ebrahim/mmcif/{tid}.cif","w").write(convert(open(cif).read(),tid))
    pi=0; info=[]; ok=True
    for e in obj["sequences"]:
        if "proteinChain" not in e: continue
        q=qseqs[pi]
        match=next((cn for cn,s in cs.items() if s==q or (len(q)>20 and (s in q or q in s))),None)
        if match is None: info.append((pi,"NOMATCH")); ok=False; pi+=1; continue
        L=len(q); hs=list(q)
        for p in (L//3,2*L//3): hs[p]=mut(hs[p])
        a3mp=f"{A3M}/{pdb}_q{pi}.a3m"
        open(a3mp,"w").write(f">query\n{q}\n>{tid}_{match}/1-{L} [subseq from] mol:protein length:{L}  topiptm\n{''.join(hs)}\n")
        e["proteinChain"]["templatesPath"]=a3mp
        info.append((pi,match)); pi+=1
    obj["name"]=f"{pdb}_topiptm"
    json.dump([obj],open(f"{TMPLDIR}/{pdb}-topiptm.json","w"))
    built.append((pdb,ok)); print(f"{pdb}: seed{pk['seed']}/s{pk['sample']} ipTM{pk['pair_iptm']:.3f} chains->{info}  {'OK' if ok else 'CHECK'}")
json.dump(cache,open("/home/ebrahim/common/release_date_cache.json","w"))
jobs=[{"pdb":pdb,"json":f"{TMPLDIR}/{pdb}-topiptm.json","seeds":list(range(101,151))} for pdb,ok in built if ok]
json.dump(jobs,open(f"{SC}/topiptm_decoys_jobs.json","w"))
print(f"BUILD_TOPIPTM_DONE {len(jobs)} jobs")
