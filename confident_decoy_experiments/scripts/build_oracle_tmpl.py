import json,os,sys,glob
sys.path.insert(0,"/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad")
from cifconv import convert
import gemmi
SC="/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl"
branch=sys.argv[1]  # decoys | goods
folds=json.load(open(f"{SC}/oracle_folds{'' if branch=='decoys' else '_goods'}.json"))
cache=json.load(open("/home/ebrahim/common/release_date_cache.json"))
TMPLDIR=f"{SC}/oracle_tmpl"; A3M=f"{SC}/oracle_a3m"; os.makedirs(TMPLDIR,exist_ok=True); os.makedirs(A3M,exist_ok=True)
def mut(a): return "A" if a!="A" else "G"
tc=[900]
built=[]
for pdb,chains in folds.items():
    base=json.load(open(f"/home/ebrahim/Protenix/foldbench_abag/jsons/{pdb}-final-updated.json"))
    obj=json.loads(json.dumps(base[0]))
    qseqs=[e["proteinChain"]["sequence"] for e in obj["sequences"] if "proteinChain" in e]
    cif2tid={}; pi=0; info=[]
    for e in obj["sequences"]:
        if "proteinChain" not in e: continue
        qi=str(pi)
        best=chains.get(qi)
        if best is None:  # no oracle fold for this chain -> leave it templated? better: skip pdb
            info.append((pi,"NO_FOLD")); pi+=1; continue
        cif=best["cif"]; match=best["model_chain"]
        if cif not in cif2tid:
            tid=f"o{tc[0]:03d}"; tc[0]+=1
            if tid not in cache: cache[tid]={"release_date":"2000-01-01"}
            open(f"/home/ebrahim/mmcif/{tid}.cif","w").write(convert(open(cif).read(),tid))
            cif2tid[cif]=tid
        tid=cif2tid[cif]; q=qseqs[pi]; L=len(q); hs=list(q)
        for p in (L//3,2*L//3): hs[p]=mut(hs[p])
        a3mp=f"{A3M}/{pdb}_q{pi}.a3m"
        open(a3mp,"w").write(f">query\n{q}\n>{tid}_{match}/1-{L} [subseq from] mol:protein length:{L}  oracle\n{''.join(hs)}\n")
        e["proteinChain"]["templatesPath"]=a3mp
        info.append((pi,f"{match} lddt{best['lddt']:.3f}")); pi+=1
    obj["name"]=f"{pdb}_oracle"
    json.dump([obj],open(f"{TMPLDIR}/{pdb}-oracle.json","w"))
    allok=all(x[1]!="NO_FOLD" for x in info)
    built.append((pdb,allok))
    print(f"{pdb}: {info}  {'OK' if allok else 'MISSING_FOLD'}")
json.dump(cache,open("/home/ebrahim/common/release_date_cache.json","w"))
# jobs
seeds=list(range(101,151))
jobs=[{"pdb":pdb,"json":f"{TMPLDIR}/{pdb}-oracle.json","seeds":seeds} for pdb,ok in built if ok]
json.dump(jobs,open(f"{SC}/oracle_{branch}_jobs.json","w"))
print(f"BUILD_ORACLE_DONE {branch}: {len(jobs)} jobs, {sum(1 for _,ok in built if not ok)} skipped")
