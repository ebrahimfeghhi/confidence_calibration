import json,os,sys,glob
SC="/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl"
TMPLDIR=f"{SC}/abonly_tmpl"; os.makedirs(TMPLDIR,exist_ok=True)
branch=sys.argv[1]  # "decoys" or "goods"
antigen_idx=json.load(open(f"{SC}/agonly_antigen_idx.json"))

def build(pdb, native_json, antigen_qidx):
    o=json.load(open(native_json)); obj=o[0] if isinstance(o,list) else o
    obj=json.loads(json.dumps(obj))  # copy
    pi=0; kept=[]; freed=[]
    for e in obj["sequences"]:
        if "proteinChain" not in e: continue
        if pi in antigen_qidx:
            e["proteinChain"].pop("templatesPath",None); freed.append(pi)   # antigen FREE
        else:
            kept.append((pi, bool(e["proteinChain"].get("templatesPath"))))  # antibody keeps native template
        pi+=1
    obj["name"]=f"{pdb}_abonly"
    json.dump([obj],open(f"{TMPLDIR}/{pdb}-abonly.json","w"))
    return kept,freed

if branch=="decoys":
    for pdb,aidx in antigen_idx.items():
        nj=f"{SC}/../tmpl/all/{pdb}-nativetmpl.json"
        if not os.path.exists(nj): print(f"MISSING native template {pdb}: {nj}"); continue
        kept,freed=build(pdb,nj,aidx)
        # sanity: every kept antibody chain MUST still carry a template
        ok=all(t for _,t in kept)
        print(f"{pdb}: antibody kept(qidx,hasTmpl)={kept} antigen freed={freed}  {'OK' if ok and kept else 'CHECK'}")
elif branch=="goods":
    good_aidx=json.load(open(f"{SC}/abonly_good_antigen_idx.json"))
    safety=json.load(open(f"{SC}/safety15.json"))
    for x in safety:
        pdb=x["pdb"]; aidx=good_aidx[pdb]
        nj=f"{SC}/safety_tmpl/{pdb}-nativetmpl.json"
        if not os.path.exists(nj): print(f"MISSING good native template {pdb}: {nj}"); continue
        kept,freed=build(pdb,nj,aidx)
        ok=all(t for _,t in kept)
        print(f"{pdb}: antibody kept={kept} antigen freed={freed}  {'OK' if ok and kept else 'CHECK'}")
print("BUILD_DONE",branch)
