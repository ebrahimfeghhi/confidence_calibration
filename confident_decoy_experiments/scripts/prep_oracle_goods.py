import json,os
import gemmi
SC="/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl"
GT="/home/ebrahim/FoldBench/examples/ground_truths"
NATDIR=f"{SC}/native_chains_goods"; os.makedirs(NATDIR,exist_ok=True)
TLC={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def seqof(ch): return "".join(TLC[r.name] for r in ch if r.name in TLC)
safety=json.load(open(f"{SC}/safety15.json"))
goodpdbs=[]
for x in safety:
    if x["pdb"] not in goodpdbs: goodpdbs.append(x["pdb"])
chainmap={}
for pdb in goodpdbs:
    base=json.load(open(f"/home/ebrahim/Protenix/foldbench_abag/jsons/{pdb}-final-updated.json"))
    qseqs=[e["proteinChain"]["sequence"] for e in base[0]["sequences"] if "proteinChain" in e]
    st=gemmi.read_structure(f"{GT}/{pdb}-assembly1.cif"); st.setup_entities()
    natchains=[(ch.name,seqof(ch)) for ch in st[0] if seqof(ch)]
    used=set(); cm={}
    for qi,q in enumerate(qseqs):
        best=None;bs=-1
        for name,s in natchains:
            if name in used: continue
            if s==q: sc=1.0
            elif len(q)>20 and (q in s or s in q): sc=0.97
            else:
                m=sum(1 for a,b in zip(s,q) if a==b); sc=m/max(len(q),1)
            if sc>bs: bs=sc; best=name
        if best is None or bs<0.7:
            print(f"{pdb} q{qi}: NO NATIVE MATCH (bs={bs:.2f})"); continue
        used.add(best); cm[str(qi)]=best
        # extract native chain
        outp=f"{NATDIR}/{pdb}_{best}.pdb"
        if not os.path.exists(outp):
            ns=gemmi.Structure(); ns.add_model(gemmi.Model("1"))
            for ch in st[0]:
                if ch.name==best: ns[0].add_chain(ch); break
            ns.setup_entities(); ns.write_pdb(outp)
    chainmap[pdb]=cm
    print(f"{pdb}: {cm}  ({'ALL' if len(cm)==len(qseqs) else 'PARTIAL %d/%d'%(len(cm),len(qseqs))})")
json.dump(chainmap,open(f"{SC}/oracle_chainmap_goods.json","w"))
print("PREP_GOODS_DONE",len(chainmap))
