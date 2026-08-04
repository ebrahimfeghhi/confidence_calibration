import json,os
import gemmi
SC="/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl"
PRED="/home/ebrahim/Protenix/foldbench_abag/predictions_template"
cm=json.load(open(f"{SC}/oracle_chainmap.json"))
antigen=json.load(open(f"{SC}/agonly_antigen_idx.json"))
man=json.load(open(f"{SC}/paratope_manifest.json"))
# index oracle-tag entries to reuse paratope/native/ab_seq
ref={(e["pdb"],e["qi"]):e for e in man if e["tag"]=="oracle"}
MDIR=f"{SC}/para_models"; os.makedirs(MDIR,exist_ok=True)
def cl(pdb,qi,s,k):
    try:
        d=json.load(open(f"{SC}/ost_oracle/{pdb}_{qi}_{s}_{k}.json")); return d.get("lddt"),d.get("_mc")
    except: return None,None
def extract(cif,mc,outp):
    stm=gemmi.read_structure(cif); stm.setup_entities()
    ch=next((c for c in stm[0] if c.name==mc),None)
    if ch is None: return False
    ns=gemmi.Structure(); ns.add_model(gemmi.Model("1")); nc=gemmi.Chain("A")
    for r in ch: nc.add_residue(r)
    ns[0].add_chain(nc); ns.setup_entities(); ns.write_pdb(outp); return True
new=[]
for pdb,d in cm.items():
    abqis=[qi for qi in d if int(qi) not in antigen[pdb]]
    if len(abqis)<2: continue
    q1,q2=abqis[0],abqis[1]
    poses={}
    for s in range(101,151):
        for k in range(5):
            a,mca=cl(pdb,q1,s,k); b,mcb=cl(pdb,q2,s,k)
            if a is not None and b is not None: poses[(s,k)]=(a,b,mca,mcb)
    (s,k),(a,b,mca,mcb)=max(poses.items(),key=lambda x:(x[1][0]+x[1][1])/2)
    cif=f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif"
    for qi,mc in [(q1,mca),(q2,mcb)]:
        e=ref.get((pdb,qi))
        if not e: continue
        outp=f"{MDIR}/{pdb}_{qi}_samepose.pdb"
        if extract(cif,mc,outp):
            new.append(dict(pdb=pdb,qi=qi,letter=e["letter"],tag="samepose",model=outp,
                native=e["native"],paratope=e["paratope"],npar=e["npar"],ab_seq=e["ab_seq"]))
    print(f"{pdb}: joint pose seed{s}/s{k}",flush=True)
json.dump(new,open(f"{SC}/samepose_manifest.json","w"))
print("SAMEPOSE_S1_DONE",len(new))
