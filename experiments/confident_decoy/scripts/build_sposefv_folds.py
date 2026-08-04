import json
SC="/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl"
PRED="/home/ebrahim/Protenix/foldbench_abag/predictions_template"
cm=json.load(open(f"{SC}/oracle_chainmap.json"))
antigen=json.load(open(f"{SC}/agonly_antigen_idx.json"))
oracle=json.load(open(f"{SC}/oracle_folds.json"))
def cl(pdb,qi,s,k):
    try:
        d=json.load(open(f"{SC}/ost_oracle/{pdb}_{qi}_{s}_{k}.json")); return d.get("lddt"),d.get("_mc")
    except: return None,None
out={}; nfab=0
for pdb,d in cm.items():
    abqis=[qi for qi in d if int(qi) not in antigen[pdb]]
    agqis=[qi for qi in d if int(qi) in antigen[pdb]]
    chains={}
    for qi in agqis: chains[qi]=oracle[pdb][qi]      # antigen unchanged (oracle mix-match)
    if len(abqis)>=2:                                 # Fab: H+L from single joint-best pose
        nfab+=1
        poses={}
        for s in range(101,151):
            for k in range(5):
                vals={}; ok=True
                for qi in abqis:
                    ld,mc=cl(pdb,qi,s,k)
                    if ld is None or mc is None: ok=False; break
                    vals[qi]=(ld,mc)
                if ok: poses[(s,k)]=vals
        (s,k),vals=max(poses.items(),key=lambda x:sum(v[0] for v in x[1].values())/len(x[1]))
        cif=f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif"
        for qi in abqis:
            ld,mc=vals[qi]; chains[qi]={"lddt":ld,"cif":cif,"model_chain":mc,"seed":s,"sample":k}
        print(f"{pdb} (Fab): H+L from seed{s}/s{k}  {[ (qi,round(vals[qi][0],3)) for qi in abqis]}")
    else:                                             # VHH: antibody unchanged (oracle)
        for qi in abqis: chains[qi]=oracle[pdb][qi]
        print(f"{pdb} (VHH): antibody unchanged from oracle")
    out[pdb]=chains
json.dump(out,open(f"{SC}/oracle_folds_samepose.json","w"),indent=1)
print(f"SPOSEFV_FOLDS_DONE  ({nfab} Fabs changed, {len(cm)-nfab} VHH unchanged)")
