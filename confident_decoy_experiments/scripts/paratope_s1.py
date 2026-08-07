import json,glob,os,statistics as st
import numpy as np, gemmi
SC="/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl"
PRED="/home/ebrahim/Protenix/foldbench_abag/predictions_template"
GT="/home/ebrahim/FoldBench/examples/ground_truths"
chainmap=json.load(open(f"{SC}/oracle_chainmap.json"))
antigen=json.load(open(f"{SC}/agonly_antigen_idx.json"))
oracle=json.load(open(f"{SC}/oracle_folds.json"))
TLC={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
MDIR=f"{SC}/para_models"; os.makedirs(MDIR,exist_ok=True)
def cached(pdb,qi,s,k):
    try: d=json.load(open(f"{SC}/ost_oracle/{pdb}_{qi}_{s}_{k}.json")); return d.get("lddt"),d.get("_mc")
    except: return None,None
def pose_feat(pdb,qi,s,k):
    ld,mc=cached(pdb,qi,s,k)
    if ld is None or mc is None: return None
    cif=f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif"
    try: stm=gemmi.read_structure(cif)
    except: return None
    ch=next((c for c in stm[0] if c.name==mc),None)
    if ch is None: return None
    bs=[];ca=[]
    for r in ch:
        a=next((at for at in r if at.name=="CA"),None)
        if a is None: continue
        bs.append(a.b_iso); ca.append([a.pos.x,a.pos.y,a.pos.z])
    if not ca: return None
    return dict(s=s,k=k,ld=ld,mc=mc,plddt=float(np.mean(bs)),ca=np.array(ca),cif=cif)
def paratope_positions(pdb,ab_letter,ag_letters,cutoff=5.0):
    # returns 0-based positions (over the antibody chain's amino-acid residues, in order)
    # that contact antigen, plus the antibody native sequence (for stage-2 alignment).
    st_=gemmi.read_structure(f"{GT}/{pdb}-assembly1.cif"); st_.setup_entities()
    m=st_[0]
    ab=next((c for c in m if c.name==ab_letter),None)
    agatoms=[a.pos for c in m if c.name in ag_letters for r in c for a in r]
    if ab is None or not agatoms: return set(),""
    pos=set(); seq=""; i=0
    for r in ab:
        if r.name not in TLC: continue
        seq+=TLC[r.name]
        hit=False
        for a in r:
            for p in agatoms:
                if a.pos.dist(p)<cutoff: hit=True;break
            if hit: break
        if hit: pos.add(i)
        i+=1
    return pos,seq
def extract(cif,mc,outp):
    stm=gemmi.read_structure(cif); stm.setup_entities()
    ch=next((c for c in stm[0] if c.name==mc),None)
    if ch is None: return False
    ns=gemmi.Structure(); ns.add_model(gemmi.Model("1")); nc=gemmi.Chain("A")
    for r in ch: nc.add_residue(r)
    ns[0].add_chain(nc); ns.setup_entities(); ns.write_pdb(outp); return True
manifest=[]
for pdb,cm in chainmap.items():
    agq=[str(q) for q in antigen[pdb]]
    ag_letters=set(cm[q] for q in agq)
    for qi,letter in cm.items():
        if int(qi) in antigen[pdb]: continue   # antibody chains only
        feats=[pose_feat(pdb,qi,s,k) for s in range(101,151) for k in range(5)]
        feats=[f for f in feats if f]
        if len(feats)<10: continue
        L=st.mode([len(f["ca"]) for f in feats]); feats=[f for f in feats if len(f["ca"])==L]
        # consensus pose (closest to mean structure)
        ref=feats[0]["ca"]-feats[0]["ca"].mean(0); al=[]
        for f in feats:
            Pc=f["ca"]-f["ca"].mean(0); V,S,Wt=np.linalg.svd(Pc.T@ref); d=np.sign(np.linalg.det(V@Wt)); al.append(Pc@(V@np.diag([1,1,d])@Wt))
        al=np.stack(al); ms=al.mean(0)
        rm=[np.sqrt(((al[i]-ms)**2).sum(1).mean()) for i in range(len(feats))]
        i_pl=max(range(len(feats)),key=lambda i:feats[i]["plddt"])
        i_co=min(range(len(feats)),key=lambda i:rm[i])
        picks={"plddt":feats[i_pl],"cons":feats[i_co]}
        orc=oracle[pdb][qi]; picks["oracle"]={"s":orc["seed"],"k":orc["sample"],"mc":orc["model_chain"],"cif":orc["cif"]}
        para_set,ab_seq=paratope_positions(pdb,letter,ag_letters); para=sorted(para_set)
        for tag,pk in picks.items():
            outp=f"{MDIR}/{pdb}_{qi}_{tag}.pdb"
            if extract(pk["cif"],pk["mc"],outp):
                manifest.append(dict(pdb=pdb,qi=qi,letter=letter,tag=tag,model=outp,
                    native=f"{SC}/native_chains/{pdb}_{letter}.pdb",paratope=para,npar=len(para),ab_seq=ab_seq))
        print(f"{pdb} q{qi} paratope positions: {len(para)}",flush=True)
json.dump(manifest,open(f"{SC}/paratope_manifest.json","w"))
print("S1_DONE",len(manifest),"entries")
