import json,glob,os,csv,statistics as st
import gemmi
SC="/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl"
PRED="/home/ebrahim/Protenix/foldbench_abag/predictions_template"
chainmap=json.load(open(f"{SC}/oracle_chainmap.json"))
antigen=json.load(open(f"{SC}/agonly_antigen_idx.json"))
oracle=json.load(open(f"{SC}/oracle_folds.json"))
meta=json.load(open(f"{SC}/meta.json"))
CSV="/home/ebrahim/tts_abag/results/csv_files/dockq_ost_protenix_full.csv"
# per-pose interface DockQ: (pdb, frozenset{c1,c2}, seed, sample) -> dockq  (max over interfaces of the pdb)
posedq={}
for r in csv.DictReader(open(CSV)):
    pdb=r["assembly_id"].replace("-assembly1","")
    try: key=(pdb,int(r["seed"]),int(r["sample"])); dq=float(r["dockq"])
    except: continue
    posedq[key]=max(posedq.get(key,-1),dq)   # best interface dockq for that pose
def plddt_of(cif,mc):
    try: stm=gemmi.read_structure(cif)
    except: return None
    ch=next((c for c in stm[0] if c.name==mc),None)
    if ch is None: return None
    bs=[a.b_iso for r in ch for a in r if a.name=="CA"]
    return st.mean(bs) if bs else None
def cached(pdb,qi,s,k):
    try: d=json.load(open(f"{SC}/ost_oracle/{pdb}_{qi}_{s}_{k}.json")); return d.get("lddt"),d.get("_mc")
    except: return None,None
print(f"{'chain':11} {'poolLDDT':13} {'oracle(seed,lddt,dock)':26} {'pLDDTpick(seed,lddt,dock)':26} {'same?':5}")
rows=[]
for pdb,cm in chainmap.items():
    for qi in cm:
        role="ag" if int(qi) in antigen[pdb] else "ab"
        # gather poses
        recs=[]
        for s in range(101,151):
            for k in range(5):
                ld,mc=cached(pdb,qi,s,k)
                if ld is None or mc is None: continue
                cif=f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif"
                pl=plddt_of(cif,mc)
                if pl is None: continue
                recs.append((s,k,ld,pl))
        if len(recs)<10: continue
        lds=[r[2] for r in recs]
        orc=oracle[pdb][qi]; orc_s,orc_k,orc_ld=orc["seed"],orc["sample"],orc["lddt"]
        pl_rec=max(recs,key=lambda r:r[3])   # argmax plddt
        odq=posedq.get((pdb,orc_s,orc_k)); pdq=posedq.get((pdb,pl_rec[0],pl_rec[1]))
        same=(orc_s,orc_k)==(pl_rec[0],pl_rec[1])
        rows.append(dict(pdb=pdb,qi=qi,role=role,pool_min=min(lds),pool_max=max(lds),
            orc_ld=orc_ld,orc_dock=odq,pl_ld=pl_rec[2],pl_dock=pdq,same=same))
        if role=="ab":
            print(f"{pdb} q{qi}(ab) {min(lds):.2f}-{max(lds):.2f}   ({orc_s},{orc_ld:.3f},{'%.2f'%odq if odq is not None else 'NA'})      ({pl_rec[0]},{pl_rec[2]:.3f},{'%.2f'%pdq if pdq is not None else 'NA'})      {'YES' if same else 'no'}")
json.dump(rows,open(f"{SC}/verify_ab_rows.json","w"),indent=1)
ab=[r for r in rows if r["role"]=="ab"]
print(f"\n=== ANTIBODY summary ({len(ab)} chains) ===")
print(f"pLDDT picks the SAME pose as oracle: {sum(r['same'] for r in ab)}/{len(ab)}")
print(f"mean antibody pool lDDT spread (max-min): {st.mean([r['pool_max']-r['pool_min'] for r in ab]):.3f}")
print(f"mean regret (oracle-pLDDT): {st.mean([r['orc_ld']-r['pl_ld'] for r in ab]):.3f}")
gd=[r for r in ab if r['orc_dock'] is not None]
print(f"\noracle-antibody-fold source pose is a GOOD dock (DockQ>=0.23): {sum(r['orc_dock']>=0.23 for r in gd)}/{len(gd)}")
print(f"  mean DockQ of oracle-antibody source pose: {st.mean([r['orc_dock'] for r in gd]):.3f}")
print(f"  (base rate: fraction of all poses that are good docks is ~0.14 from mech.py)")
print("DONE")
