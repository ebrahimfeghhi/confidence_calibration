"""why_iptm_washes.py — is pair_iptm blind to interface quality because it AVERAGES over all cross-chain
pairs (the interface being <1%)? Recompute the TM score two ways from the SAME saved PAE:
  - global : max-anchor over the whole antibody, mean over the WHOLE antigen  (the standard pair_iptm)
  - iface  : max-anchor over INTERFACE antibody residues, mean over INTERFACE antigen residues only
Same d0 for both, so the ONLY difference is the residue set. If iface tracks DockQ and global doesn't,
the dilution/averaging IS the mechanism; if neither tracks, even the interface PAE is uninformative.

  python steer/experiments/why_iptm_washes.py
"""
import csv, numpy as np
from pathlib import Path
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
import forward_grad as fg  # noqa
PRED=Path.home()/"esm/foldbench_predictions/esmfold2fast_tts/8vzo"
DEEP="results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"
AA3={'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'}
D0=10.0
def tm(x): return 1.0/(1.0+(x/D0)**2)

def chain_cb(cif):
    d=MMCIF2Dict(cif); g=d['_atom_site.group_PDB']; ch=d['_atom_site.auth_asym_id']; comp=d['_atom_site.label_comp_id']
    at=d['_atom_site.label_atom_id']; sid=d['_atom_site.label_seq_id']
    x=d['_atom_site.Cartn_x']; y=d['_atom_site.Cartn_y']; z=d['_atom_site.Cartn_z']; res={}
    for i in range(len(g)):
        if g[i]!='ATOM' or comp[i] not in AA3: continue
        try: s=int(sid[i])
        except: continue
        res.setdefault((ch[i],s),{})[at[i]]=np.array([float(x[i]),float(y[i]),float(z[i])])
    out={}
    for (c,s),a in sorted(res.items(),key=lambda kv:(kv[0][0],kv[0][1])):
        r=a.get('CB',a.get('CA'))
        if r is not None: out.setdefault(c,[]).append(r)
    return {c:np.stack(v) for c,v in out.items()}

def main():
    fg.configure_target("8vzo"); AB=fg.AB; C=fg.C   # 8vzo: C=antigen [0:551], AB=antibody [551:1006]
    dq={}
    for r in csv.DictReader(open(DEEP)):
        if r['assembly_id'][:4]!='8vzo': continue
        try: k=(int(r['seed']),int(r['sample'])); v=float(r['dockq'])
        except: continue
        dq[k]=max(dq.get(k,0.0),v)
    good=[(166,3),(809,4),(201,2),(246,3),(981,0),(95,1),(946,3)]
    allk=sorted(dq,key=lambda k:dq[k])
    mid=[k for k in allk if 0.10<=dq[k]<=0.20][:8]
    bad=allk[:8]
    poses=good+mid+bad
    print(f"{'seed':>5} {'samp':>4} {'DockQ':>6} {'global_iptm':>11} {'iface_iptm':>10} {'nIface':>6}  {'tier'}")
    rows=[]
    for seed,samp in poses:
        cif=PRED/f"seed_{seed}"/"predictions"/f"8vzo_sample_{samp}.cif"
        pae=PRED/f"seed_{seed}"/"predictions"/f"8vzo_pae_sample_{samp}.npy"
        if not cif.exists() or not pae.exists(): continue
        P=np.load(pae).astype(np.float32)                 # [L,L]
        cb=chain_cb(str(cif)); order=sorted(cb.keys()); coords=np.concatenate([cb[k] for k in order],0)
        ag=coords[C]; ab=coords[AB]
        Dm=np.linalg.norm(ab[:,None]-ag[None],axis=-1)    # [nab, nag]
        iface_ab=np.where((Dm<8).any(1))[0]+AB.start       # antibody interface indices in full numbering
        iface_ag=np.where((Dm<8).any(0))[0]+C.start
        # global pair_iptm: max over antibody anchor of mean over WHOLE antigen
        Fg=tm(P[AB, C])                                    # [nab, nag]
        glob=Fg.mean(1).max()
        # iface: max over interface-antibody anchor of mean over INTERFACE antigen only
        if len(iface_ab)>0 and len(iface_ag)>0:
            Fi=tm(P[np.ix_(iface_ab, iface_ag)]); ifa=Fi.mean(1).max()
        else: ifa=float('nan')
        tier=("good" if (seed,samp) in good else "mid" if (seed,samp) in mid else "wrong")
        rows.append((dq[(seed,samp)],glob,ifa,len(iface_ag),tier))
        print(f"{seed:>5} {samp:>4} {dq[(seed,samp)]:>6.2f} {glob:>11.3f} {ifa:>10.3f} {len(iface_ag):>6}  {tier}")
    r=np.array([(d,g,i) for d,g,i,_,_ in rows if i==i])
    if len(r)>3:
        cg=np.corrcoef(r[:,0],r[:,1])[0,1]; ci=np.corrcoef(r[:,0],r[:,2])[0,1]
        print(f"\ncorrelation with DockQ:  global_iptm r={cg:+.2f}   iface_iptm r={ci:+.2f}")
        for tier in ("good","mid","wrong"):
            gs=[g for d,g,i,n,t in rows if t==tier]; is_=[i for d,g,i,n,t in rows if t==tier and i==i]
            print(f"  {tier:>5}: global {np.mean(gs):.3f}   iface {np.mean(is_):.3f}")

if __name__=="__main__":
    main()
