"""cluster_basins_samples.py — same as cluster_basins.py but at the DIFFUSION-SAMPLE level.

Clusters ALL 5 diffusion samples of every seed (up to 5*1024 structures) by interface pose, so the good
poses that live in samples 1-4 (not just sample 0) are actually in the data. Answers: once every decoded
structure is included, do the good poses show up as distinct points / a cluster, or are they too rare?

  python steer/experiments/cluster_basins_samples.py --target 8vzo
"""
import sys, csv, argparse
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
import forward_grad as fg  # noqa: E402
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
try:
    from sklearn.cluster import HDBSCAN; HAVE_HDB=True
except Exception:
    from sklearn.cluster import AgglomerativeClustering; HAVE_HDB=False

PRED=Path.home()/"esm/foldbench_predictions/esmfold2fast_tts"
DEEP="results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"; CAP=24.0
AA3={'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'}

def chain_cb(cif):
    d=MMCIF2Dict(cif); grp=d['_atom_site.group_PDB']; ch=d['_atom_site.auth_asym_id']
    comp=d['_atom_site.label_comp_id']; atom=d['_atom_site.label_atom_id']; sid=d['_atom_site.label_seq_id']
    x=d['_atom_site.Cartn_x']; y=d['_atom_site.Cartn_y']; z=d['_atom_site.Cartn_z']; res={}
    for i in range(len(grp)):
        if grp[i]!='ATOM' or comp[i] not in AA3: continue
        try: s=int(sid[i])
        except: continue
        res.setdefault((ch[i],s),{})[atom[i]]=np.array([float(x[i]),float(y[i]),float(z[i])],np.float32)
    out={}
    for (c,s),at in sorted(res.items(),key=lambda kv:(kv[0][0],kv[0][1])):
        rep=at.get('CB',at.get('CA'))
        if rep is not None: out.setdefault(c,[]).append(rep)
    return {c:np.stack(v) for c,v in out.items()}

_ARGS=None
def feat_one(sk):
    seed,samp=sk
    cif=PRED/_ARGS.target/f"seed_{seed}"/"predictions"/f"{_ARGS.target}_sample_{samp}.cif"
    if not cif.exists(): return sk, None
    try:
        cb=chain_cb(str(cif)); order=sorted(cb.keys()); coords=np.concatenate([cb[k] for k in order],0)
        ab=coords[_ARGS.ab_lo:_ARGS.ab_hi]; ag=coords[_ARGS.c_lo:_ARGS.c_hi]
        D=np.linalg.norm(ab[:,None,:]-ag[None,:,:],axis=-1)
        return sk, np.concatenate([np.minimum(D.min(1),CAP),np.minimum(D.min(0),CAP)]).astype(np.float32)
    except Exception:
        return sk, None

def main():
    global _ARGS
    ap=argparse.ArgumentParser(); ap.add_argument("--target",required=True)
    ap.add_argument("--nseeds",type=int,default=1024); ap.add_argument("--nsamp",type=int,default=5)
    a=ap.parse_args()
    fg.configure_target(a.target)
    a.ab_lo,a.ab_hi=fg.AB.start,fg.AB.stop; a.c_lo,a.c_hi=fg.C.start,fg.C.stop; _ARGS=a
    tasks=[(s,k) for s in range(a.nseeds) for k in range(a.nsamp)]
    print(f"=== {a.target}: clustering {len(tasks)} diffusion samples ({a.nseeds}x{a.nsamp}) by interface pose ===",flush=True)
    with Pool(16) as p: res=p.map(feat_one, tasks)
    keys=[sk for sk,f in res if f is not None]; X=np.stack([f for sk,f in res if f is not None])
    print(f"parsed {len(keys)}/{len(tasks)}, feature dim {X.shape[1]}",flush=True)

    dq={}
    for r in csv.DictReader(open(DEEP)):
        if r['assembly_id'][:4]!=a.target: continue
        try: s=int(r['seed']); k=int(r['sample']); v=float(r['dockq'])
        except: continue
        dq[(s,k)]=max(dq.get((s,k),0.0),v)
    lab=np.array([dq.get(sk,np.nan) for sk in keys])

    Xp=PCA(n_components=min(30,X.shape[1])).fit_transform(StandardScaler().fit_transform(X))
    cl=(HDBSCAN(min_cluster_size=20).fit_predict(Xp) if HAVE_HDB
        else __import__('sklearn.cluster',fromlist=['AgglomerativeClustering']).AgglomerativeClustering(n_clusters=8).fit_predict(Xp))
    ngood=int(np.nansum(lab>0.6))
    print(f"\nclusters: {len(set(cl)-{-1})}  noise: {int((cl==-1).sum())}")
    print(f"good samples (DockQ>0.6): {ngood}/{len(keys)} = {100*ngood/len(keys):.2f}%")
    print(f"{'cluster':>7} {'size':>6} {'meanDockQ':>10} {'maxDockQ':>9} {'nGood':>6}")
    for u in sorted(set(cl)):
        m=cl==u; d=lab[m]; d=d[~np.isnan(d)]
        if len(d)==0: continue
        print(f"{('noise' if u==-1 else u):>7} {int(m.sum()):>6} {np.mean(d):>10.3f} {np.max(d):>9.3f} {int((d>0.6).sum()):>6}")

    # scatter: draw dark/low-DockQ first, bright/high-DockQ last so good points sit on top
    o=np.argsort(np.nan_to_num(lab,nan=-1))
    fig,ax=plt.subplots(figsize=(6.8,5.6))
    sc=ax.scatter(Xp[o,0],Xp[o,1],c=np.nan_to_num(lab[o],nan=-0.05),cmap="viridis",s=8,vmin=0,vmax=1)
    plt.colorbar(sc,label="DockQ"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"{a.target}: {len(keys)} diffusion samples by interface pose (color=DockQ)")
    out=Path("steer/experiments")/f"cluster_samples_{a.target}.png"; fig.tight_layout(); fig.savefig(out,dpi=130)
    print(f"\nwrote {out}",flush=True)

if __name__=="__main__":
    main()
