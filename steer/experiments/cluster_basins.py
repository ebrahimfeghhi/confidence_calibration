"""cluster_basins.py — do the 1024 seeds cluster into discrete docking basins, and is the GOOD basin an
extractable cluster? (The core assumption of the cluster-then-rerank architecture.)

For each seed we take sample 0's predicted structure and build a label-free pose feature: the interface
profile = per-residue min distance across the antibody<->antigen interface (paratope profile ++ epitope
profile), capped at 24A. Two seeds docking the same surface/orientation have similar profiles. We PCA +
HDBSCAN the 1024 profiles UNSUPERVISED, then use DockQ ONLY to validate: does the good-DockQ set fall into
one clean cluster, how rare is it, and would 1-representative-per-cluster keep a good-pose representative?

  python steer/experiments/cluster_basins.py --target 8ts0
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
    from sklearn.cluster import HDBSCAN
    HAVE_HDB=True
except Exception:
    from sklearn.cluster import AgglomerativeClustering; HAVE_HDB=False

PRED=Path.home()/"esm/foldbench_predictions/esmfold2fast_tts"
DEEP="results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"; CAP=24.0
AA3={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
     'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

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
def feat_one(seed):
    """interface profile for one seed's sample 0: [paratope min-dists ++ epitope min-dists], capped."""
    cif=PRED/_ARGS.target/f"seed_{seed}"/"predictions"/f"{_ARGS.target}_sample_0.cif"
    if not cif.exists(): return seed, None
    try:
        cb=chain_cb(str(cif)); order=sorted(cb.keys())
        coords=np.concatenate([cb[k] for k in order],0)
        ab=coords[_ARGS.ab_lo:_ARGS.ab_hi]; ag=coords[_ARGS.c_lo:_ARGS.c_hi]
        if _ARGS.ab_lo> _ARGS.c_lo:  # antigen block comes first (e.g. 8vzo): slices already handle it
            pass
        D=np.linalg.norm(ab[:,None,:]-ag[None,:,:],axis=-1)   # [n_ab, n_ag]
        para=np.minimum(D.min(1),CAP); epi=np.minimum(D.min(0),CAP)
        return seed, np.concatenate([para,epi]).astype(np.float32)
    except Exception:
        return seed, None

def main():
    global _ARGS
    ap=argparse.ArgumentParser(); ap.add_argument("--target",required=True); ap.add_argument("--nseeds",type=int,default=1024)
    a=ap.parse_args()
    fg.configure_target(a.target)
    a.ab_lo,a.ab_hi=fg.AB.start,fg.AB.stop; a.c_lo,a.c_hi=fg.C.start,fg.C.stop
    _ARGS=a
    print(f"=== {a.target}: clustering {a.nseeds} seeds by interface pose (AB {a.ab_lo}:{a.ab_hi}, C {a.c_lo}:{a.c_hi}) ===",flush=True)

    with Pool(16) as p:
        res=p.map(feat_one, range(a.nseeds))
    seeds=[s for s,f in res if f is not None]; X=np.stack([f for s,f in res if f is not None])
    print(f"parsed {len(seeds)}/{a.nseeds} seeds, feature dim {X.shape[1]}",flush=True)

    # per-seed sample-0 DockQ (max over interfaces) for VALIDATION only
    dq={}
    for r in csv.DictReader(open(DEEP)):
        if r['assembly_id'][:4]!=a.target or r['sample']!='0': continue
        try: v=float(r['dockq']); s=int(r['seed'])
        except: continue
        dq[s]=max(dq.get(s,0.0),v)
    labels_dq=np.array([dq.get(s,np.nan) for s in seeds])

    Xs=StandardScaler().fit_transform(X); Xp=PCA(n_components=min(30,Xs.shape[1])).fit_transform(Xs)
    if HAVE_HDB:
        cl=HDBSCAN(min_cluster_size=15).fit_predict(Xp)
    else:
        cl=AgglomerativeClustering(n_clusters=8).fit_predict(Xp)
    uniq=sorted(set(cl)); ngood=int(np.nansum(labels_dq>0.6))
    print(f"\nclusters: {len([u for u in uniq if u!=-1])}  (noise pts: {int((cl==-1).sum())})")
    print(f"good seeds (DockQ>0.6): {ngood}/{len(seeds)} = {100*ngood/len(seeds):.1f}%\n")
    print(f"{'cluster':>7} {'size':>5} {'%tot':>5} {'meanDockQ':>10} {'maxDockQ':>9} {'nGood>0.6':>9}")
    rows=[]
    for u in uniq:
        m=cl==u; d=labels_dq[m]; d=d[~np.isnan(d)]
        if len(d)==0: continue
        ng=int((d>0.6).sum())
        rows.append((u,int(m.sum()),np.nanmean(d),np.nanmax(d),ng))
        print(f"{('noise' if u==-1 else u):>7} {int(m.sum()):>5} {100*m.sum()/len(seeds):>4.1f}% "
              f"{np.mean(d):>10.3f} {np.max(d):>9.3f} {ng:>9}")
    # good-cluster purity: what fraction of good seeds land in the single best-DockQ cluster
    real=[u for u in uniq if u!=-1]
    if real:
        best=max(real, key=lambda u: np.nanmean(labels_dq[cl==u]))
        gm=(labels_dq>0.6)
        in_best=int(((cl==best)&gm).sum())
        print(f"\nbest-DockQ cluster = {best}: size {int((cl==best).sum())} "
              f"({100*(cl==best).sum()/len(seeds):.1f}% of seeds), "
              f"holds {in_best}/{ngood} good seeds ({100*in_best/max(ngood,1):.0f}% purity)")
        print(f"=> extractable: {'YES clean' if in_best/max(ngood,1)>0.7 else 'PARTIAL/scattered'} "
              f"(good basin is {'a distinct cluster' if in_best/max(ngood,1)>0.7 else 'fragmented across clusters'})")

    # 2D scatter colored by DockQ
    fig,ax=plt.subplots(figsize=(6.5,5.5))
    sc=ax.scatter(Xp[:,0],Xp[:,1],c=np.nan_to_num(labels_dq,nan=-0.05),cmap="viridis",s=10,vmin=0,vmax=1)
    plt.colorbar(sc,label="DockQ (sample 0)"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"{a.target}: {a.nseeds} seeds by interface pose (color=DockQ)")
    out=Path("steer/experiments")/f"cluster_basins_{a.target}.png"; fig.tight_layout(); fig.savefig(out,dpi=130)
    print(f"\nwrote {out}",flush=True)

if __name__=="__main__":
    main()
