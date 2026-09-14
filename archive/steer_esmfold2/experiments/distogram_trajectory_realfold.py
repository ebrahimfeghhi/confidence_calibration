"""distogram_trajectory_realfold.py — FAITHFUL per-loop basin trajectory (dropout ON).

The fg steering trunk hard-codes lm_dropout=0, which collapses good seeds into the decoy basin
(dropout is load-bearing for the good basin). So to see a good seed's real late escape we must run the
ACTUAL fold with dropout 0.3. Here we register a forward hook on model.folding_trunk (called once per
recycling loop, before diffusion), run builder.fold(seed, mask0.1, dropout0.3) for the good seed and the
decoy seed, capture the first N_LOOPS+1 pair states, and measure each loop's cross-block (AB x C)
distogram W1 to the good vs decoy target structure. Tests whether the decoy seed occupies the decoy basin
all loops (fully settled -> confident) while the good seed escapes late (few refine loops -> low ipTM).

  CUDA_VISIBLE_DEVICES=5 python steer/experiments/distogram_trajectory_realfold.py --target 8ts0 \
      --good-seed 617 --good-samp 2 --decoy-seed 46 --decoy-samp 3 --mask 0.1 --dropout 0.3
"""
import sys, csv, argparse
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "run" / "esm"))
import forward_grad as fg  # noqa: E402

N_BINS=64; BIN_LO,BIN_HI=2.0,22.0
PRED=Path.home()/"esm/foldbench_predictions/esmfold2fast_tts"; OUT=Path(__file__).resolve().parent
AA3TO1={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
        'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

def chain_cb(cif):
    d=MMCIF2Dict(cif); grp=d['_atom_site.group_PDB']; ch=d['_atom_site.auth_asym_id']
    comp=d['_atom_site.label_comp_id']; atom=d['_atom_site.label_atom_id']; sid=d['_atom_site.label_seq_id']
    x=d['_atom_site.Cartn_x']; y=d['_atom_site.Cartn_y']; z=d['_atom_site.Cartn_z']; res={}
    for i in range(len(grp)):
        if grp[i]!='ATOM' or comp[i] not in AA3TO1: continue
        try: s=int(sid[i])
        except: continue
        res.setdefault((ch[i],s),{})[atom[i]]=np.array([float(x[i]),float(y[i]),float(z[i])],dtype=np.float32)
    out={}
    for (c,s),at in sorted(res.items(),key=lambda kv:(kv[0][0],kv[0][1])):
        rep=at.get('CB',at.get('CA'))
        if rep is not None: out.setdefault(c,[]).append(rep)
    return {c:np.stack(v) for c,v in out.items()}

def build_target(cif, order, ab, c):
    cb=chain_cb(cif); coords=torch.from_numpy(np.concatenate([cb[k] for k in order],0))
    A=coords[ab]; C=coords[c]; d=torch.cdist(A.unsqueeze(0),C.unsqueeze(0))[0]
    m=torch.isfinite(d); d=torch.nan_to_num(d,nan=BIN_HI)
    idx=torch.bucketize(d,torch.linspace(BIN_LO,BIN_HI,N_BINS-1))
    return F.one_hot(idx.reshape(-1),N_BINS).float().cuda(), m.reshape(-1).cuda()

def masked_w1(P,Q,mask): return (P.cumsum(-1)-Q.cumsum(-1)).abs().sum(-1)[mask].mean().item()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--target",required=True)
    ap.add_argument("--good-seed",type=int,required=True); ap.add_argument("--good-samp",type=int,required=True)
    ap.add_argument("--decoy-seed",type=int,required=True); ap.add_argument("--decoy-samp",type=int,required=True)
    ap.add_argument("--mask",type=float,default=0.1); ap.add_argument("--dropout",type=float,default=0.3)
    ap.add_argument("--chain-order",default=None)
    a=ap.parse_args()
    fg.configure_target(a.target)
    model,builder,feat,chain_infos,cond=fg.load(keep_esmc=True, kernel_backend="fused")
    pair_mask=cond["pair_mask"]; steps=fg.N_LOOPS+1
    spi=fg.load_complex(str(fg.JSON_DIR), fg._TARGET)

    captured=[]
    def hook(mod,inp,out): captured.append(out.detach().float().cpu())
    h=model.folding_trunk.register_forward_hook(hook)

    gcif=str(PRED/a.target/f"seed_{a.good_seed}"/"predictions"/f"{a.target}_sample_{a.good_samp}.cif")
    dcif=str(PRED/a.target/f"seed_{a.decoy_seed}"/"predictions"/f"{a.target}_sample_{a.decoy_samp}.cif")
    order=a.chain_order.split(",") if a.chain_order else sorted(chain_cb(gcif).keys())
    Qg,mg=build_target(gcif,order,fg.AB,fg.C); Qd,md=build_target(dcif,order,fg.AB,fg.C); m=mg&md

    print(f"=== REAL-FOLD trajectory {a.target} (dropout {a.dropout}): good {a.good_seed} vs decoy {a.decoy_seed} ===",flush=True)
    rows=[]
    for name,seed in [("good",a.good_seed),("decoy",a.decoy_seed)]:
        captured.clear()
        with torch.no_grad():
            builder.fold(model, spi, seed=seed, num_loops=fg.N_LOOPS, num_sampling_steps=16,
                         num_diffusion_samples=1, lm_mask_pct=a.mask, lm_dropout=a.dropout)
        loops=[z for z in captured if z.shape[0]==1][:steps]   # recycling passes (before diffusion)
        print(f"  seed {name}: captured {len(captured)} folding_trunk calls -> using first {len(loops)} recycling loops",flush=True)
        for loop,z in enumerate(loops):
            P=fg.cross_distogram_from_z(model, z.cuda(), pair_mask)
            wg=masked_w1(P,Qg,m); wd=masked_w1(P,Qd,m)
            rows.append((name,loop,wg,wd,"good" if wg<wd else "decoy"))
        nd=sum(1 for r in rows if r[0]==name and r[4]=="decoy")
        snap=next((r[1] for r in rows if r[0]==name and r[4]=="good"),None)
        print(f"  seed {name}: loops-in-decoy={nd}/{len(loops)}  first-good-loop={snap}",flush=True)
    h.remove()

    csvp=OUT/f"distotraj_real_{a.target}_{a.good_seed}v{a.decoy_seed}.csv"
    with open(csvp,"w",newline="") as f:
        csv.writer(f).writerows([("seed","loop","w1_to_good","w1_to_decoy","basin")]+rows)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
    for ax,name,seed in [(axes[0],"good",a.good_seed),(axes[1],"decoy",a.decoy_seed)]:
        r=[x for x in rows if x[0]==name]; lp=[x[1] for x in r]
        ax.plot(lp,[x[2] for x in r],"-o",ms=3,color="#2a7",label="W1 → good struct")
        ax.plot(lp,[x[3] for x in r],"-o",ms=3,color="#c33",label="W1 → decoy struct")
        ax.set_title(f"seed {seed} ({name})"); ax.set_xlabel("recycling loop"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    axes[0].set_ylabel("cross-block distogram W1")
    fig.suptitle(f"{a.target}: REAL-FOLD per-loop basin trajectory (dropout {a.dropout})")
    fig.tight_layout(); png=OUT/f"distotraj_real_{a.target}_{a.good_seed}v{a.decoy_seed}.png"
    fig.savefig(png,dpi=130); print(f"\nwrote {csvp}\nwrote {png}",flush=True)

if __name__=="__main__":
    main()
