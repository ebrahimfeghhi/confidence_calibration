"""distogram_trajectory.py — per-loop basin trajectory for a good seed vs a decoy seed.

For each seed, reproduce its exact masked s_lm + z0 (the fold's RNG order: mask draw -> z0 draw), run
the trunk loop-by-loop (N_LOOPS+1 loops, forward-only, NO steering), and at each loop read the trunk's
cross-block (AB x C) distogram. Then measure each loop's W1 distance to the GOOD target structure and to
the DECOY target structure. Tests the hypothesis: the decoy seed sits in the decoy basin ALL loops (fully
refined -> confident), while the good seed only escapes to the good basin late (fewer refine loops -> less
settled -> lower ipTM). Outputs a per-loop CSV + a 2-panel plot.

  CUDA_VISIBLE_DEVICES=5 python steer/experiments/distogram_trajectory.py --target 8ts0 \
      --good-seed 617 --good-samp 2 --decoy-seed 46 --decoy-samp 3 --mask 0.1
"""
import sys, csv, argparse
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "run" / "esm"))
import forward_grad as fg  # noqa: E402
from esm.models.esmfold2.processor import _seed_context  # noqa: E402

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

def masked_hidden_and_z0(model,feat,z_init,seed,mask_pct):
    with torch.no_grad(), _seed_context(seed):
        lm_hidden=model._compute_lm_hidden_states(feat["input_ids"],feat["asym_id"],feat["residue_index"],
                    feat["mol_type"],feat["token_attention_mask"], lm_mask_pct=mask_pct)
        s_lm=fg.single_rep_from_hidden(model,lm_hidden).detach().clone()
        z0=model._init_pair_state(z_init).detach().clone()
    return s_lm, z0

def per_loop_distos(model,z_init,lm_z,pair_mask,z0,steps):
    a,b=model._discretized_dynamics()
    a=a.view(1,1,1,-1).to(z_init.device,z_init.dtype); b_mat=b.to(z_init.device,z_init.dtype)
    z=z0; distos=[]
    with torch.no_grad(), torch.amp.autocast("cuda",dtype=torch.bfloat16):
        for _ in range(steps):
            z=fg._loop_body(model,z,z_init,lm_z,pair_mask,a,b_mat)
            distos.append(fg.cross_distogram_from_z(model,z.float(),pair_mask).detach())
    return distos

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--target",required=True)
    ap.add_argument("--good-seed",type=int,required=True); ap.add_argument("--good-samp",type=int,required=True)
    ap.add_argument("--decoy-seed",type=int,required=True); ap.add_argument("--decoy-samp",type=int,required=True)
    ap.add_argument("--mask",type=float,default=0.1); ap.add_argument("--chain-order",default=None)
    a=ap.parse_args()
    fg.configure_target(a.target)
    model,builder,feat,chain_infos,cond=fg.load(keep_esmc=True, kernel_backend="fused")
    pair_mask=cond["pair_mask"]; z_init=fg.compute_z_init(model,cond)
    steps=fg.N_LOOPS+1

    # reproduce each seed's masked start (both need ESM-C), then offload
    sg,z0g=masked_hidden_and_z0(model,feat,z_init,a.good_seed,a.mask)
    sd,z0d=masked_hidden_and_z0(model,feat,z_init,a.decoy_seed,a.mask)
    model._esmc.to("cpu"); torch.cuda.empty_cache()

    # targets (order/slices from the good structure's chains)
    gcif=str(PRED/a.target/f"seed_{a.good_seed}"/"predictions"/f"{a.target}_sample_{a.good_samp}.cif")
    dcif=str(PRED/a.target/f"seed_{a.decoy_seed}"/"predictions"/f"{a.target}_sample_{a.decoy_samp}.cif")
    order=a.chain_order.split(",") if a.chain_order else sorted(chain_cb(gcif).keys())
    Qg,mg=build_target(gcif,order,fg.AB,fg.C); Qd,md=build_target(dcif,order,fg.AB,fg.C)
    m=mg&md  # compare on pairs finite in both

    print(f"=== distogram trajectory {a.target}: good seed {a.good_seed} vs decoy seed {a.decoy_seed}, {steps} loops ===",flush=True)
    rows=[]
    for name,s_lm,z0 in [("good",sg,z0g),("decoy",sd,z0d)]:
        lm_z=fg.lift_to_lm_z(model,s_lm)
        distos=per_loop_distos(model,z_init,lm_z,pair_mask,z0,steps)
        for loop,P in enumerate(distos):
            wg=masked_w1(P,Qg,m); wd=masked_w1(P,Qd,m)
            basin="good" if wg<wd else "decoy"
            rows.append((name,loop,wg,wd,basin))
        nloops_decoy=sum(1 for r in rows if r[0]==name and r[4]=="decoy")
        snap=next((r[1] for r in rows if r[0]==name and r[4]=="good"),None)
        print(f"  seed {name}: loops-in-decoy={nloops_decoy}/{steps}  first-good-loop={snap}",flush=True)

    csvp=OUT/f"distotraj_{a.target}_{a.good_seed}v{a.decoy_seed}.csv"
    with open(csvp,"w",newline="") as f:
        csv.writer(f).writerows([("seed","loop","w1_to_good","w1_to_decoy","basin")]+rows)

    # plot: 2 panels, W1-to-good vs W1-to-decoy across loops
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
    for ax,name,seed in [(axes[0],"good",a.good_seed),(axes[1],"decoy",a.decoy_seed)]:
        r=[x for x in rows if x[0]==name]; lp=[x[1] for x in r]
        ax.plot(lp,[x[2] for x in r],"-o",ms=3,color="#2a7",label="W1 → good struct")
        ax.plot(lp,[x[3] for x in r],"-o",ms=3,color="#c33",label="W1 → decoy struct")
        ax.set_title(f"seed {seed} ({name})"); ax.set_xlabel("recycling loop"); ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("cross-block distogram W1")
    fig.suptitle(f"{a.target}: per-loop basin trajectory (which structure is the trunk near?)")
    fig.tight_layout(); png=OUT/f"distotraj_{a.target}_{a.good_seed}v{a.decoy_seed}.png"
    fig.savefig(png,dpi=130); print(f"\nwrote {csvp}\nwrote {png}",flush=True)

if __name__=="__main__":
    main()
