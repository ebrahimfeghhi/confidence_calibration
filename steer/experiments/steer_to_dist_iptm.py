"""steer_to_dist_iptm.py — the "force the good structure, read ipTM" experiment.

For a confident-decoy complex: build the target cross-block distogram from a chosen predicted
structure (the GOOD dock, or the DECOY as control), steer s_lm+z0 so the trunk reproduces it
(short-horizon gradient, which transfers to full recycling), then run the FULL trunk + diffusion +
confidence head and read ipTM. Tests whether the good dock's low ipTM is a settledness artifact
(steer-to-good -> ipTM rises) or intrinsic to the structure (stays low).

  CUDA_VISIBLE_DEVICES=6 python steer/experiments/steer_to_dist_iptm.py --target 8ts0 \
      --which good  --seed 617 --samp 2 --K 3
  ... --which decoy --seed 46 --samp 3 --K 3     # control (should read high ipTM)
"""
import sys, csv, argparse
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "run" / "esm"))
import forward_grad as fg  # noqa: E402

N_BINS=64; BIN_LO, BIN_HI=2.0,22.0
PRED=Path.home()/ "esm/foldbench_predictions/esmfold2fast_tts"
OUT=Path(__file__).resolve().parent
AA3TO1={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
        'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
        'TYR':'Y','VAL':'V'}

def chain_cb(cif):
    """{chain: ordered list of CB(or CA) coords} from a predicted CIF (auth chains, in residue order)."""
    d=MMCIF2Dict(cif); grp=d['_atom_site.group_PDB']; ch=d['_atom_site.auth_asym_id']
    comp=d['_atom_site.label_comp_id']; atom=d['_atom_site.label_atom_id']
    sid=d['_atom_site.label_seq_id']; x=d['_atom_site.Cartn_x']; y=d['_atom_site.Cartn_y']; z=d['_atom_site.Cartn_z']
    res={}
    for i in range(len(grp)):
        if grp[i]!='ATOM' or comp[i] not in AA3TO1: continue
        try: s=int(sid[i])
        except: continue
        rk=(ch[i],s); r=res.setdefault(rk,{})
        r[atom[i]]=np.array([float(x[i]),float(y[i]),float(z[i])],dtype=np.float32)
    out={}
    for (c,s),at in sorted(res.items(), key=lambda kv:(kv[0][0],kv[0][1])):
        rep=at.get('CB', at.get('CA'))
        if rep is not None: out.setdefault(c,[]).append(rep)
    return {c:np.stack(v) for c,v in out.items()}

def build_target(cif, chain_order, ab_slice, c_slice):
    cb=chain_cb(cif)
    coords=np.concatenate([cb[c] for c in chain_order],0)      # [L,3] model token order
    L=coords.shape[0]
    print(f"  target from {Path(cif).name}: L={L} chains={[(c,len(cb[c])) for c in chain_order]}", flush=True)
    coords=torch.from_numpy(coords)
    ab=coords[ab_slice]; c=coords[c_slice]
    d=torch.cdist(ab.unsqueeze(0), c.unsqueeze(0))[0]
    mask=torch.isfinite(d); d=torch.nan_to_num(d,nan=BIN_HI)
    boundaries=torch.linspace(BIN_LO,BIN_HI,N_BINS-1)
    idx=torch.bucketize(d,boundaries)
    Q=F.one_hot(idx.reshape(-1),N_BINS).float(); m=mask.reshape(-1)
    return Q.cuda(), m.cuda()

def masked_w1(P,Q,mask):
    per=(P.cumsum(-1)-Q.cumsum(-1)).abs().sum(-1)
    return per[mask].mean()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--target",required=True); ap.add_argument("--which",required=True,choices=["good","decoy"])
    ap.add_argument("--seed",type=int,required=True); ap.add_argument("--samp",type=int,required=True)
    ap.add_argument("--K",type=int,default=3); ap.add_argument("--lr",type=float,default=0.5)
    ap.add_argument("--max-steps",type=int,default=250); ap.add_argument("--patience",type=int,default=25)
    ap.add_argument("--tol",type=float,default=5e-4); ap.add_argument("--chain-order",default=None,
        help="comma list e.g. A,B,C ; default = sorted chain ids")
    ap.add_argument("--fused",action="store_true",help="use the fused trimul kernel (~3x faster)")
    ap.add_argument("--init-seed",type=int,default=0,help="varies z0 init + decode seed -> trial-to-trial variance")
    a=ap.parse_args()

    cfg=fg.configure_target(a.target)
    model,builder,feat,chain_infos,cond=fg.load(kernel_backend="fused" if a.fused else None)
    pair_mask=cond["pair_mask"]; z_init=fg.compute_z_init(model,cond); z0_base=fg.fixed_z0(model,z_init,seed=a.init_seed)

    cif=str(PRED/a.target/f"seed_{a.seed}"/"predictions"/f"{a.target}_sample_{a.samp}.cif")
    order=a.chain_order.split(",") if a.chain_order else sorted(chain_cb(cif).keys())
    Q,mask=build_target(cif, order, fg.AB, fg.C)

    hidden=fg.load_hidden(fg.BASELINE)
    s_lm=fg.single_rep_from_hidden(model,hidden).detach().clone().requires_grad_(True)
    z0=z0_base.detach().clone().requires_grad_(True)
    opt=torch.optim.Adam([s_lm,z0],lr=a.lr)
    tag=f"{a.target}_{a.which}_K{a.K}_i{a.init_seed}"
    print(f"=== steer {tag}: to {a.which} distogram (seed{a.seed}/samp{a.samp}), K={a.K} ===",flush=True)
    best=float("inf"); best_step=0; rows=[]
    for step in range(a.max_steps):
        opt.zero_grad()
        with torch.enable_grad():
            lm_z=fg.lift_to_lm_z(model,s_lm)
            z,_=fg.run_trunk(model,z_init,lm_z,pair_mask,z0,a.K)
            P=fg.cross_distogram_from_z(model,z,pair_mask)
            loss=masked_w1(P,Q,mask)
        loss.backward(); opt.step(); L=loss.item(); rows.append((step,L))
        if L<best-a.tol: best,best_step=L,step
        if step%10==0 or step-best_step>=a.patience:
            print(f"  step {step:3d} loss={L:.4f} best={best:.4f}@{best_step}",flush=True)
        if step-best_step>=a.patience: break

    # full trunk with steered levers -> decode -> ipTM
    with torch.no_grad():
        lm_z=fg.lift_to_lm_z(model,s_lm)
        z_final=fg.final_trunk_z(model,z_init,lm_z,pair_mask,z0,fg.N_LOOPS)
        samples=fg.decode_from_ztrunk(model,builder,feat,chain_infos,cond,z_final,5,16,a.target,a.init_seed)
    iptms=[float(s.iptm) for s in samples]
    pcs=[]
    for s in samples:
        try: pcs.append(s.pair_chains_iptm.tolist())
        except Exception: pass
    print(f"\n[{tag}] final distogram W1={rows[-1][1]:.4f} (best {best:.4f})")
    print(f"[{tag}] STEERED ipTM per sample: {[round(x,3) for x in iptms]}  best={max(iptms):.3f}")
    if pcs: print(f"[{tag}] pair_chains_iptm (sample0): {pcs[0]}")
    csv.writer(open(OUT/f"steer_dist_iptm_{tag}.csv","w")).writerows([("step","loss")]+rows)
    with open(OUT/f"steer_dist_iptm_summary.csv","a") as f:
        pc=pcs[0] if pcs else None
        pcflat="|".join(f"{v:.3f}" for row in pc for v in row) if pc else ""
        f.write(f"{a.target},{a.which},{a.seed},{a.samp},{a.K},{a.init_seed},{best:.4f},{max(iptms):.4f},{np.mean(iptms):.4f},{pcflat}\n")

if __name__=="__main__":
    main()
