"""steer_from_masked_dropout.py — experiment 1, done right: steer from a seed's masked start WITH
lm-dropout ON, so the good basin is a real head start (not the thrown-away, dropout-0 version).

The fg steering trunk hard-codes dropout=0, which collapses good seeds into the decoy basin — so the
earlier masked-start experiment wasn't actually starting in the good basin. Here the trunk applies the
model's own per-loop dropout (F.dropout(lm_z, p, training=True), matching modeling_esmfold2 line 781),
with a FIXED dropout realization (re-seed each pass) so the steering loss stays deterministic and the
gradient is clean. We reproduce the seed's masked s_lm + z0, steer s_lm+z0 to lock its target distogram
at horizon K with dropout live, then decode at N_LOOPS+1 loops (dropout live) and read INTERFACE pair_iptm.

  CUDA_VISIBLE_DEVICES=3 python steer/experiments/steer_from_masked_dropout.py --target 8ts0 \
      --seed 617 --samp 2 --K 1 --mask 0.1 --dropout 0.3
"""
import sys, argparse
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "run" / "esm"))
import forward_grad as fg  # noqa: E402
from esm.models.esmfold2.processor import _seed_context  # noqa: E402

N_BINS=64; BIN_LO,BIN_HI=2.0,22.0
PRED=Path.home()/"esm/foldbench_predictions/esmfold2fast_tts"
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

def masked_w1(P,Q,mask): return (P.cumsum(-1)-Q.cumsum(-1)).abs().sum(-1)[mask].mean()

def trunk_dropout(model, z_init, lm_z, pair_mask, z0, steps, p, dseed, grad):
    """Recycling trunk with the model's own per-loop lm-dropout (F.dropout, training=True), but a FIXED
    realization: re-seed to dseed each call so the dropout masks are identical every Adam step -> the
    steering loss is a deterministic function of (s_lm, z0). No checkpointing (K small, L~600 fits)."""
    a,b=model._discretized_dynamics()
    a=a.view(1,1,1,-1).to(z_init.device,z_init.dtype); b_mat=b.to(z_init.device,z_init.dtype)
    z=z0; torch.manual_seed(dseed)
    cm=torch.enable_grad() if grad else torch.no_grad()
    with cm, torch.amp.autocast("cuda",dtype=torch.bfloat16):
        for _ in range(steps):
            lm_z_i=F.dropout(lm_z, p=p, training=True) if p>0 else lm_z
            z=fg._loop_body(model, z, z_init, lm_z_i, pair_mask, a, b_mat)
    return z if grad else z.float()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--target",required=True); ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--samp",type=int,required=True); ap.add_argument("--which",default="good")
    ap.add_argument("--K",type=int,default=1); ap.add_argument("--mask",type=float,default=0.1)
    ap.add_argument("--dropout",type=float,default=0.3); ap.add_argument("--lr",type=float,default=0.5)
    ap.add_argument("--max-steps",type=int,default=150); ap.add_argument("--patience",type=int,default=25)
    ap.add_argument("--dseed",type=int,default=None,help="dropout realization seed (default = fold seed)")
    ap.add_argument("--chain-order",default=None)
    a=ap.parse_args()
    dseed=a.dseed if a.dseed is not None else a.seed
    fg.configure_target(a.target)
    model,builder,feat,chain_infos,cond=fg.load(keep_esmc=True, kernel_backend="fused")
    pair_mask=cond["pair_mask"]; z_init=fg.compute_z_init(model,cond)

    with torch.no_grad(), _seed_context(a.seed):
        lm_hidden=model._compute_lm_hidden_states(feat["input_ids"],feat["asym_id"],feat["residue_index"],
                    feat["mol_type"],feat["token_attention_mask"], lm_mask_pct=a.mask)
        s_lm_init=fg.single_rep_from_hidden(model,lm_hidden).detach().clone()
        z0_init=model._init_pair_state(z_init).detach().clone()
    model._esmc.to("cpu"); torch.cuda.empty_cache()

    cif=str(PRED/a.target/f"seed_{a.seed}"/"predictions"/f"{a.target}_sample_{a.samp}.cif")
    order=a.chain_order.split(",") if a.chain_order else sorted(chain_cb(cif).keys())
    Q,mask=build_target(cif,order,fg.AB,fg.C)

    s_lm=s_lm_init.requires_grad_(True); z0=z0_init.requires_grad_(True)
    opt=torch.optim.Adam([s_lm,z0],lr=a.lr)
    tag=f"{a.target}_{a.which}_maskedseed{a.seed}_K{a.K}_drop{a.dropout}"
    print(f"=== {tag}: masked s_lm+z0 seed {a.seed}, DROPOUT {a.dropout} (dseed {dseed}), lock samp{a.samp} at K={a.K} ===",flush=True)
    best=float("inf"); bstep=0
    for step in range(a.max_steps):
        opt.zero_grad()
        lm_z=fg.lift_to_lm_z(model,s_lm)
        z=trunk_dropout(model,z_init,lm_z,pair_mask,z0,a.K,a.dropout,dseed,grad=True)
        loss=masked_w1(fg.cross_distogram_from_z(model,z,pair_mask),Q,mask)
        loss.backward(); opt.step(); L=loss.item()
        if L<best-5e-4: best,bstep=L,step
        if step%10==0: print(f"  step {step:3d} loss={L:.4f} best={best:.4f}@{bstep}",flush=True)
        if step-bstep>=a.patience: break

    with torch.no_grad():
        lm_z=fg.lift_to_lm_z(model,s_lm)
        z_final=trunk_dropout(model,z_init,lm_z,pair_mask,z0,fg.N_LOOPS+1,a.dropout,dseed,grad=False)
        samples=fg.decode_from_ztrunk(model,builder,feat,chain_infos,cond,z_final,5,16,a.target,a.seed)
    pcs=[s.pair_chains_iptm.tolist() for s in samples if s.pair_chains_iptm is not None]
    # interface pair_iptm (8ts0: A,B antibody, C antigen -> A-C, B-C); best over samples
    ac=max(M[0][2] for M in pcs); bc=max(M[1][2] for M in pcs)
    print(f"\n[{tag}] final W1={loss.item():.4f} (best {best:.4f})")
    print(f"[{tag}] INTERFACE pair_iptm (best/samples): A-C={ac:.3f}  B-C={bc:.3f}")
    print(f"[{tag}] pair_chains_iptm (sample0): {pcs[0] if pcs else None}")

if __name__=="__main__":
    main()
