"""steer_from_masked.py — experiment 1: steer from the ACTUAL masked seed that produced the good dock.

Instead of starting from the unmasked baseline, reproduce seed N's exact masked s_lm + z0 (same RNG
order the fold uses: mask draw -> z0 draw), then steer s_lm+z0 to LOCK that seed's good-dock distogram
by horizon K, and let the full recycling refine it. Decode at N_LOOPS+1 (=17) to match the natural model.
Tests: does forcing early commitment of a naturally-good masked seed give a more settled / higher-ipTM
good dock than its natural late escape?

  CUDA_VISIBLE_DEVICES=N python steer/experiments/steer_from_masked.py --target 8ts0 --seed 617 \
      --good-samp 2 --K 1 --mask 0.1
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
    print(f"  target {Path(cif).name}: L={coords.shape[0]} {[(k,len(cb[k])) for k in order]}",flush=True)
    A=coords[ab]; C=coords[c]; d=torch.cdist(A.unsqueeze(0),C.unsqueeze(0))[0]
    m=torch.isfinite(d); d=torch.nan_to_num(d,nan=BIN_HI)
    idx=torch.bucketize(d,torch.linspace(BIN_LO,BIN_HI,N_BINS-1))
    return F.one_hot(idx.reshape(-1),N_BINS).float().cuda(), m.reshape(-1).cuda()

def masked_w1(P,Q,mask): return (P.cumsum(-1)-Q.cumsum(-1)).abs().sum(-1)[mask].mean()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--target",required=True); ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--good-samp",type=int,required=True); ap.add_argument("--K",type=int,default=1)
    ap.add_argument("--mask",type=float,default=0.1); ap.add_argument("--lr",type=float,default=0.5)
    ap.add_argument("--max-steps",type=int,default=150); ap.add_argument("--patience",type=int,default=25)
    ap.add_argument("--chain-order",default=None)
    a=ap.parse_args()
    fg.configure_target(a.target)
    model,builder,feat,chain_infos,cond=fg.load(keep_esmc=True, kernel_backend="fused")
    pair_mask=cond["pair_mask"]; z_init=fg.compute_z_init(model,cond)

    # reproduce seed's masked s_lm + z0 in the fold's RNG order (mask draw, then z0 draw)
    with torch.no_grad(), _seed_context(a.seed):
        lm_hidden=model._compute_lm_hidden_states(feat["input_ids"],feat["asym_id"],feat["residue_index"],
                    feat["mol_type"],feat["token_attention_mask"], lm_mask_pct=a.mask)
        s_lm_init=fg.single_rep_from_hidden(model,lm_hidden).detach().clone()
        z0_init=model._init_pair_state(z_init).detach().clone()
    model._esmc.to("cpu"); torch.cuda.empty_cache()   # free ESM-C; steering uses s_lm leaf

    cif=str(PRED/a.target/f"seed_{a.seed}"/"predictions"/f"{a.target}_sample_{a.good_samp}.cif")
    order=a.chain_order.split(",") if a.chain_order else sorted(chain_cb(cif).keys())
    Q,mask=build_target(cif,order,fg.AB,fg.C)

    s_lm=s_lm_init.requires_grad_(True); z0=z0_init.requires_grad_(True)
    opt=torch.optim.Adam([s_lm,z0],lr=a.lr)
    tag=f"{a.target}_maskedseed{a.seed}_K{a.K}"
    print(f"=== steer_from_masked {tag}: masked s_lm+z0 of seed {a.seed}, lock good(samp{a.good_samp}) at K={a.K} ===",flush=True)
    best=float("inf"); bstep=0; rows=[]
    for step in range(a.max_steps):
        opt.zero_grad()
        with torch.enable_grad():
            lm_z=fg.lift_to_lm_z(model,s_lm); z,_=fg.run_trunk(model,z_init,lm_z,pair_mask,z0,a.K)
            loss=masked_w1(fg.cross_distogram_from_z(model,z,pair_mask),Q,mask)
        loss.backward(); opt.step(); L=loss.item(); rows.append((step,L))
        if L<best-5e-4: best,bstep=L,step
        if step%10==0: print(f"  step {step:3d} loss={L:.4f} best={best:.4f}@{bstep}",flush=True)
        if step-bstep>=a.patience: break

    with torch.no_grad():
        lm_z=fg.lift_to_lm_z(model,s_lm)
        z_final=fg.final_trunk_z(model,z_init,lm_z,pair_mask,z0,fg.N_LOOPS+1)   # 17 loops (matches natural)
        samples=fg.decode_from_ztrunk(model,builder,feat,chain_infos,cond,z_final,5,16,a.target,a.seed)
    iptms=[float(s.iptm) for s in samples]
    pc=[s.pair_chains_iptm.tolist() for s in samples if s.pair_chains_iptm is not None]
    print(f"\n[{tag}] final W1={rows[-1][1]:.4f} (best {best:.4f})")
    print(f"[{tag}] STEERED ipTM/sample: {[round(x,3) for x in iptms]} best={max(iptms):.3f}")
    if pc: print(f"[{tag}] pair_chains_iptm (sample0): {pc[0]}")

if __name__=="__main__":
    main()
