"""FORCED GRADIENT STEERING to reproduce the good (675) and bad (101) loop-17 distogram, then 64 loops.
For each target seed: start from baseline s_lm, Adam-minimize w1(distogram_from_s_lm(s_lm), Pg_target)
until the trunk reproduces that seed's loop-17 interface distogram. Then decode the steered s_lm at
17 AND 65 loops -> DockQ (OST later) + ipTM (trunk-on & trunk-free). Q: does forcing the GOOD distogram
recover the good structure at high confidence, and does 64 loops help vs 16?"""
import sys, json, os
sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/steer/core")
sys.path.insert(0, "/home/ebrahim/tts_abag/steer/experiments")
import torch, numpy as np, time
import transformers.models.esmfold2.modeling_esmfold2 as M
# trunk-free ipTM monkeypatch (zero z_norm affine)
_orig = M.ConfidenceHead.forward; _ST = {}
def _patched(self, s_inputs, z, *a, **kw):
    out = _orig(self, s_inputs, z, *a, **kw)
    zln = self.z_norm; w = zln.weight.data.clone(); b = zln.bias.data.clone()
    zln.weight.data.zero_(); zln.bias.data.zero_()
    try: nt = _orig(self, s_inputs, z, *a, **kw)
    finally: zln.weight.data.copy_(w); zln.bias.data.copy_(b)
    _ST['iptm_tf'] = nt['iptm'].detach().float().cpu().reshape(-1).tolist(); del nt
    return out
M.ConfidenceHead.forward = _patched

import forward_grad as fg
from paths import ESM_PREDICTIONS

cfg = fg.configure_target("8ts0")
fg.REPS = ESM_PREDICTIONS / "8ts0_reps_gb" / "8ts0"          # captured good/bad seed reps live here
print("loading model...", flush=True)
model, builder, feat, chain_infos, cond = fg.load(chunk_size=128)
pair_mask = cond["pair_mask"]
z_init = fg.compute_z_init(model, cond)
z0 = fg.fixed_z0(model, z_init, seed=0)
base_hidden = fg.load_hidden(cfg["baseline"])
OUT = ESM_PREDICTIONS / "steer_8ts0_force"
LR, M_STEPS = 0.5, 30
STEER_LOOPS = 3           # O(L^3) per loop at L=619 -> keep loops tiny; distogram already r~0.98 by loop 3-4
recs = []

def disto_fast(s_lm, steps=STEER_LOOPS):
    """cross-block distogram from s_lm at STEER_LOOPS (distogram converged by loop 8; recon r>0.999).
    Memory-safe outer checkpoint (29GB) — decode still runs full 17/65 loops."""
    lm_z = fg.lift_to_lm_z(model, s_lm)
    z, _ = fg.run_trunk(model, z_init, lm_z, pair_mask, z0, steps, use_ckpt=True)
    return fg.cross_distogram_from_z(model, z, pair_mask)

for tag, seed in [("good", 675), ("bad", 101)]:
    Pg = fg.good_reference(model, pair_mask, z0, z_init, seeds=[seed])   # target loop-17 interface distogram
    s_lm = fg.single_rep_from_hidden(model, base_hidden).detach().requires_grad_(True)
    with torch.no_grad():
        w0 = fg.w1(disto_fast(s_lm), Pg).item()
    print(f"[{tag}] seed {seed}  W1_start={w0:.5f}", flush=True)
    opt = torch.optim.Adam([s_lm], lr=LR)
    traj = [round(w0, 5)]
    best_w, best_s = w0, s_lm.detach().clone()
    t0 = time.time()
    for it in range(M_STEPS):
        with torch.enable_grad():
            loss = fg.w1(disto_fast(s_lm), Pg)
        opt.zero_grad(); loss.backward(); opt.step()
        lv = loss.item()
        if lv < best_w:
            best_w, best_s = lv, s_lm.detach().clone()
        if (it + 1) % 2 == 0:
            dt = (time.time() - t0) / (it + 1)
            traj.append(round(lv, 5))
            print(f"[{tag}] step {it+1}/{M_STEPS}  W1={lv:.5f}  best={best_w:.5f}  ({dt:.1f}s/step)", flush=True)
            # early-stop: last two logged W1s within 1% of each other
            if len(traj) >= 3 and abs(traj[-1] - traj[-2]) < 0.01 * traj[-2]:
                print(f"[{tag}] early stop at step {it+1} (W1 plateau)", flush=True)
                break
    wf = best_w
    print(f"[{tag}] steered: W1 {w0:.5f} -> {wf:.5f}  (best)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    torch.save(best_s.cpu(), OUT / f"{tag}_slm.pt")                 # persist so decode failure never re-costs steering
    # decode BEST steered s_lm at 17 and 65 loops
    s_det = best_s
    lm_z = fg.lift_to_lm_z(model, s_det)
    for loops in [16, 64]:
        z = fg.final_trunk_z(model, z_init, lm_z, pair_mask, z0, loops + 1)
        _ST.clear()
        res = fg.decode_from_ztrunk(model, builder, feat, chain_infos, cond, z, 2, 16, "8ts0", loops)
        for k, sm in enumerate(res):
            d = OUT / f"{tag}_loops{loops}" / "8ts0" / f"seed_{loops}" / "predictions"
            os.makedirs(d, exist_ok=True)
            open(d / f"8ts0_sample_{k}.cif", "w").write(sm.complex.to_mmcif())
        rec = dict(tag=tag, seed=seed, loops=loops, w1_start=w0, w1_final=wf, w1_traj=traj,
                   iptm=[float(r.iptm) for r in res], iptm_tf=_ST.get('iptm_tf'))
        recs.append(rec)
        print(f"[{tag}] loops={loops}  iptm={rec['iptm']}  iptm_tf={rec['iptm_tf']}", flush=True)
        torch.cuda.empty_cache()

os.makedirs(OUT, exist_ok=True)
json.dump(recs, open(OUT / "recs.json", "w"), indent=2)
print("FORCE_DONE", flush=True)
