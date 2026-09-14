"""RECONSTRUCTION SANITY: can the steering engine regenerate the baseline's own distogram?
Free-run trunk from baseline s_lm + fixed_z0(seed=0) for K loops, compare reconstructed pair-state z
AND its distogram to the SAVED reference pairformer_z_loop{K-1}. If this matches, the engine is faithful
and random-z0 / good-seed steering results are meaningful. Reference snapshots: loop00..loop16 (17 iters)."""
import sys, json
sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/steer/core")
sys.path.insert(0, "/home/ebrahim/tts_abag/steer/experiments")
import torch, numpy as np
import forward_grad as fg

cfg = fg.configure_target("8ts0")
REPDIR = cfg["baseline"] / "reps"
print("loading model...", flush=True)
model, builder, feat, chain_infos, cond = fg.load(chunk_size=16)
z_init = fg.compute_z_init(model, cond)
pair_mask = cond["pair_mask"]
z0 = fg.fixed_z0(model, z_init, seed=0)                       # SAME z0 the baseline used (seed 0)
s_lm = fg.single_rep_from_hidden(model, fg.load_hidden(cfg["baseline"]))
lm_z = fg.lift_to_lm_z(model, s_lm)
AB, C = cfg["ab"], cfg["c"]
print(f"z0 {tuple(z0.shape)} lm_z {tuple(lm_z.shape)} AB={AB} C={C}", flush=True)

def dgram(z):
    # full [L,L,nbins] distogram distribution + expected-distance map
    P = fg.cross_distogram_from_z(model, z, pair_mask, full_disto=True)  # [L*L, nbins]
    L = z.shape[1]; nb = P.shape[-1]
    P = P.reshape(L, L, nb)
    bins = torch.linspace(2.0, 22.0, nb, device=P.device)                # approx bin centers (rel. compare only)
    ed = (P * bins).sum(-1)                                              # [L,L] expected dist
    return P.float().cpu(), ed.float().cpu()

def cmp_maps(edA, edB, sl_r, sl_c):
    a = edA[sl_r, sl_c].reshape(-1).numpy(); b = edB[sl_r, sl_c].reshape(-1).numpy()
    r = float(np.corrcoef(a, b)[0, 1])
    mae = float(np.abs(a - b).mean())
    return r, mae

recs = []
for K in [1, 2, 4, 8, 12, 16, 17]:
    zK = fg.final_trunk_z(model, z_init, lm_z, pair_mask, z0, K)          # free-run K loops from z0
    refidx = min(K - 1, 16)                                               # loop index (0-based)
    ref = torch.from_numpy(np.load(REPDIR / f"pairformer_z_loop{refidx:02d}.npy").astype(np.float32)).cuda()
    # raw pair-state agreement
    zc = torch.nn.functional.cosine_similarity(zK.reshape(-1), ref.reshape(-1), dim=0).item()
    zmae = (zK - ref).abs().mean().item(); zscale = ref.abs().mean().item()
    # distogram agreement (the real target)
    Pk, edk = dgram(zK); Pr, edr = dgram(ref)
    r_if, mae_if = cmp_maps(edk, edr, AB, C)                              # interface AB x C
    r_full, mae_full = cmp_maps(edk, edr, slice(None), slice(None))
    js = float((0.5 * (Pk * (Pk / (0.5*(Pk+Pr)+1e-9)).clamp_min(1e-9).log()
                       + Pr * (Pr / (0.5*(Pk+Pr)+1e-9)).clamp_min(1e-9).log())).sum(-1).mean())
    rec = dict(K=K, ref=f"loop{refidx:02d}", z_cos=round(zc,4), z_mae_rel=round(zmae/zscale,4),
               dgram_r_interface=round(r_if,4), dgram_mae_interface=round(mae_if,3),
               dgram_r_full=round(r_full,4), dgram_JS_mean=round(js,5))
    recs.append(rec)
    print(rec, flush=True)
    del ref, zK; torch.cuda.empty_cache()
json.dump(recs, open("/home/ebrahim/esm/foldbench_predictions/steer_8ts0_recon.json", "w"))
print("RECON_DONE", flush=True)
