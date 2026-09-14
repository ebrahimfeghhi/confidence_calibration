"""Test-time compute on the ESM confidence head: jitter ONLY the confidence-head weights, score the
good (675) and decoy (101) 8ts0 poses N times per noise scale, compare the A-C pair-ipTM DISTRIBUTIONS.
Q: does the ensemble MEAN re-rank good>decoy, or does the VARIANCE discriminate (one fragile under noise)?"""
import sys, json, os
sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/steer/core")
sys.path.insert(0, "/home/ebrahim/tts_abag/steer/experiments")
import torch, numpy as np
import transformers.models.esmfold2.modeling_esmfold2 as M
import forward_grad as fg
from paths import ESM_PREDICTIONS

REPS = ESM_PREDICTIONS / "8ts0_reps_gb" / "8ts0"
SIGMAS = [0.0, 0.02, 0.05, 0.10]     # gaussian weight noise as fraction of each param's std
NDRAW = 30

# monkeypatch: on each confidence-head call, run 1 clean + NDRAW noisy passes per sigma, stash A-C pci dist
_orig = M.ConfidenceHead.forward
_ST = {}
def _patched(self, s_inputs, z, *a, **kw):
    base = _orig(self, s_inputs, z, *a, **kw)
    params = [p for p in self.parameters() if p.dim() > 0]   # model is frozen (requires_grad=False); perturb anyway
    saved = [p.detach().clone() for p in params]
    stds = [p.detach().std() for p in params]
    dists = {}
    g = torch.Generator(device=z.device)
    for sig in SIGMAS:
        vals = []
        ndraw = 1 if sig == 0.0 else NDRAW
        for n in range(ndraw):
            with torch.no_grad():
                for p, s0, sd in zip(params, saved, stds):
                    if sig > 0:
                        g.manual_seed(1000 * int(sig * 1000) + n)
                        p.copy_(s0 + torch.randn(p.shape, generator=g, device=p.device, dtype=p.dtype) * (sig * sd))
                    else:
                        p.copy_(s0)
                out = _orig(self, s_inputs, z, *a, **kw)
                pci = out["pair_chains_iptm"].detach().float().cpu().numpy()
                # per-sample matrix [B,nc,nc] or [nc,nc]; take sample 0, A-C=[0][2], B-C=[1][2]
                m = pci[0] if pci.ndim == 3 else pci
                vals.append(0.5 * (float(m[0][2]) + float(m[1][2])))
        dists[str(sig)] = vals
    with torch.no_grad():
        for p, s0 in zip(params, saved):
            p.copy_(s0)
    _ST["dists"] = dists
    return base
M.ConfidenceHead.forward = _patched

cfg = fg.configure_target("8ts0")
AB, C = cfg["ab"], cfg["c"]
print("loading model...", flush=True)
model, builder, feat, chain_infos, cond = fg.load(chunk_size=128)
pair_mask = cond["pair_mask"]

def trunk_diag(z):
    """How CERTAIN is the trunk about the interface? distogram entropy (low=sharp/confident) + z-norm."""
    with torch.no_grad():
        P = fg.cross_distogram_from_z(model, z, pair_mask, full_disto=True)   # [L*L, nbins]
    L = z.shape[1]; nb = P.shape[-1]; P = P.reshape(L, L, nb)
    block = P[AB, C, :].float()                                               # [nAB, nC, nbins]
    ent = -(block * (block + 1e-9).log()).sum(-1)                            # per-pair entropy
    bins = torch.linspace(2.0, 22.0, nb, device=P.device)
    ed = (block * bins).sum(-1)                                               # expected distance
    contact = ed < 8.0
    znorm = z[0][AB][:, C].float().norm(dim=-1)                               # [nAB,nC] interface z magnitude
    return dict(mean_ent=ent.mean().item(),
                contact_ent=(ent[contact].mean().item() if contact.any() else float("nan")),
                n_contact=int(contact.sum().item()),
                min_ent=ent.min().item(),
                z_if_norm=znorm.mean().item(),
                z_contact_norm=(znorm[contact].mean().item() if contact.any() else float("nan")))

results = {}; diags = {}
for tag, seed in [("good_675", 675), ("decoy_101", 101)]:
    z = torch.from_numpy(np.load(REPS / f"seed_{seed}" / "reps" / "pairformer_z_loop16.npy").astype(np.float32)).cuda()
    diags[tag] = trunk_diag(z)
    print(f"[{tag}] trunk-rep: {diags[tag]}", flush=True)
    _ST.clear()
    _ = fg.decode_from_ztrunk(model, builder, feat, chain_infos, cond, z, 1, 16, "8ts0", seed)
    results[tag] = _ST["dists"]
    print(f"[{tag}] scored", flush=True)
    torch.cuda.empty_cache()

print("\n=== TRUNK-REP diagnostic (interface AB x C block): is the decoy's trunk MORE certain? ===")
print(f"{'metric':<16}" + "".join(f"{t:>16}" for t in diags))
for m in ["mean_ent", "contact_ent", "min_ent", "n_contact", "z_if_norm", "z_contact_norm"]:
    print(f"{m:<16}" + "".join(f"{diags[t][m]:>16.4f}" for t in diags))
print("  (lower entropy = SHARPER/more-confident distogram; higher z-norm = stronger interface signal)")

print("\n=== A-C pair-ipTM under confidence-head weight noise (mean +/- std over draws) ===")
print(f"{'sigma':>7}" + "".join(f"{t:>22}" for t in results))
for sig in map(str, SIGMAS):
    row = f"{sig:>7}"
    for t in results:
        v = np.array(results[t][sig]); row += f"   {v.mean():.4f}+/-{v.std():.4f}"
    print(row)
os.makedirs("/home/ebrahim/esm/foldbench_predictions/conf_noise_8ts0", exist_ok=True)
json.dump(results, open("/home/ebrahim/esm/foldbench_predictions/conf_noise_8ts0/dists.json", "w"))
print("CONFNOISE_DONE", flush=True)
