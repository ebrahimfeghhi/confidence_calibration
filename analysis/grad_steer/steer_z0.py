"""Steer the INITIAL recurrent pair state z0 (instead of s_lm) into the good basin.

z0 = model._init_pair_state(z_init): a [1,L,L,256] state drawn from trunc_normal(0, std),
std=sqrt(2/(5*256))~=0.04. It's the loop-0 value of the parcae SSM recurrence
z = a*z + (input from z_init + lm_z, re-injected every loop). Because a<1, z0 is a
DECAYING TRANSIENT — its influence fades over the 17 loops while the s_lm-derived input
is re-added each loop. We already know z0 perturbations can flip the 7txw basin (CUDA
nondeterminism in the random init does it for watershed seeds); here we steer it on purpose.

Method (mirrors steer_horizon.py, but the leaf is z0, and s_lm is held at baseline):
  for K in {1,3,6,9}: steer z0 to minimize w1(P_K(z0), P_G) read after K loops, restart
  each step; then run the SAME z0 the full 17 loops and report transfer = w1(P_full, P_G).
  Also report how far z0 strayed from its trunc_normal init (rel norm + frac outside +-3std)
  — z0's native manifold is just "stay trunc-normal".

If transfer collapses back to ~baseline, the recurrence forgot the nudge (z0 too transient);
if it drops toward the within-G band (~0.033-0.079), the initial-state steer survived.

Run (esm venv, GPU): CUDA_VISIBLE_DEVICES=5 ~/esm/.venv/bin/python \
    analysis/grad_steer/steer_z0.py [--lr 0.02] [--steps 40]
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forward_grad import (  # noqa: E402
    load, compute_z_init, fixed_z0, single_rep_from_hidden, load_hidden,
    lift_to_lm_z, run_trunk, cross_distogram_from_z, w1, good_reference,
    BASELINE, N_LOOPS,
)
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import ESM_PREDICTIONS  # noqa: E402

OUT_RUN = ESM_PREDICTIONS / "7txw_steer_z0" / "7txw"
HORIZONS = [1, 3, 6, 9]


def main():
    torch.manual_seed(0)
    lr = float(_argval("--lr", 0.02))
    n_steps = int(_argval("--steps", 40))
    hz = _argval("--horizons", None)
    horizons = [int(x) for x in str(hz).split(",")] if hz else HORIZONS
    print(f"loading model (ESMC -> CPU; baseline s_lm from disk)... lr={lr} steps={n_steps}",
          flush=True)
    model, builder, feat, chain_infos, cond = load()          # keep_esmc=False (old default)
    pair_mask = cond["pair_mask"]
    z_init = compute_z_init(model, cond)
    z0_init = fixed_z0(model, z_init, seed=0)                  # the reference trunc_normal init
    std = math.sqrt(2.0 / (5.0 * z_init.shape[-1]))
    z0n = z0_init.norm().item()

    Pg16 = good_reference(model, pair_mask, z0_init, z_init)   # mature-G target (loop16)

    # s_lm held FIXED at the no-mask/no-dropout baseline; only z0 varies.
    lm_z = lift_to_lm_z(model, single_rep_from_hidden(model, load_hidden(BASELINE))).detach()

    def P_at(z0, K):
        z, _ = run_trunk(model, z_init, lm_z, pair_mask, z0, total_steps=K)
        return cross_distogram_from_z(model, z, pair_mask)

    def full_w1(z0):
        with torch.no_grad():
            return w1(P_at(z0, N_LOOPS + 1), Pg16).item()

    def deviation(z0):
        d = z0.detach() - z0_init
        rel = d.norm().item() / z0n
        frac_out = (z0.detach().abs() > 3 * std).float().mean().item()
        return rel, frac_out

    base_full = full_w1(z0_init)
    print(f"\nbaseline (unsteered z0) full-horizon W1->G(16) = {base_full:.4f}   "
          f"||z0||={z0n:.1f}  std={std:.4f}", flush=True)
    print("(target: within-G band ~0.033-0.079; transfer must approach this)\n", flush=True)
    print(f"  {'K':>3} {'sep@K':>7} {'shortW1 start->best':>22} "
          f"{'rel_dz0':>8} {'frac>3std':>10} {'FULL W1->G':>11}", flush=True)

    for K in horizons:
        with torch.no_grad():
            sep = w1(P_at(z0_init, K), Pg16).item()            # G vs init separation at this K

        z = z0_init.clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)
        sh_start = sh_best = None
        for t in range(n_steps):
            P = P_at(z, K)
            loss = w1(P, Pg16)
            opt.zero_grad(); loss.backward(); opt.step()
            sh_start = loss.item() if t == 0 else sh_start
            sh_best = loss.item() if sh_best is None else min(sh_best, loss.item())

        rel, frac_out = deviation(z)
        ft = full_w1(z.detach())
        flag = "<- transferred" if ft < base_full - 0.1 else "(collapsed back)"
        print(f"  {K:>3d} {sep:7.3f} {sh_start:9.3f}->{sh_best:<9.3f} "
              f"{rel:8.2f} {frac_out:10.3f} {ft:11.4f} {flag}", flush=True)

        rd = OUT_RUN / f"K{K:02d}" / "seed_0" / "reps"
        rd.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            z_final, _ = run_trunk(model, z_init, lm_z, pair_mask, z.detach(), N_LOOPS + 1)
        np.save(rd / "pairformer_z_loop16.npy", z_final.float().cpu().numpy().astype(np.float16))

    print("\nz0 horizon sweep OK.", flush=True)


def _argval(name, default):
    for a in sys.argv[1:]:
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
        if a == name:
            i = sys.argv.index(a)
            if i + 1 < len(sys.argv):
                return sys.argv[i + 1]
    return default


if __name__ == "__main__":
    main()
