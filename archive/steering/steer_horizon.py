"""Short-horizon steering: read the distogram loss after only K recycling loops
(where the trajectory is still shallow/responsive to s_lm), step s_lm, restart.

This is the user's idea: evaluate after a few loops -> gradient -> move s_lm ->
rerun from loop 0, so we never read the loss from deep inside the B1 attractor
(where the full-17-loop landscape was flat).

THE decisive check is transfer: a steered s_lm that makes loop-K look like G is
only useful if, run for the FULL 17 loops, it still lands near G. So for each K we
steer at horizon K, then evaluate the same s_lm at the full horizon (loop 16) and
report W1->G(16). If full-horizon W1 stays ~1.0, the short-horizon win collapsed
back to B1; if it drops toward the within-G band, the steer transferred.

Run (esm venv, GPU): python analysis/grad_steer/steer_horizon.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forward_grad import (  # noqa: E402
    load, compute_z_init, fixed_z0, single_rep_from_hidden, load_hidden,
    distogram_from_s_lm, cross_distogram_from_z, run_trunk, lift_to_lm_z,
    w1, good_reference, BASELINE, REPS, GOOD, N_LOOPS,
)
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import ESM_PREDICTIONS  # noqa: E402

OUT_RUN = ESM_PREDICTIONS / "7txw_steer_horizon" / "7txw"
HORIZONS = [1, 3, 6, 9]          # loops to read the loss after (K)
N_STEPS = 40
LR = 0.5


def Pg_at_loop(model, pair_mask, L):
    """G-centroid distogram from good seeds' SAVED loop-L reps (horizon-matched ref)."""
    Ps = []
    for s in GOOD:
        z = torch.from_numpy(
            np.load(REPS / f"seed_{s}" / "reps" / f"pairformer_z_loop{L:02d}.npy").astype(np.float32)
        ).cuda()
        with torch.no_grad():
            Ps.append(cross_distogram_from_z(model, z, pair_mask))
    Pg = torch.stack(Ps).mean(0)
    return (Pg / Pg.sum(-1, keepdim=True)).detach()


def main():
    torch.manual_seed(0)
    print("loading model...", flush=True)
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]
    z_init = compute_z_init(model, cond)
    z0 = fixed_z0(model, z_init, seed=0)

    Pg16 = good_reference(model, pair_mask, z0, z_init)           # mature-G target (loop 16)
    s_lm0 = single_rep_from_hidden(model, load_hidden(BASELINE)).detach()
    s0n = s_lm0.norm().item()

    def full_w1(x):                                              # full-horizon W1 -> G(16)
        with torch.no_grad():
            P, _ = distogram_from_s_lm(model, x, z_init, pair_mask, z0, total_steps=N_LOOPS + 1)
            return w1(P, Pg16).item()

    base_full = full_w1(s_lm0)
    print(f"\nbaseline full-horizon W1->G(16) = {base_full:.4f}   ||s_lm||={s0n:.1f}",
          flush=True)
    print("(target: within-G band ~0.033-0.079; this is what 'transfer' must approach)\n",
          flush=True)

    for K in HORIZONS:
        Lref = K - 1                                            # total_steps=K -> saved loop K-1
        Pg_match = Pg_at_loop(model, pair_mask, Lref)
        # how distinguishable is G from baseline at THIS horizon? (small K -> tiny)
        with torch.no_grad():
            Pb, _ = distogram_from_s_lm(model, s_lm0, z_init, pair_mask, z0, total_steps=K)
            sep = w1(Pb, Pg_match).item()

        # --- steer at horizon K toward mature-G (user's idea), restart each step ---
        s = s_lm0.clone().requires_grad_(True)
        opt = torch.optim.Adam([s], lr=LR)
        sh_start = sh_best = None
        for t in range(N_STEPS):
            P, _ = distogram_from_s_lm(model, s, z_init, pair_mask, z0, total_steps=K)
            loss = w1(P, Pg16)
            opt.zero_grad(); loss.backward(); opt.step()
            if t == 0:
                sh_start = loss.item()
            sh_best = loss.item() if sh_best is None else min(sh_best, loss.item())
        disp = (s.detach() - s_lm0).norm().item() / s0n

        # --- THE transfer test: same steered s_lm, run full 17 loops ---
        ft = full_w1(s.detach())
        print(f"K={K:2d} (loop{Lref:02d}): baseline-vs-G sep@K={sep:.3f} | "
              f"short-horizon W1->G(16) {sh_start:.3f}->{sh_best:.3f} | "
              f"rel_disp={disp:.2f} | FULL-horizon W1->G(16)={ft:.4f} "
              f"{'<-- transferred!' if ft < base_full - 0.1 else '(collapsed back)'}",
              flush=True)

        # save the full-horizon z for decode/DockQ
        with torch.no_grad():
            z_final, _ = run_trunk(model, z_init, lift_to_lm_z(model, s.detach()),
                                   pair_mask, z0, N_LOOPS + 1)
        rd = OUT_RUN / f"K{K:02d}" / "seed_0" / "reps"
        rd.mkdir(parents=True, exist_ok=True)
        np.save(rd / "pairformer_z_loop16.npy",
                z_final.float().cpu().numpy().astype(np.float16))

    print("\nHorizon sweep OK.", flush=True)


if __name__ == "__main__":
    main()
