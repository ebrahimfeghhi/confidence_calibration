"""Phase 2: actually steer the baseline single LM rep `s_lm` toward the G basin and
see how far the cross-block distogram W1 can be driven (and, separately, whether one
big step helps vs iterating).

Key fix over the first probe: steps are scaled RELATIVE to ||s_lm|| (~2.5e3) so they
exceed the bf16 resolution floor (~1% relative); absolute eps<~25 round away.

Outputs: a relative-scale line search (one-step probe) + an Adam optimization trace
(W1 per step), compared against the within-G spread and the B1 baseline. Saves the
final steered loop16 z for decode/DockQ.

Run (esm venv, GPU): python analysis/grad_steer/steer_lmz.py
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

OUT_RUN = ESM_PREDICTIONS / "7txw_steer" / "7txw"   # steered z's land here for decode


def within_g_spread(model, pair_mask, Pg):
    """W1 of each good seed's saved loop16 distogram to the G centroid = the target band."""
    vals = []
    for s in GOOD:
        z = torch.from_numpy(
            np.load(REPS / f"seed_{s}" / "reps" / "pairformer_z_loop16.npy").astype(np.float32)
        ).cuda()
        with torch.no_grad():
            vals.append(w1(cross_distogram_from_z(model, z, pair_mask), Pg).item())
    return vals


def main():
    torch.manual_seed(0)
    print("loading model...", flush=True)
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]
    z_init = compute_z_init(model, cond)
    z0 = fixed_z0(model, z_init, seed=0)

    Pg = good_reference(model, pair_mask, z0, z_init)
    g_band = within_g_spread(model, pair_mask, Pg)
    print(f"within-G W1 band (good seeds -> G centroid): "
          f"min={min(g_band):.4f} max={max(g_band):.4f}  "
          f"mean={np.mean(g_band):.4f}", flush=True)

    s_lm0 = single_rep_from_hidden(model, load_hidden(BASELINE)).detach()
    s0n = s_lm0.norm().item()

    def loss_at(x):
        with torch.no_grad():
            P, _ = distogram_from_s_lm(model, x, z_init, pair_mask, z0)
            return w1(P, Pg).item()

    L0 = loss_at(s_lm0)
    print(f"\nbaseline (B1) start: W1->G = {L0:.4f}   ||s_lm||={s0n:.1f}", flush=True)

    # gradient at the baseline point
    s = s_lm0.clone().requires_grad_(True)
    with torch.enable_grad():
        P, _ = distogram_from_s_lm(model, s, z_init, pair_mask, z0)
        w1(P, Pg).backward()
    ghat = (s.grad / (s.grad.norm() + 1e-12)).detach()

    # --- (A) one-step probe at RELATIVE scale: step -frac*||s_lm||*ghat ---
    print("\none-step line search (x = s_lm - frac*||s_lm||*ghat):", flush=True)
    print(f"  {'frac':>6} {'|step|':>8} {'W1->G':>8} {'dL':>9}", flush=True)
    for frac in (0.01, 0.03, 0.1, 0.3, 0.6, 1.0):
        L = loss_at(s_lm0 - frac * s0n * ghat)
        print(f"  {frac:6.2f} {frac*s0n:8.1f} {L:8.4f} {L-L0:+9.4f}", flush=True)

    # --- (B) iterative Adam optimization (re-linearizes each step) ---
    print("\niterative steering (Adam, no trust region first):", flush=True)
    s = s_lm0.clone().requires_grad_(True)
    opt = torch.optim.Adam([s], lr=0.5)
    best = (L0, s_lm0.clone())
    for t in range(60):
        P, _ = distogram_from_s_lm(model, s, z_init, pair_mask, z0)
        loss = w1(P, Pg)
        opt.zero_grad(); loss.backward(); opt.step()
        lv = loss.item()
        if lv < best[0]:
            best = (lv, s.detach().clone())
        if t % 5 == 0 or t == 59:
            disp = (s.detach() - s_lm0).norm().item() / s0n
            print(f"  step {t:2d}  W1->G={lv:.4f}  rel_disp={disp:.3f}", flush=True)
    print(f"\nbest W1->G = {best[0]:.4f}  (B1 start {L0:.4f}, within-G band "
          f"{min(g_band):.4f}-{max(g_band):.4f})", flush=True)

    # --- save the best steered loop16 z for decode/DockQ ---
    with torch.no_grad():
        lm_z = lift_to_lm_z(model, best[1])
        z_final, _ = run_trunk(model, z_init, lm_z, pair_mask, z0, N_LOOPS + 1)
    reps_dir = OUT_RUN / "seed_0" / "reps"
    reps_dir.mkdir(parents=True, exist_ok=True)
    np.save(reps_dir / "pairformer_z_loop16.npy",
            z_final.detach().float().cpu().numpy().astype(np.float16))
    print(f"saved steered loop16 z -> {reps_dir}/pairformer_z_loop16.npy", flush=True)
    print("\nPhase 2 line search + opt OK.", flush=True)


if __name__ == "__main__":
    main()
