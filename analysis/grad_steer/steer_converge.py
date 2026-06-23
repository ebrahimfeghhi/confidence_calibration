"""Per-step convergence diagnostics for supervised (attract-to-G) steering, to tell
apart 'plateaued at a legibility floor' from 'optimizer starved / lr too low'.

For each config (lever in {slm, z0}, horizon K, lr, steps) run the supervised steer
toward the mature-G centroid P_G(16) and LOG EVERY STEP:
  short_loss = w1(P_K(current), P_G16)        # the thing being optimized
  full_w1    = w1(P_full(current), P_G16)     # transfer (every --full_every steps)
  dev        = manifold deviation             # slm: ||s-s0||/||s0||; z0: frac |z|>3std

s_lm leaf = baseline single LM rep (re-injected every loop -> persistent).
z0   leaf = fixed_z0 init (trunc_normal, std=sqrt(2/(5*256))), s_lm held at baseline
            (one-shot initial state -> decays).

Configs are "lever:K:lr:steps" (e.g. slm:1:0.5:150  z0:1:0.02:150  z0:1:0.1:150).
Run several per GPU; each writes results/csv_files/steer_converge_<lever>_K<k>_lr<lr>.csv.

Run (esm venv, GPU): CUDA_VISIBLE_DEVICES=N ~/esm/.venv/bin/python \
    analysis/grad_steer/steer_converge.py --configs slm:1:0.5:150,slm:3:0.5:150
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forward_grad import (  # noqa: E402
    load, compute_z_init, fixed_z0, single_rep_from_hidden, load_hidden,
    lift_to_lm_z, run_trunk, cross_distogram_from_z, distogram_from_s_lm,
    w1, good_reference, BASELINE, N_LOOPS,
)
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import CSV_DIR, ESM_PREDICTIONS  # noqa: E402

ZOUT = ESM_PREDICTIONS / "7txw_steer_converge"


def _flag(name, default):
    for a in sys.argv[1:]:
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
        if a == name:
            i = sys.argv.index(a)
            return sys.argv[i + 1]
    return default


def main():
    full_every = int(_flag("--full_every", 5))
    save_z = "--save_z" in sys.argv[1:]
    configs = _flag("--configs", "slm:1:0.5:150,slm:3:0.5:150,"
                                 "z0:1:0.02:150,z0:3:0.02:150,z0:1:0.1:150").split(",")
    print(f"loading model... configs={configs}", flush=True)
    model, builder, feat, chain_infos, cond = load()       # ESMC->CPU (old default)
    pair_mask = cond["pair_mask"]
    z_init = compute_z_init(model, cond)
    z0_init = fixed_z0(model, z_init, seed=0)
    std = math.sqrt(2.0 / (5.0 * z_init.shape[-1]))

    Pg16 = good_reference(model, pair_mask, z0_init, z_init)         # mature-G target
    s0 = single_rep_from_hidden(model, load_hidden(BASELINE)).detach()
    s0n = s0.norm().item()
    lm_z_base = lift_to_lm_z(model, s0).detach()                    # fixed s_lm for z0 runs

    def P_slm(s, K):
        P, _ = distogram_from_s_lm(model, s, z_init, pair_mask, z0_init, total_steps=K)
        return P

    def P_z0(z, K):
        z_out, _ = run_trunk(model, z_init, lm_z_base, pair_mask, z, total_steps=K)
        return cross_distogram_from_z(model, z_out, pair_mask)

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        lever, K, lr, steps = cfg.split(":")
        K, lr, steps = int(K), float(lr), int(steps)
        print(f"\n===== {lever} K={K} lr={lr} steps={steps} =====", flush=True)

        if lever == "slm":
            leaf = s0.clone().requires_grad_(True)
            P_at = lambda x, KK: P_slm(x, KK)
            def dev_of(x):
                return (x.detach() - s0).norm().item() / s0n
        elif lever == "z0":
            leaf = z0_init.clone().requires_grad_(True)
            P_at = lambda x, KK: P_z0(x, KK)
            def dev_of(x):
                return (x.detach().abs() > 3 * std).float().mean().item()   # frac off-manifold
        else:
            raise SystemExit(f"unknown lever {lever!r}")

        opt = torch.optim.Adam([leaf], lr=lr)
        out = CSV_DIR / f"steer_converge_{lever}_K{K}_lr{lr}.csv"
        rows = []
        best_loss, best_step, best_leaf = float("inf"), -1, None   # best by SHORT loss
        for t in range(steps):
            P = P_at(leaf, K)
            loss = w1(P, Pg16)
            if loss.item() < best_loss:                            # snapshot pre-step leaf
                best_loss, best_step, best_leaf = loss.item(), t, leaf.detach().clone()
            opt.zero_grad(); loss.backward(); opt.step()
            full = ""
            if t % full_every == 0 or t == steps - 1:
                with torch.no_grad():
                    full = round(w1(P_at(leaf.detach(), N_LOOPS + 1), Pg16).item(), 4)
            rows.append({"step": t, "short_loss": round(loss.item(), 4),
                         "full_w1": full, "dev": round(dev_of(leaf), 4)})
            if t % 10 == 0 or t == steps - 1:
                print(f"  step {t:3d}: short={loss.item():.4f}  full_w1={full}  "
                      f"dev={dev_of(leaf):.3f}", flush=True)
        with open(out, "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=["step", "short_loss", "full_w1", "dev"])
            wcsv.writeheader(); wcsv.writerows(rows)
        print(f"  wrote {out}", flush=True)

        # Save the BEST-short-loss leaf's full-horizon z so it can be decoded -> DockQ.
        if save_z:
            with torch.no_grad():
                if lever == "slm":
                    z_full, _ = run_trunk(model, z_init, lift_to_lm_z(model, best_leaf),
                                          pair_mask, z0_init, N_LOOPS + 1)
                else:
                    z_full, _ = run_trunk(model, z_init, lm_z_base, pair_mask,
                                          best_leaf, N_LOOPS + 1)
                best_full = w1(cross_distogram_from_z(model, z_full, pair_mask), Pg16).item()
            rd = ZOUT / f"{lever}_K{K}_lr{lr}" / "seed_0" / "reps"
            rd.mkdir(parents=True, exist_ok=True)
            np.save(rd / "pairformer_z_loop16.npy",
                    z_full.float().cpu().numpy().astype(np.float16))
            print(f"  best short_loss={best_loss:.4f} @step {best_step}; "
                  f"its full_w1->G={best_full:.4f}; saved z -> {rd}", flush=True)

    print("\nsteer_converge OK.", flush=True)


if __name__ == "__main__":
    main()
