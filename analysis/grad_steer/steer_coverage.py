"""Label-free coverage: can we recover G WITHOUT using G as the target?

Annealed multi-particle steering over s_lm (the valuable part of SMC — tempering —
without correctness-resampling, which would collapse onto the confident decoy B2
since no internal signal cleanly separates G from B2).

Each particle: start = baseline + small random nudge (direction diversity), then a
shell loss pushes its K-loop cross-distogram to a distance X from the baseline
(loop16) distogram, with X ANNEALED 0.3 -> ~1.04 (the G-radius; B2 sits at 0.66).
No G in the loss anywhere.

Readout (free, no decode): after steering, run the particle's s_lm at the FULL 17
loops and classify by nearest basin centroid (G / B1 / B2). Question: across N blind
directions at the G-shell, how many land in G vs B2 vs elsewhere? Decode the G-hits.

Run (esm venv, GPU): python analysis/grad_steer/steer_coverage.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forward_grad import (  # noqa: E402
    load, compute_z_init, fixed_z0, single_rep_from_hidden, load_hidden,
    distogram_from_s_lm, cross_distogram_from_z, run_trunk, lift_to_lm_z,
    w1, BASELINE, REPS, GOOD, N_LOOPS,
)
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import ESM_PREDICTIONS  # noqa: E402

B1 = [1, 2, 3, 4, 5, 7, 8, 9, 11, 16, 17, 18, 19, 20, 21, 22, 23, 26, 28, 29, 30, 31, 32,
      34, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49, 52, 55, 56, 57, 58, 59]
B2 = [0, 6, 12, 13, 15, 24, 27, 35, 43, 47, 50, 53, 54]
OUT = ESM_PREDICTIONS / "7txw_steer_coverage" / "7txw"

K = 1
N_PARTICLES = 8
STEPS = 30
LR = 0.5
INIT_FRAC = 0.05          # random init nudge as fraction of ||s_lm||
X_LO, X_HI = 0.30, 1.05   # annealing schedule (radius from baseline) ramps lo->hi
RAMP_FRAC = 0.8           # reach X_HI by 80% of steps, then hold
REPEL_MARGIN = 0.60       # stay >= this W1 from each repelled attractor (B2)
REPEL_MU = 4.0            # repulsion strength (B2 is a deep well, needs to be strong)


def centroid(model, pair_mask, seeds):
    Ps = [cross_distogram_from_z(
              model, torch.from_numpy(np.load(
                  REPS / f"seed_{s}" / "reps" / "pairformer_z_loop16.npy").astype(np.float32)
              ).cuda(), pair_mask)
          for s in seeds]
    P = torch.stack(Ps).mean(0)
    return (P / P.sum(-1, keepdim=True)).detach()


def main():
    print("loading model...", flush=True)
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]
    z_init = compute_z_init(model, cond)
    z0 = fixed_z0(model, z_init, seed=0)

    with torch.no_grad():
        Pbase = cross_distogram_from_z(
            model, torch.from_numpy(np.load(
                BASELINE / "reps" / "pairformer_z_loop16.npy").astype(np.float32)).cuda(),
            pair_mask)
        Pg, Pb1, Pb2 = (centroid(model, pair_mask, s) for s in (GOOD, B1, B2))
    s_lm0 = single_rep_from_hidden(model, load_hidden(BASELINE)).detach()
    s0n = s_lm0.norm().item()

    args = sys.argv[1:]
    use_repel = "--no-repel" not in args
    use_anneal = "--no-anneal" not in args

    def _flagval(name, default):
        for a in args:
            if a.startswith(name + "="):
                return type(default)(a.split("=", 1)[1])
        return default
    k = _flagval("--k", K)
    steps = _flagval("--steps", STEPS)
    tol = _flagval("--tol", 0.02)            # early-stop when |W1 - X_p| < tol
    xladder = None                            # per-particle target distance, indexed by pid
    for a in args:
        if a.startswith("--xladder="):
            xladder = [float(v) for v in a.split("=", 1)[1].split(",")]
    lr = _flagval("--lr", LR)
    tag = None                                # output subdir suffix (avoid clobbering runs)
    for a in args:
        if a.startswith("--tag="):
            tag = a.split("=", 1)[1]
    pids = [int(x) for x in args if not x.startswith("--")] or list(range(N_PARTICLES))

    def x_target(t):
        if not use_anneal:
            return X_HI
        f = min(1.0, t / (RAMP_FRAC * steps))
        return X_LO + (X_HI - X_LO) * f

    if xladder is not None:
        shell_desc = f"ladder {xladder} (per-particle X_p, early-stop tol {tol})"
    else:
        shell_desc = f"anneal {X_LO}->{X_HI}" if use_anneal else f"fixed {X_HI}"
    repel_desc = (f"REPEL B2 (margin {REPEL_MARGIN}, mu {REPEL_MU})" if use_repel
                  else "no repel")
    print(f"\ncoverage: K={k}, steps={steps}, shell {shell_desc}, {repel_desc} "
          f"(B2~0.66, G~1.04 from baseline)", flush=True)
    print(f"this instance runs particles {pids}", flush=True)
    counts = {"G": 0, "B1": 0, "B2": 0, "other": 0}
    for p in pids:
        xp = xladder[p] if xladder is not None else None
        torch.manual_seed(1000 + p)
        d = torch.randn_like(s_lm0); d = d / d.norm() * (INIT_FRAC * s0n)
        s = (s_lm0 + d).requires_grad_(True)
        opt = torch.optim.Adam([s], lr=lr)
        tgt_desc = f"X={xp:.2f}" if xp is not None else "annealed/fixed"
        print(f"  particle {p}: steering ({steps} steps @ K={k}, lr {lr}, target {tgt_desc})",
              flush=True)
        dist = None
        for t in range(steps):
            P, _ = distogram_from_s_lm(model, s, z_init, pair_mask, z0, total_steps=k)
            dist = w1(P, Pbase)                                  # push off the B1/baseline floor
            xt = xp if xp is not None else x_target(t)
            shell = (dist - xt) ** 2
            repel = (torch.relu(REPEL_MARGIN - w1(P, Pb2)) ** 2 if use_repel
                     else torch.zeros((), device=dist.device))
            loss = shell + REPEL_MU * repel
            opt.zero_grad(); loss.backward(); opt.step()
            print(f"    step {t:2d}: dist={dist.item():.3f} target={xt:.3f} "
                  f"shell={shell.item():.4f} repel={repel.item():.4f} "
                  f"loss={loss.item():.4f}", flush=True)
            if xp is not None and abs(dist.item() - xp) < tol:
                print(f"    converged to X={xp:.2f} at step {t}", flush=True)
                break
        k_dist = dist.item()
        target_ref = xp if xp is not None else X_HI
        reached = ("yes" if abs(k_dist - target_ref) < 0.05
                   else f"no(@{k_dist:.2f},target{target_ref:.2f})")
        # classify at full horizon
        with torch.no_grad():
            Pf, _ = distogram_from_s_lm(model, s.detach(), z_init, pair_mask, z0,
                                        total_steps=N_LOOPS + 1)
            wg, wb1, wb2 = (w1(Pf, c).item() for c in (Pg, Pb1, Pb2))
            df = w1(Pf, Pbase).item()
        basin = min((("G", wg), ("B1", wb1), ("B2", wb2)), key=lambda kv: kv[1])[0]
        counts[basin if basin in counts else "other"] += 1
        flag = "  <== G!" if basin == "G" else ""
        print(f"  -> particle {p}: K-dist={k_dist:.3f} reached={reached} | "
              f"full_dist={df:.3f} | W1->G={wg:.3f} B1={wb1:.3f} B2={wb2:.3f} | "
              f"basin={basin}{flag}", flush=True)
        # save EVERY particle's settled z (for run-2 repulsion / clustering)
        with torch.no_grad():
            zf, _ = run_trunk(model, z_init, lift_to_lm_z(model, s.detach()),
                              pair_mask, z0, N_LOOPS + 1)
        rd = OUT / (f"particle{p}" + (f"_{tag}" if tag else "")) / "seed_0" / "reps"
        rd.mkdir(parents=True, exist_ok=True)
        np.save(rd / "pairformer_z_loop16.npy",
                zf.float().cpu().numpy().astype(np.float16))

    print(f"\nbasin distribution over particles {pids}: {counts}", flush=True)
    print(f"  -> recovered G in {counts['G']}/{len(pids)} (decode these for DockQ)",
          flush=True)
    print("\nCoverage run OK.", flush=True)


if __name__ == "__main__":
    main()
