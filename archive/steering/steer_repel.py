"""Label-free sequential repulsion — multi-trial, for the z0 OR s_lm lever.

An "experiment" = T independent TRIALS; each trial = R runs that accumulate repulsion
anchors (the leaf is RESET each run). One trial works like this: keep a growing list of
visited distograms ("anchors"), seeded with the baseline (the B1 floor); each run resets
the leaf, takes M gradient steps pushing its distogram as FAR as possible from every
anchor in the list, then appends the result. So run k is repelled from the baseline + the
k-1 runs before it. Each trial clears the list and starts over (replication for robustness).

With no symmetry-breaker the s_lm leaf starts byte-identical to its own seed anchor, where
the repulsion gradient is exactly ZERO (W1 is minimized at the anchor — abs'(0)=0), so it
never moves. Two ways to break that symmetry for the s_lm lever:

  --diversity nudge : z0 context FIXED (z0_base); leaf = baseline s_lm + a fresh random
                      nudge (INIT_FRAC*||s_lm||) each run. The proven steer_sequential
                      recipe — diversity lives in the lever's own space, the repulsion W1
                      stays a PURE s_lm-basin signal, and any G is cleanly attributable to
                      s_lm steering.
  --diversity z0    : leaf = baseline s_lm (NO nudge), optimized; the FIXED z0 CONTEXT is
                      RESAMPLED per run (fixed_z0(per-run seed)). z0 is the network's
                      natural noise input, so this is the more principled entropy source —
                      but the K=3 repulsion W1 then MIXES s_lm- and z0-differences (z0 is
                      strong at K=3), and a G hit is confounded (lucky z0 vs s_lm steering).

The z0 LEVER (--lever z0) optimizes z0 itself (s_lm fixed); --diversity is ignored (each
run already draws a fresh z0 as the leaf).

Anchors and the live loss are read at horizon K (small, so z0's gradient is alive; s_lm is
re-injected every loop so K matters less). Snapshots are 17-loop evals. Run 0 of each trial
is the canonical baseline (s0, z0_base, no nudge) = the B1 floor.

--carry : carry-forward "single-leaf walk" — do NOT reset the leaf between runs; keep
optimizing the already-optimized leaf, so within a trial it's one continuous walk
repelled from its whole trail (the nudge is applied once, to start run 1 off the cusp).
dev then accumulates run-over-run, so expect off-manifold drift by the late runs.

Run (esm venv, GPU):
  CUDA_VISIBLE_DEVICES=0 python steer_repel.py --lever slm --diversity nudge --M 5 --K 3 --trials 10 --runs 10
  CUDA_VISIBLE_DEVICES=1 python steer_repel.py --lever slm --diversity z0    --M 5 --K 3 --trials 10 --runs 10
  CUDA_VISIBLE_DEVICES=2 python steer_repel.py --lever slm --diversity nudge --carry --M 5 --K 3 --trials 10 --runs 10
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
    lift_to_lm_z, run_trunk, cross_distogram_from_z, w1, BASELINE, GOOD, N_LOOPS,
)
from steer_sequential import B1, B2          # noqa: E402
from steer_deviation import centroid          # noqa: E402

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import CSV_DIR, ESM_PREDICTIONS  # noqa: E402

ZOUT = ESM_PREDICTIONS / "7txw_repel"          # lever-neutral (holds z0 + slm; nudge + z0-var)
INIT_FRAC = 0.05                               # s_lm nudge size (fraction of ||s_lm||)


def _flag(name, default, cast=str):
    for a in sys.argv[1:]:
        if a.startswith(name + "="):
            return cast(a.split("=", 1)[1])
        if a == name:
            i = sys.argv.index(a)
            return cast(sys.argv[i + 1])
    return default


def main():
    lever = _flag("--lever", "slm")            # slm | z0
    div = _flag("--diversity", "nudge")        # nudge | z0   (slm lever only)
    M = _flag("--M", 5, int)                    # gradient steps per run
    R = _flag("--runs", 10, int)                # steered runs per trial (+ run 0 = baseline seed)
    T = _flag("--trials", 10, int)              # independent trials (each resets its anchor list)
    lr = _flag("--lr", 0.3, float)             # proven repulsion lr (s_lm); z0 lever wants ~0.1
    K = _flag("--K", 3, int)                    # horizon for BOTH the live loss and the anchors
    carry = "--carry" in sys.argv[1:]           # carry-forward walk: don't reset the leaf between runs
    window = _flag("--window", 0, int)          # repel from only the last `window` anchors (0 = all)
    leash = _flag("--leash", 0.0, float)        # clamp ||leaf-s0|| <= leash*||s0|| each step (slm; 0 = off)
    keepbase = "--keepbase" in sys.argv[1:]     # always include the baseline anchor (anchors[0]) in the loss
    renudge = "--renudge" in sys.argv[1:]       # carry: re-nudge the leaf off the cusp at the start of each run
    tagsuffix = _flag("--tagsuffix", "", str)   # appended to the output tag (avoid clobbering prior runs)
    assert lever in ("z0", "slm")
    assert div in ("nudge", "z0")
    print(f"loading model... lever={lever} diversity={div} carry={carry} window={window} "
          f"keepbase={keepbase} renudge={renudge} leash={leash} M={M} trials={T} runs={R} "
          f"lr={lr} K={K}", flush=True)
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]
    z_init = compute_z_init(model, cond)
    std = math.sqrt(2.0 / (5.0 * z_init.shape[-1]))

    s0 = single_rep_from_hidden(model, load_hidden(BASELINE)).detach()    # baseline s_lm
    s0n = s0.norm().item()
    lm_z_base = lift_to_lm_z(model, s0).detach()                          # fixed for z0 lever
    z0_base = fixed_z0(model, z_init, seed=0)                             # canonical baseline z0

    with torch.no_grad():
        Pg, Pb1, Pb2 = (centroid(model, pair_mask, s) for s in (GOOD, B1, B2))

    slm_z0 = [z0_base]      # the live z0 context for the slm lever (reset per run in z0-var mode)

    def run(leaf, total_steps, grad):
        """leaf -> trunk(total_steps) -> cross-distogram. Branches on lever."""
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            if lever == "z0":
                z_out, _ = run_trunk(model, z_init, lm_z_base, pair_mask, leaf, total_steps)
            else:
                z_out, _ = run_trunk(model, z_init, lift_to_lm_z(model, leaf), pair_mask,
                                     slm_z0[0], total_steps)
            return z_out, cross_distogram_from_z(model, z_out, pair_mask)

    def dev_of(leaf):
        if lever == "z0":
            return (leaf.detach().abs() > 3 * std).float().mean().item()    # frac off-manifold
        return (leaf.detach() - s0).norm().item() / s0n                      # rel s_lm displacement

    def gseed(trial, run_idx):
        return 10_000 + 1000 * trial + run_idx

    def set_context(trial, run_idx):
        """Set the per-run fixed z0 context. Resampled only in slm z0-var mode (run>0)."""
        if lever == "slm" and div == "z0" and run_idx > 0:
            slm_z0[0] = fixed_z0(model, z_init, seed=gseed(trial, run_idx))
        else:
            slm_z0[0] = z0_base

    def init_leaf(trial, run_idx):
        if lever == "z0":
            return fixed_z0(model, z_init, seed=gseed(trial, run_idx)).clone().requires_grad_(True)
        # s_lm lever
        if div == "nudge" and run_idx > 0:
            torch.manual_seed(gseed(trial, run_idx))               # reproducible per (trial, run)
            d = torch.randn_like(s0)
            d = d / d.norm() * (INIT_FRAC * s0n)                   # 5%-of-norm random restart
            return (s0 + d).clone().requires_grad_(True)
        return s0.clone().requires_grad_(True)                     # baseline (run-0 seed; z0-var diversity)

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    sfx = ((f"_w{window}" if window else "") + ("_rn" if renudge else "") + ("_kb" if keepbase else "")
           + (f"_leash{int(round(leash*100))}" if leash else ""))
    tag = ((f"slm_{div}" if lever == "slm" else "z0") + ("_carry" if carry else "") + sfx + f"_M{M}_K{K}" + tagsuffix)
    out_csv = CSV_DIR / f"steer_repel_{tag}.csv"
    fields = ["trial", "run", "basin", "wg", "wb1", "wb2", "dev", "repel"]
    rows = []
    steps_csv = CSV_DIR / f"steer_repel_{tag}_steps.csv"   # per-step optimization trajectory
    step_fields = ["trial", "run", "step", "repel", "dev"]
    step_rows = []

    def flush_steps():
        with open(steps_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=step_fields); w.writeheader(); w.writerows(step_rows)

    def flush_csv():
        with open(out_csv, "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=fields)
            wcsv.writeheader()
            wcsv.writerows(rows)

    def snapshot(leaf, trial, run_idx, repel):
        """17-loop eval of the current copy: distogram, basin, dev. Save z for decode. (Eval only.)"""
        z17, P17 = run(leaf, N_LOOPS + 1, grad=False)
        wg, wb1, wb2 = (w1(P17, c).item() for c in (Pg, Pb1, Pb2))
        basin = min((("G", wg), ("B1", wb1), ("B2", wb2)), key=lambda kv: kv[1])[0]
        dev = dev_of(leaf)
        rd = ZOUT / tag / f"trial{trial:02d}" / f"run{run_idx:02d}" / "seed_0" / "reps"
        rd.mkdir(parents=True, exist_ok=True)
        np.save(rd / "pairformer_z_loop16.npy", z17.float().cpu().numpy().astype(np.float16))
        zK, _ = run(leaf, K, grad=False)                                 # K-horizon z for K-vs-17 decode
        np.save(rd / f"pairformer_z_K{K}.npy", zK.float().cpu().numpy().astype(np.float16))
        rows.append({"trial": trial, "run": run_idx, "basin": basin,
                     "wg": round(wg, 4), "wb1": round(wb1, 4), "wb2": round(wb2, 4),
                     "dev": round(dev, 4), "repel": round(repel, 4)})
        flag = "  <== G!" if basin == "G" else ""
        print(f"  t{trial:02d} run {run_idx:2d}: basin={basin:>2} W1->G={wg:.3f} "
              f"dev={dev:.3f} repel={repel:.3f}{flag}", flush=True)
        return P17.detach()

    # ---- T independent trials; each accumulates its own anchor list (RESET between trials) ----
    for t in range(T):
        print(f"\n===== trial {t+1}/{T} =====", flush=True)
        # Step 0: seed the anchor list with the baseline's K-distogram (the B1 floor).
        set_context(t, 0)
        seed_leaf = init_leaf(t, 0)
        with torch.no_grad():
            _, P0 = run(seed_leaf, K, grad=False)
        anchors = [P0.detach()]
        snapshot(seed_leaf, t, 0, repel=0.0)
        flush_csv()

        leaf = init_leaf(t, 1) if carry else None           # carry: nudge once, leaf persists across runs
        for r in range(1, R + 1):
            set_context(t, r)                               # fixed z0 context for this run
            if not carry:
                leaf = init_leaf(t, r)                      # RESET each run: nudged s_lm / baseline copy
            elif renudge and r > 1 and lever == "slm":      # carry: kick off the cusp each run (run 1 already nudged)
                with torch.no_grad():
                    torch.manual_seed(gseed(t, r) + 7)
                    d = torch.randn_like(leaf); d = d / d.norm() * (INIT_FRAC * s0n)
                    leaf.add_(d)
            opt = torch.optim.Adam([leaf], lr=lr)           # fresh optimizer each run; leaf persists if carry
            last = 0.0
            for s in range(M):
                _, P_K = run(leaf, K, grad=True)            # live distogram at K
                active = anchors[-window:] if window else anchors   # window: repel from the last `window` only
                if keepbase and window and len(anchors) > window:   # always keep the baseline anchor too
                    active = [anchors[0]] + active
                mean_dist = sum(w1(P_K, a) for a in active) / len(active)
                step_rows.append({"trial": t, "run": r, "step": s,        # per-step trajectory (pre-update state)
                                  "repel": round(mean_dist.item(), 4),
                                  "dev": round(dev_of(leaf), 4)})
                loss = -mean_dist                           # minimize -> maximize distance -> repel
                opt.zero_grad(); loss.backward(); opt.step()
                if leash and lever == "slm":                # project back onto the on-manifold ball
                    with torch.no_grad():
                        disp = leaf - s0; d = disp.norm()
                        if d > leash * s0n:
                            leaf.copy_(s0 + disp * (leash * s0n / d))
                last = mean_dist.item()
            flush_steps()
            with torch.no_grad():                           # converged: its K-distogram is the new anchor
                _, P_conv = run(leaf, K, grad=False)
            anchors.append(P_conv.detach())
            snapshot(leaf, t, r, repel=last)                # 17-loop classify/save/decode (eval only)
            flush_csv()

    counts = {}
    for row in rows:
        counts[row["basin"]] = counts.get(row["basin"], 0) + 1
    print(f"\nbasin counts over {len(rows)} snapshots: {counts}", flush=True)
    print(f"wrote {out_csv}", flush=True)
    print("steer_repel OK.", flush=True)


if __name__ == "__main__":
    main()
