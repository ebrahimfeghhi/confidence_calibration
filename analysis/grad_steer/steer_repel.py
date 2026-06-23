"""Label-free sequential repulsion — a single-leaf 'novelty walk', for z0 OR s_lm.

The honest test of a steering lever: with NO G target, can pushing the leaf away from
its own visited distograms walk it out of B1 into the good basin, or does it flee
off-manifold into unfoldable distograms?

Per seed (one fixed leaf, carried throughout — NEVER resampled within a walk):
  anchors = [ 17-loop distogram of the initial leaf ]            # the B1 start
  repeat ITERS times:
    M gradient steps maximizing mean W1(P_K(leaf), anchor)        # repel, read at horizon K
    z17 = 17-loop distogram of the updated leaf
    append z17 to anchors; save z17; classify (nearest centroid) + dev

Anchors are the 17-LOOP distograms (the actual basin reps, detached constants); the
loss/gradient is read at horizon K (kept small so z0's gradient is alive — it decays ~a^K;
s_lm is re-injected every loop so K is less critical).

Levers:
  --lever z0  : leaf = the initial pair state (fixed_z0(seed)); s_lm pinned to baseline.
                dev = frac of |z0| beyond +-3*std (off the trunc-normal manifold).
  --lever slm : leaf = the SAME baseline s_lm for every seed (no nudge); the per-seed
                diversity comes from a different z0 draw (fixed_z0(seed)) used as the fixed
                trunk context, plus CUDA nondeterminism. dev = ||s - s0|| / ||s0||.

Run (esm venv, GPU):
  CUDA_VISIBLE_DEVICES=4 ... steer_repel.py --lever z0  --M 5  --seeds 0,..,9
  CUDA_VISIBLE_DEVICES=7 ... steer_repel.py --lever slm --M 5  --seeds 0,..,9
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

ZOUT = ESM_PREDICTIONS / "7txw_z0_repel"
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
    lever = _flag("--lever", "z0")             # z0 | slm
    M = _flag("--M", 5, int)                    # gradient steps per run
    R = _flag("--runs", 10, int)                # number of runs (anchors) in the chain
    lr = _flag("--lr", 0.1, float)
    K = _flag("--K", 3, int)                    # horizon for BOTH the live loss and the anchors
    assert lever in ("z0", "slm")
    print(f"loading model... lever={lever} M={M} runs={R} lr={lr} K={K}", flush=True)
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]
    z_init = compute_z_init(model, cond)
    std = math.sqrt(2.0 / (5.0 * z_init.shape[-1]))

    s0 = single_rep_from_hidden(model, load_hidden(BASELINE)).detach()    # baseline s_lm
    s0n = s0.norm().item()
    lm_z_base = lift_to_lm_z(model, s0).detach()                          # fixed for z0 mode
    z0_base = fixed_z0(model, z_init, seed=0)                             # fixed for slm mode

    with torch.no_grad():
        Pg, Pb1, Pb2 = (centroid(model, pair_mask, s) for s in (GOOD, B1, B2))

    slm_z0 = [z0_base]      # per-seed z0 context for slm mode (set in the seed loop)

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
        return (leaf.detach() - s0).norm().item() / s0n                      # rel displacement

    def init_leaf(seed):
        if lever == "z0":
            return fixed_z0(model, z_init, seed=seed).clone().requires_grad_(True)
        # s_lm: z0 context stays FIXED at z0_base (never reassigned) so the repulsion W1
        # reflects pure s_lm-basin differences, not z0-context noise.
        return s0.clone().requires_grad_(True)             # same frozen baseline s_lm every run

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{lever}_M{M}_K{K}"
    out_csv = CSV_DIR / f"steer_repel_{tag}.csv"
    fields = ["run", "basin", "wg", "wb1", "wb2", "dev", "repel"]
    rows = []

    def snapshot(leaf, run_idx, repel):
        """17-loop eval of the current copy: distogram, basin, dev. Save z for decode. (Eval only.)"""
        z17, P17 = run(leaf, N_LOOPS + 1, grad=False)
        wg, wb1, wb2 = (w1(P17, c).item() for c in (Pg, Pb1, Pb2))
        basin = min((("G", wg), ("B1", wb1), ("B2", wb2)), key=lambda kv: kv[1])[0]
        dev = dev_of(leaf)
        rd = ZOUT / tag / f"run{run_idx:02d}" / "seed_0" / "reps"
        rd.mkdir(parents=True, exist_ok=True)
        np.save(rd / "pairformer_z_loop16.npy", z17.float().cpu().numpy().astype(np.float16))
        rows.append({"run": run_idx, "basin": basin,
                     "wg": round(wg, 4), "wb1": round(wb1, 4), "wb2": round(wb2, 4),
                     "dev": round(dev, 4), "repel": round(repel, 4)})
        flag = "  <== G!" if basin == "G" else ""
        print(f"  run {run_idx:2d}: basin={basin:>2} W1->G={wg:.3f} "
              f"dev={dev:.3f} repel={repel:.3f}{flag}", flush=True)
        return P17.detach()

    # ---- single anchor-accumulating chain of R runs; RESET the copy each run ----
    # Step 0: seed the anchor set with the first sample's K-distogram (B1).
    seed_leaf = init_leaf(0)
    with torch.no_grad():
        _, P0 = run(seed_leaf, K, grad=False)               # K-horizon distogram (the anchor type)
    anchors = [P0.detach()]
    snapshot(seed_leaf, 0, repel=0.0)                       # record/save/classify the B1 start (17-loop)

    for r in range(1, R + 1):
        print(f"\n----- {lever} run {r}/{R} (repelling {len(anchors)} anchors) -----", flush=True)
        leaf = init_leaf(r)                                 # RESET: fresh z0 draw / baseline copy
        opt = torch.optim.Adam([leaf], lr=lr)
        last = 0.0
        for _ in range(M):
            _, P_K = run(leaf, K, grad=True)                # live distogram at K (=3)
            mean_dist = sum(w1(P_K, a) for a in anchors) / len(anchors)
            loss = -mean_dist                               # minimize -> maximize distance -> repel
            opt.zero_grad(); loss.backward(); opt.step()
            last = mean_dist.item()
        with torch.no_grad():                               # converged: its K-distogram is the new anchor
            _, P_conv = run(leaf, K, grad=False)
        anchors.append(P_conv.detach())
        snapshot(leaf, r, repel=last)                       # 17-loop classify/save/decode (eval only)
        with open(out_csv, "w", newline="") as f:           # flush incrementally
            wcsv = csv.DictWriter(f, fieldnames=fields); wcsv.writeheader(); wcsv.writerows(rows)

    print(f"\nwrote {out_csv}  ({len(rows)} rows)", flush=True)
    print("steer_repel OK.", flush=True)


if __name__ == "__main__":
    main()
