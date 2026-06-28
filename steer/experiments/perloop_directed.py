"""Per-loop distogram trace for the DIRECTED (steered) runs vs lottery references.

Question: the lottery G seeds snap to G late (~loop 12). The directed runs are
FORMED by a K=3 gradient acting early — does that make the directed G runs commit
EARLIER? We reproduce the proven recipe (lever=slm, nudge, fixed z0, M=5, K=3,
lr=0.3) trial-faithfully (same seeds + anchor accumulation), but score wg/wb1/wb2
at EVERY one of the 17 eval loops for every run. Plus references: each lottery-G
seed (GOOD) and the B1 baseline, scored the same per-loop way.

Output: perloop_directed.csv  (kind, id, basin_final, loop, wg, wb1, wb2)
  kind = directed | seedG | baseline

Run: CUDA_VISIBLE_DEVICES=N python perloop_directed.py [--trials 1 --runs 10]
"""
import sys
import csv
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from forward_grad import (load, compute_z_init, fixed_z0, single_rep_from_hidden, load_hidden,  # noqa
                          lift_to_lm_z, run_trunk, cross_distogram_from_z, _loop_body, w1,
                          BASELINE, REPS, GOOD, N_LOOPS)
from basins import load_centroids  # noqa: E402

OUT = Path(__file__).resolve().parent
LR, INIT_FRAC, REAL_G, M, K = 0.3, 0.05, 0.08, 5, 3


def flag(name, default, cast=str):
    for a in sys.argv[1:]:
        if a.startswith(name + "="):
            return cast(a.split("=", 1)[1])
        if a == name:
            return cast(sys.argv[sys.argv.index(a) + 1])
    return default


def main():
    T = flag("--trials", 1, int)
    R = flag("--runs", 10, int)
    print(f"config: T={T} R={R} M={M} K={K} lr={LR} (proven slm/nudge/fixed-z0)", flush=True)

    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]
    Pg, Pb1, Pb2 = load_centroids(model, pair_mask)
    z_init = compute_z_init(model, cond)
    z0_base = fixed_z0(model, z_init, seed=0)
    s0 = single_rep_from_hidden(model, load_hidden(BASELINE)).detach()
    s0n = s0.norm().item()
    lm_z_base = lift_to_lm_z(model, s0).detach()
    a, b = model._discretized_dynamics()
    a = a.view(1, 1, 1, -1).to(z_init.device, z_init.dtype)
    b_mat = b.to(z_init.device, z_init.dtype)

    def disto_grad(lm_z, z0, steps):                     # differentiable (for the repel loss)
        z, _ = run_trunk(model, z_init, lm_z, pair_mask, z0, steps)
        return cross_distogram_from_z(model, z, pair_mask)

    @torch.no_grad()
    def disto_fwd(lm_z, z0, steps):                      # forward-only final distogram (anchors)
        z = z0
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(steps):
                z = _loop_body(model, z, z_init, lm_z, pair_mask, a, b_mat)
        return cross_distogram_from_z(model, z, pair_mask)

    @torch.no_grad()
    def perloop(lm_z, z0):                               # score wg/wb1/wb2 at EVERY loop
        z, trace = z0, []
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for lp in range(N_LOOPS + 1):
                z = _loop_body(model, z, z_init, lm_z, pair_mask, a, b_mat)
                P = cross_distogram_from_z(model, z, pair_mask)
                trace.append((lp, w1(P, Pg).item(), w1(P, Pb1).item(), w1(P, Pb2).item()))
        return trace

    rows = []

    def emit(kind, ident, basin, trace):
        for lp, wg, wb1, wb2 in trace:
            rows.append(dict(kind=kind, id=ident, basin_final=basin, loop=lp,
                             wg=round(wg, 4), wb1=round(wb1, 4), wb2=round(wb2, 4)))

    def basin_of(trace):
        wg, wb1, wb2 = trace[-1][1], trace[-1][2], trace[-1][3]
        return min([("G", wg), ("B1", wb1), ("B2", wb2)], key=lambda x: x[1])[0]

    # --- references: B1 baseline + each lottery-G seed, scored per loop ---
    base_tr = perloop(lm_z_base, z0_base)
    emit("baseline", "B1base", basin_of(base_tr), base_tr)
    print(f"  baseline B1: final basin={basin_of(base_tr)}", flush=True)
    for S in GOOD:
        sG = single_rep_from_hidden(model, load_hidden(REPS / f"seed_{S}")).detach()
        tr = perloop(lift_to_lm_z(model, sG), z0_base)
        emit("seedG", f"seed{S}", basin_of(tr), tr)
        print(f"  lottery seed_{S}: final basin={basin_of(tr)}", flush=True)

    # --- directed runs: reproduce the proven trial faithfully, score per loop ---
    t0 = time.time()
    for t in range(T):
        print(f"\n===== directed trial {t} =====", flush=True)
        anchors = [disto_fwd(lm_z_base, z0_base, K).detach()]
        for r in range(1, R + 1):
            torch.manual_seed(10000 + 1000 * t + r)
            d = torch.randn_like(s0)
            leaf = (s0 + d / d.norm() * INIT_FRAC * s0n).clone().requires_grad_(True)
            opt = torch.optim.Adam([leaf], lr=LR)
            for s in range(M):
                with torch.enable_grad():
                    P = disto_grad(lift_to_lm_z(model, leaf), z0_base, K)
                    loss = -sum(w1(P, anc) for anc in anchors) / len(anchors)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                c_eval = lift_to_lm_z(model, leaf)
            anchors.append(disto_fwd(c_eval, z0_base, K).detach())
            tr = perloop(c_eval, z0_base)
            bas = basin_of(tr)
            emit("directed", f"t{t}r{r}", bas, tr)
            wgf = tr[-1][1]
            print(f"  run {r:2d}: basin={bas:>2} wg={wgf:.3f}"
                  f"{'  <== REAL G!' if wgf < REAL_G else ''}", flush=True)
            del opt, leaf
            torch.cuda.empty_cache()
            with open(OUT / "perloop_directed.csv", "w", newline="") as f:   # incremental flush
                w = csv.DictWriter(f, fieldnames=["kind", "id", "basin_final", "loop", "wg", "wb1", "wb2"])
                w.writeheader(); w.writerows(rows)

    print(f"\ndone {time.time()-t0:.0f}s. wrote {OUT/'perloop_directed.csv'} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
