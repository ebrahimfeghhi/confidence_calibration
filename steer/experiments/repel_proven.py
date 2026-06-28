"""Configurable sequential greedy self-repulsion (the proven label-free recipe),
for sweeping diversity sources to find what hits real-G consistently within R runs.

Per trial: growing anchor list seeded with baseline B1. Each run: pick a leaf
START (nudge | masked-rep), z0 CONTEXT (fixed | per-run), take M grad steps
maximizing W1 from all anchors (horizon K), append result. Anchors self-generated
(label-free); G/B1/B2 centroids score only. Forward-only eval for speed.

Flags: --maskreps (start each run from a saved masked ESMC rep, excl. known-G unless
--include-g), --zvary (resample z0 per run), --trials T --runs R --tag NAME.

Run: CUDA_VISIBLE_DEVICES=N python repel_proven.py [flags]
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
import forward_grad as fg  # noqa: E402  — for configure_target(name) on a new complex

OUT = Path(__file__).resolve().parent
LR, INIT_FRAC, REAL_G = 0.3, 0.05, 0.08

import torch.nn as nn  # noqa: E402


class ConforNet(nn.Module):
    """Channel-wise affine on the pair conditioning lm_z (init = identity / 0)."""
    def __init__(self, dim=256):
        super().__init__()
        self.W = nn.Parameter(torch.eye(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, lm_z):
        return torch.matmul(lm_z, self.W.t()) + self.b


def flag(name, default, cast=str):
    for a in sys.argv[1:]:
        if a == name and cast == bool:
            return True
        if a.startswith(name + "="):
            return cast(a.split("=", 1)[1])
        if a == name and cast != bool:
            return cast(sys.argv[sys.argv.index(a) + 1])
    return default


def main():
    maskreps = flag("--maskreps", False, bool)
    zvary = flag("--zvary", False, bool)
    include_g = flag("--include-g", False, bool)
    T = flag("--trials", 3, int)
    toff = flag("--trial-offset", 0, int)            # global trial index = toff + local; for multi-GPU splits
    R = flag("--runs", 10, int)
    M = flag("--M", 5, int)
    K = flag("--K", 3, int)
    init_frac = flag("--init-frac", INIT_FRAC, float)
    lever = flag("--lever", "slm", str)                  # slm | affine
    lr = flag("--lr", LR, float)
    loss_mode = flag("--loss", "mean", str)              # mean | softmin (over anchors)
    tau = flag("--tau", 0.1, float)                       # soft-min temperature (-> hard min as tau->0)
    tag = flag("--tag", "nudge_fz0", str)
    target = flag("--target", "7txw", str)               # 7txw | 8im0 | ... (chain slices + rep paths)
    pairwise = flag("--pairwise", False, bool)           # log W1 to every prior run at K=3 & K=17
    bptt = flag("--bptt", 0, int)                         # truncated backprop: grad through last N loops only (0=full)
    multik = [int(x) for x in flag("--multik", "", str).split(",") if x]   # loss at multiple horizons, e.g. 3,6,9
    if multik and not bptt:
        bptt = 1                                          # multi-K only affordable with truncated backprop
    print(f"config: lever={lever} maskreps={maskreps} zvary={zvary} T={T} R={R} "
          f"M={M} K={K} bptt={bptt} multik={multik} lr={lr} loss={loss_mode} tau={tau} "
          f"init_frac={init_frac} tag={tag}", flush=True)
    kcol = f"w1_k{K}"                                     # pairwise loss-horizon column name
    pair_fields = (["trial", "run", "prev"] + [f"w1_k{k}" for k in multik] + ["w1_k17"]) if multik \
        else ["trial", "run", "prev", kcol, "w1_k17"]

    cfg = fg.configure_target(target)                    # sets chain slices (AB,C) + rep paths for this complex
    gt_mode = target != "7txw"                           # no precomputed G/B1/B2 centroids for a new complex
    good = cfg["good"] or []
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]
    Pg, Pb1, Pb2 = (None, None, None) if gt_mode else load_centroids(model, pair_mask)
    z_init = compute_z_init(model, cond)
    z0_base = fixed_z0(model, z_init, seed=0)
    s0 = single_rep_from_hidden(model, load_hidden(cfg["baseline"])).detach()
    s0n = s0.norm().item()
    lm_z_base = lift_to_lm_z(model, s0).detach()         # baseline pair conditioning (affine lever)
    eye = torch.eye(s0.shape[-1], device=s0.device)
    a, b = model._discretized_dynamics()
    a = a.view(1, 1, 1, -1).to(z_init.device, z_init.dtype)
    b_mat = b.to(z_init.device, z_init.dtype)
    pool = [s for s in range(60) if include_g or s not in good]

    def cond_of(leaf):                                    # leaf -> pair conditioning lm_z (grad-capable)
        return leaf(lm_z_base) if lever == "affine" else lift_to_lm_z(model, leaf)

    def disto_grad(lm_z, z0, steps):                      # differentiable (checkpointed) for the loss
        z, _ = run_trunk(model, z_init, lm_z, pair_mask, z0, steps)
        return cross_distogram_from_z(model, z, pair_mask)

    @torch.no_grad()
    def disto_fwd(lm_z, z0, steps):                       # forward-only (fast) for anchors/eval
        z = z0
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(steps):
                z = _loop_body(model, z, z_init, lm_z, pair_mask, a, b_mat)
        return cross_distogram_from_z(model, z, pair_mask)

    def disto_grad_trunc(lm_z, z0, steps, tb):            # K-loop forward value, grad only through last tb loops (ESMFold recycle regime)
        z = z0
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(steps - tb):
                z = _loop_body(model, z, z_init, lm_z, pair_mask, a, b_mat)
        z = z.detach()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(tb):
                z = _loop_body(model, z, z_init, lm_z, pair_mask, a, b_mat)
        return cross_distogram_from_z(model, z, pair_mask)

    rows, step_rows, pair_rows, realg, nearg = [], [], [], 0, 0
    ref0 = disto_fwd(lm_z_base, z0_base, N_LOOPS + 1).detach() if gt_mode else None  # baseline (confident-wrong) basin
    outcomes = []                                        # K17 cross-distograms, saved for post-hoc ground-truth scoring
    leaves = []                                          # optimized s_lm per run, for decode -> structure -> DockQ
    t0 = time.time()
    for t_local in range(T):
        t = toff + t_local
        print(f"\n===== trial {t} ({tag}) =====", flush=True)
        z0_seed0 = fixed_z0(model, z_init, seed=10000 + 1000 * t) if zvary else z0_base
        anchors = [disto_fwd(lm_z_base, z0_seed0, K).detach()]
        anchors17 = [disto_fwd(lm_z_base, z0_seed0, N_LOOPS + 1).detach()] if pairwise else None
        anchors_multi = {k: [disto_fwd(lm_z_base, z0_seed0, k).detach()] for k in multik} if multik else None
        g = torch.Generator(device="cpu").manual_seed(777 + t)
        perm = torch.randperm(len(pool), generator=g).tolist()
        for r in range(1, R + 1):
            t_run = time.time()
            z0 = fixed_z0(model, z_init, seed=10000 + 1000 * t + r) if zvary else z0_base
            if lever == "affine":
                torch.manual_seed(10000 + 1000 * t + r)
                net = ConforNet(s0.shape[-1]).cuda()
                with torch.no_grad():
                    net.W.add_(torch.randn_like(net.W) * 0.01)   # symmetry break
                    net.b.add_(torch.randn_like(net.b) * 0.01)
                leaf = net
                opt = torch.optim.Adam(net.parameters(), lr=lr)
                params = list(net.parameters())
                start = "affine"
            elif maskreps:
                S = pool[perm[(r - 1) % len(perm)]]
                leaf = single_rep_from_hidden(model, load_hidden(REPS / f"seed_{S}")).detach().clone().requires_grad_(True)
                opt = torch.optim.Adam([leaf], lr=lr)
                params = [leaf]
                start = f"mask{S}"
            else:
                torch.manual_seed(10000 + 1000 * t + r)
                d = torch.randn_like(s0)
                leaf = (s0 + d / d.norm() * init_frac * s0n).clone().requires_grad_(True)
                opt = torch.optim.Adam([leaf], lr=lr)
                params = [leaf]
                start = "nudge"
            rep = 0.0
            for s in range(M):
                with torch.enable_grad():
                    if multik:                                                # mean over horizons of (mean W1 to that horizon's anchors)
                        lm_z_g = cond_of(leaf)
                        per_k = []
                        for k in multik:
                            Pk = disto_grad_trunc(lm_z_g, z0, k, bptt)
                            per_k.append(torch.stack([w1(Pk, anc) for anc in anchors_multi[k]]).mean())
                        md = torch.stack(per_k).mean()
                    else:
                        P = (disto_grad_trunc(cond_of(leaf), z0, K, bptt) if bptt
                             else disto_grad(cond_of(leaf), z0, K))
                        dists = torch.stack([w1(P, anc) for anc in anchors])      # [n_anchors]
                        if loss_mode == "softmin":
                            # soft-min distance to the NEAREST anchor; recomputed each step,
                            # so the active (nearest) anchor is free to change as the leaf moves.
                            md = -tau * torch.logsumexp(-dists / tau, dim=0)
                        else:
                            md = dists.mean()                                 # mean over anchors (original)
                    loss = -md
                if lever == "affine":
                    drift = ((net.W.detach() - eye).norm() + net.b.detach().norm()).item()
                else:
                    drift = ((leaf.detach() - s0).norm() / s0n).item()  # off-manifold drift so far
                step_rows.append(dict(trial=t, run=r, step=s, repel=round(md.item(), 5),
                                      drift=round(drift, 5)))
                opt.zero_grad(); loss.backward()
                if lever == "affine":
                    torch.nn.utils.clip_grad_norm_(params, 10.0)
                opt.step()
                rep = md.item()
            with torch.no_grad():
                c_eval = cond_of(leaf)
            anchors.append(disto_fwd(c_eval, z0, K).detach())
            if multik:
                for k in multik:
                    anchors_multi[k].append(disto_fwd(c_eval, z0, k).detach())
            P17 = disto_fwd(c_eval, z0, N_LOOPS + 1)
            if pairwise:                                      # W1 of this run to every prior run, per loss-horizon + K17
                for j in range(len(anchors17)):              # j=0 baseline B1, j>=1 prior runs
                    d = {"trial": t, "run": r, "prev": j,
                         "w1_k17": round(w1(P17, anchors17[j]).item(), 5)}
                    if multik:
                        for k in multik:
                            d[f"w1_k{k}"] = round(w1(anchors_multi[k][-1], anchors_multi[k][j]).item(), 5)
                    else:
                        d[kcol] = round(w1(anchors[-1], anchors[j]).item(), 5)
                    pair_rows.append(d)
                anchors17.append(P17.detach())
                _dk = multik[-1] if multik else K            # representative loss horizon for console summary
                _d3 = [p[f"w1_k{_dk}"] for p in pair_rows if p["trial"] == t and p["run"] == r]
                _d17 = [p["w1_k17"] for p in pair_rows if p["trial"] == t and p["run"] == r]
                print(f"       vs {len(_d3)} prior: K{_dk} min={min(_d3):.3f} mean={sum(_d3)/len(_d3):.3f}"
                      f" | K17 min={min(_d17):.3f} mean={sum(_d17)/len(_d17):.3f}", flush=True)
            if gt_mode:
                wg = w1(P17, ref0).item()                # ESCAPE from baseline basin (higher = moved further off)
                wb1, wb2, basin, rg = 0.0, 0.0, "esc", False
                outcomes.append(P17.detach().to(torch.float16).cpu().numpy())
                if lever != "affine":
                    leaves.append(leaf.detach().to(torch.float16).cpu().numpy())  # optimized s_lm
            else:
                wg, wb1, wb2 = w1(P17, Pg).item(), w1(P17, Pb1).item(), w1(P17, Pb2).item()
                basin = min([("G", wg), ("B1", wb1), ("B2", wb2)], key=lambda x: x[1])[0]
                rg = wg < REAL_G
            realg += rg; nearg += (basin == "G")
            dt = round(time.time() - t_run, 1)
            rows.append(dict(trial=t, run=r, start=start, basin=basin, wg=round(wg, 4),
                             wb1=round(wb1, 4), wb2=round(wb2, 4), repel=round(rep, 4), secs=dt))
            print(f"  run {r:2d} [{start:>7}]: basin={basin:>2} wg={wg:.3f}  ({dt}s)"
                  f"{'  <== REAL G!' if rg else ('  <-- nearG' if basin == 'G' else '')}", flush=True)
            with open(OUT / f"repel_{tag}_steps.csv", "w", newline="") as sf:   # incremental flush
                sw = csv.DictWriter(sf, fieldnames=["trial", "run", "step", "repel", "drift"])
                sw.writeheader(); sw.writerows(step_rows)
            with open(OUT / f"repel_{tag}.csv", "w", newline="") as rf:          # per-run results, flushed
                rw = csv.DictWriter(rf, fieldnames=["trial", "run", "start", "basin", "wg", "wb1", "wb2", "repel", "secs"])
                rw.writeheader(); rw.writerows(rows)
            if pairwise:
                with open(OUT / f"pairwise_{tag}.csv", "w", newline="") as pf:    # full run-vs-prior matrix
                    pw = csv.DictWriter(pf, fieldnames=pair_fields)
                    pw.writeheader(); pw.writerows(pair_rows)
            del opt, leaf, P17
            torch.cuda.empty_cache()

    with open(OUT / f"repel_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trial", "run", "start", "basin", "wg", "wb1", "wb2", "repel", "secs"])
        w.writeheader(); w.writerows(rows)
    with open(OUT / f"repel_{tag}_steps.csv", "w", newline="") as f:        # per-step loss trajectory
        w = csv.DictWriter(f, fieldnames=["trial", "run", "step", "repel", "drift"])
        w.writeheader(); w.writerows(step_rows)
    if pairwise:
        with open(OUT / f"pairwise_{tag}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=pair_fields)
            w.writeheader(); w.writerows(pair_rows)
    if gt_mode and outcomes:
        import numpy as _np
        _np.save(OUT / f"outcomes_{tag}.npy", _np.stack(outcomes))
        if leaves:
            _np.save(OUT / f"leaves_{tag}.npy", _np.stack(leaves))    # [n_runs,1,L,256] optimized s_lm
        print(f"saved {len(outcomes)} outcomes + {len(leaves)} leaves -> outcomes/leaves_{tag}.npy", flush=True)
    trials_hit = len({r["trial"] for r in rows if r["wg"] < REAL_G})
    print(f"\n[{tag}] done {time.time()-t0:.0f}s. REAL G={realg}, nearest-G={nearg}, "
          f"trials with >=1 real-G: {trials_hit}/{T}, best wg={min(r['wg'] for r in rows):.3f}",
          flush=True)


if __name__ == "__main__":
    main()
