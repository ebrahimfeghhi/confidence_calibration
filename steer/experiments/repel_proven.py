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
    decode = flag("--decode", False, bool)               # decode each run's OWN trajectory -> CIF -> DockQ inline
    n_samples = flag("--n-samples", 2, int)
    n_steps = flag("--n-steps", 200, int)
    plateau = flag("--plateau", False, bool)             # run grad steps until md (repulsion) plateaus, not fixed M
    max_steps = flag("--max-steps", 60, int)             # hard cap on steps/run when --plateau
    plateau_tol = flag("--plateau-tol", 0.005, float)    # min md gain/step to count as "still improving"
    plateau_patience = flag("--plateau-patience", 4, int)  # stop after this many non-improving steps
    gt_npz = flag("--gt-npz", "", str)                   # GT distogram npz -> score & stop on W1->GT (loss stays label-free)
    gt_thresh = flag("--gt-thresh", 0.7, float)          # W1->GT below this = reached GT basin
    until_gt = flag("--until-gt", False, bool)           # run each trial until GT reached (capped at --max-runs)
    max_runs = flag("--max-runs", R, int)                # run cap per trial when --until-gt
    no_score = flag("--no-score", False, bool)           # skip the K=17 diagnostic (P17/wg/basin/outcomes); leaves & DockQ unaffected
    fast_bwd = flag("--fast-bwd", False, bool)           # drop outer grad checkpoint in run_trunk (no recompute; needs VRAM)
    fused = flag("--fused", False, bool)                 # enable fused Triton trimul kernel (method-preserving speedup)
    chunk_size = flag("--chunk-size", 32, int)           # trunk chunk_size (<=0 -> None; faster at L<600)
    # --- campaign mode: accumulate anchors in ONE process, batch-decode every N runs,
    #     stop on a DockQ hit or a wall-clock budget (the "do 16, decode, keep going" loop) ---
    cycle_decode = flag("--cycle-decode", 0, int)        # decode+eval the last N runs' leaves every N runs (0=off)
    stop_dockq = flag("--stop-dockq", 0.0, float)        # stop campaign once any ab-ag DockQ >= this (0=off)
    time_budget = flag("--time-budget", 0, int)          # stop campaign after this many wall-clock seconds (0=off)
    launch_batch = flag("--launch-batch", 2, int)        # batched 0->17 launch chunk for cycle decode (per-leaf fallback on OOM)
    plateau_warn = flag("--plateau-warn", 0.01, float)   # flag a run whose repel loss still climbs > this at its LAST step (M too few)
    if no_score and (until_gt or pairwise):
        raise SystemExit("--no-score is incompatible with --until-gt / --pairwise (both need the K=17 score)")
    if multik and not bptt:
        bptt = 1                                          # multi-K only affordable with truncated backprop
    print(f"config: lever={lever} maskreps={maskreps} zvary={zvary} T={T} R={R} "
          f"M={M} K={K} bptt={bptt} multik={multik} lr={lr} loss={loss_mode} tau={tau} "
          f"init_frac={init_frac} decode={decode} plateau={plateau}(max{max_steps}) tag={tag}", flush=True)
    kcol = f"w1_k{K}"                                     # pairwise loss-horizon column name
    pair_fields = (["trial", "run", "prev"] + [f"w1_k{k}" for k in multik] + ["w1_k17"]) if multik \
        else ["trial", "run", "prev", kcol, "w1_k17"]

    cfg = fg.configure_target(target)                    # sets chain slices (AB,C) + rep paths for this complex
    gt_mode = target != "7txw"                           # no precomputed G/B1/B2 centroids for a new complex
    good = cfg["good"] or []
    if decode or cycle_decode:                           # ab-ag DockQ of decoded structures vs native (inline OR campaign)
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
        import json as _json, tempfile as _tempfile, subprocess as _sub, csv as _csv
        from eval_ost_foldbench import postprocess_cif, OST_BIN, OST_ENV   # noqa: E402
        from paths import GROUND_TRUTHS, TARGETS_CSV               # noqa: E402
        NATIVE = GROUND_TRUTHS / f"{target}-assembly1.cif"

        def _target_meta():     # (antigen label, interface chains) from FoldBench metadata
            antigen, chains = "C", set()
            try:
                with open(TARGETS_CSV) as _f:
                    for _row in _csv.DictReader(_f):
                        if _row.get("pdb_id", "").split("-")[0] == target:
                            chains.update(c for c in (_row.get("interface_chain_id_1"),
                                                      _row.get("interface_chain_id_2")) if c)
                            if _row.get("antigen_chain"):
                                antigen = _row["antigen_chain"].strip()
            except Exception:
                pass
            return antigen, sorted(chains)
        ANTIGEN, WANT_CHAINS = _target_meta()
        FORCE_MAP = [f"{c}:{c}" for c in WANT_CHAINS] or ["A:A", "B:B", "C:C"]

        def _ost(patched, outj, chain_map=None):
            cmd = [OST_BIN, "compare-structures", "-m", str(patched), "-r", str(NATIVE),
                   "-o", str(outj), "--fault-tolerant", "--min-pep-length", "4",
                   "--min-nuc-length", "4", "--lddt", "--dockq"]
            if chain_map:
                cmd += ["-c", *chain_map]
            try:
                r = _sub.run(cmd, capture_output=True, text=True, timeout=180, env=OST_ENV)
            except Exception:                            # TimeoutExpired / OSError -> caller may retry
                return False
            return r.returncode == 0 and outj.exists() and outj.stat().st_size > 0

        def _abag(outj):        # antibody-antigen DockQ from an OST json, or None if no interface
            d = _json.load(open(outj))
            ifaces = list(zip(d.get("dockq_interfaces", []), d.get("dockq", [])))
            # iface[:2] are the REFERENCE chains; strip a copy suffix so a duplicate-copy native
            # (antigen reported as e.g. "C-2") still matches the metadata antigen label.
            abag = [dq for iface, dq in ifaces if ANTIGEN in {c.split("-")[0] for c in iface[:2]}]
            pool = abag if abag else [dq for _, dq in ifaces]   # 2-chain VHH: single interface
            return max(pool) if pool else None

        def dockq_cif(cif):
            # DockQ of the antibody-antigen interface (antigen label from FoldBench metadata,
            # not a hardcoded "C"), NOT max over all interfaces (which would report the trivial
            # heavy-light interface on a Fab). Auto chain-mapping first; if it yields no usable
            # interface (auto-mapper crashed, dropped the antigen, or mapped a duplicate copy),
            # retry once with a forced identity map. Returns nan on any failure (never raises
            # into the decode loop).
            try:
                with _tempfile.TemporaryDirectory() as td:
                    patched, outj = Path(td) / "m.cif", Path(td) / "o.json"
                    if not postprocess_cif(Path(cif), patched):
                        return float("nan")
                    dq = _abag(outj) if _ost(patched, outj) else None
                    if dq is None:                       # unusable -> force identity map & re-parse
                        dq = _abag(outj) if _ost(patched, outj, FORCE_MAP) else None
                    return dq if dq is not None else float("nan")
            except Exception:
                return float("nan")
    model, builder, feat, chain_infos, cond = load(chunk_size=chunk_size,
                                                    kernel_backend="fused" if fused else None)
    pair_mask = cond["pair_mask"]
    Qg = maskg = None
    if gt_npz:
        import numpy as _npg
        _g = _npg.load(gt_npz)
        Qg = torch.from_numpy(_g["Q"].astype("float32")).cuda()
        maskg = torch.from_numpy(_g["mask"]).bool().cuda()

    def w1_gt(P):                                          # masked W1 of a cross-distogram to the GT (label used ONLY for scoring/stop)
        return (P.cumsum(-1) - Qg.cumsum(-1)).abs().sum(-1)[maskg].mean()
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
        z, _ = run_trunk(model, z_init, lm_z, pair_mask, z0, steps, use_ckpt=not fast_bwd)
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
    dockq_rows = []                                      # inline per-(run,sample) iptm + DockQ when --decode
    campaign_stop = None                                 # "hit" | "timeout" | None (campaign mode)
    plateau_rows = []                                    # per-run last-step repel gain + plateaued flag
    cycle_best = []                                       # (cycle, best DockQ) history
    # In campaign mode anchors accumulate for the whole 12h in ONE process; store them
    # fp16 to halve VRAM (distograms are 0..1 over 64 bins; w1's cumsum promotes to fp32).
    anc_cast = (lambda x: x.half()) if cycle_decode else (lambda x: x)

    def _cycle_decode(cyc, run_lo, run_hi, batch_leaves):
        """Batched 0->17 launch + diffusion decode of ONE cycle's leaves; append per-
        (run,sample) ab-ag DockQ + ipTM to dockq_{tag}.csv (cumulative). Returns best DockQ."""
        import numpy as _np
        ddir = OUT / f"decoded_{tag}"; ddir.mkdir(exist_ok=True)
        drows, tdec, n = [], time.time(), len(batch_leaves)
        for c0 in range(0, n, launch_batch):
            idx = list(range(c0, min(c0 + launch_batch, n)))
            lm_z_B = torch.cat([lift_to_lm_z(model, torch.from_numpy(batch_leaves[i].astype("float32")).cuda())
                                for i in idx], dim=0)
            try:
                z_B = fg.final_trunk_z_batched(model, z_init, lm_z_B, pair_mask, z0_base, N_LOOPS + 1)
                z_list = [z_B[bi:bi + 1] for bi in range(len(idx))]
            except Exception as e:               # batched launch OOM/unsupported -> proven per-leaf fallback
                print(f"       [cycle {cyc}] batched launch failed ({type(e).__name__}: {e}); per-leaf", flush=True)
                z_list = [fg.final_trunk_z(model, z_init, lm_z_B[bi:bi + 1], pair_mask, z0_base, N_LOOPS + 1)
                          for bi in range(len(idx))]
            for bi, i in enumerate(idx):
                run = run_lo + i
                samples = fg.decode_from_ztrunk(model, builder, feat, chain_infos, cond,
                                                z_list[bi], n_samples, n_steps, target, seed=run)
                for k, smp in enumerate(samples):
                    cif = ddir / f"run{run}_s{k}.cif"
                    cif.write_text(smp.complex.to_mmcif())
                    ip = float(smp.iptm) if smp.iptm is not None else (float(smp.ptm) if smp.ptm is not None else 0.0)
                    dq = dockq_cif(cif)
                    drows.append(dict(run=run, sample=k, iptm=round(ip, 4), dockq=round(dq, 4)))
            del lm_z_B
            torch.cuda.empty_cache()
        dockq_rows.extend(drows)
        with open(OUT / f"dockq_{tag}.csv", "w", newline="") as df:
            dw = csv.DictWriter(df, fieldnames=["run", "sample", "iptm", "dockq"])
            dw.writeheader(); dw.writerows(dockq_rows)
        _np.save(OUT / f"leaves_{tag}.npy", _np.stack(leaves))    # crash-safety: persist s_lm's seen so far
        best = max((d["dockq"] for d in drows if d["dockq"] == d["dockq"]), default=float("nan"))
        biptm = max((d["iptm"] for d in drows), default=float("nan"))
        print(f"  == cycle {cyc} decode (runs {run_lo}-{run_hi}, {len(drows)} samples, "
              f"{time.time()-tdec:.0f}s): best DockQ={best:.3f}  best ipTM={biptm:.3f} ==", flush=True)
        return best
    t0 = time.time()
    for t_local in range(T):
        t = toff + t_local
        print(f"\n===== trial {t} ({tag}) =====", flush=True)
        z0_seed0 = fixed_z0(model, z_init, seed=10000 + 1000 * t) if zvary else z0_base
        anchors = [anc_cast(disto_fwd(lm_z_base, z0_seed0, K).detach())]
        anchors17 = [disto_fwd(lm_z_base, z0_seed0, N_LOOPS + 1).detach()] if pairwise else None
        anchors_multi = {k: [disto_fwd(lm_z_base, z0_seed0, k).detach()] for k in multik} if multik else None
        g = torch.Generator(device="cpu").manual_seed(777 + t)
        perm = torch.randperm(len(pool), generator=g).tolist()
        for r in range(1, (max_runs if until_gt else R) + 1):
            t_run = time.time()
            torch.cuda.reset_peak_memory_stats()
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
            md_hist = []                                  # per-step repel value, for plateau-by-step-M detection
            s, stale, prev_md = 0, 0, -1e9
            steps_cap = max_steps if plateau else M       # plateau: run until md stops climbing (capped at max_steps)
            while s < steps_cap:
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
                md_hist.append(rep)
                stale = stale + 1 if (plateau and md.item() - prev_md < plateau_tol) else 0
                prev_md = md.item()
                s += 1
                if plateau and stale >= plateau_patience:
                    break
            # plateau-by-step-M watch: is the repel loss still climbing at the LAST step?
            last_gain = (md_hist[-1] - md_hist[-2]) if len(md_hist) >= 2 else 0.0
            plateaued = last_gain <= plateau_warn
            plateau_rows.append(dict(trial=t, run=r, steps=len(md_hist),
                                     final_repel=round(md_hist[-1], 5) if md_hist else 0.0,
                                     last_gain=round(last_gain, 5), plateaued=int(plateaued)))
            if not plateaued:
                print(f"       [plateau-WARN] run {r}: repel still climbing at step {len(md_hist)} "
                      f"(last_gain={last_gain:+.4f} > {plateau_warn}); M={M} may be too few", flush=True)
            steer_secs = round(time.time() - t_run, 1)               # steering-only wall-clock
            steer_peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            with torch.no_grad():
                c_eval = cond_of(leaf)
            anchors.append(anc_cast(disto_fwd(c_eval, z0, K).detach()))
            if multik:
                for k in multik:
                    anchors_multi[k].append(disto_fwd(c_eval, z0, k).detach())
            if no_score:                                      # --no-score fast path: skip the K=17 diagnostic (leaves unaffected)
                wg, wb1, wb2, basin, rg = float("nan"), 0.0, 0.0, "?", False
            else:
                P17 = disto_fwd(c_eval, z0, N_LOOPS + 1)
                if pairwise:                                  # W1 of this run to every prior run, per loss-horizon + K17
                    for j in range(len(anchors17)):          # j=0 baseline B1, j>=1 prior runs
                        d = {"trial": t, "run": r, "prev": j,
                             "w1_k17": round(w1(P17, anchors17[j]).item(), 5)}
                        if multik:
                            for k in multik:
                                d[f"w1_k{k}"] = round(w1(anchors_multi[k][-1], anchors_multi[k][j]).item(), 5)
                        else:
                            d[kcol] = round(w1(anchors[-1], anchors[j]).item(), 5)
                        pair_rows.append(d)
                    anchors17.append(P17.detach())
                    _dk = multik[-1] if multik else K        # representative loss horizon for console summary
                    _d3 = [p[f"w1_k{_dk}"] for p in pair_rows if p["trial"] == t and p["run"] == r]
                    _d17 = [p["w1_k17"] for p in pair_rows if p["trial"] == t and p["run"] == r]
                    print(f"       vs {len(_d3)} prior: K{_dk} min={min(_d3):.3f} mean={sum(_d3)/len(_d3):.3f}"
                          f" | K17 min={min(_d17):.3f} mean={sum(_d17)/len(_d17):.3f}", flush=True)
                if gt_mode:
                    if Qg is not None:                       # score W1->GT (label-free loss, GT only for scoring/stop)
                        wg = w1_gt(P17).item()
                        wb1, wb2 = w1(P17, ref0).item(), 0.0  # wb1 = escape dist (still logged for diagnostics)
                        rg = wg < gt_thresh
                        basin = "G" if rg else "esc"
                    else:
                        wg = w1(P17, ref0).item()            # ESCAPE from baseline basin (higher = moved further off)
                        wb1, wb2, basin, rg = 0.0, 0.0, "esc", False
                else:
                    wg, wb1, wb2 = w1(P17, Pg).item(), w1(P17, Pb1).item(), w1(P17, Pb2).item()
                    basin = min([("G", wg), ("B1", wb1), ("B2", wb2)], key=lambda x: x[1])[0]
                    rg = wg < REAL_G
                realg += rg; nearg += (basin == "G")
                outcomes.append(P17.detach().to(torch.float16).cpu().numpy())        # K17 cross-distogram
            if lever != "affine":
                leaves.append(leaf.detach().to(torch.float16).cpu().numpy())          # optimized s_lm -> decode_batch.py
            if decode and (rg or not until_gt):           # decode (GT hits only when --until-gt) -> CIF -> DockQ
                z_trunk = fg.final_trunk_z(model, z_init, c_eval, pair_mask, z0, N_LOOPS + 1)
                samples = fg.decode_from_ztrunk(model, builder, feat, chain_infos, cond,
                                                z_trunk, n_samples, n_steps, target, seed=r)
                ddir = OUT / f"decoded_{tag}"; ddir.mkdir(exist_ok=True)
                rdq = []
                for k, s in enumerate(samples):
                    cif = ddir / f"run{r}_s{k}.cif"
                    cif.write_text(s.complex.to_mmcif())
                    ip = float(s.iptm) if s.iptm is not None else (float(s.ptm) if s.ptm is not None else 0.0)
                    dq = dockq_cif(cif)
                    dockq_rows.append(dict(run=r, sample=k, iptm=round(ip, 4), dockq=round(dq, 4)))
                    rdq.append((ip, dq))
                bestdq = max((d for _, d in rdq if d == d), default=float("nan"))
                print(f"       decode: iptm={[round(ip,3) for ip,_ in rdq]} "
                      f"dockq={[round(d,3) for _,d in rdq]} best={bestdq:.3f}", flush=True)
                with open(OUT / f"dockq_{tag}.csv", "w", newline="") as df:
                    dw = csv.DictWriter(df, fieldnames=["run", "sample", "iptm", "dockq"])
                    dw.writeheader(); dw.writerows(dockq_rows)
            dt = round(time.time() - t_run, 1)
            rows.append(dict(trial=t, run=r, start=start, basin=basin, wg=round(wg, 4),
                             wb1=round(wb1, 4), wb2=round(wb2, 4), repel=round(rep, 4), secs=dt,
                             steer_secs=steer_secs, peak_gb=steer_peak_gb))
            print(f"  run {r:2d} [{start:>7}]: basin={basin:>2} wg={wg:.3f}  ({dt}s)"
                  f"{'  <== REAL G!' if rg else ('  <-- nearG' if basin == 'G' else '')}", flush=True)
            with open(OUT / f"repel_{tag}_steps.csv", "w", newline="") as sf:   # incremental flush
                sw = csv.DictWriter(sf, fieldnames=["trial", "run", "step", "repel", "drift"])
                sw.writeheader(); sw.writerows(step_rows)
            with open(OUT / f"repel_{tag}.csv", "w", newline="") as rf:          # per-run results, flushed
                rw = csv.DictWriter(rf, fieldnames=["trial", "run", "start", "basin", "wg", "wb1", "wb2", "repel", "secs", "steer_secs", "peak_gb"])
                rw.writeheader(); rw.writerows(rows)
            if pairwise:
                with open(OUT / f"pairwise_{tag}.csv", "w", newline="") as pf:    # full run-vs-prior matrix
                    pw = csv.DictWriter(pf, fieldnames=pair_fields)
                    pw.writeheader(); pw.writerows(pair_rows)
            with open(OUT / f"plateau_{tag}.csv", "w", newline="") as plf:         # per-run step-M plateau watch
                plw = csv.DictWriter(plf, fieldnames=["trial", "run", "steps", "final_repel", "last_gain", "plateaued"])
                plw.writeheader(); plw.writerows(plateau_rows)
            del opt, leaf                                # P17 only exists when not no_score
            if not no_score:
                del P17
            torch.cuda.empty_cache()
            if until_gt and rg:
                print(f"  >> trial {t} REACHED GT at run {r} (W1->GT={wg:.3f}) — ending trial", flush=True)
                break
            # --- campaign: batch-decode this cycle's leaves, stop on a DockQ hit or the time budget ---
            if cycle_decode and r % cycle_decode == 0:
                cyc = r // cycle_decode
                best = _cycle_decode(cyc, r - cycle_decode + 1, r, leaves[-cycle_decode:])
                cycle_best.append((cyc, best))
                if stop_dockq and best == best and best >= stop_dockq:
                    campaign_stop = "hit"
                    print(f"  >> CAMPAIGN HIT [{tag}]: cycle {cyc} best DockQ {best:.3f} >= "
                          f"{stop_dockq} after {time.time()-t0:.0f}s — stopping.", flush=True)
            if time_budget and (time.time() - t0) >= time_budget and not campaign_stop:
                campaign_stop = "timeout"
                print(f"  >> CAMPAIGN TIMEOUT [{tag}]: {time.time()-t0:.0f}s >= {time_budget}s "
                      f"({r} runs done) — stopping.", flush=True)
            if campaign_stop:
                break
        if campaign_stop:
            break

    with open(OUT / f"repel_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trial", "run", "start", "basin", "wg", "wb1", "wb2", "repel", "secs", "steer_secs", "peak_gb"])
        w.writeheader(); w.writerows(rows)
    with open(OUT / f"repel_{tag}_steps.csv", "w", newline="") as f:        # per-step loss trajectory
        w = csv.DictWriter(f, fieldnames=["trial", "run", "step", "repel", "drift"])
        w.writeheader(); w.writerows(step_rows)
    if pairwise:
        with open(OUT / f"pairwise_{tag}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=pair_fields)
            w.writeheader(); w.writerows(pair_rows)
    if outcomes:
        import numpy as _np
        _np.save(OUT / f"outcomes_{tag}.npy", _np.stack(outcomes))
        if leaves:
            _np.save(OUT / f"leaves_{tag}.npy", _np.stack(leaves))    # [n_runs,1,L,256] optimized s_lm
        print(f"saved {len(outcomes)} outcomes + {len(leaves)} leaves -> outcomes/leaves_{tag}.npy", flush=True)
    if cycle_decode:                                     # campaign summary (DockQ-driven, not wg)
        overall = max((d["dockq"] for d in dockq_rows if d["dockq"] == d["dockq"]), default=float("nan"))
        n_notplat = sum(1 for p in plateau_rows if not p["plateaued"])
        print(f"\n[{tag}] campaign done {time.time()-t0:.0f}s, stop={campaign_stop or 'runs-exhausted'}. "
              f"{len(rows)} runs, {len(cycle_best)} cycles, best ab-ag DockQ={overall:.3f}. "
              f"plateau: {n_notplat}/{len(plateau_rows)} runs still climbing at step M "
              f"(see plateau_{tag}.csv).", flush=True)
        return
    _thr = gt_thresh if Qg is not None else REAL_G
    trials_hit = len({r["trial"] for r in rows if r["wg"] < _thr})
    print(f"\n[{tag}] done {time.time()-t0:.0f}s. REAL G={realg}, nearest-G={nearg}, "
          f"trials with >=1 real-G: {trials_hit}/{T}, best wg={min(r['wg'] for r in rows):.3f}",
          flush=True)


if __name__ == "__main__":
    main()
