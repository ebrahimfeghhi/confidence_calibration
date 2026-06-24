import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forward_grad import (  # noqa: E402
    load, compute_z_init, fixed_z0, single_rep_from_hidden, load_hidden,
    distogram_from_s_lm, run_trunk, lift_to_lm_z, cross_distogram_from_z,
    w1, BASELINE, GOOD, N_LOOPS, REPS
)

B1 = [1, 2, 3, 4, 5, 7, 8, 9, 11, 16, 17, 18, 19, 20, 21, 22, 23, 26, 28, 29, 30, 31, 32,
      34, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49, 52, 55, 56, 57, 58, 59]
B2 = [0, 6, 12, 13, 15, 24, 27, 35, 43, 47, 50, 53, 54]
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import ESM_PREDICTIONS  # noqa: E402

OUT = ESM_PREDICTIONS / "7txw_steer_sequential" / "7txw"

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

    K = 3
    LR = 0.3
    INIT_FRAC = 0.05

    # CLI flags: --steps=N --runs=N --tag=NAME --use-shell
    args = sys.argv[1:]
    def _flagval(name, default):
        for a in args:
            if a.startswith(name + "="):
                return type(default)(a.split("=", 1)[1])
        return default
    STEPS = _flagval("--steps", 15)
    N_RUNS = _flagval("--runs", 10)
    tag = _flagval("--tag", "")
    out_dir = (OUT.parent / f"7txw_{tag}" if tag else OUT)

    # Baseline distogram AT K=3 (same horizon as the live P_K and the converged anchors).
    # The loop-16 `Pbase` above is kept only for the full_dist report.
    with torch.no_grad():
        Pbase_K3, _ = distogram_from_s_lm(model, s_lm0, z_init, pair_mask, z0, total_steps=K)
        Pbase_K3 = Pbase_K3.detach()

    # Optional: re-add escape shell if requested later, but for now we do pure repel.
    use_shell = "--use-shell" in args
    SHELL_RADIUS = 0.48
    SHELL_WEIGHT = 1.0  # if used

    print(f"\nsequential repel: K={K}, lr={LR}, steps={STEPS}, runs={N_RUNS}", flush=True)
    if use_shell:
        print(f"  + escape shell enabled (target W1={SHELL_RADIUS} from baseline, weight={SHELL_WEIGHT})", flush=True)
    else:
        print("  pure repel mode (no radius pin)", flush=True)

    anchors = [Pbase_K3]   # seed: baseline (B1 floor) at K=3 — repelling from it = escaping B1
    counts = {"G": 0, "B1": 0, "B2": 0, "other": 0}

    for run in range(N_RUNS):
        print(f"\n--- Run {run+1}/{N_RUNS} ---", flush=True)
        torch.manual_seed(2000 + run)
        
        # Fresh init per run
        d = torch.randn_like(s_lm0)
        d = d / d.norm() * (INIT_FRAC * s0n)
        s = (s_lm0 + d).requires_grad_(True)
        
        opt = torch.optim.Adam([s], lr=LR)
        
        print(f"steering {STEPS} steps, repelling {len(anchors)} anchors...", flush=True)
        for t in range(STEPS):
            P_K, _ = distogram_from_s_lm(model, s, z_init, pair_mask, z0, total_steps=K)
            
            # Repel from all accumulated anchors (mean of distances)
            dists = [w1(P_K, a) for a in anchors]
            mean_dist = sum(dists) / len(dists)
            
            # Maximizing distance means minimizing -distance
            loss = -mean_dist
            
            if use_shell:
                # Add the shell loss relative to the first anchor (baseline)
                dist_baseline = dists[0]
                shell_loss = (dist_baseline - SHELL_RADIUS) ** 2
                loss = loss + SHELL_WEIGHT * shell_loss
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if t % 5 == 0 or t == STEPS - 1:
                # Format string showing distances to a few recent anchors
                dist_str = ", ".join(f"{d.item():.3f}" for d in dists[:3])
                if len(dists) > 3: dist_str += f", ... ({len(dists)} total)"
                shell_str = f" shell={shell_loss.item():.4f}" if use_shell else ""
                print(f"  step {t:2d}: mean_dist={mean_dist.item():.3f} [W1s: {dist_str}]{shell_str} loss={loss.item():.4f}", flush=True)
                
        # Append converged P_K to anchors (detached, no gradients)
        with torch.no_grad():
            P_K_converged, _ = distogram_from_s_lm(model, s.detach(), z_init, pair_mask, z0, total_steps=K)
            anchors.append(P_K_converged.detach())
            
        # Classify at full horizon
        with torch.no_grad():
            Pf, _ = distogram_from_s_lm(model, s.detach(), z_init, pair_mask, z0, total_steps=N_LOOPS + 1)
            wg, wb1, wb2 = (w1(Pf, c).item() for c in (Pg, Pb1, Pb2))
            df = w1(Pf, Pbase).item()
            
        basin = min((("G", wg), ("B1", wb1), ("B2", wb2)), key=lambda kv: kv[1])[0]
        counts[basin if basin in counts else "other"] += 1
        flag = "  <== G!" if basin == "G" else ""
        print(f"-> run {run}: full_dist={df:.3f} | W1->G={wg:.3f} B1={wb1:.3f} B2={wb2:.3f} | basin={basin}{flag}", flush=True)
        
        # Save settled z
        with torch.no_grad():
            zf, _ = run_trunk(model, z_init, lift_to_lm_z(model, s.detach()), pair_mask, z0, N_LOOPS + 1)
        rd = out_dir / f"run{run}" / "seed_0" / "reps"
        rd.mkdir(parents=True, exist_ok=True)
        np.save(rd / "pairformer_z_loop16.npy", zf.float().cpu().numpy().astype(np.float16))

    print(f"\nbasin distribution over {N_RUNS} runs: {counts}", flush=True)
    if counts['G'] > 0:
        print(f"  -> recovered G {counts['G']} times", flush=True)
    print("\nSequential repel OK.", flush=True)

if __name__ == "__main__":
    main()
