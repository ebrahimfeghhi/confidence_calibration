"""Step-count x seed sweep for the s_lm repulsion steer — gather more data on the
working steering approach (the 5-vs-15-step knife-edge, extended + replicated).

For each (seed, steps), run the label-free sequential repulsion from steer_sequential:
read the K=3 cross-distogram, push it AWAY from accumulated anchors (seed anchor =
baseline = the B1 floor; each converged run becomes a new anchor), `steps` Adam steps,
restart per run. 10 runs/iterations per (seed, steps). Loss = -mean_a W1(P_K, a).

Records per run (all free, no decode): basin via nearest W1 centroid {G,B1,B2},
dev_baseline=||s-s0||/||s0||, cloud_z=(||s-cloud_mean||-mu)/sigma (manifold metric).
Saves the loop-16 z so it can be decoded -> DockQ afterward.

Grid: --seeds 0,1,2,3,4  x  --steps 1,2,5,10,15  x  --runs 10  = 250 steered s_lm.
Split across GPUs by seed, e.g.:
  CUDA_VISIBLE_DEVICES=4 ... steer_sweep.py --seeds 0,1,2
  CUDA_VISIBLE_DEVICES=5 ... steer_sweep.py --seeds 3,4

Run (esm venv, GPU).
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forward_grad import (  # noqa: E402
    load, compute_z_init, fixed_z0, single_rep_from_hidden, load_hidden,
    distogram_from_s_lm, run_trunk, lift_to_lm_z, cross_distogram_from_z,
    w1, BASELINE, GOOD, N_LOOPS,
)
from steer_sequential import B1, B2          # noqa: E402
from steer_deviation import centroid, build_cloud  # noqa: E402

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import ESM_PREDICTIONS, CSV_DIR  # noqa: E402

OUT = ESM_PREDICTIONS / "7txw_steer_sweep" / "7txw"
K = 3
LR = 0.3
INIT_FRAC = 0.05
STEP_SET = [1, 2, 5, 10, 15]
N_RUNS = 10


def _list_arg(name, default):
    for a in sys.argv[1:]:
        if a.startswith(name + "="):
            return [int(x) for x in a.split("=", 1)[1].split(",")]
        if a == name:
            i = sys.argv.index(a)
            return [int(x) for x in sys.argv[i + 1].split(",")]
    return default


def main():
    seeds = _list_arg("--seeds", [0, 1, 2, 3, 4])
    step_set = _list_arg("--steps", STEP_SET)
    n_runs = _list_arg("--runs", [N_RUNS])[0]
    print(f"loading model (ESMC -> CPU)... seeds={seeds} steps={step_set} runs={n_runs}",
          flush=True)
    model, builder, feat, chain_infos, cond = load()      # keep_esmc=False (old default)
    pair_mask = cond["pair_mask"]
    z_init = compute_z_init(model, cond)
    z0 = fixed_z0(model, z_init, seed=0)

    with torch.no_grad():
        Pg, Pb1, Pb2 = (centroid(model, pair_mask, s) for s in (GOOD, B1, B2))
    cloud_S, cloud_mean, mu, sigma = build_cloud(model)

    s0 = single_rep_from_hidden(model, load_hidden(BASELINE)).detach()
    s0v = s0.flatten().cpu()
    s0n = s0.norm().item()
    with torch.no_grad():
        Pbase_K, _ = distogram_from_s_lm(model, s0, z_init, pair_mask, z0, total_steps=K)
        Pbase_K = Pbase_K.detach()

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_".join(str(s) for s in seeds)
    out_csv = CSV_DIR / f"steer_sweep_7txw_s{tag}.csv"
    fields = ["seed", "steps", "run", "basin", "wg", "wb1", "wb2",
              "dev_baseline", "cloud_z"]
    rows = []

    for seed in seeds:
        for steps in step_set:
            print(f"\n===== seed={seed} steps={steps} (K={K}, lr={LR}, {n_runs} runs) =====",
                  flush=True)
            anchors = [Pbase_K]                       # repel from baseline (B1 floor) first
            for run in range(n_runs):
                torch.manual_seed(10000 * seed + 2000 + run)
                d = torch.randn_like(s0)
                d = d / d.norm() * (INIT_FRAC * s0n)
                s = (s0 + d).requires_grad_(True)
                opt = torch.optim.Adam([s], lr=LR)
                for _ in range(steps):
                    P_K, _ = distogram_from_s_lm(model, s, z_init, pair_mask, z0,
                                                 total_steps=K)
                    loss = -sum(w1(P_K, a) for a in anchors) / len(anchors)
                    opt.zero_grad(); loss.backward(); opt.step()

                with torch.no_grad():
                    P_K_conv, _ = distogram_from_s_lm(model, s.detach(), z_init, pair_mask,
                                                      z0, total_steps=K)
                    anchors.append(P_K_conv.detach())
                    # ONE full-horizon pass, reused for both classify and the saved z
                    z_full, _ = run_trunk(model, z_init, lift_to_lm_z(model, s.detach()),
                                          pair_mask, z0, N_LOOPS + 1)
                    Pf = cross_distogram_from_z(model, z_full, pair_mask)
                    wg, wb1, wb2 = (w1(Pf, c).item() for c in (Pg, Pb1, Pb2))
                basin = min((("G", wg), ("B1", wb1), ("B2", wb2)), key=lambda kv: kv[1])[0]

                sv = s.detach().flatten().cpu()
                dev = (sv - s0v).norm().item() / s0n
                cloud_z = ((sv - cloud_mean).norm().item() - mu) / sigma

                rd = OUT / f"seed{seed}" / f"{steps}step" / f"run{run}" / "seed_0" / "reps"
                rd.mkdir(parents=True, exist_ok=True)
                np.save(rd / "pairformer_z_loop16.npy",
                        z_full.float().cpu().numpy().astype(np.float16))

                rows.append({"seed": seed, "steps": steps, "run": run, "basin": basin,
                             "wg": round(wg, 4), "wb1": round(wb1, 4), "wb2": round(wb2, 4),
                             "dev_baseline": round(dev, 4), "cloud_z": round(cloud_z, 4)})
                flag = " <== G!" if basin == "G" else ""
                print(f"  run{run}: basin={basin:>2} W1->G={wg:.3f} "
                      f"dev={dev:.3f} cloud_z={cloud_z:.2f}{flag}", flush=True)
            # flush CSV after each (seed,steps) so partial progress is saved
            with open(out_csv, "w", newline="") as f:
                wcsv = csv.DictWriter(f, fieldnames=fields); wcsv.writeheader(); wcsv.writerows(rows)

    print(f"\nwrote {out_csv}  ({len(rows)} rows)", flush=True)
    print("steer_sweep OK.", flush=True)


if __name__ == "__main__":
    main()
