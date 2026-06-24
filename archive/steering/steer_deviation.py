"""How far does the steered `s_lm` stray from the network's input manifold, as a
function of optimization steps? (5 vs 15)

Hypothesis (testing): "far from everything explored" is the right repulsion objective;
the 15-step failure is the optimizer over-perturbing the LM single rep `s_lm` off the
network's valid input distribution, so the trunk emits a novel-but-unfoldable distogram.

We re-run the sequential-repulsion protocol from `steer_sequential.py` (it only saved
the output z, not the optimized s_lm — and z is not invertible), restarting from the
baseline s_lm0 at STEPS=5 and STEPS=15 (10 runs each, anchors accumulate within a pass),
and for each run's converged s_lm record:
  dev_baseline = ||s - s0|| / ||s0||                 # the user's metric
  cloud_z      = (||s - mean_cloud|| - mu) / sigma   # cloud = 60 real-seed s_lm's
  cloud_min    = min_seed ||s - s_seed||
  z_match      = W1(reproduced full-horizon distogram, on-disk SCORED z distogram)
                 ~0 => we reproduced the scored trajectory despite CUDA nondeterminism;
                 large => basin flipped, do NOT map this run onto the old DockQ.

Existing DockQ for the scored runs: results/csv_files/dockq_ost_7txw_seqrepel.csv
  (5-step run2/run8 good ~0.85-0.91; 15-step run1 bad 0.37, run7 medium 0.72).

Run (esm venv, GPU): CUDA_VISIBLE_DEVICES=N python analysis/grad_steer/steer_deviation.py
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
    w1, BASELINE, GOOD, N_LOOPS, REPS,
)
from steer_sequential import B1, B2  # noqa: E402

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import ESM_PREDICTIONS, CSV_DIR  # noqa: E402

SCORED = ESM_PREDICTIONS / "7txw_steer_sequential"   # 7txw (15-step), 7txw_seq5 (5-step)
OUTDEV = ESM_PREDICTIONS / "7txw_steer_deviation"

K = 3
LR = 0.3
INIT_FRAC = 0.05
N_RUNS = 10
STEP_SET = [5, 15]


def centroid(model, pair_mask, seeds):
    Ps = [cross_distogram_from_z(
              model, torch.from_numpy(np.load(
                  REPS / f"seed_{s}" / "reps" / "pairformer_z_loop16.npy").astype(np.float32)
              ).cuda(), pair_mask)
          for s in seeds]
    P = torch.stack(Ps).mean(0)
    return (P / P.sum(-1, keepdim=True)).detach()


def build_cloud(model):
    """Reconstruct s_lm for all 60 real seeds -> stacked reps (cpu), mean, radius mu/sigma."""
    seeds = sorted(int(p.name.split("_")[1]) for p in REPS.glob("seed_*"))
    vecs = []
    for s in seeds:
        with torch.no_grad():
            v = single_rep_from_hidden(model, load_hidden(REPS / f"seed_{s}")).flatten().cpu()
        vecs.append(v)
        torch.cuda.empty_cache()
    S = torch.stack(vecs)                      # [60, L*256] on cpu
    mean = S.mean(0)
    radii = (S - mean).norm(dim=1)
    print(f"cloud: {len(seeds)} seeds, radius mu={radii.mean():.4e} sigma={radii.std():.4e}",
          flush=True)
    return S, mean, radii.mean().item(), radii.std().item()


def on_disk_distogram(model, pair_mask, steps, run):
    sub = "7txw_seq5" if steps == 5 else "7txw"
    zp = SCORED / sub / f"run{run}" / "seed_0" / "reps" / "pairformer_z_loop16.npy"
    if not zp.exists():
        return None
    z = torch.from_numpy(np.load(zp).astype(np.float32)).cuda()
    with torch.no_grad():
        return cross_distogram_from_z(model, z, pair_mask)


def main():
    print("loading model...", flush=True)
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]
    z_init = compute_z_init(model, cond)
    z0 = fixed_z0(model, z_init, seed=0)

    with torch.no_grad():
        Pg, Pb1, Pb2 = (centroid(model, pair_mask, s) for s in (GOOD, B1, B2))
    cloud_S, cloud_mean, mu, sigma = build_cloud(model)

    s0 = single_rep_from_hidden(model, load_hidden(BASELINE)).detach()
    s0v = s0.flatten().cpu()
    s0n = s0.norm().item()
    base_cloud_z = ((s0v - cloud_mean).norm().item() - mu) / sigma
    print(f"||s0||={s0n:.4e}  baseline cloud_z={base_cloud_z:.3f} "
          f"(sanity: should be near/inside the cloud)", flush=True)

    with torch.no_grad():
        Pbase_K, _ = distogram_from_s_lm(model, s0, z_init, pair_mask, z0, total_steps=K)
        Pbase_K = Pbase_K.detach()

    arg_steps = None
    for a in sys.argv[1:]:
        if a.startswith("--steps="):
            arg_steps = [int(a.split("=", 1)[1])]
    step_set = arg_steps or STEP_SET
    suffix = f"_{step_set[0]}step" if len(step_set) == 1 else ""

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = CSV_DIR / f"steer_deviation_7txw{suffix}.csv"
    fields = ["steps", "run", "basin", "wg", "wb1", "wb2",
              "dev_baseline", "cloud_z", "cloud_min", "z_match"]
    rows = [{"steps": 0, "run": -1, "basin": "baseline", "wg": "", "wb1": "", "wb2": "",
             "dev_baseline": 0.0, "cloud_z": round(base_cloud_z, 4),
             "cloud_min": round((cloud_S - s0v).norm(dim=1).min().item(), 4), "z_match": ""}]

    for steps in step_set:
        print(f"\n===== STEPS={steps} (K={K}, lr={LR}, {N_RUNS} runs) =====", flush=True)
        anchors = [Pbase_K]
        for run in range(N_RUNS):
            torch.manual_seed(2000 + run)
            d = torch.randn_like(s0)
            d = d / d.norm() * (INIT_FRAC * s0n)
            s = (s0 + d).requires_grad_(True)
            opt = torch.optim.Adam([s], lr=LR)
            for t in range(steps):
                P_K, _ = distogram_from_s_lm(model, s, z_init, pair_mask, z0, total_steps=K)
                mean_dist = sum(w1(P_K, a) for a in anchors) / len(anchors)
                loss = -mean_dist
                opt.zero_grad(); loss.backward(); opt.step()

            with torch.no_grad():
                P_K_conv, _ = distogram_from_s_lm(model, s.detach(), z_init, pair_mask, z0,
                                                  total_steps=K)
                anchors.append(P_K_conv.detach())
                Pf, _ = distogram_from_s_lm(model, s.detach(), z_init, pair_mask, z0,
                                            total_steps=N_LOOPS + 1)
                wg, wb1, wb2 = (w1(Pf, c).item() for c in (Pg, Pb1, Pb2))
                zf, _ = run_trunk(model, z_init, lift_to_lm_z(model, s.detach()),
                                  pair_mask, z0, N_LOOPS + 1)
                Pdisk = on_disk_distogram(model, pair_mask, steps, run)
                z_match = w1(Pf, Pdisk).item() if Pdisk is not None else float("nan")
            basin = min((("G", wg), ("B1", wb1), ("B2", wb2)), key=lambda kv: kv[1])[0]

            sv = s.detach().flatten().cpu()
            dev_baseline = (sv - s0v).norm().item() / s0n
            cloud_z = ((sv - cloud_mean).norm().item() - mu) / sigma
            cloud_min = (cloud_S - sv).norm(dim=1).min().item()

            rd = OUTDEV / f"{steps}step" / f"run{run}" / "seed_0" / "reps"
            rd.mkdir(parents=True, exist_ok=True)
            np.save(rd / "s_lm.npy", s.detach().float().cpu().numpy())
            np.save(rd / "pairformer_z_loop16.npy",
                    zf.float().cpu().numpy().astype(np.float16))

            rows.append({"steps": steps, "run": run, "basin": basin,
                         "wg": round(wg, 4), "wb1": round(wb1, 4), "wb2": round(wb2, 4),
                         "dev_baseline": round(dev_baseline, 4),
                         "cloud_z": round(cloud_z, 4), "cloud_min": round(cloud_min, 4),
                         "z_match": round(z_match, 4)})
            print(f"  run{run}: basin={basin:>2} dev_baseline={dev_baseline:.3f} "
                  f"cloud_z={cloud_z:.2f} cloud_min={cloud_min:.3e} "
                  f"z_match={z_match:.4f}", flush=True)

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out_csv}  ({len(rows)} rows)", flush=True)
    print("steer_deviation OK.", flush=True)


if __name__ == "__main__":
    main()
