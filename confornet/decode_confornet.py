"""Decode ConforNet real-G hits into structures for DockQ scoring.

`steer_confornet.py` saves only the tiny affine `(W, b)` for a real-G hit (`basin==G`
and `wg<0.08`) — never the 144 MB z. This script reads those hits from the run CSV,
reloads each affine, **regenerates the trunk z** exactly as the experiment evaluated it
(`lm_z = affine(lm_z0)` -> `run_trunk` for the full 17 loops), then runs the diffusion
structure head (reusing `diffuse_from_z` / `save_samples` from the run pipeline).

Outputs land as `7txw_confornet_decode_<label>/7txw/seed_i/predictions/` so one
`eval_ost_foldbench.py` call scores them all; a seed->hit map CSV records the mapping.

Run (esm venv, GPU):  python confornet/decode_confornet.py --tag g0
Then score:
  python eval/eval_ost_foldbench.py \
      --pred_dir <decode dir> --out_csv results/csv_files/dockq_confornet_<label>.csv
"""
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "run" / "esm"))
from run_7txw_loop_diffusion import diffuse_from_z, save_samples  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forward_grad import (  # noqa: E402
    load, compute_z_init, fixed_z0, single_rep_from_hidden, load_hidden,
    lift_to_lm_z, run_trunk, BASELINE, N_LOOPS,
)
from steer_confornet import ConforNet, REAL_G  # noqa: E402

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import CSV_DIR, ESM_PREDICTIONS  # noqa: E402

N_SAMPLES = 5
N_STEPS = 16


def _flag(name, default):
    for a in sys.argv[1:]:
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
        if a == name:
            return sys.argv[sys.argv.index(a) + 1]
    return default


def starred_runs(tag):
    """Rows with basin==G & wg<REAL_G from steer_confornet_<tag>.csv."""
    rows = []
    csv_path = CSV_DIR / f"steer_confornet_{tag}.csv"
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["basin"] == "G" and float(r["wg"]) < REAL_G:
                rows.append((int(r["run"]), int(r["particle"]), float(r["wg"])))
    return rows


def main():
    tag = _flag("--tag", "g0")
    label = _flag("--label", tag)
    affine_root = ESM_PREDICTIONS / "7txw_confornet" / tag
    out = ESM_PREDICTIONS / f"7txw_confornet_decode_{label}"

    runs = starred_runs(tag)
    print(f"tag={tag} label={label} -> {len(runs)} real-G hits: "
          f"{[(rn, p) for rn, p, _ in runs]}", flush=True)
    if not runs:
        print("nothing to decode.", flush=True)
        return

    print("loading model...", flush=True)
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]
    z_init = compute_z_init(model, cond)
    z0 = fixed_z0(model, z_init, seed=0)
    s0 = single_rep_from_hidden(model, load_hidden(BASELINE)).detach()
    lm_z0 = lift_to_lm_z(model, s0).float().detach()          # baseline pair conditioning

    mapping = []
    for i, (rn, p, wg) in enumerate(runs):
        pt = affine_root / f"run{rn:05d}_p{p}" / "confornet.pt"
        pred_dir = out / "7txw" / f"seed_{i}" / "predictions"
        mapping.append({"seed": i, "run": rn, "particle": p, "wg": wg})
        if any(pred_dir.glob("7txw_sample_*.cif")):
            print(f"  seed_{i} (run{rn:05d}_p{p}): already decoded, skip", flush=True)
            continue
        if not pt.exists():
            print(f"  seed_{i} (run{rn:05d}_p{p}): NO affine at {pt}, skip", flush=True)
            continue
        state = torch.load(pt, map_location="cpu")
        net = ConforNet().cuda()
        with torch.no_grad():
            net.W.copy_(state["W"].cuda())
            net.b.copy_(state["b"].cuda())
            z_out, _ = run_trunk(model, z_init, net(lm_z0), pair_mask, z0, N_LOOPS + 1)
            z_np = z_out.float().cpu().numpy()
            samples = diffuse_from_z(model, builder, feat, chain_infos, cond, z_np,
                                     N_SAMPLES, N_STEPS, "7txw", rng_seed=12345)
        save_samples(pred_dir, "7txw", samples)
        best = max(s.iptm or s.ptm or 0.0 for s in samples)
        print(f"  seed_{i} (run{rn:05d}_p{p}, wg={wg:.3f}): {len(samples)} samples, "
              f"best ipTM={best:.3f}", flush=True)
        torch.cuda.empty_cache()

    mp = CSV_DIR / f"dockq_confornet_{label}_seedmap.csv"
    with open(mp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "run", "particle", "wg"])
        w.writeheader()
        w.writerows(mapping)
    print(f"\nwrote seed->hit map: {mp}", flush=True)
    print(f"decode dir: {out}", flush=True)
    print("score:  python eval/eval_ost_foldbench.py "
          f"--pred_dir {out} --out_csv results/csv_files/dockq_confornet_{label}.csv", flush=True)


if __name__ == "__main__":
    main()
