"""Decode the steered-G (real-G: basin==G & wg<0.08) runs' saved z into structures,
laid out as seed_0..seed_N under ONE 7txw dir so eval_ost_foldbench.py scores them all in
one call. Each seed_i maps to a (trial,run) starred run; the mapping is saved to
dockq_<label>_seedmap.csv.

CLI:
  --tag=TAG       run tag whose CSV/reps to read   (default slm_nudge_M5_K3)
  --zfile=NAME    which z to decode                (default pairformer_z_loop16.npy; use
                  pairformer_z_K3.npy for the K=3 horizon)
  --label=NAME    output label for dirs/CSVs       (default nudgeK17)

Run (esm venv, GPU): python analysis/grad_steer/decode_starred.py --tag=... --zfile=... --label=...
Then: python eval/eval_ost_foldbench.py --pred_dir <decode dir> --out_csv results/csv_files/dockq_<label>.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "run" / "esm"))
from run_7txw_loop_diffusion import (  # noqa: E402
    load_complex, build_conditioning, diffuse_from_z, save_samples,
)
from esm.models.esmfold2 import ESMFold2InputBuilder  # noqa: E402
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model  # noqa: E402

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import ESM_PREDICTIONS, JSON_DIR, CSV_DIR  # noqa: E402

N_SAMPLES = 5
N_STEPS = 16


def _flag(name, default):
    for a in sys.argv[1:]:
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


def starred_runs(tag):
    rows = []
    with open(CSV_DIR / f"steer_repel_{tag}.csv") as f:
        for r in csv.DictReader(f):
            if r["basin"] == "G" and float(r["wg"]) < 0.08:
                rows.append((int(r["trial"]), int(r["run"]), float(r["wg"])))
    return rows


def main():
    tag = _flag("--tag", "slm_nudge_M5_K3")
    zfile = _flag("--zfile", "pairformer_z_loop16.npy")
    label = _flag("--label", "nudgeK17")
    reps = ESM_PREDICTIONS / "7txw_repel" / tag
    out = ESM_PREDICTIONS / f"7txw_steer_decode_{label}"

    runs = starred_runs(tag)
    print(f"tag={tag} zfile={zfile} label={label} -> {len(runs)} starred runs: "
          f"{[(t, rn) for t, rn, _ in runs]}", flush=True)

    print("loading model...", flush=True)
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
    model.set_chunk_size(32)
    builder = ESMFold2InputBuilder()
    spi = load_complex(str(JSON_DIR), "7txw")
    feat, chain_infos = builder.prepare_input(spi, seed=0, device=model.device)
    with torch.no_grad():
        cond = build_conditioning(model, feat)

    mapping = []
    for i, (t, rn, wg) in enumerate(runs):
        z_path = reps / f"trial{t:02d}" / f"run{rn:02d}" / "seed_0" / "reps" / zfile
        pred_dir = out / "7txw" / f"seed_{i}" / "predictions"
        mapping.append({"seed": i, "trial": t, "run": rn, "wg": wg})
        if any(pred_dir.glob("7txw_sample_*.cif")):
            print(f"  seed_{i} (t{t:02d}r{rn:02d}): already decoded, skip", flush=True)
            continue
        if not z_path.exists():
            print(f"  seed_{i} (t{t:02d}r{rn:02d}): NO z at {z_path}, skip", flush=True)
            continue
        z_np = np.load(z_path).astype(np.float32)
        with torch.no_grad():
            samples = diffuse_from_z(model, builder, feat, chain_infos, cond, z_np,
                                     N_SAMPLES, N_STEPS, "7txw", rng_seed=12345)
        save_samples(pred_dir, "7txw", samples)
        best = max(s.iptm or s.ptm or 0.0 for s in samples)
        print(f"  seed_{i} (t{t:02d}r{rn:02d}, wg={wg:.3f}): {len(samples)} samples, "
              f"best ipTM={best:.3f}", flush=True)
        torch.cuda.empty_cache()

    mp = CSV_DIR / f"dockq_{label}_seedmap.csv"
    with open(mp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "trial", "run", "wg"])
        w.writeheader()
        w.writerows(mapping)
    print(f"\nwrote seed->run map: {mp}", flush=True)
    print(f"decode dir: {out}", flush=True)
    print("score:  python eval/eval_ost_foldbench.py "
          f"--pred_dir {out} --out_csv results/csv_files/dockq_{label}.csv", flush=True)


if __name__ == "__main__":
    main()
