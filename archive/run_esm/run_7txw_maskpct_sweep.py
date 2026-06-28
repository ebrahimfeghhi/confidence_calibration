"""Random-mask-RATE sweep for 7txw: one lm_mask_pct per GPU, the SAME seeds on each.

Tests whether simply raising the random LM mask rate above the default 10% finds
good seeds more efficiently. Each GPU folds all --seeds at its own --pcts value
(pure random masking, no forced residues) into ESM_PREDICTIONS/7txw_mask{pct}_none
(reuses run_7txw_forcemask.py with --conditions none).

Usage:
    python run/esm/run_7txw_maskpct_sweep.py \
        --gpus 0 1 2 5 --pcts 0.125 0.15 0.175 0.20 --seeds $(seq 3000 3199)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import ESM_PREDICTIONS, ESM_PYTHON


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 5])
    ap.add_argument("--pcts", type=float, nargs="+", default=[0.125, 0.15, 0.175, 0.20])
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--num_diffusion_samples", type=int, default=5)
    ap.add_argument("--num_sampling_steps", type=int, default=16)
    ap.add_argument("--num_loops", type=int, default=16)
    ap.add_argument("--chunk_size", type=int, default=32)
    args = ap.parse_args()

    if len(args.gpus) != len(args.pcts):
        sys.exit(f"need one pct per gpu ({len(args.gpus)} gpus, {len(args.pcts)} pcts)")

    log_dir = ESM_PREDICTIONS / "7txw_maskpct_sweep_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / "run_7txw_forcemask.py"
    print(f"{len(args.seeds)} seeds x {len(args.pcts)} mask rates (1 rate/GPU)", flush=True)

    procs = []
    for gpu, pct in zip(args.gpus, args.pcts):
        cmd = [str(ESM_PYTHON), str(script),
               "--seeds", *[str(s) for s in args.seeds],
               "--conditions", "none",
               "--lm_mask_pct", str(pct),
               "--run_prefix", f"7txw_mask{pct}",
               "--num_diffusion_samples", str(args.num_diffusion_samples),
               "--num_sampling_steps", str(args.num_sampling_steps),
               "--num_loops", str(args.num_loops),
               "--chunk_size", str(args.chunk_size),
               "--no_config_log"]
        log_path = log_dir / f"gpu{gpu}_pct{pct}.log"
        lf = open(log_path, "w")
        proc = subprocess.Popen(
            cmd, cwd=str(Path(ESM_PYTHON).parent.parent.parent),  # esm repo root
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            stdout=lf, stderr=lf)
        print(f"GPU {gpu}: lm_mask_pct={pct}  -> 7txw_mask{pct}_none  "
              f"(pid {proc.pid})  log -> {log_path}", flush=True)
        procs.append((proc, lf))

    rc = 0
    for proc, lf in procs:
        proc.wait(); lf.close(); rc |= proc.returncode
    print(f"\nAll workers done (rc={rc}).", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
