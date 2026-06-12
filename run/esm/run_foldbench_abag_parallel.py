"""
Launch run_foldbench_abag.py in parallel across multiple GPUs, splitting seeds evenly.
Each GPU runs all structures for its subset of seeds.

Usage:
    python scripts/run_foldbench_abag_parallel.py \
        --run_name esmfold2fast_tts \
        --gpus 0 1 2 3 \
        --seeds 0 1 2 3 4 5 6 7 8 9
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# --- repo-root bootstrap so `from paths import ...` works from any cwd ---
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import ESM_PREDICTIONS, JSON_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", default=list(range(10)))
    parser.add_argument("--lm_mask_pct", type=float, default=0.1)
    parser.add_argument("--lm_dropout", type=float, default=0.3)
    parser.add_argument("--num_loops", type=int, default=16)
    parser.add_argument("--num_sampling_steps", type=int, default=16)
    parser.add_argument("--num_diffusion_samples", type=int, default=5)
    parser.add_argument("--json_dir", default=str(JSON_DIR))
    parser.add_argument("--complexes", nargs="+", default=None,
                        help="Only process these complex names (e.g. 7txw 7ua2).")
    args = parser.parse_args()

    n_gpus = len(args.gpus)
    seed_chunks = [args.seeds[i::n_gpus] for i in range(n_gpus)]

    log_dir = ESM_PREDICTIONS / args.run_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).resolve().parent / "run_foldbench_abag.py"
    procs = []
    for gpu, seeds in zip(args.gpus, seed_chunks):
        if not seeds:
            continue
        cmd = [
            sys.executable, str(script),
            "--run_name", args.run_name,
            "--seeds", *map(str, seeds),
            "--num_loops", str(args.num_loops),
            "--num_sampling_steps", str(args.num_sampling_steps),
            "--num_diffusion_samples", str(args.num_diffusion_samples),
            "--lm_mask_pct", str(args.lm_mask_pct),
            "--lm_dropout", str(args.lm_dropout),
            "--json_dir", args.json_dir,
        ]
        if args.complexes:
            cmd += ["--complexes", *args.complexes]
        log_path = log_dir / f"gpu{gpu}.log"
        log_file = open(log_path, "w")
        proc = subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                                          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
                                stdout=log_file, stderr=log_file)
        print(f"GPU {gpu}: seeds {seeds}  (pid {proc.pid})  log → {log_path}")
        procs.append((proc, log_file))

    for proc, log_file in procs:
        proc.wait()
        log_file.close()

    print("\nAll done.")


if __name__ == "__main__":
    main()
