"""Launch loop_diffusion.py across GPUs, sharding (seed, loop) cells.

Cells = seeds x loops are split round-robin so each GPU gets a balanced share
(each worker loads the model + conditioning once, then streams its cells).

Usage:
    python run/esm/loop_diffusion_parallel.py \
        --gpus 0 2 5 --seeds 10 14 33 39 12 0 32 42 --loops $(seq 0 16) \
        --num_diffusion_samples 5
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
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 2, 5])
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[10, 14, 33, 39, 12, 0, 32, 42])
    ap.add_argument("--loops", type=int, nargs="+", default=list(range(17)))
    ap.add_argument("--run_name", default="7txw_loops")
    ap.add_argument("--num_diffusion_samples", type=int, default=5)
    ap.add_argument("--num_sampling_steps", type=int, default=16)
    ap.add_argument("--chunk_size", type=int, default=32)
    args = ap.parse_args()

    cells = [f"{s}:{l}" for s in args.seeds for l in args.loops]
    n = len(args.gpus)
    shards = [cells[i::n] for i in range(n)]

    out_dir = ESM_PREDICTIONS / args.run_name
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(cells)} cells across {n} GPUs "
          f"({len(args.seeds)} seeds x {len(args.loops)} loops)", flush=True)

    script = Path(__file__).resolve().parent / "loop_diffusion.py"
    procs = []
    for gpu, shard in zip(args.gpus, shards):
        if not shard:
            continue
        cmd = [str(ESM_PYTHON), str(script),
               "--cells", *shard,
               "--run_name", args.run_name,
               "--num_diffusion_samples", str(args.num_diffusion_samples),
               "--num_sampling_steps", str(args.num_sampling_steps),
               "--chunk_size", str(args.chunk_size)]
        log_path = log_dir / f"gpu{gpu}.log"
        lf = open(log_path, "w")
        proc = subprocess.Popen(
            cmd, cwd=str(Path(ESM_PYTHON).parent.parent.parent),  # esm repo root
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            stdout=lf, stderr=lf)
        print(f"GPU {gpu}: {len(shard)} cells  (pid {proc.pid})  log -> {log_path}",
              flush=True)
        procs.append((proc, lf))

    rc = 0
    for proc, lf in procs:
        proc.wait(); lf.close(); rc |= proc.returncode
    print(f"\nAll workers done (rc={rc}).", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
