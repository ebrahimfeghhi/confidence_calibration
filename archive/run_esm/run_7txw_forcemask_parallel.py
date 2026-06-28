"""Launch run_7txw_forcemask.py across GPUs, sharding seeds.

Each GPU worker loads the model once and folds its seed shard under ALL requested
conditions (none/candidate/control). Logs go to
ESM_PREDICTIONS/{run_prefix}_logs/gpu{g}.log.

Usage:
    python run/esm/run_7txw_forcemask_parallel.py \
        --gpus 0 1 2 5 --seeds $(seq 3000 3199) \
        --conditions none candidate control
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
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--conditions", nargs="+", default=["none", "candidate", "control"])
    ap.add_argument("--candidate", nargs="+", default=["C48", "C49", "C50"])
    ap.add_argument("--control", nargs="+", default=["C65", "C70", "C98"])
    ap.add_argument("--run_prefix", default="7txw_force")
    ap.add_argument("--num_diffusion_samples", type=int, default=5)
    ap.add_argument("--num_sampling_steps", type=int, default=16)
    ap.add_argument("--num_loops", type=int, default=16)
    ap.add_argument("--chunk_size", type=int, default=32)
    args = ap.parse_args()

    n = len(args.gpus)
    shards = [args.seeds[i::n] for i in range(n)]
    log_dir = ESM_PREDICTIONS / f"{args.run_prefix}_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(args.seeds)} seeds x {len(args.conditions)} conditions across {n} GPUs",
          flush=True)

    script = Path(__file__).resolve().parent / "run_7txw_forcemask.py"
    procs = []
    for gpu, shard in zip(args.gpus, shards):
        if not shard:
            continue
        cmd = [str(ESM_PYTHON), str(script),
               "--seeds", *[str(s) for s in shard],
               "--conditions", *args.conditions,
               "--candidate", *args.candidate,
               "--control", *args.control,
               "--run_prefix", args.run_prefix,
               "--num_diffusion_samples", str(args.num_diffusion_samples),
               "--num_sampling_steps", str(args.num_sampling_steps),
               "--num_loops", str(args.num_loops),
               "--chunk_size", str(args.chunk_size),
               "--no_config_log"]  # avoid concurrent writers to one config file
        log_path = log_dir / f"gpu{gpu}.log"
        lf = open(log_path, "w")
        proc = subprocess.Popen(
            cmd, cwd=str(Path(ESM_PYTHON).parent.parent.parent),  # esm repo root
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            stdout=lf, stderr=lf)
        print(f"GPU {gpu}: {len(shard)} seeds  (pid {proc.pid})  log -> {log_path}",
              flush=True)
        procs.append((proc, lf))

    rc = 0
    for proc, lf in procs:
        proc.wait(); lf.close(); rc |= proc.returncode
    print(f"\nAll workers done (rc={rc}).", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
