#!/usr/bin/env python
"""Run the 4 straggler jobs on GPU3 while the main scheduler finishes 7z2m on GPUs 0-2.

These were cut short by the watchdog incident and popped from the main queue, so the running scheduler
will never retry them. Its queue is empty, so there is no risk of both picking up the same job.
Seed-level resume means only the missing seeds are computed.
"""
import json, os, subprocess, time
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import EXP, CODE_ROOT
OUT, LOGD = f"{EXP}/artifacts/sweep_out", f"{EXP}/artifacts/sweep_logs"
SEEDS = list(range(101, 151)); PY = "/home/ebrahim/Protenix/.venv/bin/python"
jobs = json.load(open(f"{EXP}/artifacts/sweep_jobs.json"))
TARGETS = {("globallddt", "8gho"), ("ifacelddt", "8q7o"),
           ("globalplddt", "8q7o"), ("ifaceplddt", "8q7o")}
todo = [j for j in jobs if (j["run"], j["pdb"]) in TARGETS]
for j in todo:
    d = f"{OUT}/{j['run']}/{j['name']}"
    miss = [s for s in SEEDS
            if not os.path.exists(f"{d}/seed_{s}/predictions/{j['name']}_summary_confidence_sample_4.json")]
    if not miss:
        print(f"{j['run']}/{j['pdb']}: already complete", flush=True); continue
    print(f"[{time.strftime('%H:%M')}] GPU3 START {j['run']}/{j['pdb']} ({len(miss)} seeds)", flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="3", LAYERNORM_TYPE="openfold", OMP_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1")
    with open(f"{LOGD}/{j['run']}_{j['pdb']}_retry.log", "w") as fh:
        subprocess.run([PY, "runner/inference.py", "--model_name", "protenix-v2",
                        "--load_checkpoint_dir", "/home/ebrahim/checkpoint",
                        "--input_json_path", j["json"], "--dump_dir", f"{OUT}/{j['run']}",
                        "--seeds", ",".join(map(str, miss)),
                        "--sample_diffusion.N_sample", "5", "--sample_diffusion.N_step", "200",
                        "--use_template", "true"],
                       cwd="/home/ebrahim/Protenix", env=env, stdout=fh, stderr=subprocess.STDOUT)
    done = sum(1 for s in SEEDS
               if os.path.exists(f"{d}/seed_{s}/predictions/{j['name']}_summary_confidence_sample_4.json"))
    print(f"[{time.strftime('%H:%M')}] GPU3 DONE  {j['run']}/{j['pdb']} {done}/50", flush=True)
print("STRAGGLERS DONE")
