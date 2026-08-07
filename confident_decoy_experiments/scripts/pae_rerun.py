#!/usr/bin/env python
"""Re-run the picked seeds with token-level PAE dumping, across GPUs 0-3.

Protenix already knows how to dump the full token x token PAE -- `--need_atom_confidence true` writes
`<name>_full_data_sample_<k>.json` containing `token_pair_pae`, `token_pair_pde`, `contact_probs` and
`atom_plddt`. Nothing in the model or the dumper needs patching; the original runs simply never set the
flag, and PAE is not recoverable from a CIF, so the poses must be regenerated.

Each job re-runs ONE seed (5 samples) for one (pdb, arm). The input JSON is the exact one that produced
the pose originally -- the native arm has three different provenances (`native_tmpl/`, `sweep_tmpl/`,
`native_a3m_all/`) and using the wrong one would silently generate a DIFFERENT pose whose PAE describes
nothing we analysed. `scripts/pae_verify.py` confirms reproduction afterwards.

Resource policy: GPUs 0-3 only (0-3 are ours; 4-7 belong to a co-worker), one worker per GPU, BLAS
threads pinned to 1 so numpy in the dataloader cannot fan out (see memory: blas-threads-blowup).

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/pae_rerun.py [--gpus 0,1,2,3] [--dry-run]
Writes: artifacts/pae_out/<arm>/<name>/seed_<seed>/predictions/*_full_data_sample_*.json
        artifacts/pae_run_logs/<name>_<arm>_seed<seed>.log
        artifacts/pae_rerun_status.json
"""
import argparse, json, os, subprocess, sys, threading, queue, time

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import EXP, CODE_ROOT
PROTENIX = "/home/ebrahim/Protenix"
OUT = f"{EXP}/artifacts/pae_out"
LOGS = f"{EXP}/artifacts/pae_run_logs"
STATUS = f"{EXP}/artifacts/pae_rerun_status.json"
os.makedirs(LOGS, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--gpus", default="0,1,2,3")
ap.add_argument("--dry-run", action="store_true")
A = ap.parse_args()
GPUS = [g.strip() for g in A.gpus.split(",") if g.strip()]

jobs = json.load(open(f"{EXP}/artifacts/pae_rerun_jobs.json"))
# Longest-first: the big assemblies (many tokens) dominate wall-clock and should start earliest.
jobs.sort(key=lambda j: -os.path.getsize(j["input_json"]))

lock = threading.Lock()
state = {j["name"] + "|" + j["arm"] + "|" + str(j["seed"]): "pending" for j in jobs}


def done_already(j):
    d = f"{OUT}/{j['arm']}/{j['name']}/seed_{j['seed']}/predictions"
    return all(os.path.exists(f"{d}/{j['name']}_full_data_sample_{k}.json") for k in range(5))


def flush():
    with lock:
        json.dump(state, open(STATUS, "w"), indent=1)


def worker(gpu, q):
    while True:
        try:
            j = q.get_nowait()
        except queue.Empty:
            return
        key = j["name"] + "|" + j["arm"] + "|" + str(j["seed"])
        if done_already(j):
            state[key] = "cached"; flush(); print(f"[gpu{gpu}] CACHED {key}", flush=True); continue
        log = f"{LOGS}/{j['name']}_{j['arm']}_seed{j['seed']}.log"
        env = dict(os.environ,
                   CUDA_VISIBLE_DEVICES=gpu, LAYERNORM_TYPE="openfold",
                   OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
        cmd = [f"{PROTENIX}/.venv/bin/python", "runner/inference.py",
               "--model_name", "protenix-v2",
               "--load_checkpoint_dir", "/home/ebrahim/checkpoint",
               "--input_json_path", j["input_json"],
               "--dump_dir", f"{OUT}/{j['arm']}",
               "--seeds", str(j["seed"]),
               "--sample_diffusion.N_sample", "5",
               "--sample_diffusion.N_step", "200",
               "--use_template", "true" if j["arm"] == "native" else "false",
               "--need_atom_confidence", "true"]
        state[key] = "running"; flush()
        print(f"[gpu{gpu}] START {key}", flush=True)
        t0 = time.time()
        with open(log, "w") as lf:
            r = subprocess.run(cmd, cwd=PROTENIX, env=env, stdout=lf, stderr=subprocess.STDOUT)
        ok = (r.returncode == 0) and done_already(j)
        state[key] = "done" if ok else f"FAILED(rc={r.returncode})"
        flush()
        print(f"[gpu{gpu}] {'DONE ' if ok else 'FAIL '} {key}  {time.time()-t0:.0f}s", flush=True)


print(f"{len(jobs)} jobs on GPUs {GPUS}")
for j in jobs:
    print(f"  {j['arm']:6} {j['name']:18} seed {j['seed']:3d}  {os.path.basename(j['input_json'])}"
          f"   [{', '.join(j['interfaces'])}]")
if A.dry_run:
    sys.exit(0)

q = queue.Queue()
for j in jobs:
    q.put(j)
ts = [threading.Thread(target=worker, args=(g, q), daemon=True) for g in GPUS]
for t in ts: t.start()
for t in ts: t.join()

flush()
bad = {k: v for k, v in state.items() if v not in ("done", "cached")}
print(f"\n{sum(1 for v in state.values() if v in ('done','cached'))}/{len(state)} complete")
if bad:
    print("PROBLEMS:")
    for k, v in bad.items():
        print(f"  {k}: {v}")
print(f"WROTE {STATUS}")
