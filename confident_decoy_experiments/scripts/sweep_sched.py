#!/usr/bin/env python
"""PHASE 3 of the 7-run template sweep: run every job in artifacts/sweep_jobs.json on GPUs 0-3.

Global work queue: each GPU takes the next unfinished job as it frees up, so no GPU idles waiting for
a slow one. ONE job per GPU at a time -- a Protenix process peaks around 83 GB and the cards are 97 GB,
so two per card would OOM.

RESUMABLE: a job whose 50 seeds already have sample_4 confidence JSONs is skipped, so the scheduler can
be killed and relaunched without losing work.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/sweep_sched.py
Writes: artifacts/sweep_out/<run>/<name>/seed_<S>/predictions/...
        artifacts/sweep_logs/<run>_<pdb>.log, artifacts/sweep_status.json
"""
import argparse, json, os, subprocess, sys, time, threading

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import EXP, CODE_ROOT
# Parameterised so the same scheduler serves the template sweep and the later native-only runs;
# the resume/ordering/memory logic is identical, only the job list and output root differ.
_ap = argparse.ArgumentParser()
_ap.add_argument("--jobs", default=f"{EXP}/artifacts/sweep_jobs.json")
_ap.add_argument("--out", default=f"{EXP}/artifacts/sweep_out")
_ap.add_argument("--logs", default=f"{EXP}/artifacts/sweep_logs")
_ap.add_argument("--status", default=f"{EXP}/artifacts/sweep_status.json")
_ap.add_argument("--n-sample", type=int, default=5,
                 help="diffusion samples per seed. MUST match the completion probe below: an arm run "
                      "at N_sample=1 writes only sample_0, so a hardcoded sample_4 probe would mark "
                      "every job incomplete forever and re-run it on every pass.")
_ap.add_argument("--gpus", default="0,1,2,3")
_A = _ap.parse_args()
OUT = _A.out
LOGD = _A.logs
GPUS = [int(x) for x in _A.gpus.split(",")]
NSAMPLE = _A.n_sample
SEEDS = list(range(101, 151))
PY = "/home/ebrahim/Protenix/.venv/bin/python"
os.makedirs(OUT, exist_ok=True); os.makedirs(LOGD, exist_ok=True)
jobs = json.load(open(_A.jobs))


def job_seeds(job):
    """The seeds THIS job asks for, falling back to the global range.

    Arms differ: the template sweep and the 26-set arms use 50 seeds (101-150), but `leader20` runs
    20 templates x 5 seeds per interface. Ignoring job["seeds"] and using the global list silently
    ran 10x the intended work (454 jobs x 50 seeds instead of x5)."""
    return job.get("seeds") or SEEDS


def missing_seeds(job):
    """Seeds with no complete output yet. Passing only these to inference makes a restart cheap --
    without it, killing the scheduler throws away every partially-finished job."""
    d = f"{OUT}/{job['run']}/{job['name']}"
    return [s for s in job_seeds(job)
            if not os.path.exists(
                f"{d}/seed_{s}/predictions/"
                f"{job['name']}_summary_confidence_sample_{NSAMPLE - 1}.json")]


def done(job):
    return len(job_seeds(job)) - len(missing_seeds(job))


def cost(job):
    """Cheap proxy for runtime: total modelled residues (sum of chain length x copy count)."""
    try:
        o = json.load(open(job["json"]))
        o = o[0] if isinstance(o, list) else o
        return sum(len(e["proteinChain"]["sequence"]) * int(e["proteinChain"].get("count", 1))
                   for e in o["sequences"] if "proteinChain" in e)
    except Exception:
        return 0


# SHORTEST-FIRST. Makespan is identical either way here (4 huge jobs, 4 GPUs), but shortest-first
# banks ~104 of 108 results hours earlier and leaves only 7z2m -- the slowest pdb by 20x, and the one
# whose interface labels were wrong anyway -- outstanding at the end. Longest-first locked all four
# GPUs on 7z2m for ~10 h and produced nothing else in the meantime.
lock = threading.Lock()
queue = sorted([j for j in jobs if done(j) < len(job_seeds(j))], key=cost)
skipped = len(jobs) - len(queue)
print(f"{len(jobs)} jobs total; {skipped} already complete; {len(queue)} to run", flush=True)
state = {"total": len(jobs), "skipped": skipped, "queue": len(queue), "finished": [], "running": {}}


def save():
    json.dump(state, open(_A.status, "w"), indent=1)


BIG = 1500          # residues; above this a job needs most of a card (7z2m ~2244 -> ~92 GB)
NEED_MIB = 90000    # free memory a BIG job needs


def free_mib(gpu):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used",
                              "--format=csv,noheader,nounits", "-i", str(gpu)],
                             capture_output=True, text=True).stdout.strip()
        tot, used = [int(x) for x in out.split(",")]
        return tot - used
    except Exception:
        return NEED_MIB          # unknown -> don't block


def worker(gpu):
    while True:
        with lock:
            if not queue: return
            # A BIG job needs almost a whole card. If a co-tenant is on this GPU there is not enough
            # room and it would OOM, so take a smaller job instead and leave the big one queued.
            idx = 0
            if cost(queue[0]) > BIG and free_mib(gpu) < NEED_MIB:
                idx = next((i for i, j in enumerate(queue) if cost(j) <= BIG), None)
                if idx is None:
                    # only big jobs left and this card is occupied -- wait for the neighbour to leave
                    print(f"[{time.strftime('%H:%M')}] GPU{gpu} WAIT only big jobs left, "
                          f"{free_mib(gpu)} MiB free (< {NEED_MIB}); retry in 10 min", flush=True)
                    lock.release()
                    try:
                        time.sleep(600)
                    finally:
                        lock.acquire()
                    continue
            job = queue.pop(idx)
            state["running"][str(gpu)] = f"{job['run']}/{job['pdb']}"
            save()
        todo = missing_seeds(job)
        if not todo:
            with lock:
                state["running"].pop(str(gpu), None); save()
            continue
        # Key the log on the JOB, not the pdb: the dockqdonor arm runs one job per INTERFACE, so
        # 8slb_AB and 8slb_AC are two jobs of the same pdb and a pdb-keyed log silently overwrites one
        # with the other. Output dirs and missing_seeds() already key on job['name'], so only the log
        # was affected.
        # Key the log on job['name'] -- the only field unique per job. Keying on the interface
        # collided across leader20's 20 templates per interface: each job truncated the previous
        # one's log, so per-template template-loading could not be verified at all.
        log = f"{LOGD}/{job['run']}_{job.get('name', job.get('interface', job['pdb']))}.log"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), LAYERNORM_TYPE="openfold",
                   OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
        cmd = [PY, "runner/inference.py", "--model_name", "protenix-v2",
               "--load_checkpoint_dir", "/home/ebrahim/checkpoint",
               "--input_json_path", job["json"], "--dump_dir", f"{OUT}/{job['run']}",
               "--seeds", ",".join(map(str, todo)),
               "--sample_diffusion.N_sample", str(NSAMPLE),
               "--sample_diffusion.N_step", "200",
               "--use_template", "true"]
        t0 = time.time()
        print(f"[{time.strftime('%H:%M')}] GPU{gpu} START {job['run']}/{job['pdb']} "
              f"({len(queue)} left)", flush=True)
        with open(log, "w") as fh:
            subprocess.run(cmd, cwd="/home/ebrahim/Protenix", env=env, stdout=fh, stderr=subprocess.STDOUT)
        n = done(job)
        with lock:
            state["finished"].append({"run": job["run"], "pdb": job["pdb"], "seeds_done": n,
                                      "seeds_wanted": len(job_seeds(job)),
                                      "minutes": round((time.time() - t0) / 60, 1)})
            state["running"].pop(str(gpu), None)
            save()
        print(f"[{time.strftime('%H:%M')}] GPU{gpu} DONE  {job['run']}/{job['pdb']} "
              f"{n}/{len(job_seeds(job))} seeds in {(time.time()-t0)/60:.1f} min ({len(queue)} left)", flush=True)


ths = [threading.Thread(target=worker, args=(g,), daemon=False) for g in GPUS]
[t.start() for t in ths]
[t.join() for t in ths]
state["running"] = {}
save()
inc = [f for f in state["finished"] if f["seeds_done"] < f.get("seeds_wanted", len(SEEDS))]
print(f"\nALL DONE. {len(state['finished'])} jobs run; {len(inc)} INCOMPLETE: "
      f"{[(f['run'], f['pdb'], f['seeds_done']) for f in inc]}")
