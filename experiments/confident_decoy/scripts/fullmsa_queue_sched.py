#!/usr/bin/env python
"""Full-MSA run, GLOBAL WORK-QUEUE scheduler — never leaves a GPU idle.

Previous schedulers processed one PDB at a time and split its seeds across GPUs, so a PDB needing
fewer seeds than GPUs left GPUs idle. Here every missing (pdb, seed) is one queue item; 4 workers
(one per GPU) pull the next item as soon as they finish, so all GPUs stay busy until the queue drains.
Order still prioritises PDBs that are furthest from the target, so coverage completes early.

MSA subsampling disabled via --data.msa.min_size.test 16384 (README 6.9).
Target defaults to seeds 101-104 (N=4); override with argv[1] (e.g. 16)."""
import subprocess, os, json, glob, sys, threading, queue, time

PY   = "/home/ebrahim/Protenix/.venv/bin/python"
PROT = "/home/ebrahim/Protenix"
EXP  = "/home/ebrahim/confident_decoy_experiments"
JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
DUMP = f"{EXP}/artifacts/fullmsa_out"
LOGD = f"{EXP}/artifacts/fullmsa_logs"
os.makedirs(DUMP, exist_ok=True); os.makedirs(LOGD, exist_ok=True)

GPUS   = [0, 1, 2, 3]
NTARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 4
TARGET  = list(range(101, 101 + NTARGET))
ORDER   = ["7z2m","8slb","8yxi","8the","8fas","8ux6","8fbw","8q3j","8che","8av2","8aok","8bf4"]

GENENV = dict(os.environ)
GENENV.update(LAYERNORM_TYPE="openfold", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
              MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
lock = threading.Lock()

def has(p, s):
    return os.path.exists(f"{DUMP}/{p}/seed_{s}/predictions/{p}_summary_confidence_sample_4.json")

def progress():
    return {p: sum(1 for s in TARGET if has(p, s)) for p in ORDER}

def log(m):
    with lock:
        ts = time.strftime("%H:%M:%S")
        prog = progress()
        json.dump({"t": ts, "msg": m, "target": NTARGET, "progress": prog},
                  open(f"{LOGD}/queue_status.json", "w"), indent=1)
        print(f"{ts} {m} | {prog}", flush=True)

# build the queue: all missing (pdb, seed), ordered by ORDER
q = queue.Queue()
todo = [(p, s) for p in ORDER for s in TARGET if not has(p, s)]
for item in todo: q.put(item)
log(f"queue scheduler start; target N={NTARGET}; {len(todo)} (pdb,seed) jobs to run")

def worker(gpu):
    while True:
        try: p, s = q.get_nowait()
        except queue.Empty:
            log(f"GPU{gpu}: queue empty, worker exiting"); return
        if has(p, s):
            q.task_done(); continue
        env = dict(GENENV); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log(f"GPU{gpu}: start {p} seed {s}  (queue left ~{q.qsize()})")
        subprocess.run(
            [PY, "runner/inference.py", "--model_name", "protenix-v2",
             "--load_checkpoint_dir", "/home/ebrahim/checkpoint",
             "--input_json_path", f"{JDIR}/{p}-final-updated.json",
             "--dump_dir", DUMP, "--seeds", str(s),
             "--sample_diffusion.N_sample", "5", "--sample_diffusion.N_step", "200",
             "--use_template", "true", "--data.msa.min_size.test", "16384"],
            cwd=PROT, env=env,
            stdout=open(f"{LOGD}/q_{p}_{s}_g{gpu}.log", "w"), stderr=subprocess.STDOUT)
        log(f"GPU{gpu}: done {p} seed {s}")
        q.task_done()

threads = [threading.Thread(target=worker, args=(g,), daemon=False) for g in GPUS]
for t in threads: t.start()
for t in threads: t.join()
log(f"ALL_DONE queue N={NTARGET}")
