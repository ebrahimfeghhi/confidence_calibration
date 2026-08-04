#!/usr/bin/env python
"""Expand the free-run pool to 150 seeds for the 8 confident-decoy PDBs that
currently have only 50 seeds (101-150). Adds seeds 0-100 (101 new seeds) x 5
samples, standard templates (use_template true), matching the base pool.
Writes into the canonical predictions_template dir. One PDB per GPU, dynamic
refill across GPUs 0-3. See confident_decoy_README.md."""
import subprocess, time, os, json, glob, re

PY   = "/home/ebrahim/Protenix/.venv/bin/python"
PROT = "/home/ebrahim/Protenix"
JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
DUMP = "/home/ebrahim/Protenix/foldbench_abag/predictions_template"
LOGD = "/home/ebrahim/confident_decoy_experiments/artifacts/expand_seeds_logs"
STATUS = f"{LOGD}/status.json"
os.makedirs(LOGD, exist_ok=True)

GPUS  = [0, 1, 2, 3]
PDBS  = ["7z2m", "8aok", "8av2", "8bf4", "8che", "8fas", "8slb", "8the"]
SEEDS = list(range(0, 101))          # 0..100 inclusive = 101 seeds
NEED  = len(SEEDS)

GENENV = dict(os.environ)
GENENV.update(LAYERNORM_TYPE="openfold", OMP_NUM_THREADS="4",
              OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
              NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")

def log(m):
    ts = time.strftime("%H:%M:%S")
    prog = {p: len(glob.glob(f"{DUMP}/{p}/seed_*/predictions/{p}_summary_confidence_sample_4.json"))
            for p in PDBS}
    json.dump({"t": ts, "msg": m, "done_seeds": prog}, open(STATUS, "w"), indent=1)
    print(f"{ts} {m} | {prog}", flush=True)

def done(pdb):
    # count seeds in our 0-100 band that finished all 5 samples
    c = 0
    for s in SEEDS:
        if os.path.exists(f"{DUMP}/{pdb}/seed_{s}/predictions/{pdb}_summary_confidence_sample_4.json"):
            c += 1
    return c >= NEED

def inflight():
    d = {}
    for p in os.listdir('/proc'):
        if not p.isdigit():
            continue
        try:
            cl = open(f'/proc/{p}/cmdline', 'rb').read().replace(b'\0', b' ').decode()
        except Exception:
            continue
        if 'inference.py' not in cl or 'predictions_template' not in cl:
            continue
        m = re.search(r'/([0-9a-z]+)-final-updated\.json', cl)
        if not m or m.group(1) not in PDBS:
            continue
        try:
            env = open(f'/proc/{p}/environ', 'rb').read().replace(b'\0', b'\n').decode()
        except Exception:
            env = ''
        g = re.search(r'CUDA_VISIBLE_DEVICES=(\d+)', env)
        d.setdefault(m.group(1), []).append(int(g.group(1)) if g else -1)
    return d

def launch(pdb, gpu):
    env = dict(GENENV); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.Popen(
        [PY, "runner/inference.py", "--model_name", "protenix-v2",
         "--load_checkpoint_dir", "/home/ebrahim/checkpoint",
         "--input_json_path", f"{JDIR}/{pdb}-final-updated.json",
         "--dump_dir", DUMP, "--seeds", ",".join(map(str, SEEDS)),
         "--sample_diffusion.N_sample", "5", "--sample_diffusion.N_step", "200",
         "--use_template", "true"],
        cwd=PROT, env=env,
        stdout=open(f"{LOGD}/{pdb}_g{gpu}.log", "w"), stderr=subprocess.STDOUT)

log(f"expand-seeds scheduler start; pdbs={PDBS}; seeds 0-100 -> +{NEED}/pdb on GPUs {GPUS}")
while True:
    ifl  = inflight()
    busy = set(g for gs in ifl.values() for g in gs if g >= 0)
    free = [g for g in GPUS if g not in busy]
    pend = [p for p in PDBS if not done(p) and p not in ifl]
    for g in list(free):
        if not pend:
            break
        p = pend.pop(0)
        launch(p, g); busy.add(g); free.remove(g)
        log(f"launched {p} on GPU{g}")
    if all(done(p) for p in PDBS):
        log("ALL_DONE expand-seeds scheduler")
        break
    time.sleep(20)
