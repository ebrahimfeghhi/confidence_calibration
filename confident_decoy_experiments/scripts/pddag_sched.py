#!/usr/bin/env python
"""Run the 'pddag' experiment: best-pLDDT antigen fold as ONLY template (antibody freed),
15 confident-decoy interfaces (12 PDBs), seeds 101-150 x 5 samples, seed-matched to
free/agonly/native50. One PDB per GPU, dynamic refill. Detects ALL inference.py procs so it
never double-books a GPU another job (e.g. the 7z2m expand) is using. See README Methods."""
import subprocess, time, os, json, glob, re

PY   = "/home/ebrahim/Protenix/.venv/bin/python"
PROT = "/home/ebrahim/Protenix"
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import EXP, CODE_ROOT
DUMP = f"{EXP}/artifacts/pddag_out"
LOGD = f"{EXP}/artifacts/pddag_logs"
STATUS = f"{LOGD}/status.json"
os.makedirs(DUMP, exist_ok=True); os.makedirs(LOGD, exist_ok=True)

GPUS  = [0, 1, 2, 3]                # 5=john, 6-7=shawn: never touch
SEEDS = list(range(101, 151))       # seed-matched to free/agonly/native50
jobs  = json.load(open(f"{EXP}/artifacts/pddag_jobs.json"))
ALL   = [j["pdb"] for j in jobs]; JMAP = {j["pdb"]: j["json"] for j in jobs}

GENENV = dict(os.environ)
GENENV.update(LAYERNORM_TYPE="openfold", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
              MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")

def log(m):
    ts = time.strftime("%H:%M:%S")
    prog = {p: len(glob.glob(f"{DUMP}/{p}_pddag/seed_*/predictions/{p}_pddag_summary_confidence_sample_4.json")) for p in ALL}
    json.dump({"t": ts, "msg": m, "done": prog}, open(STATUS, "w"), indent=1)
    print(f"{ts} {m} | {prog}", flush=True)

def done(pdb):
    return len(glob.glob(f"{DUMP}/{pdb}_pddag/seed_*/predictions/{pdb}_pddag_summary_confidence_sample_4.json")) >= len(SEEDS)

def busy_gpus_and_ours():
    """Return (set of GPUs running ANY inference.py, {our pdb -> [gpus]})."""
    busy = set(); ours = {}
    for p in os.listdir('/proc'):
        if not p.isdigit(): continue
        try: cl = open(f'/proc/{p}/cmdline', 'rb').read().replace(b'\0', b' ').decode()
        except Exception: continue
        if 'inference.py' not in cl: continue
        try: env = open(f'/proc/{p}/environ', 'rb').read().replace(b'\0', b'\n').decode()
        except Exception: env = ''
        g = re.search(r'CUDA_VISIBLE_DEVICES=(\d+)', env)
        gpu = int(g.group(1)) if g else -1
        if gpu >= 0: busy.add(gpu)
        m = re.search(r'/([0-9a-z]+)-pddag\.json', cl)
        if m: ours.setdefault(m.group(1), []).append(gpu)
    return busy, ours

def launch(pdb, gpu):
    env = dict(GENENV); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.Popen(
        [PY, "runner/inference.py", "--model_name", "protenix-v2",
         "--load_checkpoint_dir", "/home/ebrahim/checkpoint",
         "--input_json_path", JMAP[pdb], "--dump_dir", DUMP,
         "--seeds", ",".join(map(str, SEEDS)),
         "--sample_diffusion.N_sample", "5", "--sample_diffusion.N_step", "200",
         "--use_template", "true"],
        cwd=PROT, env=env, stdout=open(f"{LOGD}/{pdb}_g{gpu}.log", "w"), stderr=subprocess.STDOUT)

log(f"pddag scheduler start; pdbs={ALL}; seeds 101-150 on GPUs {GPUS}")
while True:
    busy, ours = busy_gpus_and_ours()
    free = [g for g in GPUS if g not in busy]
    pend = [p for p in ALL if not done(p) and p not in ours]
    for g in list(free):
        if not pend: break
        p = pend.pop(0)
        launch(p, g); busy.add(g); free.remove(g)
        log(f"launched {p} on GPU{g}")
    if all(done(p) for p in ALL):
        log("ALL_DONE pddag scheduler"); break
    time.sleep(20)
