#!/usr/bin/env python
"""Run the label-bug re-runs: agonly / abonly / pddag for 8yxi + 8fbw with CORRECTED antigen labels.
6 jobs x 50 seeds (101-150) x 5 samples. One job per GPU (0-3), dynamic refill. Detects ALL
inference.py procs so it never double-books a GPU. See README 2 (label bug) / 6.2."""
import subprocess, time, os, json, glob, re

PY   = "/home/ebrahim/Protenix/.venv/bin/python"
PROT = "/home/ebrahim/Protenix"
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import EXP, CODE_ROOT
DUMP = f"{EXP}/artifacts/relabel_out"
LOGD = f"{EXP}/artifacts/relabel_logs"
os.makedirs(DUMP, exist_ok=True); os.makedirs(LOGD, exist_ok=True)
GPUS  = [0, 1, 2, 3]                 # 5=john, 6-7=shawn: never touch
SEEDS = list(range(101, 151))
jobsd = json.load(open(f"{EXP}/artifacts/relabel_jobs.json"))
JOBS  = [(f"{j['pdb']}_{run}", j["json"]) for run, lst in jobsd.items() for j in lst]

GENENV = dict(os.environ)
GENENV.update(LAYERNORM_TYPE="openfold", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
              MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")

def log(m):
    ts = time.strftime("%H:%M:%S")
    prog = {n: len(glob.glob(f"{DUMP}/{n}/seed_*/predictions/{n}_summary_confidence_sample_4.json")) for n, _ in JOBS}
    json.dump({"t": ts, "msg": m, "done": prog}, open(f"{LOGD}/status.json", "w"), indent=1)
    print(f"{ts} {m} | {prog}", flush=True)

def done(name):
    return len(glob.glob(f"{DUMP}/{name}/seed_*/predictions/{name}_summary_confidence_sample_4.json")) >= len(SEEDS)

def scan():
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
        m = re.search(r'relabel_tmpl/([0-9a-z]+)-(\w+)\.json', cl)
        if m: ours[f"{m.group(1)}_{m.group(2)}"] = gpu
    return busy, ours

def launch(name, jpath, gpu):
    env = dict(GENENV); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.Popen(
        [PY, "runner/inference.py", "--model_name", "protenix-v2",
         "--load_checkpoint_dir", "/home/ebrahim/checkpoint",
         "--input_json_path", jpath, "--dump_dir", DUMP,
         "--seeds", ",".join(map(str, SEEDS)),
         "--sample_diffusion.N_sample", "5", "--sample_diffusion.N_step", "200",
         "--use_template", "true"],
        cwd=PROT, env=env, stdout=open(f"{LOGD}/{name}_g{gpu}.log", "w"), stderr=subprocess.STDOUT)

log(f"relabel scheduler start; jobs={[n for n,_ in JOBS]}")
while True:
    busy, ours = scan()
    free = [g for g in GPUS if g not in busy]
    pend = [(n, p) for n, p in JOBS if not done(n) and n not in ours]
    for g in list(free):
        if not pend: break
        n, p = pend.pop(0)
        launch(n, p, g); busy.add(g); free.remove(g)
        log(f"launched {n} on GPU{g}")
    if all(done(n) for n, _ in JOBS):
        log("ALL_DONE relabel scheduler"); break
    time.sleep(20)
