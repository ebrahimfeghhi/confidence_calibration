#!/usr/bin/env python
"""FULL-MSA run: same free (no-template) setup as the canonical pool, but with MSA row
subsampling DISABLED, so every forward pass sees the complete MSA.

Mechanism (README 6.9): protenix/model/utils.py::sample_indices draws
  sample_size = randint(low=min(lower_bound, n), high=n+1)
so setting lower_bound (= data.msa.min_size.test) >= n forces sample_size == n every pass.
We pass --data.msa.min_size.test 16384 (>= any MSA depth here; deepest is 8the at ~3260).
12 PDBs x 50 seeds (101-150) x 5 samples, seed-matched to every other run."""
import subprocess, time, os, json, glob, re

PY   = "/home/ebrahim/Protenix/.venv/bin/python"
PROT = "/home/ebrahim/Protenix"
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import EXP, CODE_ROOT
JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
DUMP = f"{EXP}/artifacts/fullmsa_out"
LOGD = f"{EXP}/artifacts/fullmsa_logs"
os.makedirs(DUMP, exist_ok=True); os.makedirs(LOGD, exist_ok=True)
GPUS  = [0, 1, 2, 3]
SEEDS = list(range(101, 151))
PDBS  = ["7z2m","8aok","8av2","8bf4","8che","8fas","8fbw","8q3j","8slb","8the","8ux6","8yxi"]

GENENV = dict(os.environ)
GENENV.update(LAYERNORM_TYPE="openfold", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
              MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")

def log(m):
    ts = time.strftime("%H:%M:%S")
    prog = {p: len(glob.glob(f"{DUMP}/{p}/seed_*/predictions/{p}_summary_confidence_sample_4.json")) for p in PDBS}
    json.dump({"t": ts, "msg": m, "done": prog}, open(f"{LOGD}/status.json","w"), indent=1)
    print(f"{ts} {m} | {prog}", flush=True)

def done(p):
    return len(glob.glob(f"{DUMP}/{p}/seed_*/predictions/{p}_summary_confidence_sample_4.json")) >= len(SEEDS)

def scan():
    busy=set(); ours={}
    for pid in os.listdir('/proc'):
        if not pid.isdigit(): continue
        try: cl=open(f'/proc/{pid}/cmdline','rb').read().replace(b'\0',b' ').decode()
        except Exception: continue
        if 'inference.py' not in cl: continue
        try: env=open(f'/proc/{pid}/environ','rb').read().replace(b'\0',b'\n').decode()
        except Exception: env=''
        g=re.search(r'CUDA_VISIBLE_DEVICES=(\d+)', env); gpu=int(g.group(1)) if g else -1
        if gpu>=0: busy.add(gpu)
        m=re.search(r'/([0-9a-z]+)-final-updated\.json', cl)
        if m and 'fullmsa_out' in cl: ours[m.group(1)]=gpu
    return busy, ours

def launch(pdb, gpu):
    env=dict(GENENV); env["CUDA_VISIBLE_DEVICES"]=str(gpu)
    subprocess.Popen(
        [PY,"runner/inference.py","--model_name","protenix-v2",
         "--load_checkpoint_dir","/home/ebrahim/checkpoint",
         "--input_json_path", f"{JDIR}/{pdb}-final-updated.json",
         "--dump_dir", DUMP, "--seeds", ",".join(map(str,SEEDS)),
         "--sample_diffusion.N_sample","5","--sample_diffusion.N_step","200",
         "--use_template","true",
         "--data.msa.min_size.test","16384"],          # <-- disables MSA row subsampling
        cwd=PROT, env=env, stdout=open(f"{LOGD}/{pdb}_g{gpu}.log","w"), stderr=subprocess.STDOUT)

log(f"fullmsa scheduler start; pdbs={PDBS}")
while True:
    busy, ours = scan()
    free=[g for g in GPUS if g not in busy]
    pend=[p for p in PDBS if not done(p) and p not in ours]
    for g in list(free):
        if not pend: break
        p=pend.pop(0); launch(p,g); busy.add(g); free.remove(g); log(f"launched {p} on GPU{g}")
    if all(done(p) for p in PDBS): log("ALL_DONE fullmsa"); break
    time.sleep(20)
