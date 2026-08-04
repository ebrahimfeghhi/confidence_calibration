#!/usr/bin/env python
"""Full-MSA run, SEED-PARALLEL ordering: finish one interface at a time by splitting its seeds
across all 4 GPUs, so every interface reaches N=16 quickly (instead of 50 seeds serially per PDB).

Phase 1: seeds 101-116 (N=16) for all 12 PDBs, PDB-by-PDB, 4 chunks -> 4 GPUs.
MSA subsampling disabled via --data.msa.min_size.test 16384 (see README 6.9).
Already-completed seeds on disk are skipped."""
import subprocess, time, os, json, glob

PY="/home/ebrahim/Protenix/.venv/bin/python"; PROT="/home/ebrahim/Protenix"
EXP="/home/ebrahim/confident_decoy_experiments"
JDIR="/home/ebrahim/Protenix/foldbench_abag/jsons"
DUMP=f"{EXP}/artifacts/fullmsa_out"; LOGD=f"{EXP}/artifacts/fullmsa_logs"
os.makedirs(DUMP,exist_ok=True); os.makedirs(LOGD,exist_ok=True)
GPUS=[0,1,2,3]
TARGET=list(range(101,117))                      # N=16
# fastest / smallest first so coverage lands early
ORDER=["8fas","8ux6","8fbw","8q3j","7z2m","8slb","8yxi","8che","8av2","8aok","8bf4","8the"]
GENENV=dict(os.environ)
GENENV.update(LAYERNORM_TYPE="openfold",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1",
              MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1")

def log(m):
    ts=time.strftime("%H:%M:%S")
    prog={p:len([s for s in TARGET if os.path.exists(f"{DUMP}/{p}/seed_{s}/predictions/{p}_summary_confidence_sample_4.json")]) for p in ORDER}
    json.dump({"t":ts,"msg":m,"n16_progress":prog},open(f"{LOGD}/n16_status.json","w"),indent=1)
    print(f"{ts} {m} | {prog}",flush=True)

def missing(p):
    return [s for s in TARGET if not os.path.exists(f"{DUMP}/{p}/seed_{s}/predictions/{p}_summary_confidence_sample_4.json")]

def launch(p, seeds, gpu):
    env=dict(GENENV); env["CUDA_VISIBLE_DEVICES"]=str(gpu)
    return subprocess.Popen(
        [PY,"runner/inference.py","--model_name","protenix-v2",
         "--load_checkpoint_dir","/home/ebrahim/checkpoint",
         "--input_json_path",f"{JDIR}/{p}-final-updated.json",
         "--dump_dir",DUMP,"--seeds",",".join(map(str,seeds)),
         "--sample_diffusion.N_sample","5","--sample_diffusion.N_step","200",
         "--use_template","true","--data.msa.min_size.test","16384"],
        cwd=PROT,env=env,stdout=open(f"{LOGD}/n16_{p}_g{gpu}.log","w"),stderr=subprocess.STDOUT)

log(f"N=16 seed-parallel scheduler start; order={ORDER}")
for p in ORDER:
    need=missing(p)
    if not need:
        log(f"{p} already has N=16 - skip"); continue
    chunks=[need[i::len(GPUS)] for i in range(len(GPUS))]
    procs=[]
    for g,ch in zip(GPUS,chunks):
        if ch: procs.append(launch(p,ch,g))
    log(f"{p}: launched {len(need)} seeds across {len([c for c in chunks if c])} GPUs")
    for pr in procs: pr.wait()
    log(f"{p}: DONE N=16 ({len(TARGET)-len(missing(p))}/16 on disk)")
log("ALL_DONE phase1 N=16")
