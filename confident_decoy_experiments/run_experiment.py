#!/usr/bin/env python3
"""
Reproducible experiment runner for the confident-decoy project.

Each experiment is a YAML config in configs/. A config is fully self-describing:
motivation, method, generation params, provenance, and a `pipeline` of exact commands
(build -> generate -> score). This runner replays that pipeline.

Usage:
  python3 run_experiment.py configs/abonly_decoys.yaml            # run all stages
  python3 run_experiment.py configs/abonly_decoys.yaml --stage score
  python3 run_experiment.py configs/abonly_decoys.yaml --dry-run  # print commands only
  python3 run_experiment.py --list                                # list configs + status

Paths in a pipeline cmd that start with "scripts/" or "artifacts/" are resolved relative
to this experiments/ folder (durable copies). Everything else runs from `workdir` (the
scratch working dir) unless the stage sets its own `cwd`.
"""
import sys,os,subprocess,argparse,glob
try: import yaml
except ImportError: sys.exit("pip install pyyaml")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import CODE_ROOT, ARTIFACTS_DIR
EXP=CODE_ROOT  # kept as EXP for the pipeline-resolution code below

def resolve(tok):
    # scripts/ and configs/ moved into this repo; artifacts/ has not (Phase 2) and
    # still lives at the old standalone location -- see paths.py.
    if tok.startswith("artifacts/"): return os.path.join(str(ARTIFACTS_DIR), tok[len("artifacts/"):])
    if tok.startswith(("scripts/","configs/")): return os.path.join(EXP,tok)
    return tok

def load(cfg): return yaml.safe_load(open(cfg))

def list_configs():
    for f in sorted(glob.glob(f"{EXP}/configs/*.yaml")):
        c=load(f); print(f"{os.path.basename(f):40} [{c.get('status','?'):8}] result={ (c.get('result') or {}).get('value')}  :: {c.get('hypothesis','')[:60]}")

def run(cfg_path,stage=None,dry=False):
    c=load(cfg_path); wd=c.get("workdir",EXP)
    print(f"=== {c['id']} ({c.get('status')}) ===")
    print(f"motivation: {c.get('motivation','').strip().splitlines()[0] if c.get('motivation') else ''}")
    for st in c["pipeline"]:
        if stage and st["name"]!=stage: continue
        cmd=[resolve(t) for t in st["cmd"]]
        cwd=st.get("cwd",wd)
        log=os.path.join(EXP,"logs",f"{c['id']}_{st['name']}.log")
        print(f"\n-- stage '{st['name']}' (cwd={cwd})\n   {' '.join(cmd)}\n   log-> {log}")
        if dry: continue
        with open(log,"w") as lf:
            r=subprocess.run(cmd,cwd=cwd,stdout=lf,stderr=subprocess.STDOUT)
        print(f"   exit {r.returncode}")
        if r.returncode!=0: print("   STAGE FAILED - stopping"); return

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("config",nargs="?"); ap.add_argument("--stage"); ap.add_argument("--dry-run",action="store_true"); ap.add_argument("--list",action="store_true")
    a=ap.parse_args()
    if a.list or not a.config: list_configs()
    else: run(a.config,a.stage,a.dry_run)
