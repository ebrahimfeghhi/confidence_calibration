"""Instrument Protenix inference to log per-seed MSA subsample size.
Monkeypatches sample_indices (logs [seed, n_total, sample_size] per MSA-module call = per recycle) and
seed_everything (tracks current seed). Env: MSA_JSON, MSA_SEEDS, MSA_DUMP, MSA_PROBE_LOG."""
import sys, os, json
JSON = os.environ["MSA_JSON"]; SEEDS = os.environ["MSA_SEEDS"]
DUMP = os.environ["MSA_DUMP"]; LOGFILE = os.environ["MSA_PROBE_LOG"]
sys.argv = ["inference.py", "--model_name", "protenix-v2",
            "--load_checkpoint_dir", "/home/ebrahim/checkpoint",
            "--input_json_path", JSON, "--dump_dir", DUMP, "--seeds", SEEDS,
            "--sample_diffusion.N_sample", os.environ.get("MSA_NSAMPLE", "1"),
            "--sample_diffusion.N_step", os.environ.get("MSA_NSTEP", "10"),
            "--model.N_cycle", os.environ.get("MSA_NCYCLE", "3"),
            "--use_template", "true"]

import protenix.model.utils as U
import runner.inference as INF

_cur = {"seed": None}; LOG = []
_os = INF.seed_everything
def _sp(*a, **k):
    _cur["seed"] = int(k["seed"]) if "seed" in k else int(a[0])
    return _os(*a, **k)
INF.seed_everything = _sp

_osi = U.sample_indices
def _si(*a, **k):
    idx = _osi(*a, **k)
    n = a[0] if a else k.get("n")
    try: LOG.append([_cur["seed"], int(n), int(idx.numel())])
    except Exception: pass
    return idx
U.sample_indices = _si

try:
    INF.run()
finally:
    json.dump(LOG, open(LOGFILE, "w"))
    print(f"MSA_PROBE_WROTE {LOGFILE} entries={len(LOG)}", flush=True)
