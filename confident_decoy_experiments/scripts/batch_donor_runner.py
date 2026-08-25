#!/usr/bin/env python
"""Run many pre-built donor JSONs in ONE process, loading the model once.

WHY. sweep_sched.py launches runner/inference.py once per donor: 250 processes, each loading the
model (~36 s) to do 5-10 seeds of work (~40-80 s). At 5 seeds that is roughly half the wall-clock
spent loading. Loading once for all donors on a GPU cuts an arm nearly in half.

WHY NOT runner/batch_inference.py. It does the same loop but calls preprocess_input() on every JSON,
which runs the MSA/template search machinery and requires kalign (not installed). Our JSONs are
already fully built -- templatesPath, MSA paths and modelSeeds are set by build_allpose_generic.py --
so preprocessing is exactly what we must skip.

SEEDS STAY PER DONOR. infer_predict() reads modelSeeds from json_data[0] of the file it is given, and
each of our files holds one donor, so every donor keeps its own seed list. Nothing about the
experiment changes; only the process boundary moves.

Usage:
  .../python scripts/batch_donor_runner.py --jsons <dir-or-file-list> --dump-dir <out> [--n-sample 5]
"""
import argparse, glob, json, os, sys, time

sys.path.insert(0, "/home/ebrahim/Protenix")
from collections.abc import Mapping
from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from protenix.config.config import parse_configs
from runner.inference import InferenceRunner, infer_predict, update_gpu_compatible_configs

ap = argparse.ArgumentParser()
ap.add_argument("--jsons", required=True, help="directory of donor JSONs, or a file listing paths")
ap.add_argument("--dump-dir", required=True)
ap.add_argument("--n-sample", type=int, default=5)
ap.add_argument("--n-step", type=int, default=200)
ap.add_argument("--model-name", default="protenix-v2")
ap.add_argument("--checkpoint", default="/home/ebrahim/checkpoint")
A = ap.parse_args()

if os.path.isdir(A.jsons):
    files = sorted(glob.glob(os.path.join(A.jsons, "*.json")))
else:
    files = [l.strip() for l in open(A.jsons) if l.strip()]
if not files:
    sys.exit(f"no JSONs found at {A.jsons}")

# Seeds come from each file's modelSeeds (use_seeds_in_json=True), so the list passed here is only a
# fallback for a file that lacks the field -- which would be a build error worth failing loudly on.
missing = [f for f in files if not (json.load(open(f))[0].get("modelSeeds"))]
if missing:
    sys.exit(f"{len(missing)} JSONs have no modelSeeds, e.g. {missing[0]} -- rebuild with seeds embedded")

# Build configs exactly as runner/inference.py's CLI does (two-pass parse + model-specific merge).
# get_default_runner() is NOT used: it hard-requires a kalign binary whenever use_template=True, and
# kalign is not installed here -- our pipeline never needs it because the templates are pre-built.
arg_str = (f"--model_name {A.model_name} --load_checkpoint_dir {A.checkpoint} "
           f"--input_json_path {files[0]} --dump_dir {A.dump_dir} "
           f"--seeds 101 --use_template true --use_seeds_in_json true "
           f"--sample_diffusion.N_sample {A.n_sample} --sample_diffusion.N_step {A.n_step}")
cfg = parse_configs(configs={**configs_base, **{"data": data_configs}, **inference_configs},
                    arg_str=arg_str, fill_required_with_null=True)
base = {**configs_base, **{"data": data_configs}, **inference_configs}
def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping): deep_update(d[k], v)
        else: d[k] = v
    return d
deep_update(base, model_configs[cfg.model_name])
configs = parse_configs(configs=base, arg_str=arg_str, fill_required_with_null=True)
configs = update_gpu_compatible_configs(configs)

t0 = time.time()
runner = InferenceRunner(configs)          # <-- the one model load
os.makedirs(A.dump_dir, exist_ok=True)
print(f"[batch] model loaded once in {time.time()-t0:.1f} s; {len(files)} donors to run", flush=True)

ok = fail = 0
for i, f in enumerate(files, 1):
    configs["input_json_path"] = f
    t1 = time.time()
    try:
        infer_predict(runner, configs)
        ok += 1
    except Exception as e:
        fail += 1
        print(f"[batch] FAILED {os.path.basename(f)}: {e}", flush=True)
    print(f"[batch] {i}/{len(files)} {os.path.basename(f)} in {time.time()-t1:.1f} s", flush=True)
print(f"[batch] done: {ok} ok, {fail} failed, {time.time()-t0:.1f} s total", flush=True)
