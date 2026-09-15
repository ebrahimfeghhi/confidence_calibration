#!/usr/bin/env python
"""Fold + control + 5 pose-arms for many seeds, with the model loaded ONCE.

WHY. ops/commit11.sh spawns a fresh `runner/inference.py` for every stage, and ~50s of each run is
process startup -- 39s of it reloading the 464M-parameter checkpoint. On 9s36 an arm does 10s of real
compute inside a 71s run, so roughly 85% of the wall-clock is overhead paid 7 times per seed. Loading
once and looping cuts a 25-seed run from ~4 hours to ~1.

HOW. runner.predict() calls self.model(...) WITHOUT passing N_cycle, so the model uses the N_cycle it
captured at construction (protenix.py:115) -- settable between stages as runner.model.N_cycle. The
rest of the per-stage configuration is either an env var read inside the forward
(PROTENIX_LOAD_TRUNK / DUMP_TRUNK / MULTICHAIN_MASK / TMPL_KEEP_FRAME), or read from configs per call
(mc_dropout_apply_rate, dump_dir, input_json_path, seeds). Template caps are read when the featuriser
is built, and infer_predict rebuilds the dataloader on every call, so those follow too.

Protenix's OWN featurisation is used throughout -- nothing is reconstructed by hand. Four earlier
attempts at this kind of driver failed precisely because they rebuilt input features from a dump.

PER SEED:
  fold     10 recycles, 5 poses, PROTENIX_DUMP_TRUNK -> z/s + the five donor poses
  control  1 recycle from that trunk, stock templates and default cross-chain mask: what recycle 11
           would have done anyway, so "committing to a pose" can be told apart from "one more recycle"
  p0..p4   1 recycle from the same trunk, with pose k as the ONLY template

Usage:
  commit11_fast.py --pdb 9x1h --iface 9x1h_EF --seeds 300000-300049 --out artifacts/c11fast
"""
import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, "/home/ebrahim/Protenix")
sys.path.insert(0, "/home/ebrahim/tts_abag/confident_decoy_experiments/scripts")

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
DC = "/home/ebrahim/tts_abag/diffusion_commit"
PY = "/home/ebrahim/Protenix/.venv/bin/python"

# Same DATASET contract as fold_and_commit.sh: inputs, scoring reference and scorer move together.
DATASETS = {
    "pxm2025":   (f"{EXP}/artifacts/pxm2025/jsons",
                  lambda pdb: f"{EXP}/artifacts/pxm2025/cif/{pdb}.cif", "pxm"),
    "foldbench": ("/home/ebrahim/Protenix/foldbench_abag/jsons",
                  lambda pdb: f"/home/ebrahim/FoldBench/examples/ground_truths/{pdb}-assembly1.cif",
                  "ost"),
}


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def build_runner(ckpt_dir, model_name, job_path, dump_dir):
    """Construct InferenceRunner exactly as runner/inference.py:run() does, then keep it."""
    from collections.abc import Mapping
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from protenix.config import parse_configs
    from runner.inference import InferenceRunner, model_configs, update_gpu_compatible_configs

    base = {**configs_base, **{"data": data_configs}, **inference_configs}

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d
    deep_update(base, model_configs[model_name])
    configs = parse_configs(configs=base, arg_str="", fill_required_with_null=True)
    configs.load_checkpoint_dir = ckpt_dir
    configs.model_name = model_name
    configs.input_json_path = job_path
    configs.dump_dir = dump_dir
    configs.use_template = True
    configs = update_gpu_compatible_configs(configs)
    return InferenceRunner(configs), configs


def named_job(src, name, out_dir):
    """Write a copy of a job json under a UNIQUE name.

    configs.dump_dir turned out not to be re-readable after the runner is built -- Protenix writes to
    <dump_dir at build time>/<job name>/seed_N/predictions. So stages are separated by NAME instead,
    which Protenix does honour, rather than by directory. Without this the fold and control (both
    named "<pdb>") and all five arms (all named "<pdb>_st") would overwrite each other.
    """
    obj = json.load(open(src))
    obj = obj if isinstance(obj, list) else [obj]
    for o in obj:
        o["name"] = name
    os.makedirs(out_dir, exist_ok=True)
    dst = f"{out_dir}/{name}.json"
    json.dump(obj, open(dst, "w"))
    return dst


class StageError(RuntimeError):
    """A stage produced the wrong number of poses. Raised rather than warned: a stage that silently
    yields nothing still lets the run continue and writes a plausible-looking but empty summary."""


def stage(runner, configs, job, seed, n_cycle, env, name, out_root, n_sample=5, expect=5):
    """One inference stage.

    env entries are set and restored around the call so stages cannot contaminate each other.
    ASSERTS the stage produced `expect` poses: a stage that silently produces nothing still lets the
    run continue and yields a plausible-looking but wrong summary, which has happened twice here.
    """
    from runner.inference import infer_predict
    old = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    configs.input_json_path = job
    configs.seeds = [seed]
    configs.sample_diffusion.N_sample = n_sample
    runner.model.N_cycle = n_cycle              # predict() uses the model's captured N_cycle
    t0 = time.time()
    infer_predict(runner, configs)
    dt = time.time() - t0
    for k, v in old.items():                     # restore, so stages cannot contaminate each other
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    n = len(glob.glob(f"{out_root}/{name}/**/*_sample_*.cif", recursive=True))
    print(f"    {name:<18} {n} poses  {dt:5.0f}s"
          f"{'' if n == expect else f'   !! EXPECTED {expect}, GOT {n}'}", flush=True)
    if n != expect:
        raise StageError(f"{name}: produced {n}/{expect} poses")
    return dt


TMPL_ENV = {"PROTENIX_MULTICHAIN_MASK": "0", "PROTENIX_TMPL_KEEP_FRAME": "1",
            "PROTENIX_TMPL_NODEDUP": "1", "PROTENIX_MAX_TEMPLATES": "1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--iface", required=True)
    ap.add_argument("--seeds", required=True, help="e.g. 300000-300049 or 300000,300005")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", default="/home/ebrahim/checkpoint")
    ap.add_argument("--model-name", default="protenix-v2")
    ap.add_argument("--keep-arm-cifs", action="store_true")
    # Default to the PROJECT csv, not a private one: report_complex.py, sweep_watch.sh and every
    # analysis read that file, and rows written elsewhere are invisible to all of them.
    ap.add_argument("--csv", default="/home/ebrahim/tts_abag/confident_decoy_experiments"
                                    "/artifacts/commit11/results.csv")
    ap.add_argument("--keep-trunk", action="store_true",
                    help="keep each seed's trunk dump; default discards it once the seed is armed, "
                         "since the trunk is only needed WHILE that seed's own arms run")
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="pxm2025")
    A = ap.parse_args()
    seeds = parse_seeds(A.seeds)
    JOBS, REFF, SCORER = DATASETS[A.dataset]
    REF = REFF(A.pdb)
    job0 = f"{JOBS}/{A.pdb}-final-updated.json"
    for f in (job0, REF):
        if not os.path.exists(f):
            raise SystemExit(f"missing input: {f}")
    os.makedirs(A.out, exist_ok=True)
    csv_path = A.csv
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as fh:
            fh.write("iface,seed,arm,donor_pose,sample,dockq,iptm\n")
    done = set()
    for r in csv.DictReader(open(csv_path)):
        # KEY ON THE INTERFACE TOO. Without it, pointing this driver at the shared project csv makes
        # every complex inherit another complex's completions: 8fas_BC seeds 1-51 already had rows for
        # donor/none/p0..p4, so 8vqf_AC seed 50 was judged finished before it started. It folded, skipped
        # all seven stages, recorded nothing, and exited looking healthy -- 50 seeds of GPU for no data.
        if r["iface"] == A.iface:
            done.add((r["seed"], r["arm"]))

    t_load = time.time()
    runner, configs = build_runner(A.ckpt_dir, A.model_name, job0, A.out)
    print(f"model loaded in {time.time()-t_load:.0f}s -- now reused for {len(seeds)} seeds", flush=True)

    for seed in seeds:
      try:
        # ONE naming scheme: {out}/s<seed>_<thing>. Protenix writes predictions to
        # <dump_dir at build time>/<job name>/, so the job names below put arm outputs in the same
        # namespace as the trunk and templates. Mixing two schemes is what hid the last bug.
        pre = f"{A.out}/s{seed}"
        trunk = f"{pre}_trunk.npz"
        t0 = time.time()

        if not os.path.exists(trunk):
            j = named_job(job0, f"s{seed}_donor", f"{A.out}/jobs")
            stage(runner, configs, j, seed, 10,
                  {"PROTENIX_DUMP_TRUNK": trunk, "PROTENIX_LOAD_TRUNK": None},
                  f"s{seed}_donor", A.out)
        if not os.path.exists(trunk):
            print(f"  seed {seed}: NO TRUNK -- skipping", flush=True)
            continue
        poses = sorted(glob.glob(f"{A.out}/s{seed}_donor/**/*_sample_*.cif", recursive=True))
        if len(poses) < 5:
            print(f"  seed {seed}: only {len(poses)} donor poses -- skipping", flush=True)
            continue

        if (str(seed), "none") not in done:
            j = named_job(job0, f"s{seed}_none", f"{A.out}/jobs")
            stage(runner, configs, j, seed, 1,
                  {"PROTENIX_LOAD_TRUNK": trunk, "PROTENIX_DUMP_TRUNK": None},
                  f"s{seed}_none", A.out)

        for i, pose in enumerate(poses):
            if (str(seed), f"p{i}") in done:
                continue
            T = f"{pre}_tmpl{i}"
            r = None                      # bound only if we actually shell out to the builder
            if not os.path.exists(f"{T}/job.json"):
                # SELFTMPL_JOBS must follow the dataset: the builder clones the base job json, and its
                # default is the pxm2025 pool. Without this a foldbench complex silently fails to build
                # every template and all five p-arms are skipped -- which is exactly what happened on
                # the first validation run, and it reported only as "template build failed".
                r = subprocess.run([PY, f"{DC}/build_selftmpl_general.py", "--pdb", A.pdb,
                                    "--out", T, "--poses", pose, "--tids", f"t{i}{seed % 100:02d}"],
                                   cwd=DC, capture_output=True, text=True,
                                   env={**os.environ, "SELFTMPL_JOBS": JOBS})
            if not os.path.exists(f"{T}/job.json"):
                print(f"  seed {seed} pose {i}: template build FAILED", flush=True)
                for line in ((r.stderr if r else "") or "").strip().splitlines()[-3:]:
                    print(f"      {line}", flush=True)
                continue
            env = dict(TMPL_ENV); env["PROTENIX_LOAD_TRUNK"] = trunk
            env["PROTENIX_DUMP_TRUNK"] = None
            j = named_job(f"{T}/job.json", f"s{seed}_p{i}", f"{A.out}/jobs")
            stage(runner, configs, j, seed, 1, env, f"s{seed}_p{i}", A.out)

        # score everything, then drop the arms' structures (DockQ/ipTM survive in the pxm jsons;
        # 50 seeds x 35 poses is otherwise a lot of disk on a quota this project has already filled)
        for arm, dn in [("donor", -1), ("none", -1)] + [(f"p{i}", i) for i in range(len(poses))]:
            d = f"{A.out}/s{seed}_{arm}"
            if not os.path.isdir(d) or (str(seed), arm) in done:
                continue
            out = subprocess.run([PY, f"{DC}/row.py", "--dir", d,
                                  "--iface", A.iface, "--seed", str(seed), "--arm", arm,
                                  "--donor", str(dn), "--ref", REF, "--scorer", SCORER],
                                 cwd=DC, capture_output=True, text=True)
            if out.stdout.strip():
                with open(csv_path, "a") as fh:
                    fh.write(out.stdout)
                done.add((str(seed), arm))
            if arm != "donor" and not A.keep_arm_cifs:
                for sub in glob.glob(f"{d}/*"):
                    if os.path.isdir(sub) and "_json" not in sub:
                        shutil.rmtree(sub, ignore_errors=True)
        # The trunk is only needed WHILE this seed's arms run -- each resumes from it via
        # PROTENIX_LOAD_TRUNK. Across 1000 seeds keeping them would refill the quota.
        if not A.keep_trunk and os.path.exists(trunk):
            os.remove(trunk)
        print(f"  seed {seed} done in {time.time()-t0:.0f}s  "
              f"(rows {sum(1 for _ in open(csv_path))-1})", flush=True)
      except StageError as e:
        # One bad seed must not abandon the remaining seeds of this complex.
        print(f"  !! seed {seed} ABORTED: {e}", flush=True)
        continue

    print("all seeds complete", flush=True)


if __name__ == "__main__":
    main()
