#!/usr/bin/env python
"""RUN PHASE for self-templating: fold each arm against templates that were written beforehand.

This is deliberately read-only with respect to templates. write_templates.py creates the namespace --
every pose file, every id registration, every arm's job.json -- and this script only consumes it. A
fold therefore cannot write a template even by mistake, which is the hole that produced a day of bad
data:

  * ids keyed on an arm's internal pose index meant all five per-pose arms of a seed wrote ONE file;
    two GPUs on adjacent arms overwrote each other and the featuriser accepted the wrong pose silently,
    because a pose from the same complex still matches the query sequence
  * the shared template dir holds >10k 4-char ids, where collisions are near-certain by birthday bound
  * Protenix caches the release-date file per process, so ids appended mid-run were invisible to
    already-running workers and their hits were rejected as "date unknown"

WHAT A RUN PRODUCES

    <out>/self_template/<pdb>/<iface>/<seed>/<run_id>/
        <arm>/seed_<seed>/predictions/*.cif, *_summary_confidence_*.json
        run.json      everything needed to reproduce: precision, recycles, samples, every template's
                      id + source pose + md5, namespace, git sha, gpu, driver args

run_id is <utc timestamp>-<pid>-<hash of the defining inputs>, so folding one seed twice accumulates
instead of overwriting -- folds are not reproducible (identical config has differed by 0.5 DockQ), so
a repeat is data, not a duplicate.

GUARDS, all of which have caught a real bug:
  * template count consumed must equal the number handed in -- a silent drop fails the arm
  * pose count out must equal --nsample
  * a failed arm releases its claim so another worker retries it, instead of vanishing from the sweep

Usage:
    run_self_template.py --manifest <arms.tsv> --ns <namespace> --out <root> --csv <results.csv> \
        [--dtype fp32] [--ncycle 10] [--nsample 5] [--claim-dir <dir>] [--only all5|perpose]
    run_self_template.py --report --csv <results.csv> --manifest <arms.tsv>
"""
import argparse
import csv
import glob
import hashlib
import json
import logging
import os
import re
import socket
import subprocess
import sys
import time

sys.path.insert(0, "/home/ebrahim/Protenix")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit")

DC = "/home/ebrahim/tts_abag/diffusion_commit"
PY = "/home/ebrahim/Protenix/.venv/bin/python"
REPO = "/home/ebrahim/tts_abag/confident_decoy_experiments"
JOBS = f"{REPO}/artifacts/pxm2025/jsons"
REF = f"{REPO}/artifacts/pxm2025/cif"

# cross-chain template geometry is the entire point of self-templating: without these the template
# carries each chain's fold but nothing about how the chains sit relative to each other.
TMPL_ENV = {"PROTENIX_MULTICHAIN_MASK": "0", "PROTENIX_TMPL_KEEP_FRAME": "1",
            "PROTENIX_TMPL_NODEDUP": "1", "PROTENIX_MAX_TEMPLATE_CANDIDATES": "20"}

_TEMPLATE_COUNTS = []


class _CountHandler(logging.Handler):
    """Protenix reports the template count only to its logger, so capture it there to assert on."""

    def emit(self, record):
        try:
            m = record.getMessage()
        except Exception:
            return
        g = re.search(r"Found (\d+) templates", m)
        if g:
            _TEMPLATE_COUNTS.append(int(g.group(1)))


class ArmError(RuntimeError):
    pass


def build_runner(ckpt_dir, model_name, job_path, dump_dir, dtype):
    """Construct InferenceRunner as runner/inference.py:run() does, then keep it for every arm.

    Built once per worker: the checkpoint is 464M params and takes 40-80s to load, so per-fold
    construction would dominate the sweep.
    """
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
    if dtype:
        configs.dtype = dtype
    return InferenceRunner(configs), configs


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd="/home/ebrahim/tts_abag",
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def md5(path, n=12):
    return hashlib.md5(open(path, "rb").read()).hexdigest()[:n]


def run_id(defining):
    h = hashlib.md5(json.dumps(defining, sort_keys=True, default=str).encode()).hexdigest()[:6]
    return f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{os.getpid()}-{h}"


def named_job(src, name, out_dir):
    """Copy a job json under a unique NAME.

    Protenix derives the output directory from the job name, so two arms sharing a name would write
    into each other. The name is the only per-fold lever on that path.
    """
    obj = json.load(open(src))
    obj = obj if isinstance(obj, list) else [obj]
    for o in obj:
        o["name"] = name
    os.makedirs(out_dir, exist_ok=True)
    dst = f"{out_dir}/{name}.json"
    json.dump(obj, open(dst, "w"))
    return dst


def fold_arm(runner, configs, job, seed, name, n_cycle, n_sample, expect_templates, dump_base, env):
    """One fold. Asserts the templates were consumed and the poses were produced.

    expect_templates=None is the FREE (untemplated) case: whatever Protenix finds by its own search
    is legitimate, so the count is recorded rather than asserted. Every other arm must consume
    exactly the templates it was handed.
    """
    from runner.inference import infer_predict

    old = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, str(v))

    # Point the dumper at THIS fold's directory. runner/dumper.py builds the path as
    # <base_dir>/<dataset>/<job name>/seed_<seed>/predictions and captures base_dir when the runner is
    # constructed -- but it is a plain attribute, so setting it here gives the run-scoped layout
    # natively rather than moving files afterwards.
    saved_base = getattr(runner.dumper, "base_dir", None)
    os.makedirs(dump_base, exist_ok=True)
    runner.dumper.base_dir = dump_base
    # Protenix writes per-sample data errors to <dump>/ERR/<name>.txt without creating the dir, which
    # turns a recoverable error into FileNotFoundError and killed both workers twice.
    os.makedirs(f"{dump_base}/ERR", exist_ok=True)

    configs.input_json_path = job
    configs.seeds = [seed]
    configs.sample_diffusion.N_sample = n_sample
    runner.model.N_cycle = n_cycle
    _TEMPLATE_COUNTS.clear()
    t0 = time.perf_counter()
    try:
        infer_predict(runner, configs)
    finally:
        dt = time.perf_counter() - t0
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        if saved_base is not None:
            runner.dumper.base_dir = saved_base

    if not _TEMPLATE_COUNTS:
        raise ArmError("no template count reported -- featurisation did not run")
    if expect_templates is not None:
        bad = [c for c in _TEMPLATE_COUNTS if c != expect_templates]
        if bad:
            raise ArmError(f"templates consumed {sorted(set(_TEMPLATE_COUNTS))}, "
                           f"expected {expect_templates} on every chain")
    n = len(glob.glob(f"{dump_base}/{name}/**/*_sample_*.cif", recursive=True))
    if n != n_sample:
        raise ArmError(f"produced {n}/{n_sample} poses")
    return dt, sorted(set(_TEMPLATE_COUNTS))


def report(A):
    from fmt import table
    import statistics as st
    if not os.path.exists(A.csv):
        print("  no results yet")
        return 0
    arms = {}
    for r in csv.DictReader(open(A.csv)):
        arms.setdefault((r["iface"], r["seed"], r["arm"]), []).append(
            (float(r["iptm"]), float(r["dockq"])))
    rows = []
    for (iface, seed, arm), v in sorted(arms.items()):
        ip = [x for x, _ in v]
        dq = [q for _, q in v]
        rows.append([iface, seed, arm, len(v), st.mean(ip), max(v)[1], max(dq), st.mean(dq)])
    print(table(["interface", "seed", "arm", "poses", "arm ipTM", "sel DockQ (top ipTM)",
                 "best DockQ", "mean DockQ"], rows, align="l l l r r r r r",
                dec={"arm ipTM": 3, "sel DockQ (top ipTM)": 3, "best DockQ": 3, "mean DockQ": 3}))
    print(f"\n  arms scored: {len(rows)}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--ns", help="namespace built by write_templates.py (read-only here)")
    ap.add_argument("--out")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--claim-dir")
    ap.add_argument("--dtype", choices=("bf16", "fp32", "fp16"), default="fp32")
    ap.add_argument("--ncycle", type=int, default=10)
    ap.add_argument("--nsample", type=int, default=5)
    ap.add_argument("--only", choices=("all5", "perpose"), help="run just one arm class")
    ap.add_argument("--free", type=int, metavar="REPS",
                    help="untemplated arms instead of templated ones: fold each (iface, seed) REPS "
                         "times with stock settings. Measures the within-seed pose spread, and at "
                         "bf16 also checks this config reproduces the original pool.")
    ap.add_argument("--ckpt-dir", default="/home/ebrahim/checkpoint")
    ap.add_argument("--model-name", default="protenix-v2")
    ap.add_argument("--report", action="store_true")
    A = ap.parse_args()
    if A.report:
        return report(A)
    for req in ("manifest", "out"):
        if not getattr(A, req):
            raise SystemExit(f"--{req} is required")
    if not A.free and not A.ns:
        raise SystemExit("--ns is required unless --free")

    arms = []
    seen = set()
    for line in open(A.manifest):
        if not line.strip():
            continue
        iface, pdb, tag, seed, poses = line.rstrip("\n").split("\t")
        if A.free:
            # one free fold set per (interface, seed); the templated tags are irrelevant here
            if (iface, seed) in seen:
                continue
            seen.add((iface, seed))
            for rep in range(1, A.free + 1):
                arms.append((iface, pdb, f"free{A.dtype}r{rep}", seed, []))
            continue
        is_all = len(poses.split()) > 1
        if A.only == "all5" and not is_all:
            continue
        if A.only == "perpose" and is_all:
            continue
        arms.append((iface, pdb, tag, seed, poses.split()))
    if not arms:
        raise SystemExit("no arms selected")

    if not os.path.exists(A.csv):
        os.makedirs(os.path.dirname(A.csv) or ".", exist_ok=True)
        with open(A.csv, "w") as fh:
            fh.write("iface,seed,arm,donor_pose,sample,dockq,iptm\n")
    done = {(r["iface"], r["seed"], r["arm"]) for r in csv.DictReader(open(A.csv))}
    if A.claim_dir:
        os.makedirs(A.claim_dir, exist_ok=True)

    # the namespace supplies BOTH the template files and the release dates; nothing here writes to it.
    # A free arm must NOT see it: redirecting the mmcif dir would silently change which templates
    # Protenix can find, confounding the precision comparison this mode exists to make.
    mm = rd = None
    if not A.free:
        mm, rd = f"{A.ns}/mmcif", f"{A.ns}/release_dates.json"
        for p in (mm, rd):
            if not os.path.exists(p):
                raise SystemExit(f"namespace incomplete: {p} missing -- run write_templates.py first")
        os.environ["SELFTMPL_MMCIF_DIR"] = mm
        os.environ["SELFTMPL_RELEASE_DATES"] = rd
    else:
        for k in ("SELFTMPL_MMCIF_DIR", "SELFTMPL_RELEASE_DATES"):
            os.environ.pop(k, None)

    # runner.error_dir is derived from configs.dump_dir when the runner is BUILT, not from the
    # per-fold dump path, and Protenix writes data/OOM errors there without creating it -- so an
    # otherwise recoverable OOM became FileNotFoundError and killed the worker. Create it up front.
    os.makedirs(f"{A.out}/ERR", exist_ok=True)
    t_load = time.time()
    runner, configs = build_runner(A.ckpt_dir, A.model_name,
                                  f"{JOBS}/{arms[0][1]}-final-updated.json", A.out, A.dtype)
    if not A.free:
        configs.data.template.prot_template_mmcif_dir = mm
        configs.data.template.release_dates_path = rd
    logging.getLogger().addHandler(_CountHandler())
    print(f"  model loaded in {time.time()-t_load:.0f}s | dtype={A.dtype} | namespace={A.ns}",
          flush=True)

    ran = skipped = failed = 0
    for iface, pdb, tag, seed, poses in arms:
        arm = tag if A.free else f"tmpl_{tag}"
        if (iface, seed, arm) in done:
            skipped += 1
            continue
        if A.claim_dir:
            try:
                os.mkdir(f"{A.claim_dir}/{iface}_{tag}")
            except FileExistsError:
                continue
        if A.free:
            job_src, tids, env = f"{JOBS}/{pdb}-final-updated.json", [], {}
        else:
            job_src = f"{A.ns}/jobs/{iface}_{tag}/job.json"
            tids = [x["tid"] for x in json.load(open(f"{A.ns}/jobs/{iface}_{tag}/chosen.json"))]
            env = dict(TMPL_ENV, PROTENIX_MAX_TEMPLATES=str(len(poses)))
        if not os.path.exists(job_src):
            print(f"  {iface} {tag}: NO PREBUILT JOB at {job_src}", flush=True)
            failed += 1
            _release(A, iface, tag)
            continue
        rid = run_id({"iface": iface, "seed": seed, "tag": tag, "dtype": A.dtype,
                      "ncycle": A.ncycle, "nsample": A.nsample, "tids": tids})
        rdir = os.path.join(A.out, "free" if A.free else "self_template",
                            pdb, iface, str(seed), rid)
        name = f"{iface}_{tag}"
        job = named_job(job_src, name, f"{rdir}/jobs")
        try:
            dt, tcounts = fold_arm(runner, configs, job, int(seed), name, A.ncycle, A.nsample,
                                   None if A.free else len(poses), rdir, env)
        except ArmError as e:
            print(f"  {iface} {tag}: FAILED -- {e}", flush=True)
            failed += 1
            _release(A, iface, tag)
            continue

        json.dump({
            "run_id": rid,
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": "free" if A.free else "self_template",
            "pdb": pdb, "iface": iface, "seed": seed, "arm": arm,
            "precision": A.dtype, "n_cycle": A.ncycle, "n_sample": A.nsample,
            "fold_seconds": round(dt, 1),
            "templates": [{"tid": t, "source_pose": p, "md5": md5(p)} for t, p in zip(tids, poses)],
            "templates_found_by_search": tcounts if A.free else None,
            "template_namespace": None if A.free else A.ns,
            "host": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "layernorm_type": os.environ.get("LAYERNORM_TYPE"),
            "git_sha": git_sha(),
            "driver_args": " ".join(sys.argv[1:]),
        }, open(f"{rdir}/run.json", "w"), indent=1)

        out = subprocess.run([PY, f"{DC}/row.py", "--dir", f"{rdir}/{name}", "--iface", iface,
                              "--seed", str(seed), "--arm", arm, "--donor", "-1",
                              "--ref", f"{REF}/{pdb}.cif", "--scorer", "pxm"],
                             cwd=DC, capture_output=True, text=True)
        if out.stdout.strip():
            with open(A.csv, "a") as fh:
                fh.write(out.stdout)
            ran += 1
            print(f"  {iface} {tag} seed {seed}: scored in {dt:.0f}s -> {rid}", flush=True)
        else:
            print(f"  {iface} {tag}: SCORING produced no rows", flush=True)
            for line in (out.stderr or "").strip().splitlines()[-3:]:
                print(f"      {line}", flush=True)
            failed += 1
            _release(A, iface, tag)
    print(f"complete: {ran} arms run, {skipped} already scored, {failed} failed", flush=True)
    return 0


def _release(A, iface, tag):
    """A failed arm must give its claim back or it silently disappears from the sweep."""
    if A.claim_dir:
        try:
            os.rmdir(f"{A.claim_dir}/{iface}_{tag}")
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
