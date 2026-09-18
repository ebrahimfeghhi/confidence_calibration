#!/usr/bin/env python
"""THE driver. Folds a queue of work units with the checkpoint loaded once, scores each, records
provenance. Replaces fast_driver.py and self_template/run_self_template.py, which were two
implementations of this same harness.

WHY ONE DRIVER. The two it replaces each had their own copy of the claim logic, the model-once loop,
the output paths and the error handling. Every bug therefore had to be found and fixed twice, and the
missing-ERR-directory crash that killed workers three times was fixed separately in each. One harness
means one place for those to live.

THE ONLY REAL DISTINCTION between work units is how many of OUR OWN poses go into one fold:

    free      0 self-templated poses    one plain fold
    all5      N poses -> 1 fold         all of them as the template, averaged by TemplateEmbedder
    perpose   N poses -> N folds        one pose each, so the arms can be compared

"free" MEANS ZERO SELF-TEMPLATED POSES, NOT ZERO TEMPLATES. Protenix's own template search still runs
on a free fold and supplies ~4 real PDB templates per chain -- confirmed in the logs as "Found 4
templates for sequence ...". So the contrast these units support is "PDB templates" versus "PDB
templates replaced by our own poses", never "templated" versus "untemplated". Reading `free` as
template-free would misattribute the whole self-templating effect.

There is deliberately no "gen2" kind. Templating on a previous generation's poses is just `all5` with
pose paths that point somewhere else; where a pose came from is not the driver's business.

THERE IS NO TRUNK RESUME. Loading a dumped recurrent state and paying one further recycle instead of
folding from scratch was measured to be unusable: with dropout off, the resumed state was 94% as wrong
as skipping the cycle entirely, and the cause was never found. Every result produced that way is void.
It is absent here rather than deprecated, so it cannot be reached by accident.

WRITE PHASE FIRST. Templated units consume a namespace that write_templates.py has already built --
every pose file, every 4-char id, and a COMPLETE release_dates.json. This ordering is load-bearing,
not tidiness: Protenix caches release dates once per process (template_utils.py:838), so an id
registered mid-run is rejected as "date unknown" and every template is silently dropped while the
fold still completes and scores normally. Ids keyed on anything but the pose file's own content also
collided -- all five per-pose arms of a seed took index 0 and wrote one file, and because
same-complex sequences still match, the featuriser accepted the wrong pose with no error.

QUEUE FILE -- one unit per line, tab separated:

    kind    pdb     target      tag       seeds           poses
    free    8vqf    *           free100   300050-300099
    all5    9vca    9vca_AC     s20all    300020          /abs/p0.cif /abs/p1.cif ...
    perpose 9vca    9vca_AC     s20       300020          /abs/p0.cif /abs/p1.cif ...

  target  the interface to score, or `*` for every interface that pdb contributes to --pool. A fold
          produces the whole complex, so scoring one interface when the pdb has three would either
          discard work already done or refold identical coordinates.
  tag     names the arm in the csv and, for templated units, the write-phase job directory.
  seeds   `300020` or `300050-300099`.

Usage:
    fold.py --queue q.tsv --out <root> --csv <results.csv> [--templates <ns>] [--dtype bf16]
    fold.py --report --csv <results.csv>
"""
import argparse
import collections
import csv
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, "/home/ebrahim/Protenix")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit")
sys.path.insert(0, "/home/ebrahim/tts_abag")

import paths     # noqa: E402  every location comes from here, never a literal
import runmeta   # noqa: E402

DC = str(paths.CODE_DIR)
PY = str(paths.PROTENIX_PYTHON)

# Inputs, ground truth and scorer move together -- a foldbench complex scored with the pxm scorer
# silently produces nothing -- so they come as one triple from paths.dataset(), not assembled here.

# Cross-chain template geometry is the entire point of self-templating: without these the template
# carries each chain's fold but nothing about how the chains sit relative to each other.
TMPL_ENV = {"PROTENIX_MULTICHAIN_MASK": "0", "PROTENIX_TMPL_KEEP_FRAME": "1",
            "PROTENIX_TMPL_NODEDUP": "1", "PROTENIX_MAX_TEMPLATE_CANDIDATES": "20"}

KINDS = ("free", "all5", "perpose")
_TEMPLATE_COUNTS = []


class ArmError(RuntimeError):
    """A unit failed in a way that must not be silently skipped."""


class _CountHandler(logging.Handler):
    """Protenix reports how many templates it actually used only to its logger, so capture it there.

    Without this the number lives in a log file and nowhere the code can assert on, which is exactly
    how 20 arms ran with 3 templates while the driver believed it had handed them 5.
    """

    def emit(self, record):
        try:
            m = record.getMessage()
        except Exception:
            return
        g = re.search(r"Found (\d+) templates", m)
        if g:
            _TEMPLATE_COUNTS.append(int(g.group(1)))


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def build_runner(ckpt_dir, model_name, job_path, dump_dir, dtype):
    """Construct InferenceRunner as runner/inference.py:run() does, then keep it for every unit.

    The checkpoint is 464M parameters and takes 50-80s to load, so building one per fold would
    dominate any sweep -- on a 123-complex queue that is ~2.7 GPU-hours of pure overhead.
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
    if dtype is not None:
        configs.dtype = dtype
    configs = update_gpu_compatible_configs(configs)
    return InferenceRunner(configs), configs


def named_job(src, name, out_dir):
    """Copy a job json under a unique NAME.

    Protenix derives its output directory from the job name, so two units sharing a name write into
    each other. configs.dump_dir is NOT re-read after the runner is built, so the name is the only
    per-fold lever on that path.
    """
    obj = json.load(open(src))
    obj = obj if isinstance(obj, list) else [obj]
    for o in obj:
        o["name"] = name
    os.makedirs(out_dir, exist_ok=True)
    dst = f"{out_dir}/{name}.json"
    json.dump(obj, open(dst, "w"))
    return dst


def fold(runner, configs, job, seed, name, n_cycle, n_sample, expect_templates, dump_base, env):
    """One fold. Asserts what it produced, because a stage that silently yields nothing still lets
    the run continue and writes a plausible-looking but empty summary -- that has happened twice.

    expect_templates=None means a free unit: it has no self-templated poses, so there is no count to
    assert -- Protenix's own search still returns ~4 real PDB templates per chain and that number is
    recorded, not checked. Any other value is the number of our poses and must be matched on EVERY
    chain; a chain that quietly fell back to fewer is the silent-wrong-template failure.
    """
    from runner.inference import infer_predict

    old = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, str(v))

    # Point the dumper at THIS fold's directory. runner/dumper.py builds the path as
    # <base_dir>/<dataset>/<job name>/seed_<seed>/predictions and captures base_dir when the runner
    # is constructed -- but it is a plain attribute, so setting it here gives the run-scoped layout
    # natively instead of moving files afterwards.
    saved_base = getattr(runner.dumper, "base_dir", None)
    os.makedirs(dump_base, exist_ok=True)
    # Protenix writes per-sample data/OOM errors to <dump>/ERR/<name>.txt WITHOUT creating the
    # directory, which turns a recoverable OOM into FileNotFoundError and killed workers three times.
    os.makedirs(f"{dump_base}/ERR", exist_ok=True)
    runner.dumper.base_dir = dump_base

    configs.input_json_path = job
    configs.seeds = [seed]
    configs.sample_diffusion.N_sample = n_sample
    runner.model.N_cycle = n_cycle          # predict() uses the N_cycle the model captured
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
            raise ArmError(f"templates consumed {sorted(set(_TEMPLATE_COUNTS))}, expected "
                           f"{expect_templates} on every chain")
    n = len(glob.glob(f"{dump_base}/{name}/**/*_sample_*.cif", recursive=True))
    if n != n_sample:
        raise ArmError(f"produced {n}/{n_sample} poses")
    return dt, sorted(set(_TEMPLATE_COUNTS))


def read_queue(path):
    """Parse the queue into (kind, pdb, target, tag, seed, poses) units, one per fold.

    perpose lines expand here, so the rest of the driver never has to care: one line in, N units out,
    each with its own tag so their outputs and csv rows cannot collide.
    """
    units = []
    for lineno, raw in enumerate(open(path), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        f = raw.rstrip("\n").split("\t")
        if len(f) < 5:
            raise SystemExit(f"{path}:{lineno}: need >=5 tab-separated fields, got {len(f)}")
        kind, pdb, target, tag, seedspec = f[:5]
        poses = f[5].split() if len(f) > 5 and f[5].strip() else []
        if kind not in KINDS:
            raise SystemExit(f"{path}:{lineno}: kind must be one of {KINDS}, got {kind!r}")
        if kind == "free" and poses:
            raise SystemExit(f"{path}:{lineno}: free units take no poses")
        if kind != "free" and not poses:
            raise SystemExit(f"{path}:{lineno}: {kind} needs pose paths")
        for p in poses:
            if not os.path.isabs(p):
                raise SystemExit(f"{path}:{lineno}: pose paths must be absolute ({p})")
        for seed in parse_seeds(seedspec):
            if kind == "perpose":
                for i, p in enumerate(poses):
                    units.append(("perpose", pdb, target, f"{tag}p{i}", seed, [p]))
            else:
                units.append((kind, pdb, target, tag, seed, poses))
    if not units:
        raise SystemExit(f"{path}: no units")
    return units


def reconcile_claims(claim_dir, csv_path, pool_ifaces):
    """Release claims whose work never finished, so a killed worker's units are retried.

    A claim is taken before a fold and released only on failure, so a worker killed mid-fold leaves
    one behind -- and a claimed-but-unscored unit is skipped FOREVER, silently shrinking the sweep.
    This ran as an ad-hoc script twice and I got the parsing wrong the second time (claim names had
    changed shape between drivers, so it matched nothing and reported success). It belongs here, run
    on every start, keyed only on the first and last name components so any naming scheme works.
    """
    if not claim_dir or not os.path.isdir(claim_dir):
        return 0
    scored = collections.defaultdict(set)
    for r in csv.DictReader(open(csv_path)):
        scored[(r["iface"].split("_")[0], r["seed"])].add(r["iface"])
    released = 0
    for c in os.listdir(claim_dir):
        parts = c.split("_")
        pdb, seed = parts[0], parts[-1]
        want = pool_ifaces.get(pdb)
        if want and not want <= scored.get((pdb, seed), set()):
            try:
                os.rmdir(f"{claim_dir}/{c}")
                released += 1
            except OSError:
                pass
    if released:
        print(f"  released {released} stale claims (killed mid-fold, now retryable)", flush=True)
    return released


def report(csv_path):
    sys.path.insert(0, DC)
    from fmt import table
    import statistics as st
    arms = collections.defaultdict(list)
    for r in csv.DictReader(open(csv_path)):
        arms[(r["iface"], r["seed"], r["arm"])].append((float(r["iptm"]), float(r["dockq"])))
    rows = []
    for (iface, seed, arm), v in sorted(arms.items()):
        ip = [x for x, _ in v]
        dq = [d for _, d in v]
        # SELECTED = the pose this arm's own confidence picks, which is the number that matters; the
        # oracle max is reported beside it only to show the headroom.
        rows.append([iface, seed, arm, len(v), st.mean(ip), max(v)[1], max(dq)])
    print(table(["interface", "seed", "arm", "poses", "arm ipTM", "selected DockQ", "best DockQ"],
                rows, align="l l l r r r r",
                dec={"arm ipTM": 3, "selected DockQ": 3, "best DockQ": 3}))
    print(f"\n  arms scored: {len(rows)}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue")
    ap.add_argument("--out")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--templates", help="namespace built by write_templates.py; required for "
                                        "templated units, never written to here")
    ap.add_argument("--pool", default=str(paths.PXM2025_POOL_CSV),
                    help="csv whose interfaces define what `target=*` scores")
    ap.add_argument("--claim-dir", help="atomic mkdir claims so workers share one queue")
    ap.add_argument("--dtype", choices=("bf16", "fp32", "fp16"), default="bf16")
    ap.add_argument("--ncycle", type=int, default=10)
    ap.add_argument("--nsample", type=int, default=5)
    ap.add_argument("--dataset", choices=sorted(paths.DATASET_NAMES), default="pxm2025")
    ap.add_argument("--ckpt-dir", default=str(paths.PROTENIX_CKPT))
    ap.add_argument("--model-name", default="protenix-v2")
    ap.add_argument("--keep-cifs", action="store_true", default=True,
                    help="poses are kept by default: a pose whose score is in the csv but whose "
                         "coordinates are gone cannot be templated on later, and folds do not "
                         "reproduce")
    ap.add_argument("--drop-cifs", dest="keep_cifs", action="store_false")
    ap.add_argument("--report", action="store_true")
    A = ap.parse_args()
    if A.report:
        return report(A.csv)
    for req in ("queue", "out"):
        if not getattr(A, req):
            raise SystemExit(f"--{req} is required")

    units = read_queue(A.queue)
    JOBS, REFF, SCORER = paths.dataset(A.dataset)
    templated = [u for u in units if u[0] != "free"]
    if templated and not A.templates:
        raise SystemExit(f"{len(templated)} templated units need --templates "
                         "(run write_templates.py first)")

    # The namespace supplies BOTH the template files and the release dates. Free units must NOT see
    # it: redirecting the mmcif dir would change which templates Protenix can find, confounding any
    # comparison against a templated arm.
    mm = rd = None
    if A.templates:
        mm, rd = f"{A.templates}/mmcif", f"{A.templates}/release_dates.json"
        for p in (mm, rd):
            if not os.path.exists(p):
                raise SystemExit(f"namespace incomplete: {p} missing -- run write_templates.py")

    # interfaces per pdb, for target=*
    pool_ifaces = collections.defaultdict(set)
    if any(u[2] == "*" for u in units):
        for r in csv.DictReader(open(A.pool)):
            pool_ifaces[r["pdb"]].add(f"{r['pdb']}_{r['chain1']}{r['chain2']}")

    os.makedirs(A.out, exist_ok=True)
    if not os.path.exists(A.csv):
        os.makedirs(os.path.dirname(A.csv) or ".", exist_ok=True)
        with open(A.csv, "w") as fh:
            fh.write("iface,seed,arm,donor_pose,sample,dockq,iptm\n")
    done = {(r["iface"], r["seed"], r["arm"]) for r in csv.DictReader(open(A.csv))}
    if A.claim_dir:
        os.makedirs(A.claim_dir, exist_ok=True)
        reconcile_claims(A.claim_dir, A.csv, pool_ifaces)

    # runner.error_dir is derived from configs.dump_dir when the runner is BUILT, not per fold.
    os.makedirs(f"{A.out}/ERR", exist_ok=True)
    t_load = time.time()
    runner, configs = build_runner(A.ckpt_dir, A.model_name,
                                   f"{JOBS}/{units[0][1]}-final-updated.json", A.out, A.dtype)
    # WHERE TEMPLATES COME FROM IS PER UNIT, NOT PER RUN. Setting the namespace once at startup
    # pointed free units at our private 65-file directory too, so they found ZERO real PDB templates
    # instead of the ~4 they should -- silently stripping templates from the very arm the templated
    # ones are compared against. Caught because run.json records templates_found. The stock paths are
    # captured here and restored for every free unit.
    STOCK_MM = configs.data.template.prot_template_mmcif_dir
    STOCK_RD = configs.data.template.release_dates_path

    def point_templates_at(use_ns):
        """Switch the template source for the unit about to run."""
        m, r = (mm, rd) if use_ns else (STOCK_MM, STOCK_RD)
        configs.data.template.prot_template_mmcif_dir = m
        configs.data.template.release_dates_path = r
        for k, v in (("SELFTMPL_MMCIF_DIR", m), ("SELFTMPL_RELEASE_DATES", r)):
            os.environ.pop(k, None) if (v is None or not use_ns) else os.environ.__setitem__(k, v)

    logging.getLogger().addHandler(_CountHandler())
    print(f"  model loaded in {time.time()-t_load:.0f}s | dtype={A.dtype} | {len(units)} units "
          f"| templates={A.templates}", flush=True)

    ran = skipped = failed = 0
    for kind, pdb, target, tag, seed, poses in units:
        want = sorted(pool_ifaces.get(pdb, ())) if target == "*" else [target]
        if not want:
            print(f"  {pdb}: no interfaces to score, skipping", flush=True)
            continue
        if all((i, str(seed), tag) in done for i in want):
            skipped += 1
            continue
        claim = f"{A.claim_dir}/{pdb}_{target}_{tag}_{seed}".replace("*", "all") \
            if A.claim_dir else None
        if claim:
            try:
                os.mkdir(claim)
            except FileExistsError:
                continue

        def release():
            """A failed unit must give its claim back or it silently vanishes from the sweep."""
            if claim:
                try:
                    os.rmdir(claim)
                except OSError:
                    pass

        point_templates_at(kind != "free")
        if kind == "free":
            job_src, env, expect = f"{JOBS}/{pdb}-final-updated.json", {}, None
        else:
            job_src = f"{A.templates}/jobs/{target}_{tag}/job.json"
            env = dict(TMPL_ENV, PROTENIX_MAX_TEMPLATES=str(len(poses)))
            expect = len(poses)
        if not os.path.exists(job_src):
            print(f"  {target} {tag}: no job at {job_src}", flush=True)
            failed += 1
            release()
            continue

        rid = runmeta.run_id({"pdb": pdb, "target": target, "tag": tag, "seed": seed, "kind": kind,
                              "dtype": A.dtype, "ncycle": A.ncycle, "nsample": A.nsample,
                              "poses": [os.path.basename(p) for p in poses]})
        rdir = runmeta.run_dir(A.out, kind, pdb, target.replace("*", "all"), seed, rid)
        name = f"{target.replace('*', pdb)}_{tag}"
        try:
            dt, tcounts = fold(runner, configs, named_job(job_src, name, f"{rdir}/jobs"), int(seed),
                               name, A.ncycle, A.nsample, expect, rdir, env)
        except Exception as e:
            print(f"  {target} {tag} seed {seed}: FAILED -- {type(e).__name__}: {str(e)[:120]}",
                  flush=True)
            failed += 1
            release()
            continue

        runmeta.write_run_json(
            rdir, kind=kind, pdb=pdb, target=target, iface=None if target == "*" else target,
            seed=seed, arm=tag, precision=A.dtype, n_cycle=A.ncycle, n_sample=A.nsample,
            fold_seconds=round(dt, 1), interfaces_scored=want,
            templates=[{"source_pose": p, "md5": runmeta_md5(p)} for p in poses] or None,
            templates_found=tcounts, template_namespace=A.templates)

        got = 0
        # Score only what is MISSING. A retried unit (stale claim released) would otherwise append a
        # second row set for interfaces already scored, and duplicate rows quietly double-weight a
        # seed in every downstream average.
        for iface in [i for i in want if (i, str(seed), tag) not in done]:
            out = subprocess.run([PY, f"{DC}/row.py", "--dir", f"{rdir}/{name}", "--iface", iface,
                                  "--seed", str(seed), "--arm", tag, "--donor", "-1",
                                  "--ref", REFF(pdb), "--scorer", SCORER],
                                 cwd=DC, capture_output=True, text=True)
            if out.stdout.strip():
                with open(A.csv, "a") as fh:      # flushed per interface, not at the end
                    fh.write(out.stdout)
                done.add((iface, str(seed), tag))
                got += 1
        if got or all((i, str(seed), tag) in done for i in want):
            ran += 1
            print(f"  {kind} {target} {tag} seed {seed}: {got}/{len(want)} interfaces in {dt:.0f}s",
                  flush=True)
        else:
            print(f"  {target} {tag} seed {seed}: SCORING produced no rows", flush=True)
            for line in (out.stderr or "").strip().splitlines()[-3:]:
                print(f"      {line}", flush=True)
            failed += 1
            release()
        if not A.keep_cifs:
            shutil.rmtree(f"{rdir}/{name}", ignore_errors=True)
    print(f"queue complete: {ran} folds run, {skipped} already scored, {failed} failed", flush=True)
    return 0


def runmeta_md5(path, n=12):
    """Checksum a pose so a run.json identifies the exact file that was templated on."""
    import hashlib
    try:
        with open(path, "rb") as fh:
            return hashlib.md5(fh.read()).hexdigest()[:n]
    except OSError:
        return "missing"


if __name__ == "__main__":
    sys.exit(main())
