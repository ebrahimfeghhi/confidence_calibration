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
import collections
import csv
import glob
import json
import os
import shutil
import subprocess

import runmeta
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


def score_arm(A, PY, DC, seed, arm, donor, REF, SCORER, csv_path, done):
    """Score one arm's output into the project csv, then drop its cifs. Rows are keyed on
    (iface, seed, arm) so a re-run can never duplicate them."""
    import subprocess as _sp
    d = f"{A.out}/s{seed}_{arm}"
    out = _sp.run([PY, f"{DC}/row.py", "--dir", d, "--iface", A.iface, "--seed", str(seed),
                   "--arm", arm, "--donor", str(donor), "--ref", REF, "--scorer", SCORER],
                  cwd=DC, capture_output=True, text=True)
    if out.stdout.strip():
        with open(csv_path, "a") as fh:
            fh.write(out.stdout)
        done.add((str(seed), arm))
    if not A.keep_arm_cifs:
        for sub in glob.glob(f"{d}/*"):
            if os.path.isdir(sub) and "_json" not in sub:
                shutil.rmtree(sub, ignore_errors=True)


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


TIMING_CSV = None          # set in main(); every stage appends one row
_LAST_TEMPLATE_COUNTS = []  # filled by the log handler below, one entry per chain of the last stage


class _TemplateCountHandler(__import__("logging").Handler):
    """Protenix reports the template count only to its logger, so capture it there.

    Without this the number is visible in a log file and nowhere the code can assert on -- which is
    exactly how 20 arms ran with 3 templates while the driver believed it had given them 5.
    """

    def emit(self, record):
        try:
            m = record.getMessage()
        except Exception:
            return
        if "templates for sequence" in m:
            import re
            g = re.search(r"Found (\d+) templates", m)
            if g:
                _LAST_TEMPLATE_COUNTS.append(int(g.group(1)))
TIMING_META = {}           # iface/pdb, so the timing file stands alone


def _sync():
    """Block until the GPU has actually finished. CUDA launches are asynchronous, so a wall-clock
    reading taken without this attributes the tail of one stage to the next one."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def stage(runner, configs, job, seed, n_cycle, env, name, out_root, n_sample=5, expect=5,
          expect_templates=None, dump_base=None):
    """One inference stage.

    env entries are set and restored around the call so stages cannot contaminate each other.
    ASSERTS the stage produced `expect` poses: a stage that silently produces nothing still lets the
    run continue and yields a plausible-looking but wrong summary, which has happened twice here.

    TIMED HERE, NOT FROM THE LOGS. Stage timings were previously read out of Protenix's own "Job
    completed in Ns" lines, which cover only the part of the work Protenix chooses to bracket and
    fold the first stage's warm-up into its number. This times the whole infer_predict call on a
    monotonic clock with the device synchronised on both sides, and writes it to a csv.
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
    # PER-FOLD OUTPUT PATH. runner/dumper.py builds it as
    #     <base_dir>/<dataset_name>/<job name>/seed_<seed>/predictions
    # and base_dir is captured when the runner is constructed -- which we do ONCE per worker to avoid
    # reloading a 464M checkpoint per fold. base_dir is a plain attribute though, so pointing it at
    # this fold's own directory gives the nested, run-scoped layout natively instead of moving files
    # afterwards. Restored below so one stage cannot leak its path into the next.
    _saved_base = getattr(runner.dumper, "base_dir", None)
    if dump_base is not None:
        os.makedirs(dump_base, exist_ok=True)
        runner.dumper.base_dir = dump_base
    # Protenix writes per-sample data errors to <dump_dir>/ERR/<name>.txt and does NOT create the
    # directory at that point, so a recoverable data error becomes FileNotFoundError and kills the
    # whole worker -- it took both workers down twice on 9hvk_BE. Cheap to guarantee here.
    os.makedirs(os.path.join(dump_base or out_root, "ERR"), exist_ok=True)
    _LAST_TEMPLATE_COUNTS.clear()
    _sync()
    t0 = time.perf_counter()                     # monotonic: immune to wall-clock adjustment
    infer_predict(runner, configs)
    _sync()                                      # the GPU may still be working when predict returns
    dt = time.perf_counter() - t0
    for k, v in old.items():                     # restore, so stages cannot contaminate each other
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    # TEMPLATE COUNT IS NOT THE POSE COUNT. A self-templating arm can produce the right number of
    # poses while silently having consumed fewer templates than were handed to it -- 20 arms ran here
    # believing they averaged 5 poses when Protenix reported "Found 3 templates" on every chain.
    if expect_templates is not None and _LAST_TEMPLATE_COUNTS:
        bad = [c for c in _LAST_TEMPLATE_COUNTS if c != expect_templates]
        if bad:
            raise StageError(f"{name}: templates consumed {sorted(set(_LAST_TEMPLATE_COUNTS))}, "
                             f"expected {expect_templates} on every chain")
    n = len(glob.glob(f"{(dump_base or out_root)}/{name}/**/*_sample_*.cif", recursive=True))
    print(f"    {name:<18} {n} poses  {dt:5.0f}s"
          f"{'' if n == expect else f'   !! EXPECTED {expect}, GOT {n}'}", flush=True)
    if TIMING_CSV:
        new = not os.path.exists(TIMING_CSV)
        with open(TIMING_CSV, "a") as fh:
            if new:
                fh.write("iface,seed,stage,n_cycle,n_sample,n_poses,seconds,sec_per_pose\n")
            fh.write(f"{TIMING_META.get('iface','')},{seed},{name.split(f's{seed}_')[-1]},"
                     f"{n_cycle},{n_sample},{n},{dt:.3f},{dt/max(n,1):.3f}\n")
    if dump_base is not None and _saved_base is not None:
        runner.dumper.base_dir = _saved_base
    if n != expect:
        raise StageError(f"{name}: produced {n}/{expect} poses")
    return dt


TMPL_ENV = {"PROTENIX_MULTICHAIN_MASK": "0", "PROTENIX_TMPL_KEEP_FRAME": "1",
            "PROTENIX_TMPL_NODEDUP": "1", "PROTENIX_MAX_TEMPLATES": "1"}


def _uniq_tids(iface, seed, n, tag=""):
    """4-char template ids unique to (interface, seed, ARM, pose).

    The builder asserts exactly 4 chars and the ids name files in a directory shared by every run on
    this machine, so they must not collide across complexes, across arms, or across concurrent workers.

    THE ARM MUST BE IN THE KEY. Keying on (interface, seed) alone gave every per-pose arm of a seed the
    same id, because each such arm has a single pose and therefore always took index 0: arms p0..p4 all
    wrote different poses to one file. Same complex means the sequence still matches, so the featuriser
    ACCEPTED whatever pose happened to be on disk -- no error, no count mismatch, silently the wrong
    template. Collisions across complexes fail loudly; collisions within a complex do not.
    """
    import hashlib
    b36 = "0123456789abcdefghijklmnopqrstuvwxyz"
    h = int(hashlib.md5(f"{iface}:{seed}:{tag}".encode()).hexdigest(), 16) % (36 ** 3)
    pre = "".join(b36[(h // 36 ** k) % 36] for k in (2, 1, 0))
    return [f"{pre}{b36[i % 36]}" for i in range(n)]


def _release_claim(A, iface, tag):
    """A skipped or failed arm must give its claim back, or every later worker treats it as taken."""
    if A.claim_dir:
        try:
            os.rmdir(f"{A.claim_dir}/{iface}_{tag}")
        except OSError:
            pass


def _install_selftmpl(runner, A, configs):
    """Wire the model's in-memory self-template hook for one of the three modes.

    The anchors matter more than the experiment: `noop` isolates "does inserting a mid-loop decode
    disturb the run", `same` isolates "does splicing the feature dict disturb it". Only if both
    reproduce a stock fold is a difference under `self` attributable to the template.
    """
    model = runner.model
    model._selftmpl_after = A.selftmpl_after
    # SAME N_sample AS THE FINAL DECODE. With N_sample=1 the hook's decode runs a different shape
    # than the final one, so it can leave different kernel/workspace selections behind -- a plausible
    # cause of the no-op anchor differing while trunk, RNG and features are all provably identical.
    model._selftmpl_nsample = int(A.arm_nsample)

    import glob as _glob
    import shutil as _shutil
    import subprocess as _sp
    import tempfile as _tf

    from protenix.data.utils import save_structure_cif

    JOBS, _REFF, _SCORER = DATASETS[A.dataset]
    state = {"n": 0}

    def _cb(xyz, cycle_no):
        if A.selftmpl_mode == "noop":
            return None                      # decode only; splice nothing
        if A.selftmpl_mode == "same":
            return {}                        # splice an EMPTY update -> dict unchanged, path exercised

        # --- selftmpl-mode=self: turn the mid-loop poses into template features ---------------
        # xyz is [N_sample, N_atom, 3] in the model's atom order, which is the order of atom_array
        # (runner/dumper.py sets pred_coordinates onto that same array), so the cif writer can be used
        # directly. atom_array is stashed by runner/inference.py because it does not exist inside the
        # model.
        aa = getattr(runner, "_cur_atom_array", None)
        if aa is None:
            raise RuntimeError("selftmpl: runner._cur_atom_array missing -- the inference stash "
                               "in runner/inference.py did not run")
        pdb = state["pdb"]
        work = f"{state['workroot']}/c{cycle_no}_{state['n']}"
        os.makedirs(f"{work}/poses", exist_ok=True)
        state["n"] += 1
        poses = []
        for i in range(xyz.shape[0]):
            fp = f"{work}/poses/{pdb}_sample_{i}.cif"
            save_structure_cif(atom_array=aa, pred_coordinate=xyz[i].detach().float().cpu(),
                               output_fpath=fp, entity_poly_type=state["entity_poly_type"],
                               pdb_id=pdb)
            poses.append(fp)
        # tids must be unique per fold, or a previously cached template of the same id is reused
        # TIDS ARE EXACTLY 4 CHARS (the builder asserts it): 3 base-36 chars of a per-fold hash plus
        # the pose index, so ids are unique per fold and a cached template is never silently reused.
        _b36 = "0123456789abcdefghijklmnopqrstuvwxyz"
        _h = abs(hash((state["tag"], cycle_no))) % (36 ** 3)
        _pre = "".join(_b36[(_h // 36 ** k) % 36] for k in (2, 1, 0))
        tids = [f"{_pre}{_b36[i % 36]}" for i in range(len(poses))]
        r = _sp.run([PY, f"{DC}/build_selftmpl_general.py", "--pdb", pdb, "--out", f"{work}/tmpl",
                     "--poses", *poses, "--tids", *tids],
                    cwd=DC, capture_output=True, text=True,
                    env={**os.environ, "SELFTMPL_JOBS": JOBS})
        if not os.path.exists(f"{work}/tmpl/job.json"):
            raise RuntimeError(f"selftmpl: template build failed\n{(r.stderr or '')[-400:]}")

        # Featurise the self-template job and lift out ONLY the template_* tensors. Everything else
        # (sequence, msa, relpe) must stay as the live run has it -- swapping more than the templates
        # would make the comparison meaningless.
        from protenix.data.inference.infer_dataloader import InferenceDataset
        import copy as _copy
        cfg2 = _copy.deepcopy(configs)
        cfg2.input_json_path = f"{work}/tmpl/job.json"
        ds = InferenceDataset(configs=cfg2)
        item, _aa2, _err = ds[0]
        if _err:
            raise RuntimeError(f"selftmpl: featurisation errored: {str(_err)[:300]}")
        # The dataset item is NOT a flat feature dict -- the features live under
        # item["input_feature_dict"] (runner/inference.py:230 reads it the same way).
        feat = item["input_feature_dict"] if "input_feature_dict" in item else item
        tmpl = {k: (v.to(xyz.device) if hasattr(v, "to") else v)
                for k, v in feat.items() if k.startswith("template_")}
        if not tmpl:
            raise RuntimeError("selftmpl: featurisation produced no template_* keys")
        if not A.keep_arm_cifs:
            _shutil.rmtree(work, ignore_errors=True)
        return tmpl

    model._selftmpl_cb = _cb
    state.update(pdb="", workroot=A.out + "/_selftmpl", entity_poly_type={}, tag="x")
    runner._selftmpl_state = state
    print(f"  self-template hook installed: mode={A.selftmpl_mode} after cycle {A.selftmpl_after}",
          flush=True)


def run_queue(A, PY, DC):
    """Deepen a whole pool: fold (pdb, seed) pairs listed in a queue tsv, model loaded ONCE.

    WHY A MODE AND NOT A SHELL LOOP over --mode free. The queue is 123 complexes; calling the driver
    once per complex reloads the 464M checkpoint every time, which is ~50-80 s of pure overhead per
    complex -- about 2.7 GPU-hours across the queue, for nothing.

    WHY ONE FOLD SCORES MANY INTERFACES. A fold produces the whole complex, and several of these PDBs
    contribute more than one antibody-antigen interface to the pool. Scoring only one would either
    throw away data already computed or, worse, make the driver refold the same coordinates per
    interface.

    Queue line: <pdb>\t<seed-spec>\t<n_new>\t<n_existing>  (written by the seed100 queue builder).
    Claims are per (pdb, seed) so any number of workers can share the queue without slicing it, and a
    claim is released on failure so the pair is retried rather than silently vanishing.
    """
    JOBS, REFF, SCORER = DATASETS[A.dataset]
    lines = []
    for ln in open(A.queue):
        if not ln.strip():
            continue
        f = ln.rstrip("\n").split("\t")
        lines.append((f[0], parse_seeds(f[1])))
    if not lines:
        raise SystemExit("empty queue")

    # interfaces to score per pdb, taken from the pool this run extends
    ifaces = collections.defaultdict(set)
    for r in csv.DictReader(open(A.pool)):
        ifaces[r["pdb"]].add(f"{r['pdb']}_{r['chain1']}{r['chain2']}")

    os.makedirs(A.out, exist_ok=True)
    csv_path = A.csv
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as fh:
            fh.write("iface,seed,arm,donor_pose,sample,dockq,iptm\n")
    done = set()
    for r in csv.DictReader(open(csv_path)):
        done.add((r["iface"], r["seed"], r["arm"]))
    if A.claim_dir:
        os.makedirs(A.claim_dir, exist_ok=True)

    arm = A.arm_name or "free100"
    t_load = time.time()
    runner, configs = build_runner(A.ckpt_dir, A.model_name,
                                  f"{JOBS}/{lines[0][0]}-final-updated.json", A.out)
    if A.dtype is not None:
        configs.dtype = A.dtype
    total = sum(len(s) for _, s in lines)
    print(f"model loaded in {time.time()-t_load:.0f}s -- {total} (pdb, seed) folds queued, "
          f"arm={arm} dtype={A.dtype or 'bf16'}", flush=True)

    ran = skipped = failed = 0
    for pdb, seeds in lines:
        job0 = f"{JOBS}/{pdb}-final-updated.json"
        if not os.path.exists(job0):
            print(f"  {pdb}: no job json, skipping", flush=True)
            continue
        want = sorted(ifaces.get(pdb, ()))
        if not want:
            print(f"  {pdb}: no interfaces in the pool, skipping", flush=True)
            continue
        for seed in seeds:
            # a pair is finished only when EVERY interface of it is scored
            if all((i, str(seed), arm) in done for i in want):
                skipped += 1
                continue
            claim = f"{A.claim_dir}/{pdb}_{seed}" if A.claim_dir else None
            if claim:
                try:
                    os.mkdir(claim)
                except FileExistsError:
                    continue
            rid = runmeta.run_id({"pdb": pdb, "seed": seed, "arm": arm, "dtype": A.dtype,
                                  "ncycle": A.ncycle, "nsample": 5})
            rdir = runmeta.run_dir(A.out, "free", pdb, pdb, seed, rid)
            name = f"s{seed}_{arm}"
            t0 = time.time()
            try:
                stage(runner, configs, named_job(job0, name, f"{rdir}/jobs"), int(seed), A.ncycle,
                      {"PROTENIX_DUMP_TRUNK": None, "PROTENIX_LOAD_TRUNK": None},
                      name, rdir, dump_base=rdir)
            except Exception as e:
                print(f"  {pdb} seed {seed}: FAILED -- {type(e).__name__}: {str(e)[:120]}",
                      flush=True)
                failed += 1
                if claim:
                    try:
                        os.rmdir(claim)
                    except OSError:
                        pass
                continue
            dt = time.time() - t0
            runmeta.write_run_json(rdir, mode="free", pdb=pdb, iface=None, seed=seed, arm=arm,
                                   dtype=(A.dtype or "bf16"), n_cycle=A.ncycle, n_sample=5,
                                   templates=None, namespace=None,
                                   fold_seconds=round(dt, 1), interfaces_scored=want,
                                   driver_args=" ".join(sys.argv[1:]))
            got = 0
            for iface in want:
                out = subprocess.run([PY, f"{DC}/row.py", "--dir", f"{rdir}/{name}",
                                      "--iface", iface, "--seed", str(seed), "--arm", arm,
                                      "--donor", "-1", "--ref", REFF(pdb), "--scorer", SCORER],
                                     cwd=DC, capture_output=True, text=True)
                if out.stdout.strip():
                    with open(csv_path, "a") as fh:      # flushed per interface, not at the end
                        fh.write(out.stdout)
                    done.add((iface, str(seed), arm))
                    got += 1
            if got:
                ran += 1
                print(f"  {pdb} seed {seed}: {got}/{len(want)} interfaces in {dt:.0f}s", flush=True)
            else:
                print(f"  {pdb} seed {seed}: SCORING produced no rows", flush=True)
                failed += 1
                if claim:
                    try:
                        os.rmdir(claim)
                    except OSError:
                        pass
            if not A.keep_arm_cifs:
                shutil.rmtree(f"{rdir}/{name}", ignore_errors=True)
    print(f"queue complete: {ran} folds run, {skipped} already scored, {failed} failed", flush=True)
    return 0


def run_manifest(A, PY, DC):
    """Self-templated folds listed in a tsv, with the model built ONCE for the whole file.

    Columns: iface, pdb, tag, seed, "pose1 [pose2 ...]". Each line is one fold at that line's own
    seed number with those poses as the sole templates, scored as arm "tmpl_<tag>".

    The runner is complex-agnostic once built -- stage() sets configs.input_json_path per call -- so a
    single load serves every pdb in the manifest. That is the entire point: the shell version this
    replaces reloaded the 464M checkpoint per arm and spent ~4.5 min on work this does in well under one.
    """
    lines = [l.rstrip("\n").split("\t") for l in open(A.manifest) if l.strip()]
    if not lines:
        raise SystemExit(f"empty manifest: {A.manifest}")
    csv_path = A.csv
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as fh:
            fh.write("iface,seed,arm,donor_pose,sample,dockq,iptm\n")
    done = {(r["iface"], r["seed"], r["arm"]) for r in csv.DictReader(open(csv_path))}
    JOBS, REFF, SCORER = DATASETS[A.dataset]
    os.makedirs(A.out, exist_ok=True)
    if A.claim_dir:
        os.makedirs(A.claim_dir, exist_ok=True)

    t_load = time.time()
    # ATTACH AFTER build_runner. logging.basicConfig() is a no-op when the root logger already has a
    # handler, so attaching first silently suppressed every INFO/WARNING Protenix emits -- including
    # the LOAD_TRUNK confirmation, which then looked like evidence the trunk was not being loaded.
    import logging
    runner, configs = build_runner(A.ckpt_dir, A.model_name,
                                   f"{JOBS}/{lines[0][1]}-final-updated.json", A.out)
    logging.getLogger().addHandler(_TemplateCountHandler())
    if A.selftmpl_mode is not None:
        _install_selftmpl(runner, A, configs)
    # PRIVATE TEMPLATE NAMESPACE for this run. Self-templated jobs set templatesPath, so the real PDB
    # library is never searched for them -- a directory holding only this run's ids is sufficient, and
    # it makes 4-char id collisions impossible instead of near-certain in the shared 10k-file dir.
    if A.selftmpl_private:
        _ns = os.path.abspath(os.path.join(A.out, "_tmplns"))
        os.makedirs(_ns, exist_ok=True)
        os.environ["SELFTMPL_MMCIF_DIR"] = _ns
        os.environ["SELFTMPL_RELEASE_DATES"] = f"{_ns}/release_dates.json"
        configs.data.template.prot_template_mmcif_dir = _ns
        configs.data.template.release_dates_path = f"{_ns}/release_dates.json"
        print(f"  private template namespace: {_ns}", flush=True)
    if A.dtype is not None:
        configs.dtype = A.dtype
        print(f"  configs.dtype set to {A.dtype}", flush=True)
    if A.mc_dropout_rate is not None:
        configs.mc_dropout_apply_rate = A.mc_dropout_rate
        print(f"  mc_dropout_apply_rate set to {A.mc_dropout_rate}", flush=True)
    print(f"model loaded in {time.time()-t_load:.0f}s -- reused for up to {len(lines)} arms", flush=True)

    ran = skipped = 0
    for iface, pdb, tag, seed, poses in lines:
        # k must be part of the arm label, or the k=1,2,3,5 resume variants collide in the csv and
        # each one is skipped as "already scored" after the first.
        arm = f"tmpl_{tag}" + (f"_k{A.arm_ncycle}" if A.load_trunk_dir else "")
        # --no-template lines carry a placeholder in the poses column; there is nothing to resolve.
        pose_list = [] if A.no_template else [os.path.abspath(x) for x in poses.split()]
        for x in pose_list:
            if not os.path.exists(x):
                raise SystemExit(f"{iface} {tag}: template pose missing: {x}")
        if (iface, seed, arm) in done:
            skipped += 1
            continue
        # atomic claim so two GPUs on one manifest never take the same line
        if A.claim_dir:
            try:
                os.mkdir(f"{A.claim_dir}/{iface}_{tag}" + (f"_k{A.arm_ncycle}" if A.load_trunk_dir else ""))
            except FileExistsError:
                continue
        TIMING_META["iface"] = iface
        d = f"{A.out}/{iface}_{tag}"
        if A.no_template:
            if getattr(runner, "_selftmpl_state", None) is not None:
                # MUST be set in THIS branch too. --no-template folds return before the templated-arm
                # path below, so setting it only there left pdb empty and the template builder looked
                # for "-final-updated.json".
                runner._selftmpl_state.update(pdb=pdb, tag=f"{iface}:{seed}")
            # EQUIVALENCE TEST SUPPORT. With --load-trunk-dir this resumes an existing trunk and pays
            # --arm-ncycle further recycles on the SAME (real-template) job the trunk came from. Run
            # against a from-scratch fold at N+k recycles, it answers whether the resume is a faithful
            # continuation or merely a cheap approximation of one.
            if A.load_trunk_dir:
                trunk = f"{A.load_trunk_dir}/{iface}_s{seed}_trunk.npz"
                if not os.path.exists(trunk):
                    print(f"  {iface} {tag}: no trunk at {trunk} -- skipping", flush=True)
                    _release_claim(A, iface, tag); continue
                env0 = {"PROTENIX_LOAD_TRUNK": trunk,
                        "PROTENIX_DUMP_TRUNK": (f"{A.dump_trunk_dir}/{iface}_s{seed}_after.npz"
                                                if A.dump_trunk_dir else None)}
                ncyc0 = A.arm_ncycle
            else:
                trunk = f"{A.out}/{iface}_s{seed}_trunk.npz"
                env0 = {"PROTENIX_DUMP_TRUNK": (f"{A.dump_trunk_dir}/{iface}_s{seed}_after.npz"
                                                if A.dump_trunk_dir
                                                else (None if A.no_trunk_dump else trunk)),
                        "PROTENIX_LOAD_TRUNK": None}
                ncyc0 = A.ncycle
            name = f"{iface}_{tag}"
            j = named_job(f"{JOBS}/{pdb}-final-updated.json", name, f"{A.out}/jobs")
            try:
                stage(runner, configs, j, int(seed), ncyc0, env0,
                      name, A.out, n_sample=A.arm_nsample, expect=A.arm_nsample)
            except StageError as e:
                print(f"  {iface} {tag}: {e}", flush=True)
                if A.claim_dir:
                    try: os.rmdir(f"{A.claim_dir}/{iface}_{tag}")
                    except OSError: pass
                continue
            out = subprocess.run([PY, f"{DC}/row.py", "--dir", f"{A.out}/{name}", "--iface", iface,
                                  "--seed", str(seed), "--arm", arm, "--donor", "-1",
                                  "--ref", REFF(pdb), "--scorer", SCORER],
                                 cwd=DC, capture_output=True, text=True)
            if out.stdout.strip():
                with open(csv_path, "a") as fh:
                    fh.write(out.stdout)
                ran += 1
                sz = os.path.getsize(trunk)/1e6 if os.path.exists(trunk) else 0
                print(f"  {iface} {tag} seed {seed}: scored, trunk {sz:.0f} MB", flush=True)
            else:
                print(f"  {iface} {tag}: SCORING produced no rows", flush=True)
            continue
        T = f"{d}/tmpl"
        r = None
        if not os.path.exists(f"{T}/job.json"):
            # TIDS MUST BE GLOBALLY UNIQUE. build_selftmpl_general writes each pose into the SHARED
            # ~/mmcif directory as <tid>.cif, so a tid built only from the seed's last two digits
            # collides across complexes -- 9nn9_AB and 8uzp_AC are both seed 300024 and produced the
            # same m024..m424. Concurrent workers then overwrite each other's template structures, the
            # featuriser loads a structure whose sequence does not match, the hit is rejected, and the
            # arm silently runs on 3 templates instead of 5.
            _tids = _uniq_tids(iface, seed, len(pose_list), tag)
            r = subprocess.run([PY, f"{DC}/build_selftmpl_general.py", "--pdb", pdb, "--out", T,
                                "--poses", *pose_list,
                                "--tids", *_tids],
                               cwd=DC, capture_output=True, text=True,
                               env={**os.environ, "SELFTMPL_JOBS": JOBS})
        def _release():
            # A FAILED ARM MUST NOT KEEP ITS CLAIM. Otherwise every later worker skips it as taken and
            # the arm is silently absent from the results forever -- it looks like a completed sweep.
            if A.claim_dir:
                try:
                    os.rmdir(f"{A.claim_dir}/{iface}_{tag}")
                except OSError:
                    pass

        if not os.path.exists(f"{T}/job.json"):
            print(f"  {iface} {tag}: template build FAILED", flush=True)
            for line in ((r.stderr if r else "") or "").strip().splitlines()[-3:]:
                print(f"      {line}", flush=True)
            _release()
            continue
        env = dict(TMPL_ENV)
        env["PROTENIX_MAX_TEMPLATES"] = str(len(pose_list))
        env["PROTENIX_MAX_TEMPLATE_CANDIDATES"] = "20"
        env["PROTENIX_DUMP_TRUNK"] = None
        env["PROTENIX_LOAD_TRUNK"] = None
        if getattr(runner, "_selftmpl_state", None) is not None:
            # per-fold context: which pdb the template builder should look up, and a tag that makes
            # this fold's template ids unique so a cached template from another fold is never reused
            runner._selftmpl_state.update(pdb=pdb, tag=f"{iface}:{seed}")
        ncyc = A.ncycle
        if A.load_trunk_dir:
            trunk = f"{A.load_trunk_dir}/{iface}_s{seed}_trunk.npz"
            if not os.path.exists(trunk):
                print(f"  {iface} {tag}: no trunk at {trunk} -- skipping", flush=True)
                _release_claim(A, iface, tag)
                continue
            env["PROTENIX_LOAD_TRUNK"] = trunk
            ncyc = A.arm_ncycle
        name = f"{iface}_{tag}" + (f"_k{A.arm_ncycle}" if A.load_trunk_dir else "")
        # RUN-SCOPED OUTPUT: <out>/self_template/<pdb>/<iface>/<seed>/<run_id>/ . The run_id keeps a
        # repeat fold of the same seed from overwriting the first -- folds are not reproducible, so a
        # repeat is data, not a duplicate.
        _rid = runmeta.run_id({"iface": iface, "seed": seed, "tag": tag, "dtype": A.dtype,
                               "ncycle": ncyc, "nsample": A.arm_nsample,
                               "poses": [os.path.basename(x) for x in pose_list]})
        _rdir = runmeta.run_dir(A.out, "self_template", pdb, iface, seed, _rid)
        j = named_job(f"{T}/job.json", name, f"{A.out}/jobs")
        try:
            stage(runner, configs, j, int(seed), ncyc, env, name, A.out,
                  n_sample=A.arm_nsample, expect=A.arm_nsample,
                  expect_templates=len(pose_list), dump_base=_rdir)
        except StageError as e:
            print(f"  {iface} {tag}: {e}", flush=True)
            _release()
            continue
        import hashlib as _hl
        runmeta.write_run_json(
            _rdir, mode="self_template", pdb=pdb, iface=iface, seed=seed, arm=arm,
            dtype=(A.dtype or "bf16"), n_cycle=ncyc, n_sample=A.arm_nsample,
            templates=[{"tid": t, "source_pose": p,
                        "md5": _hl.md5(open(p, "rb").read()).hexdigest()[:12]}
                       for t, p in zip(_tids, pose_list)],
            namespace=os.environ.get("SELFTMPL_MMCIF_DIR"),
            driver_args=" ".join(sys.argv[1:]))
        out = subprocess.run([PY, f"{DC}/row.py", "--dir", f"{_rdir}/{name}", "--iface", iface,
                              "--seed", str(seed), "--arm", arm, "--donor", "-1",
                              "--ref", REFF(pdb), "--scorer", SCORER],
                             cwd=DC, capture_output=True, text=True)
        if out.stdout.strip():
            with open(csv_path, "a") as fh:
                fh.write(out.stdout)
            ran += 1
            print(f"  {iface} {tag} seed {seed}: scored", flush=True)
        else:
            print(f"  {iface} {tag}: SCORING produced no rows", flush=True)
            for line in (out.stderr or "").strip().splitlines()[-3:]:
                print(f"      {line}", flush=True)
            _release()
    print(f"manifest complete: {ran} arms run, {skipped} already scored", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb")            # not needed in manifest mode: every line carries its own
    ap.add_argument("--iface")
    ap.add_argument("--seeds", help="e.g. 300000-300049 or 300000,300005")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", default="/home/ebrahim/checkpoint")
    ap.add_argument("--model-name", default="protenix-v2")
    # DEFAULT ON: a pose whose score is in the csv but whose coordinates are gone cannot be
    # templated on later without refolding, and folds are not reproducible run-to-run.
    ap.add_argument("--keep-arm-cifs", action="store_true", default=True)
    ap.add_argument("--prune-arm-cifs", dest="keep_arm_cifs", action="store_false")
    # Default to the PROJECT csv, not a private one: report_complex.py, sweep_watch.sh and every
    # analysis read that file, and rows written elsewhere are invisible to all of them.
    ap.add_argument("--csv", default="/home/ebrahim/tts_abag/confident_decoy_experiments"
                                    "/artifacts/commit11/results.csv")
    ap.add_argument("--keep-trunk", action="store_true",
                    help="keep each seed's trunk dump; default discards it once the seed is armed, "
                         "since the trunk is only needed WHILE that seed's own arms run")
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="pxm2025")
    # all5: fold, then ONE further recycle resuming that trunk with ALL FIVE poses as the sole template.
    # The per-pose arms (mode=perpose) already answered their question -- showing one pose reproduces it
    # (r=0.985) -- so the open question is what happens when the model sees the whole spread at once.
    # free: plain folds, no templates, no trunk -- the paired budget arm.
    # gen2: no fold at all; template on an EXTERNAL set of poses (a previous generation's output) and
    #       run the full recycle count. --tmpl-root points at <hybrid25>/<pdb>, whose s<donor>/out/.../
    #       seed_<g1>/predictions dirs supply the five templates for new seed g1+<seed-offset>.
    # manifest: arms listed in a tsv (iface, pdb, tag, seed, "pose1 [pose2 ...]"). Each line is one
    # self-templated fold at that line's OWN seed number, with those poses as the sole templates.
    # It exists because the shell version spawned one inference.py per arm and reloaded the 464M
    # checkpoint every time -- 4.5 min per arm instead of the ~25 s this driver gets by loading once.
    ap.add_argument("--mode", choices=("perpose", "all5", "free", "gen2", "manifest", "queue"),
                    default="perpose")
    ap.add_argument("--manifest", default=None)
    # --mode queue: deepen a whole pool. tsv of <pdb>\t<seed-spec>; --pool names the csv whose
    # interfaces define what to score per fold.
    ap.add_argument("--queue", default=None)
    ap.add_argument("--pool", default="/home/ebrahim/tts_abag/confident_decoy_experiments"
                                    "/artifacts/pxm2025/dockq_pxm2025.csv")
    ap.add_argument("--claim-dir", default=None, help="atomic mkdir claims for multi-GPU work")
    # --no-template: manifest lines are folded WITHOUT templates and the recurrent state is dumped.
    # This is the setup step for the cheap protocol: one 10-recycle fold per seed, then each pose can
    # be committed with 1-2 extra recycles resuming that trunk instead of a fresh 10-recycle fold.
    ap.add_argument("--no-template", action="store_true")
    # --load-trunk-dir: resume the recurrent state saved by a --no-template fold and pay only
    # --arm-ncycle further recycles, instead of folding 10 from scratch. This is the cheap version of
    # per-pose self-templating; the arm label carries k so the k=1,2,3,5 variants coexist in the csv.
    ap.add_argument("--load-trunk-dir", default=None)
    # Replicate runs do not need the recurrent state, and it is ~1.1 GB per fold on these complexes.
    ap.add_argument("--no-trunk-dump", action="store_true")
    # Validation support. --mc-dropout-rate 0 removes the per-cycle 40%-dropout coin flip, which is
    # what makes a resumed run diverge from an in-run cycle; with it off the carry-over mechanism can
    # be tested on its own. --dump-trunk-dir lets a RESUMED run also dump, so the trunk state after
    # "10 cycles + 1 resumed cycle" can be compared tensor-for-tensor against "11 cycles from scratch".
    ap.add_argument("--mc-dropout-rate", type=float, default=None)
    # --dtype fp32: run the trunk in full precision instead of the default bf16 autocast. bf16 has an
    # 8-bit mantissa, so a non-deterministic CUDA reduction order perturbs a value at ~1e-3 relative
    # rather than ~1e-7; over 10 recycles that is enough to flip a knife-edge basin commitment, which
    # is the suspected cause of identical seeds landing at DockQ 0.018 or 0.578 on the same card.
    ap.add_argument("--dtype", choices=("bf16", "fp32", "fp16"), default=None)
    ap.add_argument("--selftmpl-private", action="store_true",
                    help="per-run template dir + release-date file (isolates the 4-char id namespace)")
    # IN-MEMORY SELF-TEMPLATING at recycle k: decode mid-loop, feed the pose(s) back as the template,
    # run the remaining cycle(s), decode again. z/s never leave the process, which is what the
    # disk-based trunk resume could not manage (its state came back 94% wrong).
    #   --selftmpl-mode noop  : decode, splice NOTHING   -> must reproduce a stock run
    #   --selftmpl-mode same  : decode, splice the ORIGINAL template features -> must reproduce a stock run
    #   --selftmpl-mode self  : decode, splice the decoded pose as the sole template (the experiment)
    ap.add_argument("--selftmpl-mode", choices=("noop", "same", "self"), default=None)
    ap.add_argument("--selftmpl-after", type=int, default=9, help="0-based cycle index to fire at")
    ap.add_argument("--dump-trunk-dir", default=None)
    ap.add_argument("--arm-name", default=None, help="csv arm label; defaults per mode")
    ap.add_argument("--tmpl-root", default=None, help="gen2: <hybrid25>/<pdb>")
    ap.add_argument("--seed-offset", type=int, default=25, help="gen2: new_seed = gen1_seed + offset")
    ap.add_argument("--ncycle", type=int, default=10)
    # EXTRA RECYCLES ON THE ARM. The arm was fixed at 1 ("recycle 11"). Confidence gain tracks how long
    # the self-template is present -- +0.019 for one extra recycle versus +0.058 for a full 10-recycle
    # refold on 9gs3/9gs5 -- so k is the dial between the cheap and the expensive version. The arm label
    # carries k, so all5_k1 and all5_k2 coexist in the csv instead of one silently overwriting the other.
    ap.add_argument("--arm-ncycle", type=int, default=1)
    # Diffusion samples per ARM (the donor fold always decodes 5). The per-pose arms cost ~5 s of
    # recycle and ~19 s of decode each, so the decode -- not the recycle -- is what makes five arms
    # more expensive than one refold. At 1 sample the whole per-pose sweep should undercut a refold.
    # The arm label carries the count, so n1 and n5 runs coexist instead of one being skipped as done.
    ap.add_argument("--arm-nsample", type=int, default=5)
    ap.add_argument("--timing-csv", default=None, help="append per-stage timings here")
    A = ap.parse_args()
    global TIMING_CSV
    TIMING_CSV = A.timing_csv
    if A.mode == "manifest":
        if not A.manifest:
            raise SystemExit("--mode manifest needs --manifest")
        return run_manifest(A, PY, DC)
    if A.mode == "queue":
        if not A.queue:
            raise SystemExit("--mode queue needs --queue")
        return run_queue(A, PY, DC)
    for req in ("pdb", "iface", "seeds"):
        if getattr(A, req) is None:
            raise SystemExit(f"--{req} is required for --mode {A.mode}")
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
    # TIMING_CSV is already set at the top of main() -- a second `global` after the first assignment
    # is a SyntaxError, and ast.parse does NOT catch it (it is raised by the compiler). Validate edits
    # to this file with compile(), not ast.parse.
    TIMING_META["iface"] = A.iface
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

        # ---- free: one plain fold, scored under --arm-name. No templates, no trunk.
        if A.mode == "free":
            arm = A.arm_name or "free75"
            if (str(seed), arm) in done:
                print(f"  seed {seed}: already scored, skipping", flush=True); continue
            j = named_job(job0, f"s{seed}_{arm}", f"{A.out}/jobs")
            stage(runner, configs, j, seed, A.ncycle,
                  {"PROTENIX_DUMP_TRUNK": None, "PROTENIX_LOAD_TRUNK": None},
                  f"s{seed}_{arm}", A.out)
            score_arm(A, PY, DC, seed, arm, -1, REF, SCORER, csv_path, done)
            print(f"  seed {seed} done in {time.time()-t0:.0f}s", flush=True)
            continue

        # ---- gen2: no fold. Template on the previous generation's poses for this seed.
        if A.mode == "gen2":
            arm = A.arm_name or "gen2"
            if (str(seed), arm) in done:
                print(f"  seed {seed}: already scored, skipping", flush=True); continue
            g1 = seed - A.seed_offset
            src = None
            for d in sorted(glob.glob(f"{A.tmpl_root}/s*/out/*/seed_{g1}/predictions")):
                src = d; break
            if not src:
                print(f"  seed {seed}: no gen-1 poses for seed {g1} under {A.tmpl_root}", flush=True)
                continue
            poses = sorted(glob.glob(f"{src}/*_sample_*.cif"))
            if len(poses) < 5:
                print(f"  seed {seed}: only {len(poses)} gen-1 poses", flush=True); continue
            T = f"{pre}_tmpl_gen2"
            r = None
            if not os.path.exists(f"{T}/job.json"):
                tids = [f"g{i}{seed % 100:02d}" for i in range(len(poses))]
                r = subprocess.run([PY, f"{DC}/build_selftmpl_general.py", "--pdb", A.pdb,
                                    "--out", T, "--poses", *poses, "--tids", *tids],
                                   cwd=DC, capture_output=True, text=True,
                                   env={**os.environ, "SELFTMPL_JOBS": JOBS})
            if not os.path.exists(f"{T}/job.json"):
                print(f"  seed {seed}: gen2 template build FAILED", flush=True)
                for line in ((r.stderr if r else "") or "").strip().splitlines()[-3:]:
                    print(f"      {line}", flush=True)
                continue
            env = dict(TMPL_ENV)
            env["PROTENIX_MAX_TEMPLATES"] = "5"
            env["PROTENIX_MAX_TEMPLATE_CANDIDATES"] = "20"
            env["PROTENIX_LOAD_TRUNK"] = None
            env["PROTENIX_DUMP_TRUNK"] = None
            j = named_job(f"{T}/job.json", f"s{seed}_{arm}", f"{A.out}/jobs")
            stage(runner, configs, j, seed, A.ncycle, env, f"s{seed}_{arm}", A.out)
            score_arm(A, PY, DC, seed, arm, g1, REF, SCORER, csv_path, done)
            print(f"  seed {seed} done in {time.time()-t0:.0f}s", flush=True)
            continue

        # SKIP A FINISHED SEED ENTIRELY. The per-arm checks below are not enough: the fold is gated on
        # this driver's OWN trunk path, so a seed already completed by the shell driver (which wrote its
        # trunk elsewhere, and deletes it afterwards) would be re-folded for nothing -- 39 pointless
        # folds when the sweep was switched over mid-complex.
        _sfx = "" if A.arm_nsample == 5 else f"_n{A.arm_nsample}"
        need = ["donor", f"all5_k{A.arm_ncycle}"] if A.mode == "all5" else \
            ["donor", "none"] + [f"p{i}{_sfx}" for i in range(5)]
        if all((str(seed), a) in done for a in need):
            print(f"  seed {seed}: all arms already scored, skipping", flush=True)
            continue

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

        if A.mode == "all5":
            ARM = f"all5_k{A.arm_ncycle}"
            if (str(seed), ARM) not in done:
                T = f"{pre}_tmpl_all5"
                r = None
                if not os.path.exists(f"{T}/job.json"):
                    tids = [f"f{i}{seed % 100:02d}" for i in range(len(poses))]
                    # SELFTMPL_JOBS must follow the dataset, or a foldbench complex silently builds
                    # nothing and every arm is skipped with only "template build failed" to show for it.
                    r = subprocess.run([PY, f"{DC}/build_selftmpl_general.py", "--pdb", A.pdb,
                                        "--out", T, "--poses", *poses, "--tids", *tids],
                                       cwd=DC, capture_output=True, text=True,
                                       env={**os.environ, "SELFTMPL_JOBS": JOBS})
                if not os.path.exists(f"{T}/job.json"):
                    print(f"  seed {seed}: all5 template build FAILED", flush=True)
                    for line in ((r.stderr if r else "") or "").strip().splitlines()[-3:]:
                        print(f"      {line}", flush=True)
                    raise StageError("all5 template build failed")
                env = dict(TMPL_ENV)
                env["PROTENIX_MAX_TEMPLATES"] = "5"          # trained default is 4 -- would drop a pose
                env["PROTENIX_MAX_TEMPLATE_CANDIDATES"] = "20"
                env["PROTENIX_LOAD_TRUNK"] = trunk           # TRUE continuation: recycle 11, not a refold
                env["PROTENIX_DUMP_TRUNK"] = None
                j = named_job(f"{T}/job.json", f"s{seed}_{ARM}", f"{A.out}/jobs")
                stage(runner, configs, j, seed, A.arm_ncycle, env, f"s{seed}_{ARM}", A.out)
        else:
            if (str(seed), "none") not in done:
                j = named_job(job0, f"s{seed}_none", f"{A.out}/jobs")
                stage(runner, configs, j, seed, 1,
                      {"PROTENIX_LOAD_TRUNK": trunk, "PROTENIX_DUMP_TRUNK": None},
                      f"s{seed}_none", A.out)

            SFX = "" if A.arm_nsample == 5 else f"_n{A.arm_nsample}"
            for i, pose in enumerate(poses):
                if (str(seed), f"p{i}{SFX}") in done:
                    continue
                T = f"{pre}_tmpl{i}"
                r = None
                if not os.path.exists(f"{T}/job.json"):
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
                j = named_job(f"{T}/job.json", f"s{seed}_p{i}{SFX}", f"{A.out}/jobs")
                stage(runner, configs, j, seed, 1, env, f"s{seed}_p{i}{SFX}", A.out,
                      n_sample=A.arm_nsample, expect=A.arm_nsample)

        # score everything, then drop the arms' structures (DockQ/ipTM survive in the pxm jsons;
        # 50 seeds x 35 poses is otherwise a lot of disk on a quota this project has already filled)
        # MUST follow --arm-ncycle. Hardcoding "all5_k1" here once already let a k=2 run fold
        # five poses and then silently write no rows, because the arm it scored did not exist.
        arms = ([("donor", -1), (f"all5_k{A.arm_ncycle}", -1)] if A.mode == "all5"
                else [("donor", -1), ("none", -1)]
                + [(f"p{i}" + ("" if A.arm_nsample == 5 else f"_n{A.arm_nsample}"), i)
                   for i in range(len(poses))])
        for arm, dn in arms:
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
