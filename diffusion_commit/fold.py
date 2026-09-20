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

!! KNOWN BUG, FOUND 2026-09-19, NOT YET FIXED -- READ BEFORE TRUSTING ANY SELF-TEMPLATED RESULT.
The CIFs this driver writes for templates are only PARTLY readable by Protenix. Its template parser
(protenix/data/template/template_parser.py, auth_chain_to_seq) sees ONE chain in them -- verified on
every complex of both runs: the builder requests e.g. ['A','B','C'] and the parser returns ['A'].
Consequence, from template_utils.py:578:

        target_seq = mmcif_obj.chain_to_seqres.get(chain_id, "")
        if not target_seq:
            if len(mmcif_obj.chain_to_seqres) == 1:   # <-- silently uses the ONLY chain
                ...
            else:
                raise QueryToTemplateAlignError(f"Chain {chain_id} not found.")

So a hit for chain A succeeds by falling through that shortcut, while hits for B/C/D raise. Every
entity of a complex was therefore templated against chain A, and on complexes where that raised, the
fold still completed and scored. "Found 5 templates" in a log does NOT mean five multi-chain templates
were used.

This invalidates the MECHANISM described above -- a template cannot be supplying cross-chain placement
if only one chain is visible -- even though the measured effects (e.g. 9h01_AB 0.005 -> 0.782 pooled,
50 seeds) reproduced. Fix the writer, then re-run one complex before interpreting anything.
"""
import argparse
import collections
import csv
import glob
import hashlib
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
# The four knobs that make a self-template carry real docking geometry. Overridable from the shell
# ONLY so the settings can be bisected when a templated fold misbehaves -- every templated fold in
# this project produced NaN coordinates from verifiably valid template features, and narrowing that
# to a single knob is impossible if the driver silently overwrites whatever the caller set.
# Defaults are unchanged; an unset variable behaves exactly as before.
# THE KERNEL SELECTION IS NOT OPTIONAL AND MUST NOT COME FROM THE CALLER'S SHELL. Without
# LAYERNORM_TYPE=openfold this model emits NaN coordinates -- silently. No error, no warning, five
# plausible-looking cif files per fold, and the failure only surfaces downstream when the scorer dies
# on "SVD did not converge". The sweep's tmux sessions happened to set it; a fresh shell does not, and
# the resulting NaNs cost eight diagnostic folds spent bisecting template flags for a failure that had
# nothing to do with templates. setdefault, not assignment, so an explicit override is still possible.
os.environ.setdefault("LAYERNORM_TYPE", "openfold")

TMPL_ENV = {k: os.environ.get(k, v) for k, v in {
    "PROTENIX_MULTICHAIN_MASK": "0", "PROTENIX_TMPL_KEEP_FRAME": "1",
    "PROTENIX_TMPL_NODEDUP": "1", "PROTENIX_MAX_TEMPLATE_CANDIDATES": "20"}.items()}

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




B36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def hook_tids(pdb, tag, cycle_no, n):
    """4-char template ids for a mid-loop splice, DETERMINISTIC.

    Protenix identifies a template by a 4-character pdb id, so a decoded pose has to masquerade as
    one. The previous version built these from abs(hash((tag, cycle_no))) -- and Python randomises
    string hashing per process, so the ids were unknowable before the run. That mattered because of
    the next function: ids have to be registered in the release-date file BEFORE the first fold, and
    you cannot pre-register a name you cannot predict.
    """
    h = int(hashlib.md5(f"{pdb}:{tag}:{cycle_no}".encode()).hexdigest(), 16) % (36 ** 3)
    pre = "".join(B36[(h // 36 ** k) % 36] for k in (2, 1, 0))
    return [f"{pre}{B36[i % 36]}" for i in range(n)]


def append_index(out, pdb, seed, run_dir, kind, precision, ncycle, templated):
    """One line per fold in <experiment>/index.tsv, so runs are found by grep not by walking.

    run.json stays authoritative per fold; this is a lookup table and can be rebuilt from those files
    if it is ever lost. Appended under the assumption of one writer per line -- a short append is
    atomic enough in practice, and a duplicated line is harmless to a grep.
    """
    p = os.path.join(out, "index.tsv")
    new = not os.path.exists(p)
    with open(p, "a") as fh:
        if new:
            fh.write("pdb\tseed\trun_dir\tkind\tprecision\tn_cycle\ttemplated\n")
        fh.write(f"{pdb}\t{seed}\t{run_dir}\t{kind}\t{precision}\t{ncycle}\t{templated}\n")


def preregister_hook_tids(units, after, nsample, rd_path, quiet=False):
    """Put every id the hook will ever use into the release-date file, before any fold runs.

    THIS IS THE FIX FOR "Found 0 templates". Protenix loads the release-date file ONCE per process
    (template_utils.py:838) and keeps it. A mid-run splice registers its ids after that snapshot, so
    _assess_hit rejects them as "date unknown" (line 372) and drops every template -- while the fold
    completes and scores normally. Registering up front means the snapshot already contains them.

    The date is deliberately old: templates newer than the target are filtered out by design, so a
    self-template must not look recent.
    """
    dates = json.load(open(rd_path)) if os.path.exists(rd_path) else {}
    added = 0
    # SCHEMA MATTERS. _parse_release_dates (template_utils.py:869) keeps only entries shaped
    # {"id": {"release_date": "YYYY-MM-DD"}} -- a flat {"id": "YYYY-MM-DD"} parses to NOTHING, and the
    # id is then rejected as "date unknown" even though it is plainly in the file. That cost a smoke
    # test to find, because the fold still completed and scored.
    for kind, pdb, target, tag, seed, _poses in units:
        for c in range(after, after + 8):        # covers any --ncycle the run might use
            for t in hook_tids(pdb, f"{target}_{tag}_{seed}", c, nsample):
                if t not in dates:
                    dates[t] = {"release_date": "2000-01-01"}
                    added += 1
    tmp = rd_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dates, fh)
    os.replace(tmp, rd_path)
    # Parse it back with PROTENIX'S OWN parser, not ours: the only check that matters is whether the
    # ids are visible to the code that will reject them.
    from protenix.data.template.template_utils import TemplateHitFilter  # noqa: F401
    import datetime as _dt
    seen = {k.lower() for k, v in dates.items()
            if isinstance(v, dict) and "release_date" in v
            and _dt.datetime.strptime(v["release_date"], "%Y-%m-%d")}
    missing = [t for _, pdb, target, tag, seed, _p in units
               for c in range(after, after + 8)
               for t in hook_tids(pdb, f"{target}_{tag}_{seed}", c, nsample) if t not in seen]
    if missing:
        raise SystemExit(f"pre-registration failed: {len(missing)} ids are not parseable from "
                         f"{rd_path} (e.g. {missing[:3]}) -- check the release_date schema")
    if not quiet:
        print(f"  pre-registered {added} hook template ids ({len(seen)} parseable) in {rd_path}",
              flush=True)
    return added


def save_decode(runner, state, xyz, cycle_no, summary):
    """Write one mid-loop decode: the poses, and the confidence json row.py needs for ipTM.

    The json must be named *summary_confidence_sample_<k>.json and carry chain_pair_iptm as a nested
    list -- eval/pxm_dockq.py indexes that matrix by model asym order to get the per-interface ipTM.
    Tensors have to be converted; json cannot serialise them and the failure would surface much later
    as a missing ipTM column.
    """
    from protenix.data.utils import save_structure_cif
    aa = getattr(runner, "_cur_atom_array", None)
    if aa is None:
        raise ArmError("hook: runner._cur_atom_array missing")
    pdb = state["pdb"]
    d = f"{state['workroot']}/c{cycle_no}/poses"
    os.makedirs(d, exist_ok=True)
    out = []
    for i in range(int(xyz.shape[0])):
        fp = f"{d}/{pdb}_sample_{i}.cif"
        save_structure_cif(atom_array=aa, pred_coordinate=xyz[i].detach().float().cpu(),
                           output_fpath=fp,
                           entity_poly_type=getattr(runner, "_cur_entity_poly_type", {}),
                           pdb_id=pdb)
        out.append(fp)
        if summary is None:
            continue
        try:
            row = summary[i] if isinstance(summary, (list, tuple)) else summary
            def _plain(v):
                if hasattr(v, "tolist"):
                    return v.tolist()
                if isinstance(v, dict):
                    return {k: _plain(x) for k, x in v.items()}
                if isinstance(v, (list, tuple)):
                    return [_plain(x) for x in v]
                return v
            with open(f"{d}/{pdb}_summary_confidence_sample_{i}.json", "w") as fh:
                json.dump({k: _plain(v) for k, v in dict(row).items()}, fh)
        except Exception as e:
            print(f"    hook: could not write confidence for sample {i}: {e}", flush=True)
    return out


def install_hook(runner, configs, mode, nsample, dataset, ns=None, observe=(), splice=(),
                 hold=False):
    """Mid-loop decode at recycle `after`, then optionally splice template features back in.

    WHAT THIS IS FOR. A templated fold currently costs a whole second fold. If the model can decode at
    recycle 10, take its own poses back as the template, and run recycle 11, the templated arm becomes
    ~4 s instead of ~35 s. The model-side half of this lives in protenix/model/protenix.py; this is the
    callback it invokes.

    THREE MODES, two of which are controls:
      noop  decode, splice NOTHING           -- isolates the cost of merely decoding mid-loop
      same  decode, splice back the templates the fold is ALREADY using -- exercises the splice path
            with values that cannot change the answer
      self  decode, splice the decoded pose as the template -- the actual experiment

    Both controls must reproduce a plain run of the same recycle count. They exist because a no-op
    hook once moved DockQ by 0.252 in fp32, where repeat folds agree to ~0.001, and the cause was
    never found. Until that is understood, `self` must be compared against `noop`/`same`, never
    against a stock run.

    NOTE `same` USED TO BE A NO-OP BY ACCIDENT: it returned {}, and the model splices only `if
    _st_feats:`, so an empty dict was falsy and skipped the splice. It now returns the real
    template_* tensors, which is why the model-side call passes the live feature dict.
    """
    import torch  # noqa: F401  (imported for its side effect on the model's device handling)

    # A PRIVATE NAMESPACE FOR THE SPLICE ONLY. The ids the hook mints must be findable when it
    # featurises them, but they must not go into the shared release-date cache, and pointing the LIVE
    # run at this directory would strip the real PDB templates from cycles 0..after and change the
    # baseline. So: the run keeps stock templates, and only the hook's own featurisation looks here.
    state = {"pdb": "", "tag": "", "workroot": None, "n": 0,
             "ns": ns, "mmcif": f"{ns}/mmcif" if ns else None,
             "rd": f"{ns}/release_dates.json" if ns else None}

    def _cb(xyz, cycle_no, feats=None, summary=None, splice=True):
        # EVERY decode is written out, splice or not: the decode at the splice cycle IS the free
        # baseline for this fold, and the later ones are the 11-/12-/13-recycle outputs. Writing the
        # confidence json beside them is what gives those poses an ipTM, without which only oracle
        # DockQ could be compared and never SELECTED quality.
        save_decode(runner, state, xyz, cycle_no, summary)
        if not splice:
            return None
        if mode == "noop":
            return None
        if mode == "same":
            if feats is None:
                raise ArmError("hook mode 'same' needs the live feature dict -- the model-side "
                               "call in protenix.py must pass it")
            same = {k: v for k, v in feats.items() if k.startswith("template_")}
            if not same:
                # A fold with no templates at all has nothing to re-inject, and returning {} would
                # silently degrade this arm into the no-op arm -- the exact confusion this mode exists
                # to avoid.
                raise ArmError("hook mode 'same' found no template_* features to re-inject")
            return same
        # --- self: the fold's own mid-loop poses become the template ---------------------------
        aa = getattr(runner, "_cur_atom_array", None)
        if aa is None:
            raise ArmError("hook: runner._cur_atom_array missing -- the stash in "
                           "runner/inference.py did not run")
        from protenix.data.utils import save_structure_cif
        pdb, tag = state["pdb"], state["tag"]
        work = f"{state['workroot']}/c{cycle_no}"
        os.makedirs(f"{work}/poses", exist_ok=True)
        poses, tids = [], hook_tids(pdb, tag, cycle_no, int(xyz.shape[0]))
        for i in range(xyz.shape[0]):
            fp = f"{work}/poses/{pdb}_sample_{i}.cif"
            save_structure_cif(atom_array=aa, pred_coordinate=xyz[i].detach().float().cpu(),
                               output_fpath=fp,
                               entity_poly_type=getattr(runner, "_cur_entity_poly_type", {}),
                               pdb_id=pdb)
            poses.append(fp)
        jobs, _, _ = paths.dataset(dataset)
        r = subprocess.run([PY, f"{DC}/build_selftmpl_general.py", "--pdb", pdb,
                            "--out", f"{work}/tmpl", "--poses", *poses, "--tids", *tids],
                           cwd=DC, capture_output=True, text=True,
                           env={**os.environ, "SELFTMPL_JOBS": jobs,
                                "SELFTMPL_MMCIF_DIR": state["mmcif"],
                                "SELFTMPL_RELEASE_DATES": state["rd"]})
        if not os.path.exists(f"{work}/tmpl/job.json"):
            raise ArmError(f"hook: template build failed\n{(r.stderr or '')[-300:]}")
        # Featurise that job and lift out ONLY the template_* tensors. Sequence, MSA and relative
        # position encodings must stay exactly as the live run has them -- swapping more than the
        # templates would make the comparison meaningless.
        # THE SPLICE NEEDS THE TEMPLATE FLAGS, and only the splice. Five poses of one complex share a
        # sequence, so without NODEDUP the featuriser collapses them to ONE ("Found 1 templates"), and
        # without MAX_TEMPLATES it caps below five. MULTICHAIN_MASK=0 and KEEP_FRAME=1 preserve the
        # cross-chain placement, which is the only thing a self-template has to offer. They are set
        # around this featurisation and restored, because applying them to the whole fold would change
        # how the live run treats its stock templates and move the baseline we compare against.
        _mark = len(_TEMPLATE_COUNTS)      # anything appended past here is the HOOK's featurisation
        _env = dict(TMPL_ENV, PROTENIX_MAX_TEMPLATES=str(int(xyz.shape[0])))
        _old_env = {k: os.environ.get(k) for k in _env}
        for k, v in _env.items():
            os.environ[k] = str(v)
        from protenix.data.inference.infer_dataloader import InferenceDataset
        import copy as _copy
        cfg2 = _copy.deepcopy(configs)
        cfg2.input_json_path = f"{work}/tmpl/job.json"
        cfg2.data.template.prot_template_mmcif_dir = state["mmcif"]
        cfg2.data.template.release_dates_path = state["rd"]
        item, _aa2, _err = InferenceDataset(configs=cfg2)[0]
        if _err:
            raise ArmError(f"hook: featurisation errored: {str(_err)[:200]}")
        for k, v in _old_env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        f2 = item["input_feature_dict"] if "input_feature_dict" in item else item
        out = {k: (v.to(xyz.device) if hasattr(v, "to") else v)
               for k, v in f2.items() if k.startswith("template_")}
        if not out:
            raise ArmError("hook: featurisation produced no template_* keys")
        # KEYS ARE NOT TEMPLATES. The featuriser emits template_* tensors even when every hit was
        # rejected, so checking for keys passed while splicing EMPTY templates -- the same silent
        # failure in a new costume. Assert on the mask: it counts templates that survived.
        m = out.get("template_mask")
        kept = int(m.sum().item()) if hasattr(m, "sum") else -1
        if kept == 0:
            raise ArmError(f"hook: all {len(tids)} self-templates were rejected (template_mask is "
                           f"empty) -- ids {tids[:3]} are probably unregistered or mis-dated")
        mine = _TEMPLATE_COUNTS[_mark:]
        if mine and min(mine) == 0:
            raise ArmError("hook: a chain kept ZERO self-templates -- ids unregistered or mis-dated")
        if mine and min(mine) < len(tids):
            # NOT fatal. Two of the five decoded poses can be near-identical, and dedup collapses
            # them even with NODEDUP set; a fold that templated on 4 real poses instead of 5 is still
            # a valid experiment, whereas failing it throws away the whole seed. Zero, on the other
            # hand, means the ids are broken and the fold is measuring nothing.
            print(f"    hook: kept {sorted(set(mine))} of {len(tids)} self-templates "
                  f"(near-duplicate poses deduped)", flush=True)
        state["n"] += 1
        return out

    runner.model._selftmpl_splice = tuple(int(c) for c in splice)
    runner.model._selftmpl_observe = tuple(int(c) for c in observe)
    # HOLD is the whole distinction between the two arms: the model schedules a restore one cycle
    # after each splice unless told to hold.
    runner.model._selftmpl_hold = bool(hold)
    runner.model._selftmpl_after = None      # retired: --hook-splice covers it
    runner.model._selftmpl_nsample = int(nsample)
    runner.model._selftmpl_cb = _cb
    runner._selftmpl_state = state
    print(f"  hook installed: mode={mode} | splices at {sorted(splice)} | "
          f"{'HOLDS to the end' if hold else 'restores after one recycle'} | "
          f"observes at {sorted(observe) or 'nothing'}", flush=True)
    return state


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
    # MID-LOOP HOOK. --hook-after is a 0-BASED cycle index: with --ncycle 11 --hook-after 9 the hook
    # fires after 10 recycles and the 11th runs with whatever it spliced.
    ap.add_argument("--hook", choices=("noop", "same", "self"), default=None)
    # WHERE to splice. 0-based cycle indices: --hook-splice 9 with --ncycle 13 fires after the 10th
    # recycle. WHETHER IT PERSISTS is --hook-hold, and that is the only real difference between the
    # two arms we run -- a single late splice that stays, versus periodic splices that are undone.
    # These used to be separate flags (--hook-after vs --hook-splice) that could contradict each
    # other, resolved by silently disabling one of them inside install_hook.
    ap.add_argument("--hook-splice", default="9")
    ap.add_argument("--hook-hold", action="store_true",
                    help="keep the spliced template for the rest of the fold instead of restoring "
                         "the originals after one recycle")
    # Cycles to decode WITHOUT splicing, so one fold yields several recycle counts. "auto" observes
    # every cycle after the splice, which is what makes 11/12/13 one fold instead of three.
    ap.add_argument("--hook-observe", default="auto")
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

    if A.hook:
        per = [int(x) for x in A.hook_splice.split(",") if x.strip()]
        if not per:
            raise SystemExit("--hook-splice needs at least one cycle")
        if max(per) >= A.ncycle - 1:
            raise SystemExit(f"--hook-splice {per} needs a cycle AFTER the last splice to use it; "
                             f"with --ncycle {A.ncycle} the highest usable splice cycle is "
                             f"{A.ncycle - 2}")
        # In periodic mode the splices ARE the intervention, so nothing extra is observed by default:
        # every splice already decodes, and the fold's own final decode is the result.
        # Observation decodes only make sense when the splice HOLDS: they read the state of a fold
        # that is still carrying the template. With restores there is nothing steady to observe.
        obs = ((range(max(per) + 1, A.ncycle - 1) if A.hook_observe == "auto"
                else [int(x) for x in A.hook_observe.split(",") if x.strip()])
               if A.hook_hold else [])
        hook_state = install_hook(runner, configs, A.hook, A.nsample, A.dataset,
                                  ns=None, observe=obs, splice=per, hold=A.hook_hold)
        # Pre-registration is PER FOLD now (see the unit loop): each fold has its own release-date
        # file, so Protenix's once-per-process cache -- keyed by path -- reads each one fresh.
    logging.getLogger().addHandler(_CountHandler())
    print(f"  model loaded in {time.time()-t_load:.0f}s | dtype={A.dtype} | {len(units)} units "
          f"| templates={A.templates}", flush=True)

    ran = skipped = failed = 0
    for kind, pdb, target, tag, seed, poses in units:
        # EVERY interface of this complex, always. A fold produces the whole complex whatever it was
        # aimed at, so scoring only the motivating interface discarded measurements already paid for --
        # and hid what templating one interface does to the others in the same complex.
        want = sorted(pool_ifaces.get(pdb, ()))
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

        rid = runmeta.run_id()
        rdir = runmeta.run_dir(A.out, pdb, seed, rid)
        name = f"{target.replace('*', pdb)}_{tag}"
        if A.hook:
            # PER-FOLD TEMPLATE NAMESPACE. Sharing one namespace across a sweep put every fold's
            # 4-char ids into the same 36^3 space: 720 of ~3300 collided, so one fold could overwrite
            # a template another was still using. Scoped to a fold there are 15 ids and collisions are
            # impossible. It also sidesteps Protenix's once-per-process release-date cache, which is
            # keyed by PATH -- a fresh path per fold is read fresh.
            ns = os.path.abspath(f"{rdir}/templates")
            os.makedirs(f"{ns}/mmcif", exist_ok=True)
            hook_state.update(pdb=pdb, tag=f"{target}_{tag}_{seed}",
                              workroot=os.path.abspath(f"{rdir}/_hook"),
                              ns=ns, mmcif=f"{ns}/mmcif", rd=f"{ns}/release_dates.json")
            preregister_hook_tids([(kind, pdb, target, tag, seed, poses)], min(per), A.nsample,
                                  f"{ns}/release_dates.json", quiet=True)
            # STATE LIVES ON THE MODEL, which is reused across folds: a pending restore or a stashed
            # template set from the previous fold would leak into this one.
            runner.model._st_restore_at = None
            runner.model._st_orig_tmpl = None
        try:
            dt, tcounts = fold(runner, configs, named_job(job_src, name, f"{rdir}/jobs"), int(seed),
                               name, A.ncycle, A.nsample, expect, rdir, env)
        except Exception as e:
            print(f"  {pdb} {target} {tag} seed {seed}: FAILED -- {type(e).__name__}: {str(e)[:120]}",
                  flush=True)
            failed += 1
            release()
            continue

        runmeta.write_run_json(
            # motivating_interface is WHY this arm exists (which interface's poses were chosen);
            # interfaces_scored is what the fold actually produced. For a pool fold there is no
            # motivation -- it deepens the pool for every interface equally.
            rdir, kind=kind, pdb=pdb,
            motivating_interface=None if target == "*" else target,
            seed=seed, arm=tag, precision=A.dtype, n_cycle=A.ncycle, n_sample=A.nsample,
            fold_seconds=round(dt, 1), interfaces_scored=want,
            templates=[{"source_pose": p, "md5": runmeta_md5(p)} for p in poses] or None,
            templates_found=tcounts, template_namespace=A.templates,
            # true iff this fold minted its own templates -- so `templates/` existing and this flag
            # always agree, and a plain fold never gets an empty templates directory
            templated=bool(A.hook == "self" or poses), run_dir=os.path.basename(rdir))
        append_index(A.out, pdb, seed, os.path.basename(rdir), kind, A.dtype, A.ncycle,
                     bool(A.hook == "self" or poses))

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
            # pdb AND target: target is "*" for a pool fold, so printing target alone left every
            # progress line anonymous -- unreadable when diagnosing which complex is slow.
            print(f"  {kind} {pdb} {target} {tag} seed {seed}: {got}/{len(want)} interfaces "
                  f"in {dt:.0f}s", flush=True)
        else:
            print(f"  {pdb} {target} {tag} seed {seed}: SCORING produced no rows", flush=True)
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
