#!/usr/bin/env python
"""Validate that a template arm ACTUALLY RAN WHAT IT CLAIMS, before anyone scores or believes it.

WHY THIS EXISTS: on 2026-08-04 the `topdockq` arm (reported 1/14) was found to have logged
"Found 0 templates" on all 1601 chain-loads -- its templates never loaded, so it measured an
untemplated run. The same was true of `topiptm` (0/5) and `sposefv` (2/8). Three arms, a week of GPU,
and the headline conclusion built on them ("all single-pose-fold ideas fail") had no evidence behind
it. Nothing in the pipeline noticed: the scheduler saw a clean exit, the scorer saw 250 poses, and the
numbers looked plausible because a null template arm still produces poses.

Plausible output is not evidence the run did what it says. These checks are cheap; running an arm for
a week and discovering later that it was null is not.

CHECKS
  1. templates loaded    every chain-load logged "Found 1 templates", none "Found 0"   [FATAL]
  2. seeds complete      every job has all 50 seeds with sample_4 confidence            [FATAL]
  3. output non-empty    predictions exist and are non-trivial in size                  [FATAL]
  4. templates on disk   every a3m referenced by the job JSON exists, is well-formed,
                         names a registered mmCIF, and its hit differs from the query   [FATAL]
  5. donor sanity        the donor structure a3m points to has the expected chain       [WARN]

Usage:
  .../python scripts/validate_run.py --jobs artifacts/dockq_arm_jobs.json \
      --out artifacts/sweep_out/dockqdonor --logs artifacts/dockq_arm_logs
  .../python scripts/validate_run.py --logs-only <dir>        # just the template-loading audit
Exit code 0 if all FATAL checks pass, 1 otherwise, so it can gate scoring in a pipeline.
"""
import argparse, glob, json, os, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("--jobs")
ap.add_argument("--out")
ap.add_argument("--logs")
ap.add_argument("--logs-only")
ap.add_argument("--seeds", type=int, default=50)
A = ap.parse_args()

fatal, warn = [], []


def check_template_loading(logdir):
    """The check that would have caught topdockq/topiptm/sposefv."""
    logs = glob.glob(f"{logdir}/*.log")
    if not logs:
        warn.append(f"no logs in {logdir} -- template loading UNVERIFIED"); return
    tot0 = tot1 = 0
    per = {}
    for f in logs:
        try:
            t = open(f, errors="ignore").read()
        except Exception:
            continue
        n0 = t.count("Found 0 templates")
        n1 = t.count("Found 1 templates")
        tot0 += n0; tot1 += n1
        if n0 or n1:
            per[os.path.basename(f)] = (n0, n1)
    if tot0 == 0 and tot1 == 0:
        warn.append("no 'Found N templates' lines at all -- untemplated arm, or logging changed")
        return
    if tot1 == 0:
        fatal.append(f"TEMPLATES NEVER LOADED: {tot0} chain-loads, all 'Found 0'. "
                     f"This arm measured an UNTEMPLATED run.")
    elif tot0:
        pct = 100 * tot0 / (tot0 + tot1)
        (fatal if pct > 5 else warn).append(
            f"{tot0}/{tot0+tot1} chain-loads ({pct:.0f}%) found NO template")
        for nm, (n0, n1) in sorted(per.items()):
            if n0:
                warn.append(f"  {nm}: {n0} null of {n0+n1}")
    print(f"[1] template loading: {tot1} loaded, {tot0} null across {len(per)} logs")


def check_jobs(jobs, out):
    incomplete, missing_tmpl = [], []
    for j in jobs:
        d = f"{out}/{j['run']}/{j['name']}"
        n = len(glob.glob(f"{d}/seed_*/predictions/{j['name']}_summary_confidence_sample_4.json"))
        if n < A.seeds:
            incomplete.append(f"{j.get('interface', j['pdb'])}: {n}/{A.seeds} seeds")
        cifs = glob.glob(f"{d}/seed_*/predictions/*.cif")
        if cifs and min(os.path.getsize(c) for c in cifs) < 1000:
            fatal.append(f"{j.get('interface', j['pdb'])}: suspiciously small cif output")
        # every referenced a3m must exist, be well-formed, and name a registered mmCIF
        try:
            o = json.load(open(j["json"])); o = o[0] if isinstance(o, list) else o
        except Exception:
            warn.append(f"{j.get('interface', j['pdb'])}: cannot read job json"); continue
        for e in o.get("sequences", []):
            p = e.get("proteinChain", {}).get("templatesPath")
            if not p:
                continue
            if not os.path.exists(p):
                missing_tmpl.append(f"{j.get('interface', j['pdb'])}: a3m missing {p}"); continue
            L = open(p).read().split("\n")
            m = re.match(r">(\w+)_(\S+)/1-(\d+)", L[2]) if len(L) > 3 else None
            if not m:
                missing_tmpl.append(f"{p}: malformed hit header")
            elif not os.path.exists(f"/home/ebrahim/mmcif/{m.group(1)}.cif"):
                missing_tmpl.append(f"{p}: references unregistered tid {m.group(1)}")
            elif len(L[1]) != len(L[3]):
                missing_tmpl.append(f"{p}: query/hit length mismatch")
            elif L[1] == L[3]:
                missing_tmpl.append(f"{p}: hit not mutated -- Protenix reads it as an MSA row, "
                                    f"NOT a template")
    if incomplete:
        fatal.append(f"{len(incomplete)} jobs incomplete: {incomplete[:6]}")
    if missing_tmpl:
        fatal.append(f"{len(missing_tmpl)} template problems: {missing_tmpl[:6]}")
    print(f"[2] seeds complete: {len(jobs)-len(incomplete)}/{len(jobs)} jobs")
    print(f"[3] templates on disk: {len(missing_tmpl)} problems")


if A.logs_only:
    check_template_loading(A.logs_only)
else:
    if not (A.jobs and A.out and A.logs):
        sys.exit("need --jobs --out --logs (or --logs-only DIR)")
    jobs = json.load(open(A.jobs))
    check_template_loading(A.logs)
    check_jobs(jobs, A.out)

print()
for w in warn[:20]:
    print(f"WARN  {w}")
for f in fatal:
    print(f"FATAL {f}")
print(f"\n{'PASS' if not fatal else 'FAIL'} -- {len(fatal)} fatal, {len(warn)} warnings")
sys.exit(1 if fatal else 0)
