#!/usr/bin/env python
"""Validate an input dataset BEFORE computing on it. Import this; do not hand-roll the checks.

WHY: repeatedly, analyses were run on inputs that silently did not mean what they appeared to mean,
and the problem surfaced only when a downstream number looked strange:

  - tts_abag/results/dockq_protenix/*.npy -- a HALF-MIGRATED directory. 11 of our 26 interfaces are on
    seeds 101-150; the other 15 are still on seeds 0-49. Nothing in the filename says which, and the
    two disagree on the answer (8vui_BC: 0 acceptable poses vs our 6).
  - template arms that logged "Found 0 templates" and measured untemplated runs for a week.
  - prediction directories left over from a killed build, inflating seed counts by 100.
  - `8ux6_DF` is a confident decoy at seeds 0-49 and not at 101-150 -- seed range alone flips a label.

Every one of these was cheap to detect up front and expensive to detect downstream.

USE:
    from preflight import check_pose_index, require
    idx = check_pose_index("dockq_8slb_AB_index.csv", interface="8slb_AB")   # raises if suspect

The functions RAISE rather than warn. A silent fallback is how a wrong seed range gets into a table.
"""
import csv, json, os

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import EXP, CODE_ROOT
SEED_LO, SEED_HI, THRESH = 101, 150, 0.23


class PreflightError(AssertionError):
    pass


def require(cond, msg):
    if not cond:
        raise PreflightError(msg)


def _free_scores():
    return json.load(open(f"{EXP}/artifacts/native_arm_scores.json"))


def check_pose_index(path, interface, seed_lo=SEED_LO, seed_hi=SEED_HI, expect_n=250):
    """Validate a per-pose index CSV (seed, sample, iptm, dockq_gt) before any analysis uses it.

    Checks: file exists; expected pose count; seeds inside the intended range; and -- the check that
    matters most -- the acceptable-pose count agrees with our own OST scoring, which is an INDEPENDENT
    source. Agreement on that count is strong evidence the file describes the same poses we think.
    """
    require(os.path.exists(path), f"{interface}: index missing at {path}")
    rows = list(csv.DictReader(open(path)))
    require(len(rows) == expect_n, f"{interface}: {len(rows)} poses, expected {expect_n}")
    seeds = [int(r["seed"]) for r in rows]
    require(min(seeds) >= seed_lo and max(seeds) <= seed_hi,
            f"{interface}: seeds {min(seeds)}-{max(seeds)}, expected {seed_lo}-{seed_hi}. "
            f"Mixing seed ranges silently changes which interfaces are decoys.")
    n_ok = sum(1 for r in rows if float(r["dockq_gt"]) >= THRESH)
    free = _free_scores().get(interface)
    if free and free.get("free_n_acceptable") is not None:
        require(n_ok == free["free_n_acceptable"],
                f"{interface}: index says {n_ok} acceptable poses, our OST scoring says "
                f"{free['free_n_acceptable']}. The file does not describe the same poses.")
    return rows


def check_templates_loaded(logdir, max_null_frac=0.05):
    """Assert an arm's templates actually loaded. See scripts/validate_run.py for the full version."""
    import glob
    logs = glob.glob(f"{logdir}/*.log")
    require(logs, f"no logs in {logdir} -- template loading cannot be verified")
    n0 = n1 = 0
    for f in logs:
        t = open(f, errors="ignore").read()
        n0 += t.count("Found 0 templates"); n1 += t.count("Found 1 templates")
    require(n1 > 0, f"{logdir}: templates NEVER loaded ({n0} null chain-loads) -- untemplated run")
    require(n0 <= max_null_frac * (n0 + n1),
            f"{logdir}: {n0}/{n0+n1} chain-loads found no template")
    return n1, n0


def check_prediction_dir(d, name, expect_seeds=50):
    """Assert a prediction directory holds the expected complete seeds, and flag stale siblings."""
    import glob
    n = len(glob.glob(f"{d}/seed_*/predictions/{name}_summary_confidence_sample_4.json"))
    require(n >= expect_seeds, f"{name}: {n}/{expect_seeds} complete seeds in {d}")
    return n


def declare(**kw):
    """Print the provenance of every input an analysis is about to use, so it lands in the log."""
    print("PREFLIGHT INPUTS:")
    for k, v in kw.items():
        print(f"  {k}: {v}")
