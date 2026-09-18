"""Run provenance: a stable id, a run-scoped output directory, and a run.json per fold.

WHY THIS EXISTS. Folds are not reproducible run-to-run (bf16 reductions reorder; the same seed has
landed at DockQ 0.03 and 0.59 on the same card), so a repeat fold of a seed is DATA, not a duplicate
-- it must not overwrite the first. Every fold therefore gets its own directory keyed by a hash of
the inputs that define it, plus a run.json recording enough to tell two runs of the same seed apart
after the fact: precision, recycle count, which poses were used as templates and their checksums.

The pxm2025 pool was written without any of this, and its exact configuration is now unrecoverable --
we cannot tell from disk whether it was bf16 or fp32, or what templates it saw. That is the failure
this module is meant to prevent repeating.

The id deliberately mixes a timestamp, the pid and a content hash: the hash makes an identical
re-request collapse onto one directory, while the timestamp and pid keep two concurrent workers on
the same arm from colliding.
"""
import hashlib
import json
import os
import socket
import subprocess
import sys
import time


def run_id(defining: dict) -> str:
    """`<utc>-<pid>-<hash>` for the inputs that define a fold."""
    h = hashlib.md5(json.dumps(defining, sort_keys=True, default=str).encode()).hexdigest()[:6]
    return f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{os.getpid()}-{h}"


def run_dir(out: str, mode: str, pdb: str, iface: str, seed, rid: str) -> str:
    """<out>/<mode>/<pdb>/<iface>/<seed>/<run_id>/ -- granular enough that repeats coexist."""
    return os.path.join(out, mode, str(pdb), str(iface), str(seed), rid)


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_run_json(rdir: str, **fields) -> str:
    """Record everything needed to tell this fold apart from another run of the same seed.

    Callers pass what they know (mode, pdb, iface, seed, arm, dtype, n_cycle, n_sample, templates,
    namespace, driver_args); the environment facts that turn out to matter -- host, visible devices,
    layernorm kernel, git sha -- are filled in here so no caller can forget them.
    """
    os.makedirs(rdir, exist_ok=True)
    doc = {
        "run_id": os.path.basename(rdir.rstrip("/")),
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "layernorm_type": os.environ.get("LAYERNORM_TYPE"),
        "git_sha": git_sha(),
        "argv": " ".join(sys.argv[1:]),
        **fields,
    }
    p = os.path.join(rdir, "run.json")
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, p)          # atomic: a reader never sees a half-written run.json
    return p
