#!/usr/bin/env python
"""Capture everything needed to reproduce an experiment: code versions, inputs, tool versions, configs.

Run this alongside any experiment. The per-experiment scripts record WHAT was computed; this records
the environment it was computed in, which is the part that silently drifts and cannot be reconstructed
after the fact (repo commits move, checkpoints get replaced, conda envs get upgraded).

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/capture_provenance.py [tag]
Writes: artifacts/provenance/<tag>_<utc-ish stamp from mtime>.json  (and prints a summary)
"""
import hashlib, json, os, subprocess, sys, platform

EXP = "/home/ebrahim/confident_decoy_experiments"
OUTD = f"{EXP}/artifacts/provenance"
os.makedirs(OUTD, exist_ok=True)
tag = sys.argv[1] if len(sys.argv) > 1 else "run"


def sh(cmd, cwd=None):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd,
                              timeout=60).stdout.strip()
    except Exception as e:
        return f"<error: {e}>"


def git(repo):
    if not os.path.isdir(os.path.join(repo, ".git")):
        return {"path": repo, "git": "NOT A GIT REPO"}
    return {"path": repo,
            "commit": sh("git rev-parse HEAD", repo),
            "branch": sh("git rev-parse --abbrev-ref HEAD", repo),
            "remote": sh("git config --get remote.origin.url", repo),
            "dirty_files": int(sh("git status --porcelain | wc -l", repo) or 0),
            "head_subject": sh("git log -1 --format=%s", repo)}


def digest(path, cap=64 * 1024 * 1024):
    """md5 of up to `cap` bytes - full hashing of a multi-GB checkpoint is not worth the time."""
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    n = 0
    with open(path, "rb") as f:
        while n < cap:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b); n += len(b)
    return {"path": path, "size_bytes": os.path.getsize(path),
            "md5_first_%dMB" % (cap >> 20): h.hexdigest(),
            "mtime": os.path.getmtime(path)}


def sha_files(paths):
    out = {}
    for p in paths:
        if os.path.exists(p):
            out[os.path.relpath(p, EXP)] = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    return out


prov = {
    "tag": tag,
    "host": platform.node(),
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "repos": {"FoldBench": git("/home/ebrahim/FoldBench"),
              "Protenix": git("/home/ebrahim/Protenix"),
              "tts_abag": git("/home/ebrahim/tts_abag")},
    "note_experiments_dir": "/home/ebrahim/confident_decoy_experiments is NOT version controlled",
    "model": {
        "checkpoint": digest("/home/ebrahim/checkpoint/protenix-v2.pt"),
        "model_name": "protenix-v2",
    },
    "tools": {
        "ost": sh("LD_LIBRARY_PATH=/home/ebrahim/miniconda3/envs/foldbench/lib "
                  "/home/ebrahim/miniconda3/envs/foldbench/bin/ost --version"),
        "python_pkgs": {p: sh(f"{sys.executable} -c \"import {p}; print({p}.__version__)\"")
                        for p in ("gemmi", "numpy")},
    },
    "inputs": {
        "targets_csv": "/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv",
        "targets_csv_sha256_16": sha_files(
            ["/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv"]),
        "ground_truths_dir": "/home/ebrahim/FoldBench/examples/ground_truths",
        "predictions_dir": "/home/ebrahim/Protenix/foldbench_abag/predictions_template",
        "seeds": "101-150", "samples_per_seed": 5, "poses_per_interface": 250,
    },
    "definitions": {
        "acceptable_pose": "DockQ >= 0.23 (CAPRI acceptable)",
        "selection_rule": "pose with the highest interface pair_iptm",
        "confident_decoy": "at least one acceptable pose exists AND the selected pose is not acceptable",
    },
    "scoring_protocol": {
        "source": "tts_abag/eval/eval_ost_foldbench.py",
        "command": ("ost compare-structures -m <pose>.cif -r <pdb>-assembly1.cif -o out.json "
                    "--fault-tolerant --min-pep-length 4 --min-nuc-length 4 --dockq"),
        "chain_mapping": "OST's own global mapping over the FULL native assembly vs the FULL model; "
                         "the labelled interface is then extracted BY NAME from dockq_interfaces",
        "never": "do not take dockq[0]; do not subset the native per interface pair",
    },
    "scripts_sha256_16": sha_files([f"{EXP}/scripts/{s}" for s in (
        "abag_all_101_150.py", "decoy_set.py", "fold_quality.py", "fold_selection_table.py",
        "build_sweep_templates.py", "sweep_sched.py", "score_sweep.py", "capture_provenance.py")]),
}
stamp = sh("date -u +%Y%m%dT%H%M%SZ")
out = f"{OUTD}/{tag}_{stamp}.json"
json.dump(prov, open(out, "w"), indent=1)
print(json.dumps({k: prov[k] for k in ("tag", "host", "python", "repos", "tools")}, indent=1)[:1400])
print(f"\nWROTE {out}")
