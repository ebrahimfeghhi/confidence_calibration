#!/usr/bin/env python
"""ALERT on any result artifact not built on seeds 101-150.

Seed range silently changes conclusions: `8ux6_DF` is a confident decoy at seeds 0-49 and NOT at
101-150; `8vui_BC` has 0 acceptable poses at 0-49 and 6 at 101-150. A file whose name does not state
its seed range can be mistaken for one that shares yours -- exactly what happened with
tts_abag/results/dockq_protenix, where 11 of 26 interfaces are on 101-150 and 15 are on 0-49.

Checks every artifact that records per-pose provenance, plus the prediction directories each arm was
generated from. Prints a single ALERT block for anything outside the range.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/audit_seed_ranges.py
Writes: artifacts/seed_range_audit.json
"""
import csv, glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP

LO, HI = 101, 150
alerts, ok = [], []


def note(kind, name, detail, bad):
    (alerts if bad else ok).append({"kind": kind, "name": name, "detail": detail})


# ---- 1. pose-pose matrices (ours, and the shared tts_abag ones) ----
for d, tag in ((f"{EXP}/artifacts/pose_pose_dockq", "ours"),
               ("/home/ebrahim/tts_abag/results/dockq_protenix", "tts_abag shared"),
               ("/home/ebrahim/tts_abag/results/dockq_full", "tts_abag shared"),
               ("/home/ebrahim/tts_abag/results/dockq_esm250", "tts_abag shared")):
    for f in sorted(glob.glob(f"{d}/*index*.csv")):
        try:
            seeds = [int(r["seed"]) for r in csv.DictReader(open(f)) if r["seed"].isdigit()]
        except Exception:
            continue
        if not seeds:
            continue
        bad = not (min(seeds) >= LO and max(seeds) <= HI)
        note("pose-pose matrix", f"{tag}: {os.path.basename(f)}",
             f"seeds {min(seeds)}-{max(seeds)}", bad)

# ---- 2. arm score files: check the prediction dirs they were built from ----
for f in sorted(glob.glob(f"{EXP}/artifacts/*_scores.json")):
    arm = os.path.basename(f)[:-len("_scores.json")]
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if not isinstance(d, dict) or not d:
        continue
    npose = {v.get("n") for v in d.values() if isinstance(v, dict) and "n" in v}
    bad = any(n is not None and n not in (250,) for n in npose)
    note("arm scores", arm, f"{len(d)} interfaces, pose counts {sorted(x for x in npose if x)}", bad)

# ---- 3. prediction directories: which seeds exist ----
for root in sorted(glob.glob(f"{EXP}/artifacts/sweep_out/*/*/*")):
    if not os.path.isdir(root):
        continue
    seeds = [int(m.group(1)) for p in glob.glob(f"{root}/seed_*")
             if (m := re.search(r"seed_(\d+)$", p))]
    if not seeds:
        continue
    bad = not (min(seeds) >= LO and max(seeds) <= HI)
    note("predictions", os.path.relpath(root, EXP), f"seeds {min(seeds)}-{max(seeds)}", bad)

json.dump({"expected": [LO, HI], "alerts": alerts, "ok": ok},
          open(f"{EXP}/artifacts/seed_range_audit.json", "w"), indent=1)

print(f"checked {len(alerts)+len(ok)} artifacts against seeds {LO}-{HI}\n")
if alerts:
    print(f"*** ALERT: {len(alerts)} artifacts are NOT on seeds {LO}-{HI} ***")
    for a in alerts:
        print(f"  [{a['kind']}] {a['name']}: {a['detail']}")
else:
    print("no alerts -- every checked artifact is on the expected seed range")
print(f"\n{len(ok)} artifacts OK")
print(f"WROTE artifacts/seed_range_audit.json")
