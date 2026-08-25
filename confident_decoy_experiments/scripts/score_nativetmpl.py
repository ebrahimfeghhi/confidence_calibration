#!/usr/bin/env python
"""Score the native-template factorial: {multichain_mask ON, OFF} x {MSA, no MSA}, 16 complexes.

WHY A SEPARATE SCORER. score_allpose.py assumes ONE pdb per arm (250 donors of the same complex), so
its ground-truth path, interface list and output root are all single-pdb. These arms are the other
shape: 16 different complexes, one job each. The SCORING PROTOCOL is copied verbatim from
score_allpose.py -- same OST call, same flags, interface matched BY NAME (never dockq[0]), ipTM read
from chain_pair_iptm using the MODEL chain letters OST reports.

CONDITION CHECKS, because a silently-wrong arm is this project's recurring failure:
  - any 'Found 0 templates' -> refuse to score that arm (it was untemplated)
  - a *maskoff arm with no 'PROTENIX_MULTICHAIN_MASK=0' line -> refuse (the patch did not apply)
  - a *maskon  arm WITH that line -> refuse (the env leaked across arms)

Interfaces come from artifacts/native_arm_scores.json (the curated per-interface chain pairs), not
from guessing chain letters.

Usage: .../python scripts/score_nativetmpl.py [--arms nat_msa_maskon,...] [--cores 0-63]
Writes: artifacts/nativetmpl_factorial_scores.json  (+ per-arm OST caches artifacts/<arm>_ost/)
"""
import argparse, glob, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, OST, ENV, GT

ap = argparse.ArgumentParser()
ap.add_argument("--arms", default="nat_msa_maskon,nat_msa_maskoff,nat_nomsa_maskon,nat_nomsa_maskoff")
ap.add_argument("--cores", default="0-63", help="taskset core list for OST (shared box: keep <=64)")
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--skip-incomplete", action="store_true",
                help="score arms that are still generating instead of skipping them")
A = ap.parse_args()

ART = f"{EXP}/artifacts"
NAT = json.load(open(f"{ART}/native_arm_scores.json"))
TRAIN = set(json.load(open(f"{ART}/train_test_split.json"))["train"])

# pdb -> {label: (chain1, chain2)}, restricted to the training split
IFACES = {}
for lab, v in NAT.items():
    if v["pdb"] in TRAIN:
        c1, c2 = v["chains"].split("-")
        IFACES.setdefault(v["pdb"], {})[lab] = (c1, c2)


def check_conditions(arm):
    """Return None if the arm is safe to score, else the reason it is not."""
    logs = glob.glob(f"{ART}/{arm}_logs/*.log")
    if not logs:
        return "no logs"
    nulls = xwarn = 0
    for f in logs:
        for line in open(f, errors="ignore"):
            if "Found 0 templates" in line:
                nulls += 1
            if "PROTENIX_MULTICHAIN_MASK=0" in line:
                xwarn += 1
    if nulls:
        return f"{nulls} chain-loads reported 'Found 0 templates' -- arm is untemplated"
    if arm.endswith("maskoff") and xwarn == 0:
        return "mask-OFF arm but no PROTENIX_MULTICHAIN_MASK=0 line -- patch did not apply"
    if arm.endswith("maskon") and xwarn > 0:
        return "mask-ON arm but found the mask-off warning -- env leaked between arms"
    return None


def score_pose(task):
    """{label: (iptm, dockq)} for one output pose, or None."""
    arm, pdb, cif = task
    base = os.path.basename(cif)[:-4]                      # <name>_sample_<k>
    cache = f"{ART}/{arm}_ost"
    o = f"{cache}/{base}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset", "-c", A.cores, OST, "compare-structures", "-m", cif,
                        "-r", f"{GT}/{pdb}-assembly1.cif", "-o", o, "--fault-tolerant",
                        "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq"],
                       env=ENV, capture_output=True, timeout=1800)
    conf = os.path.join(os.path.dirname(cif),
                        base.replace("_sample_", "_summary_confidence_sample_") + ".json")
    try:
        d = json.load(open(o))
        M = json.load(open(conf))["chain_pair_iptm"]
    except Exception:
        return None
    qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
    got = {}
    for lab, (c1, c2) in IFACES.get(pdb, {}).items():
        for i, t in enumerate(ifs):
            if {t[0], t[1]} == {c1, c2} and i < len(qs):    # BY NAME
                x, y = ord(t[2]) - 65, ord(t[3]) - 65
                if 0 <= x < len(M) and 0 <= y < len(M):
                    got[lab] = (M[x][y], qs[i])
                break
    return got or None


out = {}
for arm in A.arms.split(","):
    os.makedirs(f"{ART}/{arm}_ost", exist_ok=True)
    why = check_conditions(arm)
    if why:
        print(f"*** SKIP {arm}: {why}", flush=True)
        continue
    cifs = sorted(glob.glob(f"{ART}/sweep_out/{arm}/*/*/seed_*/predictions/*_sample_*.cif"))
    tasks = []
    for c in cifs:
        pdb = os.path.basename(c)[:4]
        if pdb in IFACES:
            tasks.append((arm, pdb, c))
    if not tasks:
        print(f"*** SKIP {arm}: no poses found", flush=True)
        continue
    with ThreadPoolExecutor(A.workers) as ex:
        res = list(ex.map(score_pose, tasks))

    rows = []
    for (a, pdb, c), r in zip(tasks, res):
        if not r:
            continue
        rows.append({"pdb": pdb, "cif": os.path.relpath(c, ART),
                     "scores": {lab: {"iptm": v[0], "dockq": v[1]} for lab, v in r.items()}})
    out[arm] = rows
    npdb = len({r["pdb"] for r in rows})
    print(f"{arm}: {len(rows)} poses over {npdb} complexes", flush=True)

json.dump(out, open(f"{ART}/nativetmpl_factorial_scores.json", "w"), indent=1)
print(f"\nWROTE {ART}/nativetmpl_factorial_scores.json")

# ---- summary: per interface, best-ipTM pick and the oracle over the 5 diffusion samples ----
print(f"\n{'interface':<11}" + "".join(f"{a.replace('nat_',''):>20}" for a in out))
allabs = sorted({lab for rows in out.values() for r in rows for lab in r["scores"]})
for lab in allabs:
    line = f"{lab:<11}"
    for arm in out:
        v = [(r["scores"][lab]["iptm"], r["scores"][lab]["dockq"])
             for r in out[arm] if lab in r["scores"]]
        if not v:
            line += f"{'-':>20}"
            continue
        sel = max(v, key=lambda x: x[0])[1]
        orc = max(x[1] for x in v)
        line += f"{f'{sel:.3f}/{orc:.3f}':>20}"
    print(line)
print("\n(each cell = selected DockQ by top ipTM / oracle DockQ, over the 5 diffusion samples)")
