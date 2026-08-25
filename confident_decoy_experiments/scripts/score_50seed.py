#!/usr/bin/env python
"""Score the 50-seed arms by pooling their two halves.

Seeds 101-110 and 111-150 were generated as separate arms, so each interface's 50-seed pool is the
union of two directories:
  sigma=1e-4  wnm_s1e4_<pdb>   + d50_s1e4_<pdb>
  both-off    bo/bo2/dose_s0   + bo50_<pdb>
Only complexes complete on BOTH halves are reported -- a complex with 10 of its 50 seeds would
otherwise silently enter the average as a shallower arm and drag it down.

Carries the same OST fallbacks as the other scorers (identity chain map for mapper failures,
isolated-pair rescoring when the labelled pair is absent), without which 8dtn/8f60/8fxv vanish.

Usage: .../python scripts/score_50seed.py [--jobs 6] [--arm 1e-4|bothoff|both]
"""
import argparse, collections, glob, json, os, subprocess, sys, tempfile, statistics as st
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import OST, ENV, GT

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
A = f"{EXP}/artifacts"
THRESH = 0.23

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", type=int, default=6)
ap.add_argument("--arm", default="both", choices=["1e-4", "bothoff", "both"])
ARGS = ap.parse_args()

ALL = json.load(open(f"{A}/abag_all_101_150.json"))
by_pdb = collections.defaultdict(list)
for iface, v in ALL.items():
    by_pdb[v["pdb"]].append(iface)


def _sub(src, dst, keep):
    import gemmi
    stx = gemmi.read_structure(src); stx.setup_entities()
    for mdl in stx:
        for nm in [c.name for c in mdl]:
            if nm not in keep: mdl.remove_chain(nm)
    stx.setup_entities(); stx.make_mmcif_document().write_file(dst)


def _iso(pdb, c1, c2, cif, o):
    if not (os.path.exists(o) and os.path.getsize(o)):
        with tempfile.TemporaryDirectory() as td:
            try:
                _sub(cif, f"{td}/m.cif", {c1, c2})
                _sub(f"{GT}/{pdb}-assembly1.cif", f"{td}/n.cif", {c1, c2})
            except Exception:
                return None
            subprocess.run(["taskset", "-c", "0-47", OST, "compare-structures",
                            "-m", f"{td}/m.cif", "-r", f"{td}/n.cif", "-o", o, "--fault-tolerant",
                            "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq"],
                           env=ENV, capture_output=True, timeout=900)
    try:
        q = json.load(open(o)).get("dockq") or []
    except Exception:
        return None
    return q[0] if len(q) == 1 else None


def run_pose(t):
    arm, pdb, cif = t
    name = os.path.basename(cif)[:-4]
    seed = int(cif.split("/seed_")[1].split("/")[0])
    od = f"{A}/{arm}_ost"; os.makedirs(od, exist_ok=True)
    o = f"{od}/{seed}_{name}.json"
    cf = os.path.join(os.path.dirname(cif),
                      name.replace("_sample_", "_summary_confidence_sample_") + ".json")
    if not os.path.exists(cf): return {}
    try:
        conf = json.load(open(cf))["chain_pair_iptm"]
    except Exception:
        return {}
    base = ["taskset", "-c", "0-47", OST, "compare-structures", "-m", cif,
            "-r", f"{GT}/{pdb}-assembly1.cif", "-o", o, "--fault-tolerant",
            "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq"]
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(base, env=ENV, capture_output=True, timeout=1800)
    try: d = json.load(open(o))
    except Exception: d = {}
    if d.get("status") != "SUCCESS":
        cmap = sorted({c for i in by_pdb[pdb] for c in ALL[i]["chains"].split("-", 1)})
        subprocess.run(base + ["-c"] + [f"{c}:{c}" for c in cmap],
                       env=ENV, capture_output=True, timeout=1800)
        try: d = json.load(open(o))
        except Exception: d = {}
    qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
    out = {}
    for iface in by_pdb[pdb]:
        c1, c2 = ALL[iface]["chains"].split("-", 1)
        hit = None
        for i, tt in enumerate(ifs):
            if {tt[0], tt[1]} == {c1, c2} and i < len(qs):
                x, y = ord(tt[2][0]) - 65, ord(tt[3][0]) - 65
                if 0 <= x < len(conf) and 0 <= y < len(conf): hit = (conf[x][y], qs[i])
                break
        if hit is None:
            dq = _iso(pdb, c1, c2, cif, f"{od}/iso_{c1}{c2}_{seed}_{name}.json")
            if dq is not None:
                x, y = ord(c1[0]) - 65, ord(c2[0]) - 65
                if 0 <= x < len(conf) and 0 <= y < len(conf): hit = (conf[x][y], dq)
        if hit: out[iface] = hit
    return out


def halves(pdb, which):
    """(first-half arms, second-half arm) for this complex, or None if either half is short."""
    if which == "1e-4":
        a, b = [f"wnm_s1e4_{pdb}"], f"d50_s1e4_{pdb}"
    else:
        a = [x for x in (f"bo_{pdb}", f"bo2_{pdb}") if
             len(glob.glob(f"{A}/sweep_out/{x}/**/*_sample_*.cif", recursive=True)) >= 50]
        a = a or [f"dose_s0_{i}" for i in by_pdb[pdb]]
        b = f"bo50_{pdb}"
    n1 = sum(len(glob.glob(f"{A}/sweep_out/{x}/**/*_sample_*.cif", recursive=True)) for x in a)
    n2 = len(glob.glob(f"{A}/sweep_out/{b}/**/*_sample_*.cif", recursive=True))
    if n1 < 50 or n2 < 200: return None
    return a, b


def score_arm(which, label):
    pdbs = sorted({ALL[i]["pdb"] for i in ALL})
    tasks, covered = [], []
    for p in pdbs:
        h = halves(p, which)
        if not h: continue
        covered.append(p)
        for arm in list(h[0]) + [h[1]]:
            tasks += [(arm, p, c) for c in
                      glob.glob(f"{A}/sweep_out/{arm}/**/*_sample_*.cif", recursive=True)]
    if not tasks:
        print(f"{label}: no complex complete on both halves yet"); return

    # Per-arm result cache. OST output was already cached, but re-parsing every JSON each hourly
    # pass is thousands of reads for arms that cannot change. Key on the arm's pose count: if it
    # matches, the arm is finished and its summary is reused untouched.
    CF = f"{A}/score50_cache.json"
    try: cache = json.load(open(CF))
    except Exception: cache = {}
    by_arm = collections.defaultdict(list)
    for t in tasks: by_arm[t[0]].append(t)

    acc = collections.defaultdict(list)
    fresh, reused = [], 0
    for arm, ts in by_arm.items():
        c = cache.get(arm)
        if c and c["n"] == len(ts):
            for iface, v in c["poses"].items(): acc[iface] += [tuple(x) for x in v]
            reused += 1
        else:
            fresh += ts
    if fresh:
        got = collections.defaultdict(lambda: collections.defaultdict(list))
        with ThreadPoolExecutor(max_workers=ARGS.jobs) as ex:
            for (arm, _, _), res in zip(fresh, ex.map(run_pose, fresh)):
                for iface, v in res.items(): got[arm][iface].append(v)
        for arm, per in got.items():
            cache[arm] = {"n": len(by_arm[arm]), "poses": {k: [list(x) for x in v]
                                                           for k, v in per.items()}}
            for iface, v in per.items(): acc[iface] += v
        json.dump(cache, open(CF, "w"))
    print(f"  [{reused} arms from cache, {len(fresh)} poses read fresh]")
    sel = {i: max(v, key=lambda x: x[0])[1] for i, v in acc.items() if v}
    orc = {i: max(q for _, q in v) for i, v in acc.items() if v}
    fr = {i: sum(q >= THRESH for _, q in v) / len(v) for i, v in acc.items() if v}
    ii = sorted(sel)
    print(f"{label:<26}{st.mean(sel[i] for i in ii):>8.3f}{st.median(sel[i] for i in ii):>9.3f}"
          f"{st.mean(orc[i] for i in ii):>9.3f}{st.mean(fr[i] for i in ii):>10.3f}"
          f"{sum(sel[i] >= THRESH for i in ii)/len(ii):>9.3f}{len(ii):>5}"
          f"   [{len(covered)} complexes]")


print(f"{'arm':<26}{'mean':>8}{'median':>9}{'oracle':>9}{'frac_acc':>10}{'rescued':>9}{'n':>5}")
print("-" * 88)
if ARGS.arm in ("1e-4", "both"): score_arm("1e-4", "sigma=1e-4  50 seeds")
if ARGS.arm in ("bothoff", "both"): score_arm("bothoff", "both-off    50 seeds")
