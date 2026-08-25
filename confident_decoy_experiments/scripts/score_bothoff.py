#!/usr/bin/env python
"""Score the both-OFF arm (no MSA subsampling, no dropout, no weight noise) at seeds 101-110.

The arm's poses live in TWO places for historical reasons, and both must be read or interfaces go
silently missing:
  dose_s0_<interface>   from the earlier sigma=0 sweep, keyed per INTERFACE
  bo_<pdb>              from the phase-2 gap fill, keyed per COMPLEX
Per-complex arms are preferred when both exist, since one OST call there serves every interface.

Carries the same two OST fallbacks as the other scorers -- an identity chain map for assemblies whose
global mapper throws IndexError (8dtn), and isolated-pair rescoring for duplicate-copy assemblies
whose labelled pair never gets emitted (8fxv). Without them those interfaces score as absent rather
than as a number, which is how they quietly fell out of an earlier table.

Writes artifacts/bothoff_status.json. Usage: .../python scripts/score_bothoff.py [--jobs 6]
"""
import argparse, collections, glob, json, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import OST, ENV, GT

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
A = f"{EXP}/artifacts"
THRESH = 0.23

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", type=int, default=6)
ARGS = ap.parse_args()

ALL = json.load(open(f"{A}/abag_all_101_150.json"))
by_pdb = collections.defaultdict(list)
for iface, v in ALL.items():
    by_pdb[v["pdb"]].append(iface)


def _subset(src, dst, keep):
    import gemmi
    st = gemmi.read_structure(src); st.setup_entities()
    for mdl in st:
        for nm in [c.name for c in mdl]:
            if nm not in keep: mdl.remove_chain(nm)
    st.setup_entities(); st.make_mmcif_document().write_file(dst)


def _iso(pdb, c1, c2, cif, o):
    if not (os.path.exists(o) and os.path.getsize(o)):
        with tempfile.TemporaryDirectory() as td:
            ms, ns = f"{td}/m.cif", f"{td}/n.cif"
            try:
                _subset(cif, ms, {c1, c2}); _subset(f"{GT}/{pdb}-assembly1.cif", ns, {c1, c2})
            except Exception:
                return None
            subprocess.run(["taskset", "-c", "0-47", OST, "compare-structures", "-m", ms, "-r", ns,
                            "-o", o, "--fault-tolerant", "--min-pep-length", "4",
                            "--min-nuc-length", "4", "--dockq"],
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
    if not os.path.exists(cf):
        return arm, {}
    try:
        conf = json.load(open(cf))["chain_pair_iptm"]
    except Exception:
        return arm, {}
    base = ["taskset", "-c", "0-47", OST, "compare-structures", "-m", cif,
            "-r", f"{GT}/{pdb}-assembly1.cif", "-o", o, "--fault-tolerant",
            "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq"]
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(base, env=ENV, capture_output=True, timeout=1800)
    try:
        d = json.load(open(o))
    except Exception:
        d = {}
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
    return arm, out


# ---- locate each interface's poses: per-complex arm preferred, per-interface as fallback ---------
tasks, src = [], {}
for pdb, ifaces in by_pdb.items():
    # bo_  = phase-2 gap fill; bo2_ = the later fill for complexes the phase-2 chain missed.
    # Both are the same arm; listing only one prefix silently drops the other's complexes.
    hit = None
    for pre in ("bo", "bo2"):
        cifs = glob.glob(f"{A}/sweep_out/{pre}_{pdb}/**/*_sample_*.cif", recursive=True)
        if len(cifs) >= 50:
            hit = (f"{pre}_{pdb}", cifs); break
    if hit:
        arm, cifs = hit
        tasks += [(arm, pdb, c) for c in cifs]
        for i in ifaces: src[i] = arm
        continue
    for i in ifaces:
        cifs = glob.glob(f"{A}/sweep_out/dose_s0_{i}/**/*_sample_*.cif", recursive=True)
        if len(cifs) >= 50:
            tasks += [(f"dose_s0_{i}", pdb, c) for c in cifs]
            src[i] = f"dose_s0_{i}"
print(f"{len(src)} interfaces have both-off data; {len(tasks)} poses to read", flush=True)

acc = collections.defaultdict(lambda: collections.defaultdict(list))
with ThreadPoolExecutor(max_workers=ARGS.jobs) as ex:
    for n, (arm, res) in enumerate(ex.map(run_pose, tasks), 1):
        for iface, v in res.items():
            if src.get(iface) == arm: acc[iface][arm].append(v)
        if n % 500 == 0: print(f"  {n}/{len(tasks)}", flush=True)

status = {}
for iface, per in acc.items():
    p = [x for v in per.values() for x in v]
    if not p: continue
    status[iface] = {"interface": iface, "poses": len(p), "source": src[iface],
                     "selected": max(p, key=lambda x: x[0])[1],
                     "oracle": max(q for _, q in p),
                     "frac": sum(q >= THRESH for _, q in p) / len(p)}
json.dump(status, open(f"{A}/bothoff_status.json", "w"), indent=1)
print(f"\nscored {len(status)} interfaces -> artifacts/bothoff_status.json")
