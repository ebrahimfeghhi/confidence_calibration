#!/usr/bin/env python
"""Score the MSA-module weight-noise ladder: selected/oracle DockQ per interface per sigma.

Arms are per COMPLEX (wnm_<tag>_<pdb>) but the benchmark is per INTERFACE, so one OST call per pose
serves every labelled interface of that pdb -- the same protocol as abag_all_101_150.py, not a
re-derivation. Multi-interface complexes therefore cost the same as single-interface ones.

Only complexes complete on all CORE rungs are reported. A partial ladder cannot be compared across
sigma, and including it would let the set of complexes differ between rungs -- which is exactly the
confound that made the earlier dose-sweep aggregate unreadable. Rungs added later (5e-5) or retired
(1e-1) print as extra rows carrying their own n.

SELECTION is the label-free rule used throughout: pool an arm's poses, take the single highest
interface pair_iptm pose, report ITS DockQ.

Writes artifacts/wnm_status.json (per arm x interface) and prints the sigma curve.

Usage: .../python scripts/score_wnm.py [--jobs 16]
"""
import argparse, collections, glob, json, os, subprocess, sys, statistics as st
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import OST, ENV, GT

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
A = f"{EXP}/artifacts"
THRESH = 0.23
# CORE rungs are run for every complex and define the comparable set. s5e5 was added later and s1e1
# was retired once it proved dead (0.04 per-seed hit), so neither exists for every complex --
# requiring them would silently report zero complete complexes. They are reported as extra rows with
# their own n instead.
CORE = ["s1e4", "s5e4", "s1e3", "s5e3", "s1e2", "s5e2"]
EXTRA = ["s5e5", "s1e1"]
LADDER = CORE + EXTRA

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", type=int, default=16, help="parallel OST calls (cap: keep well under 64 cores)")
A_ = ap.parse_args()

ALL = json.load(open(f"{A}/abag_all_101_150.json"))
man = json.load(open(f"{A}/wnoise_manifest.json"))
T = json.load(open(f"{A}/train_complexes.json"))
runnable = (T.get("complexes_runnable", T["complexes"])
            + T.get("train2_complexes", []) + T.get("train3_complexes", []))

by_pdb = collections.defaultdict(list)
for iface, v in ALL.items():
    by_pdb[v["pdb"]].append(iface)


def arm_poses(arm):
    return sorted(glob.glob(f"{A}/sweep_out/{arm}/**/*_sample_*.cif", recursive=True))


def complete(pdb):
    return all(len(arm_poses(f"wnm_{t}_{pdb}")) >= 50 for t in CORE)


ready = [p for p in runnable if complete(p)]
print(f"complexes complete on all {len(CORE)} core rungs: {len(ready)}/{len(runnable)}")
print(f"  {' '.join(ready)}\n")
if not ready:
    sys.exit("nothing complete yet")


def _iso(pdb, c1, c2, cif, o):
    """Rescore just the labelled pair, with both model and native subset to those two chains."""
    import tempfile
    try:
        import gemmi
    except Exception:
        return None
    def sub(src, dst, keep):
        st = gemmi.read_structure(src); st.setup_entities()
        for mdl in st:
            for nm in [c.name for c in mdl]:
                if nm not in keep: mdl.remove_chain(nm)
        st.setup_entities(); st.make_mmcif_document().write_file(dst)
    if not (os.path.exists(o) and os.path.getsize(o)):
        with tempfile.TemporaryDirectory() as td:
            try:
                sub(cif, f"{td}/m.cif", {c1, c2})
                sub(f"{GT}/{pdb}-assembly1.cif", f"{td}/n.cif", {c1, c2})
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


def run_pose(args):
    """One OST call -> {(c1,c2): (pair_iptm, dockq)} for every labelled interface of the pdb."""
    arm, pdb, cif = args
    name = os.path.basename(cif)[:-4]
    seed = int(cif.split("/seed_")[1].split("/")[0])
    o = f"{A}/{arm}_ost/{seed}_{name}.json"
    os.makedirs(os.path.dirname(o), exist_ok=True)
    base = ["taskset", "-c", "0-47", OST, "compare-structures", "-m", cif,
            "-r", f"{GT}/{pdb}-assembly1.cif", "-o", o, "--fault-tolerant",
            "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq"]
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(base, env=ENV, capture_output=True, timeout=1800)
    # OST's global chain mapper raises IndexError on some assemblies (8dtn, 8f60, 8t59), writing
    # status FAILURE with no dockq. Every pose of those complexes then silently vanishes from the
    # results rather than scoring 0 -- so retry with an explicit identity chain map.
    try:
        _bad = json.load(open(o)).get("status") != "SUCCESS"
    except Exception:
        _bad = True
    if _bad:
        cmap = [f"{c}:{c}" for c in sorted({c for i in by_pdb[pdb]
                                            for c in ALL[i]["chains"].split("-", 1)})]
        subprocess.run(base + ["-c"] + cmap, env=ENV, capture_output=True, timeout=1800)
    try:
        d = json.load(open(o))
        conf = json.load(open(os.path.join(
            os.path.dirname(cif),
            name.replace("_sample_", "_summary_confidence_sample_") + ".json")))["chain_pair_iptm"]
    except Exception:
        return arm, {}
    qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
    out = {}
    for iface in by_pdb[pdb]:
        c1, c2 = ALL[iface]["chains"].split("-", 1)
        hit = None
        for i, t in enumerate(ifs):
            if {t[0], t[1]} == {c1, c2} and i < len(qs):
                x, y = ord(t[2][0]) - 65, ord(t[3][0]) - 65
                if 0 <= x < len(conf) and 0 <= y < len(conf):
                    hit = (conf[x][y], qs[i])
                break
        # OST can return SUCCESS while emitting a DIFFERENT pair than the labelled one -- 8f60 has
        # model chains A,B,C,D against native A,B,C,K and the global mapping yields only A-B, so
        # A-C and B-C vanish from every table with no error anywhere. The status retry above cannot
        # catch this (status is SUCCESS); only rescoring the isolated pair can.
        if hit is None:
            dq = _iso(pdb, c1, c2, cif, f"{os.path.dirname(o)}/iso_{c1}{c2}_{seed}_{name}.json")
            if dq is not None:
                x, y = ord(c1[0]) - 65, ord(c2[0]) - 65
                if 0 <= x < len(conf) and 0 <= y < len(conf):
                    hit = (conf[x][y], dq)
        if hit: out[iface] = hit
    return arm, out


tasks = [(f"wnm_{t}_{p}", p, cif) for p in ready for t in LADDER
         for cif in arm_poses(f"wnm_{t}_{p}")]
print(f"{len(tasks)} poses to score with {A_.jobs} parallel OST calls ...", flush=True)

acc = collections.defaultdict(lambda: collections.defaultdict(list))
with ThreadPoolExecutor(max_workers=A_.jobs) as ex:
    for n, (arm, res) in enumerate(ex.map(run_pose, tasks), 1):
        for iface, v in res.items():
            acc[arm][iface].append(v)
        if n % 500 == 0:
            print(f"  {n}/{len(tasks)}", flush=True)

status = {}
for arm, per in acc.items():
    sig = man[arm]["sigma"]
    for iface, p in per.items():
        if not p: continue
        status[f"{arm}|{iface}"] = {
            "arm": arm, "interface": iface, "sigma": sig, "poses": len(p),
            "selected": max(p, key=lambda x: x[0])[1],
            "oracle": max(q for _, q in p),
            "frac": sum(q >= THRESH for _, q in p) / len(p),
            "ref_sel": ALL[iface]["sel_dockq"], "status_ref": ALL[iface]["status"]}
json.dump(status, open(f"{A}/wnm_status.json", "w"), indent=1)

# ---- curve over interfaces present at EVERY rung -----------------------------------------------
by_if = collections.defaultdict(dict)
for v in status.values():
    by_if[v["interface"]][v["sigma"]] = v
core_sig = sorted({man[f"wnm_{t}_{ready[0]}"]["sigma"] for t in CORE})
full = sorted(i for i, d in by_if.items() if all(s in d for s in core_sig))
print(f"\ninterfaces complete on the {len(core_sig)} core rungs: {len(full)}\n")
print(f"{'sigma':>9}{'sel_mean':>10}{'sel_med':>9}{'oracle':>9}{'frac_acc':>10}{'rescued':>9}{'n':>6}")
print("-" * 62)
for s in sorted({v["sigma"] for v in status.values()}):
    # extra rungs exist for only some complexes -- restrict each row to the interfaces that have it,
    # and print n so a row built on fewer interfaces is never mistaken for a like-for-like comparison
    ii = [i for i in full if s in by_if[i]]
    if not ii: continue
    sel = [by_if[i][s]["selected"] for i in ii]
    print(f"{s:>9.4g}{st.mean(sel):>10.3f}{st.median(sel):>9.3f}"
          f"{st.mean(by_if[i][s]['oracle'] for i in ii):>9.3f}"
          f"{st.mean(by_if[i][s]['frac'] for i in ii):>10.3f}"
          f"{sum(q >= THRESH for q in sel)/len(ii):>9.3f}{len(ii):>6}")
ref = [ALL[i]["sel_dockq"] for i in full]
print("-" * 56)
print(f"{'ref(250)':>9}{st.mean(ref):>10.3f}{st.median(ref):>9.3f}"
      f"{st.mean(ALL[i]['best_dockq'] for i in full):>9.3f}"
      f"{st.mean(ALL[i]['n_acceptable']/ALL[i]['n'] for i in full):>10.3f}"
      f"{sum(q >= THRESH for q in ref)/len(full):>9.3f}")
print("\nref = stock subsampling+dropout, seeds 101-150 (250 poses) -- NOT seed-matched to the "
      "ladder's 10 seeds; it is a depth-advantaged reference, not a control.")
