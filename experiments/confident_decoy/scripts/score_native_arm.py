#!/usr/bin/env python
"""Score the NATIVE-template arm on the freshly derived interface set (seeds 101-150).

Two questions, one scoring pass:
  RUN A  the 28 confident decoys      -> does templating both chains on the native fold rescue them?
  RUN B  the 19 generation failures   -> can perfect folds CREATE an acceptable pose where the free
                                         run produced none anywhere in 250?

Run B is the more interesting half: those interfaces top out at DockQ 0.016-0.211 in the free run, so
if native folds lift any above 0.23 they were fold-limited; if none move, the docking search itself is
the wall (7z2m already points that way -- 0.94 lDDT chains, ceiling stuck at 0.36).

Predictions come from two places: artifacts/native_run_out/ (the 20 pdbs run for this experiment) and
predictions/native50_out/ (11 pdbs already run at these seeds). First match wins.

SCORING PROTOCOL (tts_abag/eval/eval_ost_foldbench.py -- not improvised): one OST call per pose, FULL
native assembly vs FULL model, --fault-tolerant --min-pep-length 4 --min-nuc-length 4 --dockq, OST's
own global chain mapping, labelled interface extracted BY NAME. Documented isolated-pair fallback
(subsetting BOTH structures) only when the labelled pair is absent from the mapping.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/score_native_arm.py
Writes: artifacts/native_arm_scores.json
"""
import csv, json, os, subprocess, sys, glob, collections, tempfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemmi
from decoy_set import OST, ENV, GT, EXP, SEEDS, SAMPLES, THRESH

CACHE = f"{EXP}/artifacts/native_arm_ost"; os.makedirs(CACHE, exist_ok=True)
OUTJ = f"{EXP}/artifacts/native_arm_scores.json"
NWORK = 14
CLS = json.load(open(f"{EXP}/artifacts/abag_all_101_150.json"))
WANT = {k: v for k, v in CLS.items() if v["status"] in ("confident_decoy", "no_acceptable_pose")}

LOCATIONS = [(f"{EXP}/artifacts/native_run_out/native50", "_native50"),
             (f"{EXP}/predictions/native50_out", "_nativetmpl"),
             (f"{EXP}/artifacts/sweep_out/native50", "_native50")]


def locate(pdb):
    """Prediction dir with the MOST seeds in 101-150.

    Not "first directory that exists": predictions/native50_out holds seeds 0-49 for 8fbw/8q3j/8yxi
    while their 101-150 runs live in sweep_out/native50. Taking the first match silently selected the
    wrong seed range and produced zero scorable poses."""
    best = (0, None, None)
    for root, suf in LOCATIONS:
        d = f"{root}/{pdb}{suf}"
        if not os.path.isdir(d):
            continue
        n = sum(1 for s_ in SEEDS
                if os.path.exists(f"{d}/seed_{s_}/predictions/{pdb}{suf}_summary_confidence_sample_4.json"))
        if n > best[0]:
            best = (n, d, f"{pdb}{suf}")
    return (best[1], best[2]) if best[0] else (None, None)


def _subset(src, dst, keep):
    st = gemmi.read_structure(src); st.setup_entities()
    for mdl in st:
        for nm in [c.name for c in mdl]:
            if nm not in keep:
                mdl.remove_chain(nm)
    st.setup_entities(); st.make_mmcif_document().write_file(dst)


def run_pose(a):
    """All labelled interfaces of one pose, from a single OST call (+ fallback if a pair is missing)."""
    pdb, d, name, pairs, s, k = a
    cif = f"{d}/seed_{s}/predictions/{name}_sample_{k}.cif"
    if not os.path.exists(cif):
        return {}
    o = f"{CACHE}/{pdb}_{s}_{k}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset", "-c", "0-71", OST, "compare-structures", "-m", cif,
                        "-r", f"{GT}/{pdb}-assembly1.cif", "-o", o, "--fault-tolerant",
                        "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq"],
                       env=ENV, capture_output=True, timeout=900)
    try:
        dd = json.load(open(o))
    except Exception:
        return {}
    qs, ifs = dd.get("dockq") or [], dd.get("dockq_interfaces") or []
    try:
        M = json.load(open(f"{d}/seed_{s}/predictions/"
                          f"{name}_summary_confidence_sample_{k}.json"))["chain_pair_iptm"]
    except Exception:
        return {}
    out = {}
    for c1, c2 in pairs:
        hit = None
        for i, t in enumerate(ifs):
            if {t[0], t[1]} == {c1, c2} and i < len(qs):
                x, y = ord(t[2]) - 65, ord(t[3]) - 65
                hit = (M[x][y] if (0 <= x < len(M) and 0 <= y < len(M)) else -1, qs[i]); break
        if hit is None:
            io = f"{CACHE}/iso_{pdb}_{c1}{c2}_{s}_{k}.json"
            if not (os.path.exists(io) and os.path.getsize(io)):
                with tempfile.TemporaryDirectory() as td:
                    try:
                        _subset(cif, f"{td}/m.cif", {c1, c2})
                        _subset(f"{GT}/{pdb}-assembly1.cif", f"{td}/n.cif", {c1, c2})
                    except Exception:
                        continue
                    subprocess.run(["taskset", "-c", "0-71", OST, "compare-structures",
                                    "-m", f"{td}/m.cif", "-r", f"{td}/n.cif", "-o", io,
                                    "--fault-tolerant", "--min-pep-length", "4",
                                    "--min-nuc-length", "4", "--dockq"],
                                   env=ENV, capture_output=True, timeout=900)
            try:
                jd = json.load(open(io)); q = jd.get("dockq") or []
                if len(q) == 1:
                    x, y = ord(c1) - 65, ord(c2) - 65
                    hit = (M[x][y] if (0 <= x < len(M) and 0 <= y < len(M)) else -1, q[0])
            except Exception:
                pass
        if hit is not None:
            out[(c1, c2)] = hit
    return out


# Chain names come from the targets CSV, NOT from splitting the "A-B-2" label: chain ids can
# themselves contain a hyphen (8im0 has a chain literally named "B-2"), so label.split("-") is wrong
# and raised "too many values to unpack" mid-run.
by_pdb = collections.defaultdict(list)
_rows = list(csv.DictReader(open("/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv")))
for r in _rows:
    pdb = r["pdb_id"].split("-")[0]
    c1, c2 = r["interface_chain_id_1"], r["interface_chain_id_2"]
    if f"{pdb}_{c1}{c2}" in WANT:
        by_pdb[pdb].append((c1, c2))

res = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}
for n, (pdb, pairs) in enumerate(sorted(by_pdb.items()), 1):
    # Only skip a pdb whose interfaces all have a FULL 250-pose pool. Scoring while generation is
    # still running would otherwise freeze a partial pool as if it were final.
    if all(res.get(f"{pdb}_{c1}{c2}", {}).get("n", 0) >= len(SEEDS) * len(SAMPLES)
           for c1, c2 in pairs):
        continue
    d, name = locate(pdb)
    if d is None:
        print(f"[{n:2d}/{len(by_pdb)}] {pdb}: NO native50 predictions yet", flush=True); continue
    jobs = [(pdb, d, name, pairs, s, k) for s in SEEDS for k in SAMPLES]
    with ThreadPoolExecutor(NWORK) as ex:
        got = list(ex.map(run_pose, jobs))
    for c1, c2 in pairs:
        lab = f"{pdb}_{c1}{c2}"
        poses = [g[(c1, c2)] for g in got if (c1, c2) in g]
        if not poses:
            continue
        sel = max(poses, key=lambda x: x[0])
        ok = [p for p in poses if p[1] >= THRESH]
        free = CLS[lab]
        res[lab] = {"pdb": pdb, "chains": f"{c1}-{c2}", "free_status": free["status"],
                    "free_n_acceptable": free.get("n_acceptable"),
                    "free_sel_dockq": free.get("sel_dockq"), "free_best_dockq": free.get("best_dockq"),
                    "n": len(poses), "n_acceptable": len(ok),
                    "top_pair_iptm": round(sel[0], 3), "sel_dockq": round(sel[1], 3),
                    "best_dockq": round(max(p[1] for p in poses), 3),
                    "rescued": sel[1] >= THRESH}
        v = res[lab]
        print(f"[{n:2d}/{len(by_pdb)}] {lab:12} {free['status']:19} free_best={v['free_best_dockq']:.3f} "
              f"-> native n_ok={v['n_acceptable']:3d} sel={v['sel_dockq']:.3f} best={v['best_dockq']:.3f} "
              f"{'RESCUED' if v['rescued'] else '-'}", flush=True)
    json.dump(res, open(OUTJ, "w"), indent=1)

print("\n=== NATIVE TEMPLATE ARM ===")
for st, title in (("confident_decoy", "RUN A: confident decoys"),
                  ("no_acceptable_pose", "RUN B: generation failures")):
    v = [x for x in res.values() if x["free_status"] == st]
    if not v:
        continue
    print(f"{title:32} rescued {sum(x['rescued'] for x in v):2d} / {len(v):<2d} scored")
    gained = [x for x in v if x["free_status"] == "no_acceptable_pose" and x["n_acceptable"] > 0]
    if st == "no_acceptable_pose":
        print(f"{'':32} gained >=1 acceptable pose: {len(gained)} / {len(v)}")
print(f"\nWROTE {OUTJ}")
