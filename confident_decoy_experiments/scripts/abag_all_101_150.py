#!/usr/bin/env python
"""Classify EVERY FoldBench antibody-antigen interface at seeds 101-150, from scratch.

Definition (the simple one):
  confident decoy  =  at least one pose in the 250-pose pool has DockQ >= 0.23
                      AND the pose with the highest interface pair_iptm has DockQ < 0.23
  solved           =  an acceptable pose exists and ipTM picks one
  no-acceptable    =  no pose in the pool reaches DockQ 0.23 (generation failure)

This replaces the inherited 15-interface set, which was assembled from a scores CSV that mixes seed
ranges (0-49 for some pdbs, 101-150 for others) and whose selection criterion was never documented.

SCORING PROTOCOL (tts_abag/eval/eval_ost_foldbench.py -- not improvised): ONE ost compare-structures
call per POSE on the full native assembly vs the full model, --fault-tolerant --min-pep-length 4
--min-nuc-length 4 --dockq, OST's own global chain mapping; every labelled interface of that pdb is
then extracted BY NAME from dockq_interfaces. One OST call serves all interfaces of a pdb, so
multi-interface entries (8che, 8slb, 8yxi, 7z2m...) cost the same as single ones.

Also records native contact residues per interface (5 A, label_asym_id) so that crystal-contact labels
like 7z2m's A-J (3 residues) can be spotted rather than silently scored.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/abag_all_101_150.py
Writes: artifacts/abag_all_101_150.json
"""
import csv, json, os, subprocess, sys, collections, tempfile
from concurrent.futures import ThreadPoolExecutor
import gemmi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import OST, ENV, GT, EXP, PRED, SEEDS, SAMPLES, THRESH, contact_residues

TARGETS = "/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv"
CACHE = f"{EXP}/artifacts/abag_all_ost"
OUTJ = f"{EXP}/artifacts/abag_all_101_150.json"
NWORK = 16            # with taskset 0-71 this measures ~80 cores (hard cap 100)
os.makedirs(CACHE, exist_ok=True)

by_pdb = collections.defaultdict(list)
for r in csv.DictReader(open(TARGETS)):
    by_pdb[r["pdb_id"].split("-")[0]].append((r["interface_chain_id_1"], r["interface_chain_id_2"]))


def _subset(src, dst, keep):
    """Write `src` keeping only chains in `keep` (by the names OST/gemmi sees)."""
    st = gemmi.read_structure(src)
    st.setup_entities()
    for mdl in st:
        for nm in [c.name for c in mdl]:
            if nm not in keep:
                mdl.remove_chain(nm)
    st.setup_entities()
    st.make_mmcif_document().write_file(dst)


def isolated_pair_dockq(pdb, c1, c2, cif, tag):
    """PROTOCOL FALLBACK, mirroring _isolated_pair_dockq in tts_abag/eval/eval_ost_foldbench.py.

    Used only when OST's global chain mapping does not emit the labelled pair at all -- e.g. 8b7h,
    whose assembly has two copies (A,B,C and A-2,B-2,C-2) and whose mapping landed entirely on the
    '-2' copy, so A-C was absent. Subsets BOTH model and native to the two chains and rescores.
    Note: subsets both sides, never the native alone."""
    o = f"{CACHE}/iso_{tag}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        with tempfile.TemporaryDirectory() as td:
            ms, ns = f"{td}/m.cif", f"{td}/n.cif"
            try:
                _subset(cif, ms, {c1, c2})
                _subset(f"{GT}/{pdb}-assembly1.cif", ns, {c1, c2})
            except Exception:
                return None
            subprocess.run(["taskset", "-c", "0-71", OST, "compare-structures", "-m", ms, "-r", ns,
                            "-o", o, "--fault-tolerant", "--min-pep-length", "4",
                            "--min-nuc-length", "4", "--dockq"],
                           env=ENV, capture_output=True, timeout=900)
    try:
        d = json.load(open(o))
    except Exception:
        return None
    q = d.get("dockq") or []
    return q[0] if len(q) == 1 else None


def run_pose(a):
    """One OST call for a pose; returns {(c1,c2): (pair_iptm, dockq)} for every labelled interface."""
    pdb, s, k = a
    cif = f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif"
    if not os.path.exists(cif):
        return {}
    o = f"{CACHE}/{pdb}_{s}_{k}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset", "-c", "0-71", OST, "compare-structures", "-m", cif,
                        "-r", f"{GT}/{pdb}-assembly1.cif", "-o", o,
                        "--fault-tolerant", "--min-pep-length", "4", "--min-nuc-length", "4",
                        "--dockq"], env=ENV, capture_output=True, timeout=900)
    try:
        d = json.load(open(o))
    except Exception:
        return {}
    qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
    try:
        M = json.load(open(f"{PRED}/{pdb}/seed_{s}/predictions/"
                          f"{pdb}_summary_confidence_sample_{k}.json"))["chain_pair_iptm"]
    except Exception:
        return {}
    out = {}
    for c1, c2 in by_pdb[pdb]:
        hit = None
        for i, t in enumerate(ifs):
            if {t[0], t[1]} == {c1, c2} and i < len(qs):
                x, y = ord(t[2]) - 65, ord(t[3]) - 65
                pi = M[x][y] if (0 <= x < len(M) and 0 <= y < len(M)) else -1
                hit = (pi, qs[i]); break
        if hit is None:
            # labelled pair absent from OST's global mapping -> documented isolated-pair fallback
            dq = isolated_pair_dockq(pdb, c1, c2, cif, f"{pdb}_{c1}{c2}_{s}_{k}")
            if dq is not None:
                x, y = ord(c1) - 65, ord(c2) - 65
                pi = M[x][y] if (0 <= x < len(M) and 0 <= y < len(M)) else -1
                hit = (pi, dq)
        if hit is not None:
            out[(c1, c2)] = hit
    return out


res = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}
pdbs = sorted(by_pdb)
for n, pdb in enumerate(pdbs, 1):
    if any(f"{pdb}_" in k for k in res):
        continue
    jobs = [(pdb, s, k) for s in SEEDS for k in SAMPLES]
    with ThreadPoolExecutor(NWORK) as ex:
        got = list(ex.map(run_pose, jobs))
    for c1, c2 in by_pdb[pdb]:
        poses = [g[(c1, c2)] for g in got if (c1, c2) in g]
        label = f"{pdb}_{c1}{c2}"
        if not poses:
            res[label] = {"pdb": pdb, "chains": f"{c1}-{c2}", "status": "not_scored", "n": 0}
            continue
        sel = max(poses, key=lambda x: x[0])
        ok = [p for p in poses if p[1] >= THRESH]
        try:
            ctc = contact_residues(pdb, c1, c2)
        except Exception:
            ctc = None
        status = ("no_acceptable_pose" if not ok
                  else ("solved" if sel[1] >= THRESH else "confident_decoy"))
        res[label] = {"pdb": pdb, "chains": f"{c1}-{c2}", "contact_residues": ctc,
                      "n": len(poses), "n_acceptable": len(ok),
                      "top_pair_iptm": round(sel[0], 3), "sel_dockq": round(sel[1], 3),
                      "best_dockq": round(max(p[1] for p in poses), 3), "status": status}
    json.dump(res, open(OUTJ, "w"), indent=1)
    c = collections.Counter(v["status"] for v in res.values())
    print(f"[{n:3d}/{len(pdbs)}] {pdb}: " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())), flush=True)

c = collections.Counter(v["status"] for v in res.values())
print("\n=== FINAL, seeds 101-150, all FoldBench ab-ag interfaces ===")
for k, v in sorted(c.items()):
    print(f"  {k:22} {v}")
dec = sorted([k for k, v in res.items() if v["status"] == "confident_decoy"])
print(f"\nconfident decoys ({len(dec)}):")
for k in dec:
    v = res[k]
    warn = "   <-- interface only %s contact residues" % v["contact_residues"] \
        if (v["contact_residues"] is not None and v["contact_residues"] < 10) else ""
    print(f"  {k:12} contacts={str(v['contact_residues']):>4} n_ok={v['n_acceptable']:3d} "
          f"top_ipTM={v['top_pair_iptm']:.3f} sel={v['sel_dockq']:.3f} best={v['best_dockq']:.3f}{warn}")
print(f"\nWROTE {OUTJ}")
