#!/usr/bin/env python
"""Score DockQ for the free Protenix run at seeds 101-150 and select the confident decoys.

Confident decoy = over an interface's pose pool, BOTH:
  (1) the top interface-pair_iptm pose has DockQ < 0.23   (selection fails)
  (2) at least one pose has DockQ >= 0.23                 (a correct pose exists)

Scores every pose itself via OST (cached, so re-runs are cheap), so the set is reproducible from
raw predictions + ground truths alone -- no dependency on any pre-existing scores CSV.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/decoy_set.py
Writes: artifacts/decoy_set_101_150.json
"""
import argparse, json, os, subprocess, collections, sys
import numpy as np
import gemmi
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import EXP, CODE_ROOT, ARTIFACTS_DIR, PREDICTIONS_DIR
PRED = "/home/ebrahim/Protenix/foldbench_abag/predictions_template"
GT = "/home/ebrahim/FoldBench/examples/ground_truths"
OST = "/home/ebrahim/miniconda3/envs/foldbench/bin/ost"
ENV = dict(os.environ, LD_LIBRARY_PATH="/home/ebrahim/miniconda3/envs/foldbench/lib")
SEEDS, SAMPLES, THRESH = range(101, 151), range(5), 0.23

# Two interface sets, scored identically so decoy/good verdicts are directly comparable:
#   decoys = the confident-decoy candidates (meta.json)
#   goods  = the control set that selection already handles (safety15.json); a template method must
#            not BREAK these, so "still good at these seeds" is the safety baseline.
_ap = argparse.ArgumentParser()
_ap.add_argument("--set", dest="which", choices=["decoys", "goods", "extra"], default="decoys")
ARGS = _ap.parse_known_args()[0]   # parse_known_args: this module is imported by other scripts that have their own flags
_TAG = {"decoys": "decoy_set", "goods": "goods_set", "extra": "extra_set"}[ARGS.which]
CACHE = f"{EXP}/artifacts/{_TAG}_ost"
OUTJ = f"{EXP}/artifacts/{_TAG}_101_150.json"
os.makedirs(CACHE, exist_ok=True)
# "extra" = the 10 confident decoys in the canonical CSV that are NOT in our 15-interface set, so we
# can check whether they still qualify at seeds 101-150 (most were only ever scored at seeds 0-49).
_SRC = {"decoys": "meta.json", "goods": "safety15.json", "extra": "extra10.json"}[ARGS.which]
_raw = json.load(open(f"{EXP}/artifacts/{_SRC}"))
if ARGS.which == "decoys":
    meta = _raw
else:
    meta = {e["iface"]: {"pdb": e["pdb"], "c1": e["c1"], "c2": e["c2"]} for e in _raw}

# 7z2m: FoldBench labels interface A-J, but native A and J touch at only 3 residues. The assembly is
# two symmetry copies -- Fab(A,B) binds antigen K, Fab(E,F) binds antigen J -- so A-J pairs a chain
# from one copy with the other copy's antigen (a crystal contact). Corrected to the real interfaces,
# following FoldBench's own one-row-per-(antibody chain, antigen) convention. Verified 2026-08-01;
# 7z2m is the ONLY interface in this set whose label is wrong (all others have 11-55 contact residues).
INTERFACE_OVERRIDE = {"7z2m_AJ": [("A", "K"), ("B", "K")]}

# NOTE ON MULTI-COPY ASSEMBLIES (7z2m): an earlier version of this script subset the NATIVE to the
# interface pair and handed OST the full model ("copy-optimal"). That is NOT the field/project
# protocol and it was wrong: scoring each pair independently let OST choose a DIFFERENT predicted copy
# for A-K than for B-K (they disagreed on 30 of 50 poses), so the two interfaces measured different
# physical Fabs. We now follow tts_abag/eval/eval_ost_foldbench.py: ONE OST call on the full native
# assembly vs the full model, OST's own global chain mapping, then extract the labelled pair BY NAME.


def native_chains(pdb):
    """Native polymer chains keyed by label_asym_id -> (coords Nx3, per-atom residue index).

    ALWAYS use this instead of gemmi.read_structure(...).name. FoldBench's targets CSV, meta.json and
    OST's dockq_interfaces all speak label_asym_id, but gemmi's structure reader names chains by
    auth_asym_id -- they differ for 8bf4/8che/8fbw/8slb/8the, so hand-rolled lookups silently compare
    the WRONG chains. Raises on a missing file rather than returning empty.
    """
    path = f"{GT}/{pdb}-assembly1.cif"
    if not os.path.exists(path):
        raise FileNotFoundError(f"no ground truth for {pdb}: {path}")
    block = gemmi.cif.read(path).sole_block()
    cols = [list(block.find_values("_atom_site." + c)) for c in
            ("label_asym_id", "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z", "group_PDB")]
    D = collections.defaultdict(lambda: ([], []))
    for a, s, x, y, z, g in zip(*cols):
        if g != "ATOM":
            continue
        D[a][0].append([float(x), float(y), float(z)])
        D[a][1].append(s)
    if not D:
        raise ValueError(f"{pdb}: no polymer atoms parsed")
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in D.items()}


def contact_residues(pdb, c1, c2, cutoff=5.0):
    """# residues within `cutoff` across the (c1, c2) interface. RAISES if either chain is absent --
    never returns a sentinel, because a sentinel printed into a table reads as evidence."""
    D = native_chains(pdb)
    missing = [c for c in (c1, c2) if c not in D]
    if missing:
        raise KeyError(f"{pdb}: chain(s) {missing} not in native (have {sorted(D)}); "
                       f"label_asym_id vs auth_asym_id mix-up?")
    XA, IA = D[c1]
    XB, IB = D[c2]
    m = np.linalg.norm(XA[:, None, :] - XB[None, :, :], axis=-1) < cutoff
    return (len(set(IA[m.any(1)])) + len(set(IB[m.any(0)]))) if m.any() else 0


def pose(args):
    """DockQ + interface pair_iptm for one (label, seed, sample); None if unavailable."""
    label, pdb, c1, c2, s, k = args
    cif = f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif"
    if not os.path.exists(cif):
        return None
    o = f"{CACHE}/{label}_{s}_{k}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset", "-c", "0-31", OST, "compare-structures",
                        "-m", cif, "-r", f"{GT}/{pdb}-assembly1.cif", "-o", o,
                        "--fault-tolerant", "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq"],
                       env=ENV, capture_output=True, timeout=600)
    try:
        d = json.load(open(o))
    except Exception:
        return None
    qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
    dq = mc = None
    for i, t in enumerate(ifs):
        # Strip OST's copy suffix ("A-2" -> "A") before matching. In a multi-copy native assembly OST
        # maps our single predicted copy to WHICHEVER native copy fits best, so a literal comparison
        # keeps only the poses that happened to land on copy 1. On 8fxv that silently discarded 169 of
        # 369 poses in d50_s1e4 and 164 of 364 in bo50 -- a BIASED subset, not merely a smaller one,
        # because the copy OST selects can correlate with pose quality. 8fxv is the only multi-copy
        # complex in this project (verified across all 1176 OST cache dirs, 2026-08-28).
        if {t[0].split("-")[0], t[1].split("-")[0]} == {c1, c2} and i < len(qs):
            if dq is None or qs[i] > dq:      # if both copies match, keep the better
                dq, mc = qs[i], (t[2], t[3])
    if dq is None:                    # labelled pair absent from OST's mapping -> no score, not a guess
        return None
    if dq is None:
        return None
    pi = -1.0
    try:  # pair_iptm at the OST-mapped MODEL chains
        M = json.load(open(f"{PRED}/{pdb}/seed_{s}/predictions/"
                           f"{pdb}_summary_confidence_sample_{k}.json"))["chain_pair_iptm"]
        a, b = ord(mc[0]) - 65, ord(mc[1]) - 65
        if 0 <= a < len(M) and 0 <= b < len(M):
            pi = M[a][b]
    except Exception:
        pass
    return pi, dq


def main():
    # Expand meta into the interfaces we actually score, applying INTERFACE_OVERRIDE.
    targets = []
    for ifc, m in sorted(meta.items()):
        for c1, c2 in INTERFACE_OVERRIDE.get(ifc, [(m["c1"], m["c2"])]):
            label = ifc if (c1, c2) == (m["c1"], m["c2"]) else f'{m["pdb"]}_{c1}{c2}'
            targets.append((label, m["pdb"], c1, c2))

    # GUARD: every scored interface must be a real contact. Raises (never warns) on a bad label, so a
    # crystal-contact interface like the original 7z2m A-J can't silently enter the benchmark again.
    for label, pdb, c1, c2 in targets:
        n = contact_residues(pdb, c1, c2)
        if n < 10:
            raise ValueError(f"{label}: native {c1}-{c2} has only {n} contact residues -- not a real "
                             f"interface. Fix the label (add to INTERFACE_OVERRIDE) before scoring.")

    per, decoys = {}, []
    for label, pdb, c1, c2 in targets:
        with ThreadPoolExecutor(8) as ex:
            poses = [r for r in ex.map(pose, [(label, pdb, c1, c2, s, k)
                                              for s in SEEDS for k in SAMPLES]) if r]
        if not poses:
            raise RuntimeError(f"{label}: no poses scored -- check predictions exist for {pdb}")
        sel = max(poses, key=lambda x: x[0])
        ok = [p for p in poses if p[1] >= THRESH]
        is_decoy = sel[1] < THRESH and len(ok) > 0
        per[label] = {"pdb": pdb, "chains": f"{c1}-{c2}", "n_poses": len(poses),
                      "n_acceptable": len(ok), "top_pair_iptm": round(sel[0], 3),
                      "sel_dockq": round(sel[1], 3), "best_dockq": round(max(p[1] for p in poses), 3),
                      "contact_residues": contact_residues(pdb, c1, c2),
                      "confident_decoy": is_decoy,
                  "selection_ok": sel[1] >= THRESH}
        if is_decoy:
            decoys.append(label)
        v = per[label]
        print(f"{label:10} n={v['n_poses']:3d} n_ok={v['n_acceptable']:3d} top_ipTM={v['top_pair_iptm']:.3f} "
              f"sel_DockQ={v['sel_dockq']:.3f} best_DockQ={v['best_dockq']:.3f} "
              f"{'CONFIDENT DECOY' if is_decoy else ('selection OK' if sel[1] >= THRESH else 'no acceptable pose')}", flush=True)

    good = [k for k, v in per.items() if v["selection_ok"]]
    json.dump({"set": ARGS.which, "seeds": "101-150", "dockq_threshold": THRESH,
               "decoys": decoys, "selection_ok": good, "per_interface": per},
              open(OUTJ, "w"), indent=1)
    print(f"\nselection SUCCEEDS (DockQ >= {THRESH}) for {len(good)}/{len(per)} -> {good}")
    print(f"{len(decoys)} confident decoys -> {decoys}\nWROTE {OUTJ}")


if __name__ == "__main__":
    main()
