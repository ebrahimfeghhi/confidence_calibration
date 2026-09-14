#!/usr/bin/env python
"""Per-pose epitope overlap with the ground truth, for the PXM-2025 ab-ag set.

The PXM-2025 twin of scripts/epitope_overlap.py. Same question, same definitions, independent
benchmark: on FoldBench, GAIN IN EPITOPE JACCARD explains r^2 = 0.82 of the gain in DockQ from seed
scaling, while paratope change adds ~0.006 uniquely. PXM-2025 shares no PDB entries with FoldBench and
is drawn from 2025 depositions, so this is the replication test for that result.

WHY NOT fnat. fnat counts native CONTACT PAIRS recovered, so it scores epitope choice and docking
register together. A pose on exactly the right antigen patch but rotated within it keeps its epitope
overlap while fnat collapses. Separating those is the whole point.

  recall  = |pred & GT| / |GT|         of the true epitope, how much did we touch? Blind to
                                       over-prediction: a pose smeared over the whole antigen scores 1.
  jaccard = |pred & GT| / |pred | GT|  penalises touching non-native surface, but is therefore also
                                       sensitive to interface size.

DEFINITIONS, identical to the FoldBench script so the two are comparable:
  - epitope of a pose = antigen residues with a heavy atom within 5.0 A of the antibody chain (the
    CAPRI interface definition), via the shared contact_residues() routine.
  - native epitope = the same computation on the reference. Fixed per interface.
  - model residue numbers are aligned into GT numbering by sequence alignment of the antigen chain
    before intersecting; the two numbering schemes are NOT interchangeable.

THE TWO PXM-SPECIFIC INPUTS, and why they are safe:

  ROLES come from artifacts/pxm2025/chain_roles.csv (scripts/pxm_chain_roles.py). PXMeter's
  interfaces.csv does not say which chain is the antibody -- the pair is stored alphabetically sorted,
  so chain_id_1 carries no role information (it agrees with the shared-chain rule 52 times and
  disagrees 93). Roles were established by hmmscan against ANARCI's Ig V-domain profiles, which splits
  the two populations disjointly: antibody chains score 55-203, antigen chains score exactly 0.0, all
  314 of them. Getting this backwards would silently measure the PARATOPE instead and still emit a
  plausible number, so it is resolved up front rather than assumed here.

  CHAIN CORRESPONDENCE comes per-pose from PXMeter's own `ref_to_model_chain_mapping`, already on disk
  in pxm_json/<pdb>_<seed>_<sample>.json -- the same mapping the DockQ in dockq_pxm2025.csv was
  computed with. It must be read PER POSE, not once per complex: multi-copy assemblies map the
  reference antigen onto a different model copy from one sample to the next. Verified that the chain
  labels in chain_roles.csv, dockq_pxm2025.csv and these mappings are the same namespace
  (label_asym_id) and join 146/146 with zero mismatches, and that OST reads these reference files with
  label_asym_id naming too.

EMPTY SETS ARE A TRAP: an interface whose native epitope is empty would give jaccard 0/0. Those are
dropped and counted, never emitted as zeros.

Usage:
  /home/ebrahim/miniconda3/envs/foldbench/bin/python scripts/epitope_overlap_pxm.py [--jobs 24]
      [--complexes 8rwb,9vca] [--out artifacts/pxm2025/epitope_overlap_pxm.csv]
Resumable: poses already present in the output file are skipped.
Outputs: pdb,ab_chain,ag_chain,seed,sample,n_native,n_pred,n_inter,recall,jaccard
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdr_h3_rmsd import _ost, align_map, chain_residues  # noqa: E402
from epitope_clusters import contact_residues  # noqa: E402

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
COLS = ["pdb", "ab_chain", "ag_chain", "seed", "sample",
        "n_native", "n_pred", "n_inter", "recall", "jaccard"]


def native_sets(roles, side="epi"):
    """{(pdb, ab, ag): native contact residues on the chosen SIDE}, computed once per interface.

    side="epi"  -> residues on the ANTIGEN within 5 A of the antibody  (the epitope)
    side="para" -> residues on the ANTIBODY within 5 A of the antigen  (the paratope)
    contact_residues(ent, X, Y) returns residues of Y near X, so the two differ only by argument
    order -- the geometry and cutoff are identical.
    """
    _, io, _ = _ost()
    out, cache, dropped = {}, {}, []
    for r in roles:
        pdb, ab, ag = r["pdb"], r["ab_chain"], r["ag_chain"]
        gt = f"{P}/cif/{pdb}.cif"
        if not os.path.exists(gt):
            dropped.append((f"{pdb}_{ab}{ag}", "no reference cif"))
            continue
        if pdb not in cache:
            cache[pdb] = io.LoadMMCIF(gt, fault_tolerant=True)
        try:
            nat = (contact_residues(cache[pdb], ab, ag) if side == "epi"
                   else contact_residues(cache[pdb], ag, ab))
        except Exception as e:
            dropped.append((f"{pdb}_{ab}{ag}", f"{type(e).__name__}: {e}"))
            continue
        if not nat:
            dropped.append((f"{pdb}_{ab}{ag}", f"native {side} set EMPTY"))
            continue
        out[(pdb, ab, ag)] = nat
    return out, dropped


def score_pose(arg):
    pdb, ab, ag, seed, smp, native, side = arg
    try:
        _, io, _ = _ost()
        # PXMeter's own reference->model correspondence FOR THIS POSE (see module docstring).
        jf = f"{P}/pxm_json/{pdb}_{seed}_{smp}.json"
        r2m = json.load(open(jf)).get("ref_to_model_chain_mapping", {})
        if ab not in r2m or ag not in r2m:
            return ["__ERR__", f"{pdb}_{ab}{ag}", f"seed {seed} sample {smp}: "
                    f"chain not in PXMeter mapping (have {sorted(r2m)})"]
        m_ab, m_ag = r2m[ab], r2m[ag]
        # For the paratope everything swaps: align the ANTIBODY chain into GT numbering and take
        # contacts on the antibody. Roles are otherwise untouched.
        ref_c, m_c, m_other = ((ag, m_ag, m_ab) if side == "epi" else (ab, m_ab, m_ag))

        cif = f"{P}/out/{pdb}/{pdb}/seed_{seed}/predictions/{pdb}_sample_{smp}.cif"
        mdl = io.LoadMMCIF(cif, fault_tolerant=True)
        ref = io.LoadMMCIF(f"{P}/cif/{pdb}.cif", fault_tolerant=True)
        mres, rres = chain_residues(mdl, m_c), chain_residues(ref, ref_c)
        if not mres or not rres:
            return ["__ERR__", f"{pdb}_{ab}{ag}",
                    f"empty chain (model {len(mres)}, GT {len(rres)})"]

        amap = align_map("".join(x[1] for x in mres), "".join(x[1] for x in rres))
        m2g = {mres[i][0]: rres[j][0] for i, j in amap.items()}
        pred = {m2g[x] for x in contact_residues(mdl, m_other, m_c) if x in m2g}
        inter, union = pred & native, pred | native
        return [pdb, ab, ag, seed, smp, len(native), len(pred), len(inter),
                round(len(inter) / len(native), 4),
                round(len(inter) / len(union), 4) if union else 0.0]
    except Exception as e:
        return ["__ERR__", f"{pdb}_{ab}{ag}", f"seed {seed} sample {smp}: {type(e).__name__}: {e}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--complexes", default="")
    ap.add_argument("--side", choices=("epi", "para"), default="epi")
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    if A.out is None:
        A.out = f"{P}/{'epitope' if A.side == 'epi' else 'paratope'}_overlap_pxm.csv"
    want = set(A.complexes.split(",")) if A.complexes else None

    roles = [r for r in csv.DictReader(open(f"{P}/chain_roles.csv"))
             if r["status"].startswith("ok") and (want is None or r["pdb"] in want)]
    if not roles:
        sys.exit("no interfaces selected")
    native, dropped = native_sets(roles, A.side)
    print(f"{len(roles)} interfaces, native {A.side} set computed for {len(native)}",
          file=sys.stderr)
    for k, why in dropped:
        print(f"  DROPPED {k}: {why}", file=sys.stderr)

    done = set()
    if os.path.exists(A.out):
        for r in csv.DictReader(open(A.out)):
            done.add((r["pdb"], r["ab_chain"], r["ag_chain"], int(r["seed"]), int(r["sample"])))
    print(f"{len(done)} poses already scored, skipping", file=sys.stderr)

    # enumerate poses from the PXMeter jsons -- exactly the poses that carry a DockQ
    jobs = []
    for (pdb, ab, ag), nat in native.items():
        for f in glob.glob(f"{P}/pxm_json/{pdb}_*.json"):
            m = re.search(rf"{pdb}_(\d+)_(\d+)\.json$", f)
            if not m:
                continue
            seed, smp = int(m.group(1)), int(m.group(2))
            if (pdb, ab, ag, seed, smp) not in done:
                jobs.append((pdb, ab, ag, seed, smp, nat, A.side))
    print(f"{len(jobs)} poses to score on {A.jobs} workers", file=sys.stderr)
    if not jobs:
        return

    new = not os.path.exists(A.out)
    errs = collections.Counter()
    n = 0
    with open(A.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(COLS)
        with Pool(A.jobs) as pool:
            for row in pool.imap_unordered(score_pose, jobs, chunksize=8):
                if row[0] == "__ERR__":
                    errs[row[1]] += 1
                    if sum(errs.values()) <= 10:
                        print(f"  ERR {row[1]}: {row[2]}", file=sys.stderr)
                    continue
                w.writerow(row)
                n += 1
                if n % 2000 == 0:
                    fh.flush()
                    print(f"  {n}/{len(jobs)}", file=sys.stderr)
    print(f"\nwrote {n} poses to {A.out}")
    if errs:
        print(f"{sum(errs.values())} errors over {len(errs)} interfaces:")
        for k, v in errs.most_common(10):
            print(f"  {v:>5}  {k}")


if __name__ == "__main__":
    main()
