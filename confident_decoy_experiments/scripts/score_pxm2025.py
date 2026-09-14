#!/usr/bin/env python
"""Score PXM-2025 ab-ag poses with PXMeter's OWN metrics, one row per (interface, seed, sample).

WHY PXMETER AND NOT OST. Every other arm in this project is scored with OST `compare-structures
--dockq`. This arm deliberately is not: PXM-2025 is PXMeter's benchmark, its DockQ is a from-scratch
reimplementation, and scoring it their way keeps our numbers comparable to their published
leaderboard. The consequence is that PXM-2025 DockQ is NOT directly comparable to the FoldBench
numbers in artifacts/abag_selection_protocol.csv -- treat the two as separate benchmarks and compare
RELATIVE effects (1-seed vs 50-seed gains), never absolute DockQ.

CHAIN IDS. PXMeter reports interfaces keyed by REFERENCE chain ("A,E"), and its target list
(artifacts/pxm2025/interfaces.csv) names ab-ag pairs in the same reference space, so the two join
directly. Protenix's confidence lives in MODEL chain space, so pair_iptm is looked up after mapping
through the `ref_to_model_chain_mapping` that PXMeter returns for each pose -- the same
reference-vs-model distinction that, when assumed away, put 72 of 161 FoldBench interfaces on the
wrong chains.

Resumable: a pose whose pxm JSON already exists is skipped, so this can run repeatedly while the
sweep is still generating.

Usage: .../envs/pxmeter/bin/python scripts/score_pxm2025.py [--jobs 24] [--only 9h1j,9h77]
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
PXM = f"{EXP}/envs/pxmeter/bin/pxm"
JSON_DIR = f"{P}/pxm_json"
OUT = f"{P}/dockq_pxm2025.csv"
N_SAMPLE = 5
PAT = re.compile(r"seed_(\d+)/predictions/(.+?)_sample_(\d+)\.cif$")


def ab_pairs():
    """-> {pdb: [(ref_chain_1, ref_chain_2), ...]} for the ab-ag interfaces only."""
    out = {}
    for r in csv.DictReader(open(f"{P}/interfaces.csv")):
        out.setdefault(r["entry_id"], []).append((r["chain_id_1"], r["chain_id_2"]))
    return out


def complete(pdb, partial=False):
    """A complex is ready to score once all 50 seeds have all N_SAMPLE poses.

    `partial` exists only to smoke-test the pipeline mid-sweep; the selection protocol needs the
    full 50 seeds, so real runs must use the default.
    """
    n = len(glob.glob(f"{P}/out/{pdb}/**/seed_*/predictions/*_sample_*.cif", recursive=True))
    return n > 0 if partial else n >= 50 * N_SAMPLE


# Pin every worker to ONE thread. numpy/scipy inside pxm will otherwise grab all cores, so a
# 24-worker pool becomes ~100 cores and starves the GPU jobs -- exactly what happened when a
# 64-worker OST pool dropped GPU0/GPU4 to 0-2% utilisation.
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")


def run_one(a):
    cif, ref, out_json = a
    if os.path.exists(out_json) and os.path.getsize(out_json) > 0:
        return 0
    try:
        subprocess.run([PXM, "-r", ref, "-m", cif, "-o", out_json],
                       env=ENV, capture_output=True, timeout=900)
    except subprocess.TimeoutExpired:
        return 0
    return 1


def chain_idx(ch):
    """model chain letter -> index into chain_pair_iptm (A=0, B=1, ... AA=26)."""
    idx = 0
    for c in ch.split("-")[0]:
        if not c.isalpha():
            break
        idx = idx * 26 + (ord(c.upper()) - ord("A") + 1)
    return idx - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--only", default=None)
    ap.add_argument("--partial", action="store_true", help="smoke-test only: score incomplete complexes")
    A = ap.parse_args()
    os.makedirs(JSON_DIR, exist_ok=True)
    pairs = ab_pairs()
    pdbs = [p for p in sorted(pairs) if complete(p, A.partial)]
    if A.only:
        want = set(A.only.split(","))
        pdbs = [p for p in pdbs if p in want]
    if not pdbs:
        print("no complete complexes yet")
        return

    jobs = []
    for pdb in pdbs:
        for cif in glob.glob(f"{P}/out/{pdb}/**/seed_*/predictions/*_sample_*.cif", recursive=True):
            m = PAT.search(cif)
            jobs.append((cif, f"{P}/cif/{pdb}.cif",
                         f"{JSON_DIR}/{pdb}_{m.group(1)}_{m.group(3)}.json"))
    todo = [j for j in jobs if not (os.path.exists(j[2]) and os.path.getsize(j[2]) > 0)]
    print(f"{len(pdbs)} complete complexes, {len(jobs)} poses, {len(todo)} to score")
    if todo:
        with ProcessPoolExecutor(max_workers=A.jobs) as ex:
            for _ in ex.map(run_one, todo, chunksize=4):
                pass

    rows = []
    for pdb in pdbs:
        for f in sorted(glob.glob(f"{JSON_DIR}/{pdb}_*.json")):
            m = re.search(rf"{pdb}_(\d+)_(\d+)\.json$", f)
            if not m:
                continue
            seed, smp = m.group(1), m.group(2)
            try:
                d = json.load(open(f))
            except Exception:
                continue
            r2m = d.get("ref_to_model_chain_mapping", {})
            conf = None
            cf = (f"{P}/out/{pdb}/{pdb}/seed_{seed}/predictions/"
                  f"{pdb}_summary_confidence_sample_{smp}.json")
            if os.path.exists(cf):
                try:
                    conf = json.load(open(cf)).get("chain_pair_iptm")
                except Exception:
                    conf = None
            for c1, c2 in pairs[pdb]:
                v = d.get("interface", {}).get(f"{c1},{c2}") or \
                    d.get("interface", {}).get(f"{c2},{c1}")
                if v is None:
                    continue
                pi = None
                if conf and c1 in r2m and c2 in r2m:
                    i, j = chain_idx(r2m[c1]), chain_idx(r2m[c2])
                    if 0 <= i < len(conf) and 0 <= j < len(conf[i]):
                        pi = conf[i][j]
                rows.append(dict(pdb=pdb, chain1=c1, chain2=c2, seed=int(seed), sample=int(smp),
                                 dockq=v.get("dockq"), lddt=v.get("lddt"),
                                 bb_lddt=v.get("bb_lddt"), pair_iptm=pi))
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_if = len({(r["pdb"], r["chain1"], r["chain2"]) for r in rows})
    print(f"wrote {OUT}: {len(rows)} rows, {n_if} ab-ag interfaces, {len(pdbs)} complexes")
    miss = sum(1 for r in rows if r["pair_iptm"] is None)
    if miss:
        print(f"  WARNING {miss} rows have no pair_iptm")


if __name__ == "__main__":
    main()
