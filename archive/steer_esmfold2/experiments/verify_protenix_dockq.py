#!/usr/bin/env python3
"""verify_protenix_dockq.py — certify the Protenix pose-vs-pose DockQ matrices against OST.

dockq_torch.py was originally certified against OST on ESMFold2 poses (diffs <= 0.0005, which is
OST's 3-decimal rounding; docs/dockq_ground_truth_and_gpu.md). It has never been run on PROTENIX
files. Different model, different CIF writer, different chain conventions -- so the certification
does not transfer for free, and after the 17/48 chain-mapping disaster the only acceptable
standard is: check it against the independent tool, not against our own earlier code.

METHOD. Take pose pairs spanning the DockQ range. Feed BOTH to OST compare-structures, one as the
"reference" (-r) and one as the "model" (-m) -- pose-vs-pose, the exact regime the matrices live
in, where residue correspondence is free (same sequence, same model, same residue order). Compare
OST's DockQ to our matrix entry.

PASS = every |diff| <= 0.0005.  Anything larger means the matrices are wrong and every downstream
conclusion built on them is void.
"""
import csv, json, os, subprocess, sys, tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths                                       # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PRO_DIR = REPO / "results/dockq_protenix"
PRED = Path.home() / "Protenix/foldbench_abag/predictions_template"
OST = str(paths.OST_BIN)
ENV = dict(os.environ)
ENV["LD_LIBRARY_PATH"] = f"{paths.OST_LIB}:{ENV.get('LD_LIBRARY_PATH','')}"


def ost_dockq(ref_cif, mod_cif, ref_chains, mdl_chains):
    """OST DockQ, reference = ref_cif (chains ref_chains), model = mod_cif (chains mdl_chains).

    THE CORRESPONDENCE MUST BE FORCED. Our matrix defines D[i][j] as: take pose i's copy of the
    native interface (chains given by pose i's chain_mapping) as the reference, and pose j's copy
    (pose j's chain_mapping) as the model. Left to itself, OST re-derives its own chain
    correspondence between the two poses and may pair up a DIFFERENT copy -- on a multi-copy
    assembly it will happily map ref A onto model B. Then OST and we are scoring two different
    comparisons and the diff is meaningless. That is exactly what the first run of this script
    measured (6/12 "mismatches", worst 0.2018): not a DockQ bug, a correspondence mismatch.

    So pass -c REF:MDL explicitly, pinning OST to the same pairing our matrix used.

    OST's JSON reports `dockq` as a LIST parallel to `dockq_interfaces`, entries
    [ref_c1, ref_c2, mdl_c1, mdl_c2]. There is no `dockq_scores` key.
    """
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "o.json"
        cmap = [f"{r}:{m}" for r, m in zip(ref_chains, mdl_chains)]
        cmd = [OST, "compare-structures", "-m", str(mod_cif), "-r", str(ref_cif), "-o", str(out),
               "--fault-tolerant", "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq",
               "-c", *cmap]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=ENV)
        if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            return None
        d = json.loads(out.read_text())
        ifs = d.get("dockq_interfaces") or []
        scores = d.get("dockq") or []
        if not isinstance(scores, list):
            return float(scores)
        want = set(ref_chains)
        for i, iface in enumerate(ifs):
            if i >= len(scores):
                break
            if len(iface) >= 2 and {iface[0], iface[1]} == want:
                return float(scores[i])
        return None


def main():
    tags = sorted(p.name[6:-4] for p in PRO_DIR.glob("dockq_*.npy"))
    if not tags:
        print("no Protenix matrices built yet"); return
    rng = np.random.default_rng(0)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--only", nargs="*", help="check these tags specifically")
    a = ap.parse_args()
    check = ([t for t in a.only if t in tags] if a.only
             else list(rng.choice(tags, min(a.n, len(tags)), replace=False)))

    # model chain letters PER POSE, from the scoring CSV's chain_mapping (native -> model).
    # Using the NATIVE letters here would score the wrong molecules -- the 17/48 bug.
    cm = {}
    for r in csv.DictReader(open(REPO / "results/csv_files/dockq_ost_protenix_full.csv")):
        try:
            m = json.loads(r["chain_mapping"])
            cm[(r["assembly_id"][:4], r["chain1"], r["chain2"], r["seed"], r["sample"])] = \
                (m[r["chain1"]], m[r["chain2"]])
        except Exception:
            pass

    print("CERTIFYING results/dockq_protenix/ AGAINST OST (pose-vs-pose)\n")
    print(f"{'interface':<10}{'pose i':>7}{'pose j':>7}{'our GPU':>10}{'OST':>8}{'diff':>9}  verdict")
    print("-" * 60)
    worst = 0.0
    n_ok = n_bad = n_skip = 0
    for tag in check:
        cx, cc = tag.split("_")
        c1, c2 = cc[0], cc[1]
        D = np.load(PRO_DIR / f"dockq_{tag}.npy").astype(np.float64)
        idx = list(csv.DictReader(open(PRO_DIR / f"dockq_{tag}_index.csv")))
        # rows of the matrix are sorted by ipTM desc; pick pairs spanning the DockQ range
        pairs = [(0, 1), (0, len(idx) // 2), (5, len(idx) - 1), (2, 40)]
        for i, j in pairs:
            if i >= len(idx) or j >= len(idx) or i == j:
                continue
            ri, rj = idx[i], idx[j]
            fi = PRED / cx / f"seed_{ri['seed']}" / "predictions" / f"{cx}_sample_{ri['sample']}.cif"
            fj = PRED / cx / f"seed_{rj['seed']}" / "predictions" / f"{cx}_sample_{rj['sample']}.cif"
            if not (fi.exists() and fj.exists()):
                continue
            # our matrix: D[i][j] = DockQ(reference = pose i, model = pose j), where each pose
            # contributes ITS OWN copy of the native interface (per-pose chain_mapping).
            mine = float(D[i, j])
            ref_ch = cm.get((cx, c1, c2, ri["seed"], ri["sample"]))
            mdl_ch = cm.get((cx, c1, c2, rj["seed"], rj["sample"]))
            if ref_ch is None or mdl_ch is None:
                continue
            got = ost_dockq(fi, fj, ref_ch, mdl_ch)
            if got is None:
                print(f"{tag:<10}{i:>7}{j:>7}{mine:>10.4f}{'--':>8}{'--':>9}  OST failed")
                n_skip += 1
                continue
            diff = abs(mine - got)
            worst = max(worst, diff)
            ok = diff <= 0.0005
            n_ok += ok; n_bad += (not ok)
            print(f"{tag:<10}{i:>7}{j:>7}{mine:>10.4f}{got:>8.3f}{diff:>9.4f}  "
                  f"{'ok' if ok else 'MISMATCH'}")

    print("-" * 60)
    print(f"\n  {n_ok} within tolerance, {n_bad} mismatched, {n_skip} OST failures")
    print(f"  worst |diff| = {worst:.4f}   (tolerance 0.0005 = OST's 3-decimal rounding)")
    if n_bad == 0 and n_ok >= 3:
        print("\n  PASS — the Protenix matrices reproduce OST. Downstream analysis is safe.")
    else:
        print("\n  *** FAIL — do NOT trust results/dockq_protenix/ or anything built on it. ***")


if __name__ == "__main__":
    main()
