#!/usr/bin/env python3
"""msa_ablation_result.py — the three-way comparison that settles the MSA confound.

    A. ESMFold2-Fast  single-sequence  (msa_encoder.enabled = false)   the model we have been using
    B. ESMFold2 base  WITH Protenix's MSAs (msa_encoder.enabled = true)  <- the ablation
    C. Protenix       WITH its own MSAs                                the model we are explaining

All three DEPTH-MATCHED to the same 25 seeds x 5 diffusion samples = 125 poses, on the 14
complexes of the selection-critical band (pools 10-50% acceptable -- where the entire selector gap
between ESM and Protenix lives), all scored by OST, all ranked by raw pair_iptm.

THE LOGIC
  If B jumps to C's selector accuracy -> Protenix's advantage was MSAs all along, and the
     confidence-head account in docs/confidence_head_hypothesis.md is DEAD.
  If B stays at A's level             -> MSAs are not the explanation. The confidence-head
     account survives its largest challenge.
  If B lands between                  -> both contribute; report the split honestly.

The generator/selector split matters here. MSAs could plausibly improve the POOL (find the right
epitope more often) without improving SELECTION (knowing which pose is right). Those are different
claims and this reports them separately.
"""
import argparse, collections, csv, glob, re
from pathlib import Path

import numpy as np

ACC = 0.23
REPO = Path(__file__).resolve().parents[2]
ESM_CSV = REPO / "results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"
PRO_CSV = REPO / "results/csv_files/dockq_ost_protenix_full.csv"
MSA_CSV = REPO / "results/csv_files/dockq_ost_esmfold2_msa.csv"       # produced by the ablation
LOGS = "/home/ebrahim/Protenix/foldbench_abag/predictions_template/gpu*.log"
SCORE = "pair_iptm"
BAND_CX = "7txw 7zoz 8che 8fxv 8ido 8oni 8opr 8q6r 8qz3 8the 8ts0 8uky 8w84 8w90".split()


def load(path):
    by = collections.defaultdict(list)
    if not Path(path).exists():
        return by
    for r in csv.DictReader(open(path)):
        if r.get("dockq_status") not in (None, "", "ok"):
            continue
        try:
            dq, sc = float(r["dockq"]), float(r[SCORE])
        except (TypeError, ValueError, KeyError):
            continue
        if np.isnan(dq) or np.isnan(sc):
            continue
        by[(r["assembly_id"][:4], r["chain1"], r["chain2"])].append((r["seed"], dq, sc))
    return by


def subsample(poses, seeds_keep):
    return [p for p in poses if p[0] in seeds_keep]


def stats(poses):
    if not poses:
        return None
    dq = np.array([p[1] for p in poses]); sc = np.array([p[2] for p in poses])
    has = bool((dq > ACC).any())
    hit = bool(dq[int(np.argmax(sc))] > ACC) if has else False
    g, b = sc[dq > ACC], sc[dq <= ACC]
    auc = None
    if len(g) and len(b):
        auc = float(((g[:, None] > b[None, :]).sum() + 0.5 * (g[:, None] == b[None, :]).sum())
                    / (len(g) * len(b)))
    return dict(n=len(dq), has=has, hit=hit, purity=float((dq > ACC).mean()),
                best=float(dq.max()), auc=auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nseeds", type=int, default=25)
    a = ap.parse_args()
    rng = np.random.default_rng(20260713)

    esm, pro, msa = load(ESM_CSV), load(PRO_CSV), load(MSA_CSV)
    if not msa:
        print(f"!! {MSA_CSV} not found — score the ablation with eval_ost_foldbench.py first")
        return

    keys = sorted(k for k in msa if k[0] in BAND_CX and k in esm and k in pro)
    print(f"{len(keys)} interfaces scored for ALL THREE models "
          f"(selection-critical band, {len(BAND_CX)} complexes)\n")

    # the ablation ran seeds 0..nseeds-1. Match the other two to the SAME count.
    msa_seeds = {str(s) for s in range(a.nseeds)}
    rows = []
    for k in keys:
        e_seeds = sorted({p[0] for p in esm[k]})
        p_seeds = sorted({p[0] for p in pro[k]})
        e = stats(subsample(esm[k], set(rng.choice(e_seeds, min(a.nseeds, len(e_seeds)), replace=False))))
        p = stats(subsample(pro[k], set(rng.choice(p_seeds, min(a.nseeds, len(p_seeds)), replace=False))))
        m = stats(subsample(msa[k], msa_seeds))
        if not (e and p and m):
            continue
        rows.append((k, dict(A=e, B=m, C=p)))

    names = {"A": "ESMFold2-Fast   single-sequence",
             "B": "ESMFold2 base   WITH MSA  <-- the ablation",
             "C": "Protenix        WITH MSA"}
    print(f"{'model':<42}{'GENERATOR':>11}{'SELECTOR':>10}{'TOP-1':>8}{'pool purity':>13}{'AUC':>8}")
    print("-" * 92)
    out = {}
    for key in ("A", "B", "C"):
        v = [r[1][key] for r in rows]
        has = np.array([x["has"] for x in v])
        hit = np.array([x["hit"] for x in v])
        if has.sum() == 0:
            continue
        gen = has.mean()
        sel = hit[has].mean()
        aucs = [x["auc"] for x in v if x["auc"] is not None]
        out[key] = (gen, sel, hit.mean())
        print(f"{names[key]:<42}{gen:>10.0%}{sel:>10.0%}{hit.mean():>8.0%}"
              f"{np.mean([x['purity'] for x in v]):>13.0%}"
              f"{(np.mean(aucs) if aucs else float('nan')):>8.3f}")

    if not {"A", "B", "C"} <= set(out):
        return
    print("\n" + "=" * 92)
    print("THE VERDICT")
    print("=" * 92)
    (gA, sA, tA), (gB, sB, tB), (gC, sC, tC) = out["A"], out["B"], out["C"]
    gap = sC - sA
    closed = (sB - sA) / gap if abs(gap) > 1e-9 else float("nan")
    print(f"\n  selector: single-seq {sA:.0%}  ->  +MSA {sB:.0%}  ->  Protenix {sC:.0%}")
    print(f"  the ESM->Protenix selector gap is {gap:+.0%}")
    print(f"  giving ESM an MSA closes {closed:.0%} of it\n")
    if closed > 0.6:
        print("  >>> MSAs EXPLAIN THE GAP. The confidence-head account is DEAD -- it was MSAs.")
    elif closed < 0.25:
        print("  >>> MSAs DO NOT EXPLAIN THE GAP. Giving ESMFold2 the same MSAs Protenix uses does")
        print("      not make it select like Protenix. The confidence-head account SURVIVES its")
        print("      largest challenge.")
    else:
        print(f"  >>> PARTIAL. MSAs account for {closed:.0%} of the gap; the rest is not MSAs.")
    print(f"\n  generator: single-seq {gA:.0%}  ->  +MSA {gB:.0%}  ->  Protenix {gC:.0%}")
    print("  (MSAs may improve the POOL without improving SELECTION -- different claims.)")

    print("\n\nPER-INTERFACE")
    print(f"{'interface':<12}{'A single-seq':>14}{'B +MSA':>10}{'C Protenix':>12}{'A best':>9}{'B best':>9}{'C best':>9}")
    print("-" * 78)
    for k, d in rows:
        t = f"{k[0]}_{k[1]}{k[2]}"
        f = lambda x: ("WIN" if x["hit"] else ("trap" if x["has"] else "no-pose"))
        print(f"{t:<12}{f(d['A']):>14}{f(d['B']):>10}{f(d['C']):>12}"
              f"{d['A']['best']:>9.3f}{d['B']['best']:>9.3f}{d['C']['best']:>9.3f}")


if __name__ == "__main__":
    main()
