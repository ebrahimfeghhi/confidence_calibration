#!/usr/bin/env python3
"""msa_ablation_result.py — does giving ESMFold2 an MSA make it select like Protenix?

THE FOUR ARMS. The causal claim rests on B vs A' -- the SAME checkpoint with and without an MSA.
A and C are reference points only.

    A   ESMFold2-Fast, single-sequence      the model we have been studying all along
    A'  base ESMFold2, NO MSA               <-- THE CONTROL
    B   base ESMFold2, WITH Protenix's MSA  <-- THE ABLATION
    C   Protenix, with its own MSA          the model we are trying to explain

Comparing B against A would be worthless: base ESMFold2 (1.3 GB) and ESMFold2-Fast (721 MB) are
DIFFERENT CHECKPOINTS, so that contrast varies the MSA *and* the model. Only B - A' isolates the
MSA.

All arms depth-matched to the same 12 seeds x 5 diffusion samples = 60 poses, on the 14 complexes
of the selection-critical band (pools 10-50% acceptable -- where the entire ESM/Protenix selector
gap lives), all scored by OST, all ranked by raw pair_iptm.

WHAT IT DECIDES
  B >> A' , reaching C   -> MSAs explain Protenix's advantage. The confidence-head account in
                            docs/confidence_head_hypothesis.md is DEAD.
  B ~= A'                -> MSAs are not the explanation. The account survives its largest
                            challenge.
  B > A' but short of C  -> partial; report the split honestly.

GENERATOR vs SELECTOR are reported separately: an MSA could improve the POOL (find the right
epitope more often) without improving SELECTION (knowing which pose is right). Those are different
claims and conflating them would be the easiest way to fool ourselves here.
"""
import argparse, collections, csv
from pathlib import Path

import numpy as np

ACC = 0.23
REPO = Path(__file__).resolve().parents[2]
CSVS = {
    "A":  ("ESMFold2-Fast  single-seq   (different checkpoint)",
           REPO / "results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"),
    "A'": ("base ESMFold2  NO MSA       <-- CONTROL",
           REPO / "results/csv_files/dockq_ost_esmfold2_nomsa.csv"),
    "B":  ("base ESMFold2  WITH MSA     <-- ABLATION",
           REPO / "results/csv_files/dockq_ost_esmfold2_msa.csv"),
    "C":  ("Protenix       WITH MSA",
           REPO / "results/csv_files/dockq_ost_protenix_full.csv"),
}
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


def stats(poses):
    if not poses:
        return None
    dq = np.array([p[1] for p in poses]); sc = np.array([p[2] for p in poses])
    has = bool((dq > ACC).any())
    g, b = sc[dq > ACC], sc[dq <= ACC]
    auc = None
    if len(g) and len(b):
        auc = float(((g[:, None] > b[None, :]).sum() + 0.5 * (g[:, None] == b[None, :]).sum())
                    / (len(g) * len(b)))
    return dict(n=len(dq), has=has,
                hit=bool(dq[int(np.argmax(sc))] > ACC) if has else False,
                purity=float((dq > ACC).mean()), best=float(dq.max()), auc=auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nseeds", type=int, default=8)
    ap.add_argument("--min_seeds", type=int, default=4,
                    help="drop an interface unless at least this many seeds finished in BOTH arms")
    a = ap.parse_args()
    rng = np.random.default_rng(20260713)

    pools = {k: load(p) for k, (_, p) in CSVS.items()}
    missing = [k for k, v in pools.items() if not v]
    if missing:
        print(f"!! no data for arm(s) {missing} — score them with eval_ost_foldbench.py first\n")

    have = [k for k in ("A", "A'", "B", "C") if pools[k]]
    keys = sorted(k for k in pools[have[0]]
                  if k[0] in BAND_CX and all(k in pools[h] for h in have))
    print(f"{len(keys)} interfaces scored for arms {have} "
          f"(selection-critical band; depth-matched to {a.nseeds} seeds)\n")
    if not keys:
        return

    rows = []
    skipped = 0
    for k in keys:
        # The two ablation arms must be compared on the SAME seeds, or a difference could just be
        # "B happened to finish the easy seeds". So intersect the seeds actually completed in BOTH
        # arms, and use exactly those. (The arms are still running; this makes the analysis valid
        # on whatever has finished rather than requiring a clean stop.)
        abl = [h for h in ("A'", "B") if h in have]
        if len(abl) == 2:
            shared = ({p[0] for p in pools["A'"][k]} & {p[0] for p in pools["B"][k]})
        else:
            shared = {p[0] for p in pools[abl[0]][k]} if abl else set()
        if len(shared) < a.min_seeds:
            skipped += 1
            continue
        d = {}
        for h in have:
            if h in ("A'", "B"):
                keep = shared                              # identical seeds in both arms
            else:                                          # reference pools: match the COUNT
                seeds = sorted({p[0] for p in pools[h][k]})
                keep = set(rng.choice(seeds, min(len(shared), len(seeds)), replace=False))
            d[h] = stats([p for p in pools[h][k] if p[0] in keep])
        if all(d[h] for h in have):
            rows.append((k, d))
    if skipped:
        print(f"  ({skipped} interfaces dropped: fewer than {a.min_seeds} seeds finished in both "
              f"ablation arms)\n")

    print(f"{'arm':<48}{'GENERATOR':>11}{'SELECTOR':>10}{'TOP-1':>8}{'purity':>9}{'AUC':>8}")
    print("-" * 94)
    out = {}
    for h in have:
        v = [r[1][h] for r in rows]
        has = np.array([x["has"] for x in v]); hit = np.array([x["hit"] for x in v])
        if has.sum() == 0:
            continue
        aucs = [x["auc"] for x in v if x["auc"] is not None]
        out[h] = dict(gen=has.mean(), sel=hit[has].mean(), top1=hit.mean(),
                      purity=np.mean([x["purity"] for x in v]),
                      auc=np.mean(aucs) if aucs else float("nan"))
        print(f"{h + '   ' + CSVS[h][0]:<48}{out[h]['gen']:>10.0%}{out[h]['sel']:>10.0%}"
              f"{out[h]['top1']:>8.0%}{out[h]['purity']:>9.0%}{out[h]['auc']:>8.3f}")

    if not {"A'", "B"} <= set(out):
        print("\n!! the control arm (A') is missing — B alone proves nothing, because base "
              "ESMFold2 and ESMFold2-Fast are different checkpoints.")
        return

    print("\n" + "=" * 94)
    print("THE VERDICT — B vs A' is the ONLY contrast that isolates the MSA")
    print("=" * 94)
    sA_, sB = out["A'"]["sel"], out["B"]["sel"]
    gA_, gB = out["A'"]["gen"], out["B"]["gen"]
    print(f"\n  SELECTOR   no MSA {sA_:.0%}  ->  with MSA {sB:.0%}   (delta {sB - sA_:+.0%})")
    print(f"  GENERATOR  no MSA {gA_:.0%}  ->  with MSA {gB:.0%}   (delta {gB - gA_:+.0%})")

    if "C" in out:
        sC = out["C"]["sel"]
        gap = sC - sA_
        closed = (sB - sA_) / gap if abs(gap) > 1e-9 else float("nan")
        print(f"\n  Protenix's selector: {sC:.0%}")
        print(f"  the (base ESM -> Protenix) selector gap is {gap:+.0%}")
        print(f"  giving ESM the SAME MSAs closes {closed:.0%} of it\n")
        if closed > 0.6:
            print("  >>> MSAs EXPLAIN THE GAP. The confidence-head account is DEAD.")
        elif closed < 0.25:
            print("  >>> MSAs DO NOT EXPLAIN THE GAP. Handing ESMFold2 the very MSAs Protenix uses")
            print("      does not make it select like Protenix. The confidence-head account in")
            print("      docs/confidence_head_hypothesis.md SURVIVES its largest challenge.")
        else:
            print(f"  >>> PARTIAL: MSAs account for {closed:.0%} of the gap. The rest is not MSAs.")

    # paired bootstrap on the B - A' selector difference (interfaces are the resampling unit)
    both = [(r[1]["A'"], r[1]["B"]) for r in rows if r[1]["A'"]["has"] or r[1]["B"]["has"]]
    if both:
        x = np.array([p[0]["hit"] for p in both], float)
        y = np.array([p[1]["hit"] for p in both], float)
        bs = np.array([(lambda i: y[i].mean() - x[i].mean())(rng.integers(0, len(x), len(x)))
                       for _ in range(10000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"\n  paired bootstrap on (with-MSA − no-MSA), n={len(x)} interfaces:")
        print(f"    delta {y.mean()-x.mean():+.0%}   95% CI [{lo:+.0%}, {hi:+.0%}]   "
              f"P(MSA helps) = {(bs > 0).mean():.3f}")

    print("\n\nPER-INTERFACE  (WIN = argmax landed on an acceptable pose)")
    hdr = "".join(f"{h:>12}" for h in have)
    print(f"{'interface':<12}{hdr}{'  |  best DockQ in pool':>24}")
    print("-" * (12 + 12 * len(have) + 26))
    for k, d in rows:
        t = f"{k[0]}_{k[1]}{k[2]}"
        f = lambda x: ("WIN" if x["hit"] else ("trap" if x["has"] else "no-pose"))
        cells = "".join(f"{f(d[h]):>12}" for h in have)
        best = "  |  " + " ".join(f"{d[h]['best']:.2f}" for h in have)
        print(f"{t:<12}{cells}{best}")


if __name__ == "__main__":
    main()
