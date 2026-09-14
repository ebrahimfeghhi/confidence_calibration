#!/usr/bin/env python3
"""confidence_discrimination.py — is Protenix's confidence a CORRECTNESS discriminator while
ESMFold2's is a SELF-CONSISTENCY readout?

BACKGROUND (established from source, 2026-07-13):
  Protenix's confidence head is trained via an AF3 mini-rollout: sample a structure under
  no_grad (protenix/model/protenix.py:720-755), then regress the head onto THAT SAMPLE'S REAL
  ERROR vs ground truth (loss.py:1757-1795), with the trunk frozen (confidence.py:65).
  It is a discriminator on the model's own decoys, by construction.

  ESMFold2's head is ALSO structure-conditioned -- it takes the sampled coords as input and
  re-runs per diffusion sample (modeling_esmfold2.py:1061-1074). So the architectures are NOT
  as different as first hypothesised. What ESM's head was TRAINED against is not determinable
  (no training code ships). Hence the FUNCTIONAL test below, which needs no training code.

THREE CONFOUNDS THIS SCRIPT KILLS BEFORE MEASURING ANYTHING:

  1. SCORE MISMATCH. ESM's `ranking_score` is bit-identical to its `iptm`; Protenix's is an AF3
     composite (0.8*iptm + 0.2*ptm - 100*has_clash; `disorder` is hard-coded to 0 --
     sample_confidence.py:171-177). Comparing them is comparing a composite to a raw score.
     -> score every model with EVERY column, and report the grid.

  2. DEPTH. ESM has 1025 seeds x 5 samples = 5125 poses; Protenix has 50 x 5 = 250.  Picking the
     best of 5125 is a harder task than best of 250, and 20x more draws inflates coverage.
     -> subsample ESM to 50 seeds, 20 independent draws, report mean +- sd.

  3. TEMPLATE LEAKAGE. Protenix ran with templates ON and 28 complexes found 4 templates each.
     If a template is a homolog of the native, Protenix's skill is partly leakage.
     -> split every result by template status. The zero-template subset is the honest comparison.

METRICS
  GENERATOR : fraction of interfaces whose pool contains an acceptable pose (DockQ > 0.23).
  SELECTOR  : given one exists, fraction where argmax(score) lands on an acceptable pose.
  TOP-1     : generator x selector -- the number a user actually experiences.
  AUC       : P(a random acceptable pose outscores a random bad one), within one pool.
              0.5 = no signal, <0.5 = ANTI-correlated. This is the discrimination measure.
"""
import argparse, collections, csv, json, re, glob, os
import numpy as np

ACC = 0.23
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ESM_CSV = f"{REPO}/results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"
PRO_CSV = f"{REPO}/results/csv_files/dockq_ost_protenix_full.csv"
PRO_LOGS = "/home/ebrahim/Protenix/foldbench_abag/predictions_template/gpu*.log"

# Every score column present in BOTH OST CSVs. Scoring each model with each one is how confound
# #1 dies: if Protenix still wins on the SAME column, the gap is the model, not the formula.
SCORES = ["pair_iptm", "global_iptm", "ptm", "pair_ranking_score", "global_ranking_score"]


def template_status():
    """complex -> did Protenix find any template?

    In the inference logs, "Found N templates for sequence ..." lines PRECEDE the
    "[Rank r (i/n)] <complex> [seed:s]" line for the structure they belong to. So walk FORWARD.
    (Grepping backwards attributes them to the previous complex -- which is the bug that made my
    first attempt report 0 template interfaces.)
    """
    pend, seen = [], collections.defaultdict(set)
    for path in sorted(glob.glob(PRO_LOGS)):
        for line in open(path, errors="ignore"):
            m = re.search(r"Found (\d+) templates", line)
            if m:
                pend.append(int(m.group(1)))
                continue
            m = re.search(r"\]\s+([0-9a-z]{4})\s+\[seed:(\d+)\]", line)
            if m:
                if pend:
                    seen[m.group(1)].update(pend)
                pend = []
    return {c: (max(v) > 0) for c, v in seen.items()}


def load(path):
    """interface -> list of (seed, dockq, {score: value}).  Only OST rows that scored cleanly."""
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r.get("dockq_status") not in (None, "", "ok"):
            continue
        try:
            dq = float(r["dockq"])
        except (TypeError, ValueError):
            continue
        if np.isnan(dq):
            continue
        sc = {}
        for s in SCORES:
            try:
                v = float(r[s])
                sc[s] = None if np.isnan(v) else v
            except (TypeError, ValueError, KeyError):
                sc[s] = None
        key = (r["assembly_id"].split("-")[0], r["chain1"], r["chain2"])   # 7ucx-assembly1 -> 7ucx
        by[key].append((r["seed"], dq, sc))
    return by


def classify(poses, score):
    """WIN / TRAP / NOPOSE under a given scoring column."""
    usable = [(d, s[score]) for _, d, s in poses if s.get(score) is not None]
    if not usable:
        return None
    dq = np.array([u[0] for u in usable])
    sc = np.array([u[1] for u in usable])
    if not (dq > ACC).any():
        return "NOPOSE"
    return "WIN" if dq[int(np.argmax(sc))] > ACC else "TRAP"


def auc(poses, score):
    """P(random acceptable pose outscores a random bad one). 0.5 = nothing, <0.5 = backwards."""
    usable = [(d, s[score]) for _, d, s in poses if s.get(score) is not None]
    dq = np.array([u[0] for u in usable]); sc = np.array([u[1] for u in usable])
    g, b = sc[dq > ACC], sc[dq <= ACC]
    if len(g) == 0 or len(b) == 0:
        return None
    gt = (g[:, None] > b[None, :]).sum()
    eq = (g[:, None] == b[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(g) * len(b)))


def rates(counts):
    n = sum(counts.values())
    cov = counts["WIN"] + counts["TRAP"]
    if n == 0:
        return None
    return dict(n=n,
                generator=cov / n,
                selector=(counts["WIN"] / cov) if cov else float("nan"),
                top1=counts["WIN"] / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20, help="ESM 50-seed subsample repeats")
    ap.add_argument("--out", default=f"{REPO}/results/csv_files/confidence_analysis.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(20260713)

    tmpl = template_status()
    esm, pro = load(ESM_CSV), load(PRO_CSV)
    common = sorted(set(esm) & set(pro))
    n_t = sum(1 for k in common if tmpl.get(k[0], False))
    print(f"common interfaces (both models scored by OST): {len(common)}")
    print(f"  complexes with template status known : {len({k[0] for k in common if k[0] in tmpl})}")
    print(f"  interfaces whose complex USED templates: {n_t}")
    print(f"  interfaces with ZERO templates         : {len(common) - n_t}   <- clean control\n")

    subsets = {"ALL": common,
               "ZERO-TEMPLATE": [k for k in common if not tmpl.get(k[0], False)],
               "TEMPLATE": [k for k in common if tmpl.get(k[0], False)]}

    rows = []
    for sname, keys in subsets.items():
        if not keys:
            continue
        print("=" * 92)
        print(f"{sname}  ({len(keys)} interfaces)")
        print("=" * 92)
        print(f"{'model / score':<42}{'GENERATOR':>11}{'SELECTOR':>10}{'TOP-1':>8}{'mean AUC':>10}")
        print("-" * 92)
        for score in SCORES:
            # --- Protenix, native depth (50 seeds) ---
            c = collections.Counter(x for x in (classify(pro[k], score) for k in keys) if x)
            r = rates(c)
            aucs = [x for x in (auc(pro[k], score) for k in keys) if x is not None]
            if r:
                print(f"{'Protenix  ' + score:<42}{r['generator']:>10.0%}{r['selector']:>10.0%}"
                      f"{r['top1']:>8.0%}{np.mean(aucs):>10.3f}")
                rows.append(dict(subset=sname, model="protenix", score=score, depth="50 seeds",
                                 n=r["n"], generator=r["generator"], selector=r["selector"],
                                 top1=r["top1"], mean_auc=np.mean(aucs), median_auc=np.median(aucs)))

            # --- ESM, FULL depth (1025 seeds) ---
            c = collections.Counter(x for x in (classify(esm[k], score) for k in keys) if x)
            r = rates(c)
            aucs = [x for x in (auc(esm[k], score) for k in keys) if x is not None]
            if r:
                print(f"{'ESM(1025 seeds)  ' + score:<42}{r['generator']:>10.0%}{r['selector']:>10.0%}"
                      f"{r['top1']:>8.0%}{np.mean(aucs):>10.3f}")
                rows.append(dict(subset=sname, model="esm", score=score, depth="1025 seeds",
                                 n=r["n"], generator=r["generator"], selector=r["selector"],
                                 top1=r["top1"], mean_auc=np.mean(aucs), median_auc=np.median(aucs)))

            # --- ESM, DEPTH-MATCHED to Protenix (50 seeds) ---
            g, s, t = [], [], []
            for _ in range(a.reps):
                c = collections.Counter()
                for k in keys:
                    seeds = sorted({p[0] for p in esm[k]})
                    pick = set(rng.choice(seeds, min(50, len(seeds)), replace=False))
                    v = classify([p for p in esm[k] if p[0] in pick], score)
                    if v:
                        c[v] += 1
                rr = rates(c)
                if rr:
                    g.append(rr["generator"]); s.append(rr["selector"]); t.append(rr["top1"])
            if g:
                print(f"{'ESM(50 seeds, MATCHED)  ' + score:<42}{np.mean(g):>10.0%}{np.mean(s):>10.0%}"
                      f"{np.mean(t):>8.0%}{'':>10}")
                rows.append(dict(subset=sname, model="esm", score=score, depth="50 seeds (matched)",
                                 n=len(keys), generator=np.mean(g), selector=np.mean(s),
                                 top1=np.mean(t), mean_auc="", median_auc=""))
            print()

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
