#!/usr/bin/env python3
"""confidence_top_end.py — WHERE in its own confidence distribution does each model fail?

confidence_discrimination.py turned up a dissociation that the whole project hinges on:

    on the 114 zero-template interfaces        SELECTOR (argmax)      AUC (average)
      Protenix pair_iptm                            83%                  0.798
      ESMFold2 pair_iptm                            56%                  0.772

Average discrimination is nearly IDENTICAL. Argmax accuracy differs by 27 points.

AUC averages over all good-vs-bad pose pairs. argmax reads exactly ONE point: the top. So a
model can be well-calibrated in bulk and still be catastrophic at the top -- if its extreme
right tail is populated by decoys. That is precisely the confident-decoy disease, and it
predicts a NON-MONOTONIC precision curve: accuracy rises with confidence, then COLLAPSES at
the very top.

This script measures that curve directly.

For each interface, rank the poses by confidence and compute, at each depth k:
    precision@k  = fraction of the top-k poses that are acceptable (DockQ > 0.23)
    enrichment@k = precision@k / (base rate of acceptable poses in that pool)

enrichment = 1.0 means "no better than picking at random". >1 means confidence helps. <1 means
confidence is ACTIVELY MISLEADING at that depth. A healthy selector's enrichment rises
monotonically as k shrinks. A confident-decoy model's should peak and then crash.

Interfaces are pooled by averaging per-interface enrichment (so a 5125-pose interface does not
drown out a 250-pose one). Only interfaces that HAVE an acceptable pose are included -- on a
NO-POSE interface precision is 0 at every depth and says nothing about the selector.
"""
import collections, csv, glob, os, re
import numpy as np

ACC = 0.23
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ESM_CSV = f"{REPO}/results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"
PRO_CSV = f"{REPO}/results/csv_files/dockq_ost_protenix_full.csv"
PRO_LOGS = "/home/ebrahim/Protenix/foldbench_abag/predictions_template/gpu*.log"
SCORE = "pair_iptm"          # raw head output for BOTH models -- no composite, no confound


def template_status():
    pend, seen = [], collections.defaultdict(set)
    for path in sorted(glob.glob(PRO_LOGS)):
        for line in open(path, errors="ignore"):
            m = re.search(r"Found (\d+) templates", line)
            if m:
                pend.append(int(m.group(1))); continue
            m = re.search(r"\]\s+([0-9a-z]{4})\s+\[seed:(\d+)\]", line)
            if m:
                if pend: seen[m.group(1)].update(pend)
                pend = []
    return {c: (max(v) > 0) for c, v in seen.items()}


def load(path):
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r.get("dockq_status") not in (None, "", "ok"):
            continue
        try:
            dq, sc = float(r["dockq"]), float(r[SCORE])
        except (TypeError, ValueError, KeyError):
            continue
        if np.isnan(dq) or np.isnan(sc):
            continue
        by[(r["assembly_id"].split("-")[0], r["chain1"], r["chain2"])].append((r["seed"], dq, sc))
    return by


def curve(poses, fracs):
    """precision and enrichment at each top-fraction of the confidence ranking."""
    dq = np.array([p[1] for p in poses]); sc = np.array([p[2] for p in poses])
    base = (dq > ACC).mean()
    if base == 0:
        return None, None                      # NO-POSE: says nothing about the selector
    order = np.argsort(-sc)
    dq = dq[order]
    prec, enr = [], []
    for f in fracs:
        k = max(1, int(round(f * len(dq))))
        p = (dq[:k] > ACC).mean()
        prec.append(p); enr.append(p / base)
    return prec, enr


def main():
    fracs = [0.0002, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
    tmpl = template_status()
    esm, pro = load(ESM_CSV), load(PRO_CSV)
    common = sorted(set(esm) & set(pro))
    clean = [k for k in common if not tmpl.get(k[0], False)]
    print(f"{len(clean)} zero-template interfaces (the honest comparison set)\n")

    # ESM subsampled to 50 seeds so DEPTH is matched to Protenix -- otherwise "top 1%" means
    # 51 poses for ESM and 2 for Protenix, and the curves aren't comparable.
    rng = np.random.default_rng(20260713)
    esm50 = {}
    for k in clean:
        seeds = sorted({p[0] for p in esm[k]})
        pick = set(rng.choice(seeds, min(50, len(seeds)), replace=False))
        esm50[k] = [p for p in esm[k] if p[0] in pick]

    out = {}
    for name, pool in (("Protenix (250 poses)", pro),
                       ("ESMFold2 (250 poses, depth-matched)", esm50),
                       ("ESMFold2 (5125 poses, full depth)", esm)):
        E, P = [], []
        for k in clean:
            p, e = curve(pool[k], fracs)
            if e is not None:
                P.append(p); E.append(e)
        out[name] = (np.mean(P, axis=0), np.mean(E, axis=0), len(E))

    print("PRECISION @ top-k%  — fraction of the k% most-confident poses that are ACCEPTABLE\n")
    hdr = "".join(f"{f*100:>8.2f}%" for f in fracs)
    print(f"{'model':<38}{hdr}")
    print("-" * (38 + 9 * len(fracs)))
    for name, (p, e, n) in out.items():
        print(f"{name:<38}" + "".join(f"{v:>8.0%} " for v in p))

    print("\n\nENRICHMENT @ top-k%  — precision / base rate.")
    print("   1.0 = no better than random.  >1 = confidence helps.  <1 = confidence MISLEADS.\n")
    print(f"{'model':<38}{hdr}")
    print("-" * (38 + 9 * len(fracs)))
    for name, (p, e, n) in out.items():
        print(f"{name:<38}" + "".join(f"{v:>8.2f} " for v in e))

    print("\n  (mean over per-interface curves; n = "
          f"{out['Protenix (250 poses)'][2]} interfaces that contain >=1 acceptable pose)")

    # the headline: does enrichment PEAK and then collapse?
    print("\n\nIS THE CURVE NON-MONOTONIC (the confident-decoy signature)?")
    for name, (p, e, n) in out.items():
        i = int(np.argmax(e))
        top = e[0]; peak = e[i]
        verdict = ("MONOTONIC — most confident poses are the best"
                   if i == 0 else
                   f"NON-MONOTONIC — peaks at top-{fracs[i]*100:g}% ({peak:.2f}x) then FALLS to "
                   f"{top:.2f}x at the very top")
        print(f"  {name:<38} {verdict}")


if __name__ == "__main__":
    main()
