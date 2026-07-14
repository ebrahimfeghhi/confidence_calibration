#!/usr/bin/env python3
"""msa_confound.py — is Protenix's selector advantage actually just MSAs?

THE THREAT. ESMFold2-Fast runs SINGLE-SEQUENCE: `msa_encoder.enabled = false` in the
biohub/ESMFold2-Fast config (base biohub/ESMFold2 has it TRUE), and our driver never passes an
MSA (run/esm/run_foldbench_abag.py:75-77; MSA-bearing JSONs are explicitly filtered out at
:46-48). Protenix uses MSAs.

So every comparison in docs/confidence_head_hypothesis.md may really be MSA vs NO-MSA, and the
whole "Protenix's confidence head is a better discriminator" story would be confounded.
Coevolutionary signal on the antigen surface is exactly the information that identifies the true
epitope -- it could plausibly explain BOTH the generator gap and the selector gap.

THE CHEAP DECISIVE TEST (no re-running any model):

    If MSAs are the cause, Protenix's advantage must SCALE WITH MSA DEPTH.
    On complexes where Protenix's MSA is shallow, it has little coevolutionary
    information that ESM lacks -- so its advantage should shrink toward zero.
    If instead the advantage is FLAT in MSA depth, MSAs are not the explanation.

Protenix logs the MSA depth per structure:
    [Rank 0 (13/17)] 8udz [seed:0]: N_asym 9, N_token 1713, N_atom 12713, N_msa 13917

This is correlational, not causal -- MSA depth also correlates with how well-studied a complex is,
which correlates with difficulty. So a POSITIVE result (advantage scales with depth) would be
suggestive but not conclusive, and would demand the real experiment (run base ESMFold2 WITH MSAs).
A FLAT result is the stronger inference: it is hard to see how MSAs could drive the gap while
contributing nothing at the shallow end.
"""
import collections, csv, glob, re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ACC = 0.23
REPO = Path(__file__).resolve().parents[2]
ESM_CSV = REPO / "results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"
PRO_CSV = REPO / "results/csv_files/dockq_ost_protenix_full.csv"
LOGS = "/home/ebrahim/Protenix/foldbench_abag/predictions_template/gpu*.log"
SCORE = "pair_iptm"


def msa_depth():
    """complex -> MSA depth Protenix actually used."""
    out = {}
    for path in sorted(glob.glob(LOGS)):
        for line in open(path, errors="ignore"):
            m = re.search(r"\]\s+([0-9a-z]{4})\s+\[seed:\d+\].*N_msa\s+(\d+)", line)
            if m:
                out[m.group(1)] = int(m.group(2))
    return out


def template_status():
    pend, seen = [], collections.defaultdict(set)
    for path in sorted(glob.glob(LOGS)):
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
        by[(r["assembly_id"][:4], r["chain1"], r["chain2"])].append((r["seed"], dq, sc))
    return by


def selector_hit(poses):
    dq = np.array([p[1] for p in poses]); sc = np.array([p[2] for p in poses])
    if not (dq > ACC).any():
        return None                                    # NO-POSE: undefined
    return bool(dq[int(np.argmax(sc))] > ACC)


def main():
    rng = np.random.default_rng(20260713)
    depth = msa_depth()
    tmpl = template_status()
    esm, pro = load(ESM_CSV), load(PRO_CSV)
    clean = [k for k in sorted(set(esm) & set(pro))
             if not tmpl.get(k[0], False) and k[0] in depth]
    print(f"{len(clean)} zero-template interfaces with a logged Protenix MSA depth\n")

    rows = []
    for k in clean:
        seeds = sorted({p[0] for p in esm[k]})
        pick = set(rng.choice(seeds, min(50, len(seeds)), replace=False))    # depth-match
        e = selector_hit([p for p in esm[k] if p[0] in pick])
        p = selector_hit(pro[k])
        if e is None or p is None:
            continue
        rows.append(dict(iface=k, msa=depth[k[0]], esm=e, pro=p, adv=int(p) - int(e)))

    d = np.array([r["msa"] for r in rows], float)
    adv = np.array([r["adv"] for r in rows], float)
    print(f"{len(rows)} interfaces where BOTH models have an acceptable pose in the pool")
    print(f"  Protenix MSA depth: min {int(d.min())}  median {int(np.median(d))}  max {int(d.max())}")
    print(f"  Protenix selector {np.mean([r['pro'] for r in rows]):.0%},"
          f"  ESM selector {np.mean([r['esm'] for r in rows]):.0%},"
          f"  mean advantage {adv.mean():+.0%}\n")

    print("=" * 78)
    print("DOES PROTENIX'S ADVANTAGE SCALE WITH MSA DEPTH?")
    print("  if MSAs cause the gap, the advantage must SHRINK where the MSA is shallow")
    print("=" * 78)
    q = np.percentile(d, [25, 50, 75])
    edges = [d.min() - 1, q[0], q[1], q[2], d.max() + 1]
    names = ["Q1 shallowest", "Q2", "Q3", "Q4 deepest"]
    print(f"\n{'MSA depth quartile':<20}{'range':>18}{'n':>5}{'Protenix':>10}{'ESM':>7}{'advantage':>12}")
    print("-" * 74)
    for nm, lo, hi in zip(names, edges[:-1], edges[1:]):
        m = (d > lo) & (d <= hi)
        if m.sum() == 0:
            continue
        pv = np.mean([rows[i]["pro"] for i in np.where(m)[0]])
        ev = np.mean([rows[i]["esm"] for i in np.where(m)[0]])
        print(f"{nm:<20}{f'{int(lo)+1}-{int(hi)}':>18}{m.sum():>5}{pv:>10.0%}{ev:>7.0%}"
              f"{pv-ev:>+12.0%}")

    rho = spearmanr(d, adv)
    print(f"\n  Spearman(MSA depth, Protenix advantage) = {rho.statistic:+.3f}   p = {rho.pvalue:.3f}")

    # shallow vs deep, split at the median, with a bootstrap on the difference of advantages
    med = np.median(d)
    lo_m, hi_m = d <= med, d > med
    a_lo, a_hi = adv[lo_m].mean(), adv[hi_m].mean()
    bs = np.array([adv[hi_m][rng.integers(0, hi_m.sum(), hi_m.sum())].mean()
                   - adv[lo_m][rng.integers(0, lo_m.sum(), lo_m.sum())].mean()
                   for _ in range(10000)])
    lo_ci, hi_ci = np.percentile(bs, [2.5, 97.5])
    print(f"\n  advantage on SHALLOW-MSA half (n={lo_m.sum()}): {a_lo:+.0%}")
    print(f"  advantage on DEEP-MSA half    (n={hi_m.sum()}): {a_hi:+.0%}")
    print(f"  difference (deep - shallow): {a_hi-a_lo:+.0%}   95% CI [{lo_ci:+.0%}, {hi_ci:+.0%}]")

    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    if rho.statistic < -0.15:
        print("  The correlation runs BACKWARDS. Protenix's advantage is LARGEST where its MSA is")
        print("  SHALLOWEST, and it LOSES to ESM where the MSA is deepest. That is the opposite of")
        print("  what the MSA hypothesis predicts (more coevolutionary signal -> bigger win).")
    elif abs(rho.statistic) < 0.15:
        print("  Protenix's advantage is FLAT in MSA depth.")
    else:
        print("  Protenix's advantage GROWS with MSA depth -- consistent with the MSA hypothesis.")

    print(f"""
  BUT THIS TEST CANNOT SETTLE THE REAL QUESTION, and it would be dishonest to pretend it does:

  1. There is no genuinely shallow MSA here. The MINIMUM depth is {int(d.min()):,} sequences, and
     N_msa appears to cap at 16,384. So this compares DEEP against DEEPER, in a regime where the
     coevolutionary signal is almost certainly saturated. The contrast that actually matters is
     0 vs {int(d.min()):,} -- ESMFold2-Fast has NO MSA AT ALL -- and that contrast has zero
     variance in this data, because Protenix always has one and ESM never does.

  2. A confound of the confound: deep MSAs mean well-studied, conserved antigens, which are
     easier for EVERYONE -- including ESM's language model, which was trained on the same
     evolutionary data in compressed form. That alone could produce the negative slope.

  So: MSA DEPTH does not drive Protenix's advantage. Whether HAVING AN MSA AT ALL does is still
  open, and it remains the largest threat to docs/confidence_head_hypothesis.md.

  THE ONLY TEST THAT SETTLES IT: run base biohub/ESMFold2 (msa_encoder.enabled = true) WITH
  Protenix's MSAs on these interfaces, and see whether ESM's selector accuracy jumps.""")
    print("=" * 78)


if __name__ == "__main__":
    main()
