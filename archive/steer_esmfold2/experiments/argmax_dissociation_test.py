#!/usr/bin/env python3
"""argmax_dissociation_test.py — is the "don't take the argmax" effect statistically real?

dont_take_argmax.py found a clean dissociation in the selection-critical band (pools 10-50%
acceptable), depth-matched at 250 poses, raw pair_iptm:

                        rank-1 (argmax)   rank-20    delta
      ESMFold2               35%            50%      +15%    backing off HELPS
      Protenix               79%            54%      -25%    backing off HURTS

That is exactly what critical_band.py's inversion predicts (ESM's top decile has Cliff's
delta -0.011; Protenix's has +0.477). But ESM's band has only n=20 interfaces, so +15% is three
interfaces and could easily be noise.

The claim that actually matters is not either delta on its own -- it is the INTERACTION: the two
models respond to backing off in OPPOSITE directions. A control that moves the other way is much
harder to get by chance than a single effect, so that is what we test.

Paired bootstrap over interfaces (the unit of resampling is an interface, since poses within an
interface are not independent).
"""
import importlib.util, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("dta", os.path.join(HERE, "dont_take_argmax.py"))
dta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dta)          # safe: dont_take_argmax.py guards main() behind __main__

ACC = dta.ACC


def main():
    rng = np.random.default_rng(7)
    tmpl = dta.template_status()
    esm, pro = dta.load(dta.ESM_CSV), dta.load(dta.PRO_CSV)
    clean = [k for k in sorted(set(esm) & set(pro)) if not tmpl.get(k[0], False)]

    r2 = np.random.default_rng(20260713)          # same seed as dont_take_argmax -> same subsample
    E, P = {}, {}
    for k in clean:
        seeds = sorted({p[0] for p in esm[k]})
        pick = set(r2.choice(seeds, min(50, len(seeds)), replace=False))
        e = [p for p in esm[k] if p[0] in pick]
        if any(p[1] > ACC for p in e):
            E[k] = e
        if any(p[1] > ACC for p in pro[k]):
            P[k] = pro[k]

    def in_band(pool, k):
        return 0.10 <= np.mean([p[1] > ACC for p in pool[k]]) < 0.50

    Ek = [k for k in E if in_band(E, k)]
    Pk = [k for k in P if in_band(P, k)]

    def hits(pool, keys, est):
        return np.array([dta.picks(pool[k])[est] > ACC for k in keys])

    B = 10000
    res = {}
    print("PAIRED BOOTSTRAP over interfaces (10,000 resamples), selection-critical band\n")
    for label, pool, keys in (("ESMFold2", E, Ek), ("Protenix", P, Pk)):
        h1 = hits(pool, keys, "rank-1")
        h20 = hits(pool, keys, "rank-20")
        d = h20.mean() - h1.mean()
        bs = np.array([(lambda i: h20[i].mean() - h1[i].mean())(rng.integers(0, len(keys), len(keys)))
                       for _ in range(B)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  {label:<10} n={len(keys):>3}   argmax {h1.mean():>4.0%} -> rank-20 {h20.mean():>4.0%}"
              f"   delta {d:>+5.0%}   95% CI [{lo:+.0%}, {hi:+.0%}]   P(delta>0) = {(bs > 0).mean():.3f}")
        res[label] = (d, bs)

    dE, bE = res["ESMFold2"]
    dP, bP = res["Protenix"]
    inter = dE - dP
    bsi = bE - bP                      # the two interface sets are disjoint, so bootstraps are independent
    lo, hi = np.percentile(bsi, [2.5, 97.5])
    print("\nTHE INTERACTION — do the two models respond to backing off in OPPOSITE directions?")
    print(f"  (ESM: rank20 - argmax) - (Protenix: rank20 - argmax) = {inter:+.0%}")
    print(f"  95% CI [{lo:+.0%}, {hi:+.0%}]")
    print(f"  P(interaction > 0) = {(bsi > 0).mean():.4f}")
    if (bsi > 0).mean() > 0.975:
        print("\n  >>> The dissociation is real: ESM's top is contaminated, Protenix's is not.")
        print("      Each model's own delta is underpowered (n=20, n=24); the CONTRAST is not.")
    else:
        print("\n  >>> Not significant. The direction is right but the sample is too small to claim it.")


if __name__ == "__main__":
    main()
