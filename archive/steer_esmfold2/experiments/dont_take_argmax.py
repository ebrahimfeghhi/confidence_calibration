#!/usr/bin/env python3
"""dont_take_argmax.py — the fix that the inversion predicts.

critical_band.py established the mechanism:

                  selector   Cliff d ALL poses   Cliff d TOP 10%
    Protenix         79%          +0.473             +0.477        top decile is HEALTHY
    ESMFold2         35%          +0.404             -0.011        top decile is INVERTED

ESMFold2's confidence discriminates correctly across the pool and stops discriminating -- or
inverts -- inside its own top decile. argmax reads exactly that spot.

THE PREDICTION: if the very top of ESM's ranking is contaminated, then DON'T READ THE VERY TOP.
Backing off a few ranks should IMPROVE accuracy, for free, with no labels and no extra compute.

Protenix is the control. Its top decile is healthy, so its accuracy should DECAY monotonically
as we back off -- there is nothing to gain by ignoring a correct ranking.

Three estimators, all label-free:
  RANK-k        take the k-th most confident pose.
  TRIM-x        discard the top x% of poses by confidence, then take the max of what remains.
  MEDIAN-TOP-x  take the pose with MEDIAN confidence among the top x% -- a robust "somewhere near
                the top but not at it" pick. (This is the pose-level analogue of the cluster
                median that has been our method all along.)

If TRIM/RANK-k beats argmax on ESM but not on Protenix, the inversion is real, actionable, and
the two models are cleanly separated by it.
"""
import collections, csv, glob, json, os, re
import numpy as np

ACC = 0.23
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ESM_CSV = f"{REPO}/results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"
PRO_CSV = f"{REPO}/results/csv_files/dockq_ost_protenix_full.csv"
PRO_LOGS = "/home/ebrahim/Protenix/foldbench_abag/predictions_template/gpu*.log"
SCORE = "pair_iptm"


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


def picks(poses):
    """Return {estimator_name: DockQ of the pose it picks}. All label-free."""
    dq = np.array([p[1] for p in poses]); sc = np.array([p[2] for p in poses])
    order = np.argsort(-sc)                      # most confident first
    n = len(dq)
    out = {}
    for k in (1, 2, 3, 5, 10, 20, 50):
        out[f"rank-{k}"] = dq[order[min(k, n) - 1]]
    for x in (0.01, 0.02, 0.05, 0.10, 0.25):
        cut = int(round(x * n))
        rest = order[cut:] if cut < n else order[-1:]
        out[f"trim-{x:.0%}"] = dq[rest[0]]        # max confidence among what survives the trim
    for x in (0.02, 0.05, 0.10, 0.25):
        k = max(1, int(round(x * n)))
        top = order[:k]
        out[f"medtop-{x:.0%}"] = dq[top[len(top) // 2]]   # median-confidence pose of the top x%
    return out


def main():
    rng = np.random.default_rng(20260713)
    tmpl = template_status()
    esm, pro = load(ESM_CSV), load(PRO_CSV)
    clean = [k for k in sorted(set(esm) & set(pro)) if not tmpl.get(k[0], False)]
    split = json.load(open(f"{REPO}/results/csv_files/train_test_split.json"))
    train = {tuple([i.split("_")[0], i.split("_")[1][0], i.split("_")[1][1]]) for i in split["train"]}

    # depth-match ESM to Protenix (250 poses); only interfaces where a good pose EXISTS,
    # otherwise the selector is undefined and every estimator scores 0.
    E, P = {}, {}
    for k in clean:
        seeds = sorted({p[0] for p in esm[k]})
        pick = set(rng.choice(seeds, min(50, len(seeds)), replace=False))
        e = [p for p in esm[k] if p[0] in pick]
        if any(p[1] > ACC for p in e):
            E[k] = e
        if any(p[1] > ACC for p in pro[k]):
            P[k] = pro[k]

    ests = list(picks(next(iter(E.values()))).keys())

    def evaluate(pool, keys):
        res = {e: [] for e in ests}
        for k in keys:
            pk = picks(pool[k])
            for e in ests:
                res[e].append(pk[e] > ACC)
        return {e: float(np.mean(v)) for e, v in res.items()}

    sets = [
        ("ESMFold2  all zero-template", E, sorted(E)),
        ("Protenix  all zero-template (CONTROL)", P, sorted(P)),
        ("ESMFold2  selection-critical band (10-50% pure)", E,
         [k for k in E if 0.10 <= np.mean([p[1] > ACC for p in E[k]]) < 0.50]),
        ("Protenix  selection-critical band (CONTROL)", P,
         [k for k in P if 0.10 <= np.mean([p[1] > ACC for p in P[k]]) < 0.50]),
    ]

    print("SELECTOR ACCURACY of label-free estimators that DO NOT take the argmax")
    print("  (depth-matched 250 poses; raw pair_iptm; only interfaces containing >=1 good pose)\n")
    for name, pool, keys in sets:
        if not keys:
            continue
        r = evaluate(pool, keys)
        base = r["rank-1"]
        print(f"{name}   (n={len(keys)})")
        print(f"    {'estimator':<14}{'accuracy':>10}{'vs argmax':>12}")
        for e in ests:
            d = r[e] - base
            flag = "  <-- BEATS ARGMAX" if d > 0.02 else ""
            print(f"    {e:<14}{r[e]:>10.0%}{d:>+12.0%}{flag}")
        print()


if __name__ == "__main__":
    main()
