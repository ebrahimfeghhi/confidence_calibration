#!/usr/bin/env python
"""Is a redundancy filter on the donor pool SAFE? Tested against arm outcomes, not donor DockQ.

A redundancy filter drops pose B because it is structurally close to a kept pose A, on the implicit
assumption that A is a substitute for B as a TEMPLATE. That assumption has never been checked. We can
check it exactly on 8che: the same 250 free poses are (a) the rows of the cached pose-pose DockQ
matrix and (b) the 250 donors of allpose8che10, each with 10 seeds of outcome.

Three questions, in increasing order of how much they decide:

  1. DIVERGENCE. For donor pairs binned by pose-pose similarity, how different are their arm
     outcomes? If near-identical donors give near-identical results, substitution is sound.

  2. ISOLATION. What sits near 137_3 -- the only donor whose arm exceeded 0.85 -- and what did those
     neighbours' arms produce? A higher-ipTM neighbour is exactly what would delete it.

  3. THE BOTTOM LINE. At each threshold: run the filter, then ask what the best arm outcome among
     SURVIVORS is. That is what the method would actually return after filtering, and it is directly
     comparable to the unfiltered 0.957.

The filter is label-free (ipTM ranking + pose-pose geometry). DockQ-vs-native and arm outcomes are
used only to SCORE what the filter did.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/filter_safety.py
Writes: artifacts/filter_safety_8che_BE.json
"""
import csv, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import EXP

IFACE = "8che_BE"
MATDIR = "/home/ebrahim/tts_abag/results/dockq_protenix"
GOLD = "137_3"

M = np.load(f"{MATDIR}/dockq_{IFACE}.npy")
rows = list(csv.DictReader(open(f"{MATDIR}/dockq_{IFACE}_index.csv")))
assert M.shape[0] == M.shape[1] == len(rows)
tag = [f"{r['seed']}_{r['sample']}" for r in rows]
iptm = np.array([float(r["iptm"]) for r in rows])
gt = np.array([float(r["dockq_gt"]) for r in rows])

# arm outcome per donor, keyed by the donor pose it was built from
sc = json.load(open(f"{EXP}/artifacts/allpose8che10_scores.json"))
arm = {}
for r in sc["rows"]:
    if IFACE in r and r["n_seeds_done"] == 10:
        arm[f"{r['donor_seed']}_{r['donor_sample']}"] = r[IFACE]["out_max"]
missing = [t for t in tag if t not in arm]
assert not missing, f"{len(missing)} poses have no arm outcome, e.g. {missing[:5]}"
out_max = np.array([arm[t] for t in tag])
n = len(tag)
gi = tag.index(GOLD)
print(f"{IFACE}: {n} donors, all with 10-seed arm outcomes")
print(f"  {GOLD}: donor DockQ {gt[gi]:.3f}, ipTM {iptm[gi]:.3f} (rank {int(np.sum(iptm > iptm[gi])) + 1}), "
      f"arm best {out_max[gi]:.3f}")
print(f"  arms above 0.85: {int(np.sum(out_max > 0.85))}/{n}; above 0.6: {int(np.sum(out_max > 0.6))}\n")

# ---- 1. divergence of arm outcome vs donor-donor similarity ---------------------------
iu = np.triu_indices(n, 1)
sim = M[iu]
d_out = np.abs(out_max[iu[0]] - out_max[iu[1]])
BANDS = [(0.9, 1.01), (0.8, 0.9), (0.6, 0.8), (0.4, 0.6), (0.23, 0.4), (0.0, 0.23)]
print("1. ARM-OUTCOME DIVERGENCE BY DONOR SIMILARITY")
print(f"{'similarity':>14} {'#pairs':>9} {'mean |d outcome|':>17} {'median':>8} {'p95':>7} {'max':>7}")
div = []
for lo, hi in BANDS:
    m = (sim >= lo) & (sim < hi)
    if not m.any():
        continue
    v = d_out[m]
    div.append({"lo": lo, "hi": hi, "n_pairs": int(m.sum()), "mean": round(float(v.mean()), 4),
                "median": round(float(np.median(v)), 4), "p95": round(float(np.percentile(v, 95)), 4),
                "max": round(float(v.max()), 4)})
    print(f"  [{lo:.2f},{hi:.2f}) {int(m.sum()):>9} {v.mean():>17.3f} {np.median(v):>8.3f} "
          f"{np.percentile(v, 95):>7.3f} {v.max():>7.3f}")

# ---- 2. what sits near the gold donor -------------------------------------------------
print(f"\n2. NEAREST NEIGHBOURS OF {GOLD} (arm best {out_max[gi]:.3f})")
nb = np.argsort(-M[gi])
nb = [j for j in nb if j != gi][:10]
print(f"{'pose':>8} {'sim to gold':>12} {'its ipTM':>9} {'higher ipTM?':>13} {'donor DockQ':>12} {'ARM BEST':>9}")
neigh = []
for j in nb:
    hi_ = iptm[j] > iptm[gi]
    neigh.append({"tag": tag[j], "sim": round(float(M[gi, j]), 3), "iptm": round(float(iptm[j]), 4),
                  "higher_iptm": bool(hi_), "donor_dockq": round(float(gt[j]), 3),
                  "arm_best": round(float(out_max[j]), 3)})
    print(f"{tag[j]:>8} {M[gi, j]:>12.3f} {iptm[j]:>9.3f} {'YES' if hi_ else 'no':>13} "
          f"{gt[j]:>12.3f} {out_max[j]:>9.3f}")

# ---- 3. the bottom line: best arm outcome among survivors ------------------------------
def survivors(thresh):
    order = np.argsort(-iptm)
    kept = []
    for i in order:
        if all(M[i, j] < thresh for j in kept):
            kept.append(int(i))
    return kept


print("\n3. BEST ARM OUTCOME AMONG SURVIVORS (what the method would return after filtering)")
print(f"{'thresh':>7} {'#kept':>7} {'seeds @10':>10} {'gold kept':>10} {'BEST ARM':>9} {'vs 0.957':>9}")
bottom = []
for th in (0.23, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.01):
    S = survivors(th)
    b = float(out_max[S].max())
    bottom.append({"thresh": th, "n_kept": len(S), "seeds_at_10": len(S) * 10,
                   "gold_kept": bool(gi in S), "best_arm": round(b, 3)})
    print(f"{th:>7.2f} {len(S):>7} {len(S) * 10:>10} {'YES' if gi in S else 'no':>10} "
          f"{b:>9.3f} {b - 0.957:>+9.3f}")

json.dump({"iface": IFACE, "gold": GOLD, "gold_arm_best": float(out_max[gi]),
           "divergence": div, "neighbours": neigh, "bottom_line": bottom},
          open(f"{EXP}/artifacts/filter_safety_{IFACE}.json", "w"), indent=1)
print(f"\nWROTE {EXP}/artifacts/filter_safety_{IFACE}.json")
