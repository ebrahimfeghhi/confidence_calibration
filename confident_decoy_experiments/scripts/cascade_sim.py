#!/usr/bin/env python
"""Can a two-stage cascade reach allpose8che10's result under a 1000-seed budget?

allpose8che10 spends 250 donors x 10 seeds = 2500 seeds and returns DockQ 0.924 on 8che_BE by
pooled ipTM. The user's ceiling is 1000 seeds. This simulates a cascade RETROSPECTIVELY over the
10 seeds we already have for every donor -- no new inference:

  stage 1  screen all 250 donors with S seeds each          (250 * S seeds)
           rank donors by the best OUTPUT ipTM those S seeds produced
  stage 2  take the top K donors and spend the rest of the budget on them

WHY OUTPUT ipTM AND NOT THE DONOR'S OWN ipTM. A donor's own ipTM does not predict its arm at all
(Pearson +0.030, Spearman +0.115 over 250 donors; the only donor to exceed 0.85, 137_3, ranks
87/250 by its own ipTM). That is what sank leader20 and every top-K-by-ipTM donor selection. But
the ipTM of the poses an arm PRODUCES separates cleanly -- 137_3's winning seeds sit at 0.68-0.73
against failing seeds at 0.40-0.49. The cascade screens on the second signal, never the first.

WHAT STAGE 2 CAN AND CANNOT BE MEASURED HERE. We hold exactly 10 seeds per donor, so a stage-2
allocation of more than 10 seeds to a donor cannot be simulated -- there is no data for seed 11.
Stage 2 is therefore evaluated as "the surviving donors' remaining seeds", which is an
UNDERESTIMATE of what a real stage 2 with more seeds per donor would return. Reported as such.

The screen must not peek: stage-1 ranking uses only the S screen seeds, and stage-2 results come
only from the held-out remaining seeds.

Usage: /home/ebrahim/Protenix/.venv/bin/python scripts/cascade_sim.py [--iface 8che_BE] [--draws 400]
Writes: artifacts/cascade_sim.json
"""
import argparse, json, os, sys, random, statistics as st
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import EXP

ap = argparse.ArgumentParser()
ap.add_argument("--iface", default="8che_BE")
ap.add_argument("--draws", type=int, default=400)
A = ap.parse_args()
IF = A.iface

d = json.load(open(f"{EXP}/artifacts/allpose8che10_scores.json"))
rows = [r for r in d["rows"] if IF in r and r["n_seeds_done"] == 10]
assert len(rows) == 250, f"expected 250 complete donors, got {len(rows)}"

# donor -> {seed: (sel_iptm, sel_dockq, max_dockq)}
D = {}
for r in rows:
    D[r["name"]] = {s: (v["sel_iptm"], v["sel"], v["max"])
                    for s, v in r[IF]["per_seed"].items()}
    assert len(D[r["name"]]) == 10, r["name"]
SEEDS = {n: sorted(v) for n, v in D.items()}
GOLD = "8che_allpose_137_3"
full_best = max(v[2] for n in D for v in D[n].values())
full_sel = max((v for n in D for v in D[n].values()), key=lambda x: x[0])
print(f"{IF}: 250 donors x 10 seeds = 2500 seeds")
print(f"  full run (2500 seeds): pool best {full_best:.3f}, pooled-ipTM pick {full_sel[1]:.3f} "
      f"(ipTM {full_sel[0]:.3f})\n")

rng = random.Random(0)
out = {"iface": IF, "full_best": full_best, "full_sel_dockq": full_sel[1], "configs": []}

for S, K in ((2, 20), (2, 50), (3, 20), (3, 50), (4, 25), (1, 20), (1, 50)):
    budget = 250 * S + K * (10 - S)
    gold_kept, sel_dq, best_dq, ranks = 0, [], [], []
    for _ in range(A.draws):
        screen = {n: rng.sample(SEEDS[n], S) for n in D}
        # stage 1: rank donors by best OUTPUT ipTM over the screen seeds only
        score = {n: max(D[n][s][0] for s in screen[n]) for n in D}
        order = sorted(D, key=lambda n: -score[n])
        top = order[:K]
        ranks.append(order.index(GOLD) + 1)
        gold_kept += GOLD in top
        # stage 2: the survivors' HELD-OUT seeds (underestimate -- see docstring)
        pool = [D[n][s] for n in top for s in SEEDS[n] if s not in screen[n]]
        # selection over everything seen: screen poses + stage-2 poses
        seen = pool + [D[n][s] for n in D for s in screen[n]]
        sel_dq.append(max(seen, key=lambda x: x[0])[1])
        best_dq.append(max(x[2] for x in seen))
    rec = {"screen_seeds": S, "top_k": K, "budget_seeds": budget,
           "gold_retained": round(gold_kept / A.draws, 3),
           "gold_rank_median": st.median(ranks),
           "sel_dockq_median": round(st.median(sel_dq), 3),
           "sel_dockq_p10": round(sorted(sel_dq)[int(.10 * len(sel_dq))], 3),
           "sel_above_085": round(sum(1 for x in sel_dq if x > 0.85) / len(sel_dq), 3),
           "best_dockq_median": round(st.median(best_dq), 3)}
    out["configs"].append(rec)
    print(f"S={S} K={K:>2}  budget {budget:>4} seeds | "
          f"137_3 retained {rec['gold_retained']:.2f} (median rank {rec['gold_rank_median']:.0f}/250) | "
          f"selected DockQ median {rec['sel_dockq_median']:.3f} p10 {rec['sel_dockq_p10']:.3f} | "
          f"P(>0.85) {rec['sel_above_085']:.2f}")

json.dump(out, open(f"{EXP}/artifacts/cascade_sim.json", "w"), indent=1)
print(f"\nWROTE {EXP}/artifacts/cascade_sim.json")
print("NOTE stage 2 reuses each survivor's remaining 10-S seeds; a real stage 2 spending more "
      "seeds per survivor would do better. These are lower bounds.")
