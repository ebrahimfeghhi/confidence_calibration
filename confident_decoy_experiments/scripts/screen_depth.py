#!/usr/bin/env python
"""How many seeds per donor does a SCREEN need before it can pick the good donor?

8che_BE is the only interface with many donors run at full depth: leader20's 20 leader templates x 50
seeds (artifacts/leader20_ost, the chedeep run) plus dockqdonor's donor x 50 seeds
(artifacts/dockqdonor_ost). 21 donors x 50 seeds x 5 samples. That is enough to SIMULATE a shallow
screen by subsampling seeds, without running anything.

PROCEDURE. For each trial: draw K seeds per donor (without replacement, independently per donor),
score each donor by its response over just those K seeds, rank the donors, and ask whether the donor
that is genuinely best at full depth lands in the top M. Repeat and average.

"Genuinely best" = the donor with the highest max-DockQ over all 50 of its seeds (here: dockqdonor's
seed137/s3 donor, which reaches 0.958). Ranking uses only response, never the native -- the native is
used solely to define the ground-truth answer and to check the screen.

Two response statistics, because the choice matters:
  max_dockq     needs the native -> an ORACLE screen, upper bound only
  max_iptm      label-free -> what a deployable screen could actually rank on

rng is seeded, so the numbers reproduce.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/screen_depth.py [--trials 2000]
Writes: artifacts/screen_depth.json
"""
import argparse, json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, SAMPLES, THRESH

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=2000)
A = ap.parse_args()
LAB, c1, c2 = "8che_BE", "B", "E"
SEEDS = list(range(101, 151))


def parse(path, pred):
    try:
        d = json.load(open(path))
    except Exception:
        return None
    qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
    for i, t in enumerate(ifs):
        if {t[0], t[1]} == {c1, c2} and i < len(qs):
            try:
                M = json.load(open(pred))["chain_pair_iptm"]
            except Exception:
                return None
            x, y = ord(t[2]) - 65, ord(t[3]) - 65
            if 0 <= x < len(M) and 0 <= y < len(M):
                return (M[x][y], qs[i])
    return None


donors = {}
# the 20 leader templates, 50 seeds each (chedeep)
LO = f"{EXP}/artifacts/sweep_out/leader20/leader20"
for r in range(1, 21):
    name = f"{LAB}_L{r:02d}"
    per = {}
    for s in SEEDS:
        v = [p for p in (parse(f"{EXP}/artifacts/leader20_ost/{LAB}_L{r:02d}_{s}_{k}.json",
                               f"{LO}/{name}/seed_{s}/predictions/{name}_summary_confidence_sample_{k}.json")
                         for k in SAMPLES) if p]
        if v:
            per[s] = v
    if len(per) >= 45:
        donors[f"leader{r:02d}"] = per
# dockqdonor's donor, 50 seeds
DO = f"{EXP}/artifacts/sweep_out/dockqdonor/dockqdonor"
jn = [j for j in json.load(open(f"{EXP}/artifacts/dockq_arm_jobs.json")) if j["interface"] == LAB][0]["name"]
per = {}
for s in SEEDS:
    v = [p for p in (parse(f"{EXP}/artifacts/dockqdonor_ost/{LAB}_{s}_{k}.json",
                           f"{DO}/{jn}/seed_{s}/predictions/{jn}_summary_confidence_sample_{k}.json")
                     for k in SAMPLES) if p]
    if v:
        per[s] = v
if len(per) >= 45:
    donors["dockqdonor"] = per

names = sorted(donors)
full = {n: max(max(x[1] for x in v) for v in donors[n].values()) for n in names}
truth = max(names, key=lambda n: full[n])
print(f"{len(names)} donors at depth, {LAB}")
print(f"  ground-truth best donor (max DockQ over its 50 seeds): {truth} at {full[truth]:.3f}")
print(f"  runner-up: " + ", ".join(f"{n} {full[n]:.3f}" for n in sorted(names, key=lambda n: -full[n])[1:4]))

rng = np.random.default_rng(0)
out = {}
for stat in ("max_dockq", "max_iptm"):
    out[stat] = {}
    for K in (1, 2, 3, 5, 10, 20, 50):
        top1 = top3 = 0
        for _ in range(A.trials):
            sc = {}
            for nme in names:
                ss = list(donors[nme])
                pick = rng.choice(ss, size=min(K, len(ss)), replace=False)
                vals = [x for s in pick for x in donors[nme][s]]
                sc[nme] = max(v[1] for v in vals) if stat == "max_dockq" else max(v[0] for v in vals)
            order = sorted(names, key=lambda n: -sc[n])
            top1 += order[0] == truth
            top3 += truth in order[:3]
        out[stat][K] = {"top1": top1 / A.trials, "top3": top3 / A.trials}
    print(f"\n  screen ranks donors by {stat} over K seeds each "
          f"({'ORACLE -- needs the native' if stat == 'max_dockq' else 'label-free'}):")
    print(f"    {'K seeds/donor':>14} {'jobs for 21 donors':>19} {'P(best donor ranked #1)':>24} {'P(in top 3)':>12}")
    for K, v in out[stat].items():
        print(f"    {K:>14} {21*K:>19} {v['top1']:>24.2f} {v['top3']:>12.2f}")
json.dump({"interface": LAB, "n_donors": len(names), "truth": truth,
           "full_depth_best": full, "results": out},
          open(f"{EXP}/artifacts/screen_depth.json", "w"), indent=1)
print(f"\nWROTE {EXP}/artifacts/screen_depth.json")
