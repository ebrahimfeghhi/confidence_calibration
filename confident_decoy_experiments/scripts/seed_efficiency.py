#!/usr/bin/env python
"""Do we need 50 seeds? And is the dockqdonor rescue SEED-SPECIFIC or broad?

Two questions, both answerable from the poses already scored -- no new GPU.

(1) SEED EFFICIENCY. The rescue rule is "the top-pair_iptm pose in the pool has DockQ >= 0.23". With
    fewer seeds the pool is smaller and the winner can change. Sub-sampling the existing 50 seeds many
    times gives the probability of the same verdict at K seeds, i.e. how much generation is actually
    needed.

(2) IS THE RESCUE SEED-SPECIFIC? The donor fold came from one particular seed. If templating only
    helps the seed that already produced the good pose, the arm is near-tautological and a matched
    single-seed run would be the honest experiment. If acceptable poses are spread across many seeds,
    the template is transferring something general.

Per-pose (pair_iptm, DockQ) are re-read from the arm's OST cache, so this re-derives from artifacts
rather than trusting the aggregate scores file.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/seed_efficiency.py [--arm dockqdonor]
Writes: artifacts/<arm>_seed_efficiency.json
"""
import argparse, json, os, random, sys, statistics as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, SEEDS, SAMPLES, THRESH

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="dockqdonor")
ap.add_argument("--trials", type=int, default=200)
A = ap.parse_args()

CACHE = f"{EXP}/artifacts/{A.arm}_ost"
OUT = f"{EXP}/artifacts/sweep_out/{A.arm}/{A.arm}"
jobs = {j["interface"]: j for j in json.load(open(f"{EXP}/artifacts/dockq_arm_jobs.json"))}
scores = json.load(open(f"{EXP}/artifacts/{A.arm}_scores.json"))
don = json.load(open(f"{EXP}/artifacts/dockq_arm_donors.json"))
rng = random.Random(0)


def per_pose(lab):
    """[(seed, sample, pair_iptm, dockq)] from the arm's OST cache + confidence JSONs."""
    job = jobs[lab]; name = job["name"]
    parts = lab.split("_", 1)[1]
    c1, c2 = (parts[0], parts[1:]) if len(parts) > 2 else (parts[0], parts[1])
    out = []
    for s in SEEDS:
        for k in SAMPLES:
            o = f"{CACHE}/{lab}_{s}_{k}.json"
            cf = f"{OUT}/{name}/seed_{s}/predictions/{name}_summary_confidence_sample_{k}.json"
            if not (os.path.exists(o) and os.path.exists(cf)):
                continue
            try:
                d = json.load(open(o)); M = json.load(open(cf))["chain_pair_iptm"]
            except Exception:
                continue
            qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
            for i, t in enumerate(ifs):
                if {t[0], t[1]} == {c1, c2} and i < len(qs):
                    x, y = ord(t[2]) - 65, ord(t[3]) - 65
                    if 0 <= x < len(M) and 0 <= y < len(M):
                        out.append((s, k, M[x][y], qs[i]))
                    break
    return out


rows = []
KS = [1, 2, 3, 5, 10, 25, 50]
print(f"{'interface':11} {'resc@50':>7} " + " ".join(f"{'K='+str(k):>7}" for k in KS)
      + f"  {'seeds with an OK pose':>22} {'donor seed OK?':>15}")
for lab in sorted(scores):
    poses = per_pose(lab)
    if len(poses) < 200:
        continue
    by_seed = {}
    for s, k, ip, dq in poses:
        by_seed.setdefault(s, []).append((ip, dq))
    seeds = sorted(by_seed)
    truth = scores[lab]["rescued"]

    # (1) probability of rescue at K seeds, over random subsets
    probs = {}
    for K in KS:
        if K > len(seeds):
            probs[K] = None; continue
        hits = 0
        for _ in range(A.trials):
            sub = rng.sample(seeds, K)
            pool = [p for s_ in sub for p in by_seed[s_]]
            hits += max(pool, key=lambda x: x[0])[1] >= THRESH
        probs[K] = round(hits / A.trials, 3)

    # (2) how many distinct seeds produced at least one acceptable pose
    ok_seeds = [s_ for s_ in seeds if any(dq >= THRESH for _, dq in by_seed[s_])]
    dseed = don.get(lab, {}).get("seed")
    rows.append({"interface": lab, "rescued_at_50": truth, "prob_by_k": probs,
                 "n_seeds": len(seeds), "n_seeds_with_acceptable": len(ok_seeds),
                 "frac_seeds_with_acceptable": round(len(ok_seeds) / len(seeds), 3),
                 "donor_seed": dseed, "donor_seed_produced_acceptable": dseed in ok_seeds})
    r = rows[-1]
    print(f"{lab:11} {'Y' if truth else '.':>7} "
          + " ".join(f"{(probs[k] if probs[k] is not None else float('nan')):7.2f}" for k in KS)
          + f"  {len(ok_seeds):>3}/{len(seeds):<3} ({r['frac_seeds_with_acceptable']:.0%})"
          + f"  {'yes' if r['donor_seed_produced_acceptable'] else 'no':>13}")

json.dump(rows, open(f"{EXP}/artifacts/{A.arm}_seed_efficiency.json", "w"), indent=1)

print(f"\nEXPECTED number of interfaces rescued if we had run only K seeds:")
for K in KS:
    v = [r["prob_by_k"][K] for r in rows if r["prob_by_k"].get(K) is not None]
    print(f"  K={K:2d}  {sum(v):5.1f} / {len(v)}   (at K=50 the actual count is "
          f"{sum(1 for r in rows if r['rescued_at_50'])})")

resc = [r for r in rows if r["rescued_at_50"]]
if resc:
    print(f"\nAmong the {len(resc)} rescued interfaces:")
    print(f"  median fraction of seeds producing at least one acceptable pose: "
          f"{st.median(r['frac_seeds_with_acceptable'] for r in resc):.0%}")
    print(f"  donor's own seed produced an acceptable pose in "
          f"{sum(1 for r in resc if r['donor_seed_produced_acceptable'])}/{len(resc)}")
print(f"\nWROTE artifacts/{A.arm}_seed_efficiency.json")
