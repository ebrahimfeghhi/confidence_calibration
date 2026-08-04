#!/usr/bin/env python
"""Pick the exact (seed, sample) to re-run with PAE dumping, for the 18 rescued confident decoys.

GOAL: for each interface where native templating flipped the model's belief, capture the token-level
PAE of the two poses that define the flip:

  DECOY pose  (free run, no templates)  -- the pose ipTM actually selected, which is WRONG.
                                           Picked as: highest pair_iptm over all 250 poses,
                                           asserted DockQ < 0.23.
  GOOD pose   (native-templated run)    -- a correct pose the model is also confident about.
                                           Picked as: highest pair_iptm among poses with DockQ >= 0.23.

Selecting the good pose by ipTM (not by best DockQ) is deliberate: the question is why the model's
BELIEF moved, so the native-arm pose should be the one its confidence would actually deploy.

No OST and no GPU here -- every pose was already scored, so this only reads the caches:
  free   artifacts/abag_all_ost/{pdb}_{seed}_{k}.json     + predictions_template/<pdb>/...
  native artifacts/native_arm_ost/{pdb}_{seed}_{k}.json   + the native50 prediction dir

Interfaces sharing a PDB (8che, 8fbw, 8q3j, 8slb, 8yxi) may land on different seeds; each distinct
(pdb, arm, seed) becomes one re-run job, so a shared seed costs one job, not two.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/pae_pick_seeds.py
Writes: artifacts/pae_seed_picks.json, artifacts/pae_rerun_jobs.json
"""
import json, os, sys, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, PRED, SEEDS, SAMPLES, THRESH
from score_native_arm import locate                       # the "most seeds in range" fix

EXCLUDE = {"7z2m_AJ", "8udz_AC"}                           # mislabelled crystal contacts, see report §5
FREE_CACHE = f"{EXP}/artifacts/abag_all_ost"
NAT_CACHE = f"{EXP}/artifacts/native_arm_ost"


def poses(cache, pdb, pred_dir, name, c1, c2):
    """[(iptm, dockq, seed, sample)] for one interface, read from cached OST + confidence files."""
    out = []
    for s in SEEDS:
        for k in SAMPLES:
            o = f"{cache}/{pdb}_{s}_{k}.json"
            cf = f"{pred_dir}/seed_{s}/predictions/{name}_summary_confidence_sample_{k}.json"
            if not (os.path.exists(o) and os.path.exists(cf)):
                continue
            try:
                d = json.load(open(o)); M = json.load(open(cf))["chain_pair_iptm"]
            except Exception:
                continue
            qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
            for i, t in enumerate(ifs):
                if {t[0], t[1]} == {c1, c2} and i < len(qs):      # match by NAME, never dockq[0]
                    x, y = ord(t[2]) - 65, ord(t[3]) - 65
                    if 0 <= x < len(M) and 0 <= y < len(M):
                        out.append((M[x][y], qs[i], s, k))
                    break
    return out


sc = json.load(open(f"{EXP}/artifacts/native_arm_scores.json"))
rescued = sorted(k for k, v in sc.items()
                 if v["free_status"] == "confident_decoy" and v["rescued"] and k not in EXCLUDE)
print(f"{len(rescued)} rescued confident decoys\n")

picks, problems = {}, []
print(f"{'interface':12} {'DECOY (free)':>28}   {'GOOD (native)':>30}")
for lab in rescued:
    v = sc[lab]; pdb = v["pdb"]
    parts = v["chains"].split("-")
    c1, c2 = (parts[0], "-".join(parts[1:])) if len(parts) > 2 else (parts[0], parts[1])

    free = poses(FREE_CACHE, pdb, f"{PRED}/{pdb}", pdb, c1, c2)
    nd, nname = locate(pdb)
    nat = poses(NAT_CACHE, pdb, nd, nname, c1, c2) if nd else []
    if not free or not nat:
        problems.append(f"{lab}: free={len(free)} native={len(nat)} poses -- skipped"); continue

    decoy = max(free, key=lambda x: x[0])                       # what selection actually picked
    good_pool = [p for p in nat if p[1] >= THRESH]
    if not good_pool:
        problems.append(f"{lab}: no acceptable native pose despite rescued=True -- skipped"); continue
    good = max(good_pool, key=lambda x: x[0])                   # confident AND correct

    if decoy[1] >= THRESH:
        problems.append(f"{lab}: top-ipTM free pose has DockQ {decoy[1]:.3f} >= {THRESH} -- not a decoy")
        continue

    picks[lab] = {
        "pdb": pdb, "chains": [c1, c2],
        "decoy": {"arm": "free", "seed": decoy[2], "sample": decoy[3],
                  "pair_iptm": round(decoy[0], 3), "dockq": round(decoy[1], 3),
                  "pred_dir": f"{PRED}/{pdb}", "name": pdb},
        "good": {"arm": "native", "seed": good[2], "sample": good[3],
                 "pair_iptm": round(good[0], 3), "dockq": round(good[1], 3),
                 "pred_dir": nd, "name": nname},
    }
    print(f"{lab:12}  seed {decoy[2]} s{decoy[3]} ipTM {decoy[0]:.3f} DockQ {decoy[1]:.3f}"
          f"   |  seed {good[2]} s{good[3]} ipTM {good[0]:.3f} DockQ {good[1]:.3f}")

json.dump(picks, open(f"{EXP}/artifacts/pae_seed_picks.json", "w"), indent=1)

# Collapse to distinct (pdb, arm, seed) re-runs -- one GPU job each, 5 samples per job.
jobs = {}
for lab, p in picks.items():
    for role in ("decoy", "good"):
        e = p[role]
        key = f"{p['pdb']}|{e['arm']}|{e['seed']}"
        jobs.setdefault(key, {"pdb": p["pdb"], "arm": e["arm"], "seed": e["seed"],
                              "name": e["name"], "interfaces": []})["interfaces"].append(f"{lab}:{role}")
jobs = list(jobs.values())
json.dump(jobs, open(f"{EXP}/artifacts/pae_rerun_jobs.json", "w"), indent=1)

print(f"\n{len(picks)}/{len(rescued)} interfaces picked")
print(f"{len(jobs)} distinct re-run jobs "
      f"({sum(1 for j in jobs if j['arm']=='free')} free, {sum(1 for j in jobs if j['arm']=='native')} native)")
if problems:
    print(f"\nPROBLEMS ({len(problems)}):")
    for p in problems:
        print("  " + p)
print(f"\nWROTE artifacts/pae_seed_picks.json, artifacts/pae_rerun_jobs.json")
