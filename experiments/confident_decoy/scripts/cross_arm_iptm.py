#!/usr/bin/env python
"""Is the model MORE confident in the good pose (good template) than in the decoy (decoy template)?

If a good template makes ipTM prefer the correct pose by a wider margin than a decoy template makes it
prefer the wrong one, then ipTM can arbitrate BETWEEN template conditions -- pool the poses from
several templated runs, take the highest ipTM overall, and land on the right answer. That is the
germ of a deployable scheme: generate under several candidate templates, then let confidence choose.

For every interface this compares:
    dockqdonor  top-`pair_iptm` pose  (correct where that arm rescues)
    topiptm26   top-`pair_iptm` pose  (the decoy -- that arm templates from the decoy's own fold)
and asks which would win if the two pools were merged and ranked by ipTM alone.

IMPORTANT CALIBRATION CAVEAT, reported alongside: the FREE run's original decoy typically has a HIGHER
ipTM than the templated good pose. So ipTM is NOT a globally comparable scale across arbitrary
conditions -- pooling free + templated poses would re-select the decoy. The claim this script can
support is narrower: *conditioned on a template*, ipTM ranks the resulting pose sensibly, and a good
template lifts confidence in the right pose more than a decoy template lifts it in the wrong one.

NOT DEPLOYABLE AS-IS: knowing which donor was docking-correct requires the answer.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/cross_arm_iptm.py
Writes: artifacts/cross_arm_iptm.json
"""
import json, os, sys, statistics as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, THRESH

ARMS = ["dockqdonor", "topiptm26", "ifacelddt26", "globallddt26"]
S = {}
for a in ARMS:
    p = f"{EXP}/artifacts/{a}_scores.json"
    if os.path.exists(p):
        S[a] = json.load(open(p))
free = json.load(open(f"{EXP}/artifacts/native_arm_scores.json"))

if "dockqdonor" not in S or "topiptm26" not in S:
    sys.exit("need both dockqdonor and topiptm26 scores")

both = sorted(set(S["dockqdonor"]) & set(S["topiptm26"]))
rows, wins, ties = [], 0, 0
for k in both:
    a, b = S["dockqdonor"][k], S["topiptm26"][k]
    f = free.get(k, {})
    good_wins = a["top_pair_iptm"] > b["top_pair_iptm"]
    rows.append({
        "interface": k, "dockqdonor_rescued": a["rescued"],
        "dq_top_iptm": a["top_pair_iptm"], "dq_sel_dockq": a["sel_dockq"],
        "dq_n_acceptable": a["n_acceptable"],
        "ti_top_iptm": b["top_pair_iptm"], "ti_sel_dockq": b["sel_dockq"],
        "ti_n_acceptable": b["n_acceptable"],
        "free_top_iptm": f.get("top_pair_iptm"), "free_sel_dockq": f.get("free_sel_dockq"),
        "iptm_margin": round(a["top_pair_iptm"] - b["top_pair_iptm"], 3),
        "pooled_winner": "dockqdonor" if good_wins else "topiptm26",
        # would merging the two templated pools and ranking by ipTM alone give an acceptable pose?
        "pooled_pick_dockq": a["sel_dockq"] if good_wins else b["sel_dockq"],
        "pooled_pick_acceptable": (a["sel_dockq"] if good_wins else b["sel_dockq"]) >= THRESH,
    })

resc = [r for r in rows if r["dockqdonor_rescued"]]
out = {
    "n_interfaces_compared": len(rows),
    "n_dockqdonor_rescued": len(resc),
    "good_wins_among_rescued": sum(1 for r in resc if r["pooled_winner"] == "dockqdonor"),
    "median_iptm_dockqdonor_rescued": round(st.median(r["dq_top_iptm"] for r in resc), 3) if resc else None,
    "median_iptm_topiptm_same": round(st.median(r["ti_top_iptm"] for r in resc), 3) if resc else None,
    "median_iptm_free_decoy_same": round(
        st.median(r["free_top_iptm"] for r in resc if r["free_top_iptm"] is not None), 3) if resc else None,
    "pooled_two_arm_acceptable": sum(1 for r in rows if r["pooled_pick_acceptable"]),
    "rows": rows,
}
json.dump(out, open(f"{EXP}/artifacts/cross_arm_iptm.json", "w"), indent=1)

print(f"{'interface':11} {'resc':>4} {'dq ipTM':>8} {'dq DockQ':>9}  {'ti ipTM':>8} {'ti DockQ':>9} "
      f"{'margin':>7}  {'pooled pick':>12}")
for r in rows:
    print(f"{r['interface']:11} {'Y' if r['dockqdonor_rescued'] else '':>4} {r['dq_top_iptm']:8.3f} "
          f"{r['dq_sel_dockq']:9.3f}  {r['ti_top_iptm']:8.3f} {r['ti_sel_dockq']:9.3f} "
          f"{r['iptm_margin']:+7.3f}  {r['pooled_pick_dockq']:7.3f}"
          f"{' OK' if r['pooled_pick_acceptable'] else '   '}")
print(f"\nAmong the {out['n_dockqdonor_rescued']} interfaces dockqdonor rescues, the GOOD pose "
      f"out-confidences the decoy in {out['good_wins_among_rescued']}.")
print(f"median ipTM  good(dockqdonor) {out['median_iptm_dockqdonor_rescued']}  "
      f"decoy(topiptm26) {out['median_iptm_topiptm_same']}  "
      f"FREE-run decoy {out['median_iptm_free_decoy_same']}")
print(f"CALIBRATION WARNING: the free-run decoy's ipTM is the highest of the three -- ipTM is not "
      f"comparable across arbitrary conditions, only within a template condition.")
print(f"\nMerging the two templated pools and ranking by ipTM alone gives an acceptable pose on "
      f"{out['pooled_two_arm_acceptable']}/{len(rows)} interfaces.")
print(f"WROTE artifacts/cross_arm_iptm.json")
