#!/usr/bin/env python
"""Per-interface comparison: leader20 (deployable) vs the two reference points.

Three ways to pick a pose, side by side:

  top-ipTM (free)   what the model does with no intervention -- the highest interface pair_iptm pose
                    of the free run. By the definition of a confident decoy this is ALWAYS wrong
                    (DockQ < 0.23), so it is the floor.
  DockQ oracle      the best pose that exists in the free run's 250. Needs the native to identify,
                    so it is not a method -- it is the ceiling perfect selection could reach.
  leader20          the deployable method: top-20 leaders as templates, 5 seeds each, pooled, then
                    the single highest interface pair_iptm pose. Uses no native at any step.

Also reports leader20's own pool ceiling (best DockQ anywhere among its ~500 poses), which separates
"the method never generated a good pose" from "it generated one and selection missed it".

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/leader20_table.py [--md]
Writes: artifacts/leader20_comparison.json  (+ markdown table to stdout with --md)
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, THRESH

ap = argparse.ArgumentParser()
ap.add_argument("--md", action="store_true", help="emit a markdown table")
A = ap.parse_args()

lead = json.load(open(f"{EXP}/artifacts/leader20_scores.json"))
free = json.load(open(f"{EXP}/artifacts/native_arm_scores.json"))
idx = json.load(open(f"{EXP}/artifacts/leader20_index.json"))
# the two template arms to compare against, both already complete
ti = json.load(open(f"{EXP}/artifacts/topiptm26_scores.json"))    # top-ipTM pose's fold as template
dq = json.load(open(f"{EXP}/artifacts/dockqdonor_scores.json"))   # best-DockQ pose's fold as template

rows = []
for lab in sorted(lead):
    v = lead[lab]; f = free.get(lab, {})
    rows.append({
        "interface": lab,
        "top_iptm_free": f.get("free_sel_dockq"),          # no intervention (always < 0.23 here)
        "dockq_oracle_free": f.get("free_best_dockq"),     # ceiling over the free pool
        "leader20": v["method_dockq"],                     # the deployable method's answer
        "topiptm26": ti.get(lab, {}).get("sel_dockq"),      # templated on the top-ipTM (decoy) fold
        "topiptm26_rescued": ti.get(lab, {}).get("rescued"),
        "dockqdonor": dq.get(lab, {}).get("sel_dockq"),     # templated on the best-DockQ fold
        "dockqdonor_rescued": dq.get(lab, {}).get("rescued"),
        "leader20_pool_best": v["best_available"],
        "leader20_rescued": v["method_rescued"],
        "winning_leader": v["winning_leader"],
        "method_iptm": v["method_iptm"],
        "shortlist_had_good_pose": v["shortlist_contained_good_pose"],
        "n_poses": v["n_poses"], "n_templates": v["n_templates"],
    })

n = len(rows)
r_lead = sum(1 for r in rows if r["leader20_rescued"])
r_orac = sum(1 for r in rows if (r["dockq_oracle_free"] or 0) >= THRESH)
r_iptm = sum(1 for r in rows if (r["top_iptm_free"] or 0) >= THRESH)
sl = sum(1 for r in rows if r["shortlist_had_good_pose"])

if A.md:
    print("| interface | top-ipTM (free, no intervention) | DockQ oracle (free pool ceiling) | "
          "**leader20 (deployable)** | leader20 pool ceiling | winning leader |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        mark = "**" if r["leader20_rescued"] else ""
        print(f"| {r['interface']} | {r['top_iptm_free']} | {r['dockq_oracle_free']} | "
              f"{mark}{r['leader20']}{mark} | {r['leader20_pool_best']} | {r['winning_leader']} |")
    print(f"\n**{r_lead}/{n} rescued by leader20** · oracle ceiling {r_orac}/{n} · "
          f"top-ipTM baseline {r_iptm}/{n} · shortlist contained a good pose {sl}/{n}")
else:
    def cell(v, ok):
        return f"{v:.3f}{'*' if ok else ' '}" if v is not None else "    -  "
    print(f"{'interface':11} {'free top-ipTM':>13} | {'topiptm26 fold':>14} {'dockqdonor fold':>15} "
          f"| {'leader20':>9}   (* = rescued)")
    for r in rows:
        print(f"{r['interface']:11} {str(r['top_iptm_free']):>13} | "
              f"{cell(r['topiptm26'], r['topiptm26_rescued']):>14} "
              f"{cell(r['dockqdonor'], r['dockqdonor_rescued']):>15} | "
              f"{cell(r['leader20'], r['leader20_rescued']):>9}")
    rt = sum(1 for r in rows if r["topiptm26_rescued"])
    rd = sum(1 for r in rows if r["dockqdonor_rescued"])
    print(f"\n  free top-ipTM, no template   {r_iptm}/{n}   (floor -- 0 by construction)")
    print(f"  topiptm26  (decoy fold)      {rt}/{n}   ORACLE-free but reinforces the decoy")
    print(f"  dockqdonor (best-DockQ fold) {rd}/{n}   ORACLE (donor chosen using the native)")
    print(f"  leader20   (deployable)      {r_lead}/{n}   <- no native used")
    print(f"  shortlist contained a good pose {sl}/{n}")
    if sl:
        print(f"  conversion (shortlist had one -> method picked one): {r_lead}/{sl}")

json.dump({"n_scored": n, "leader20_rescued": r_lead, "oracle_ceiling": r_orac,
           "top_iptm_baseline": r_iptm, "shortlist_had_good": sl, "rows": rows},
          open(f"{EXP}/artifacts/leader20_comparison.json", "w"), indent=1)
