#!/usr/bin/env python
"""Does self-templating a pose make the model's confidence track THAT pose's quality?

THE SETUP. 10 seeds chosen because their 5 diffusion samples disagree violently -- DockQ spread
0.82-0.92 within one seed, every seed straddling the 0.23 acceptability line. Free inference cannot
rank those poses: ipTM is a property of the seed, so all 5 inherit nearly the same value (on 9vp0,
DockQ spans 0.833 while ipTM spans 0.010).

THE ARMS, all fp32, 10 recycles, template present from the start:
  hsp0..hsp4  templated on ONE pose each -> each pose gets its own trunk, so its own confidence
  hspall      templated on all 5 at once -> tests whether averaging disagreeing poses destroys it

READING IT. "ranks best" asks whether the arm with the highest ipTM is the arm that was shown the
best pose. That is the selector question. "sel DockQ" is what you would actually hold if you picked
by ipTM across the per-pose arms; compare it to "free sel DockQ", what the untemplated seed gives.
"""
import collections
import csv
import os
import sys

sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit")
from fmt import table

REPO = "/home/ebrahim/tts_abag/confident_decoy_experiments"
RES = f"{REPO}/artifacts/highspread/results.csv"
POOL = f"{REPO}/artifacts/pxm2025/dockq_pxm2025.csv"
SEEDS = f"{REPO}/artifacts/highspread_seeds.tsv"


def main():
    want = {}
    for line in open(SEEDS):
        iface, pdb, seed = line.rstrip("\n").split("\t")
        want[iface] = seed

    shown = collections.defaultdict(dict)      # iface -> sample -> (dockq, iptm) from the free pool
    for r in csv.DictReader(open(POOL)):
        iface = f"{r['pdb']}_{r['chain1']}{r['chain2']}"
        if iface in want and r["seed"] == want[iface]:
            shown[iface][int(r["sample"])] = (float(r["dockq"]), float(r["pair_iptm"]))

    if not os.path.exists(RES):
        print("  no results yet")
        return
    arms = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(RES)):
        arms[r["iface"]][r["arm"]].append((float(r["iptm"]), float(r["dockq"])))

    rows = []
    ranked = tested = 0
    for iface in sorted(arms):
        per = {int(a[8:]): v for a, v in arms[iface].items()
               if a.startswith("tmpl_hsp") and a[8:].isdigit()}
        alla = arms[iface].get("tmpl_hspall")
        s = shown.get(iface, {})
        free_sel = max(s.values())[1] if s else None          # what ipTM picks untemplated
        best_shown = max(s.items(), key=lambda kv: kv[1][0])[0] if s else None

        if per:
            # per-pose arm ipTM = mean over that arm's 5 output poses
            armip = {i: sum(x for x, _ in v) / len(v) for i, v in per.items()}
            top = max(armip, key=armip.get)
            sel = max(per[top])[1]                            # DockQ of the top-ipTM pose in that arm
            ok = (top == best_shown)
            tested += 1
            ranked += int(ok)
        else:
            armip, top, sel, ok = {}, None, None, None

        rows.append([iface, want[iface], len(per),
                     " ".join(f"{s[i][0]:.2f}" for i in sorted(s)) if s else "",
                     f"{max(s[i][1] for i in s) - min(s[i][1] for i in s):.3f}" if s else "",
                     " ".join(f"{armip[i]:.3f}" for i in sorted(armip)) if armip else "",
                     best_shown, top, "yes" if ok else ("no" if ok is not None else ""),
                     free_sel, sel,
                     max(alla)[1] if alla else None])
    print(table(["interface", "seed", "arms", "shown DockQ (per pose)", "shown ipTM spread",
                 "arm ipTM (per pose)", "best pose", "ipTM picks", "ranks best",
                 "free sel DockQ", "per-pose sel DockQ", "all-5 sel DockQ"], rows,
                align="l l r l r l r r l r r r",
                dec={"free sel DockQ": 3, "per-pose sel DockQ": 3, "all-5 sel DockQ": 3}))
    if tested:
        print(f"\n  seeds with all 5 per-pose arms scored: "
              f"{sum(1 for r in rows if r[2] == 5)}/{len(rows)}")
        print(f"  ipTM picks the best pose: {ranked}/{tested} seeds (chance = 1/5)")


if __name__ == "__main__":
    main()
