#!/usr/bin/env python
"""Full per-pose picture for the high-spread self-templating sweep, one block per complex.

Each row is one of the seed's five diffusion poses:
  shown DockQ / shown ipTM   the pose as the FREE run produced and scored it
  arm ipTM / arm DockQ       what came back after re-folding with that pose as the sole template
The all-5 row is the single arm that took all five poses at once.

The point of the layout: shown ipTM is near-constant down the column (ipTM is a property of the seed,
so free inference cannot rank a seed's own poses), while shown DockQ varies hugely. If arm ipTM then
spreads out and orders with shown DockQ, self-templating has created ranking information that did not
exist before.

Only complexes with all five per-pose arms scored are shown; partial ones are listed underneath.
"""
import collections
import csv
import statistics as st
import sys

sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit")
from fmt import table

REPO = "/home/ebrahim/tts_abag/confident_decoy_experiments"


def main():
    want = {}
    for line in open(f"{REPO}/artifacts/highspread_seeds.tsv"):
        iface, _pdb, seed = line.rstrip("\n").split("\t")
        want[iface] = seed

    shown = collections.defaultdict(dict)
    for r in csv.DictReader(open(f"{REPO}/artifacts/pxm2025/dockq_pxm2025.csv")):
        iface = f"{r['pdb']}_{r['chain1']}{r['chain2']}"
        if iface in want and r["seed"] == want[iface]:
            shown[iface][int(r["sample"])] = (float(r["dockq"]), float(r["pair_iptm"]))

    arms = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(f"{REPO}/artifacts/highspread/results.csv")):
        arms[r["iface"]][r["arm"]].append((float(r["iptm"]), float(r["dockq"])))

    partial = []
    for iface in sorted(arms):
        per = {int(a[8:]): v for a, v in arms[iface].items()
               if a.startswith("tmpl_hsp") and a[8:].isdigit()}
        if len(per) < 5:
            partial.append(f"{iface} ({len(per)}/5 per-pose"
                           f"{', all-5 done' if 'tmpl_hspall' in arms[iface] else ''})")
            continue
        s = shown[iface]
        rows = []
        for i in sorted(per):
            ip = [x for x, _ in per[i]]
            dq = [q for _, q in per[i]]
            rows.append([f"pose {i}", s[i][0], s[i][1], st.mean(ip), max(dq), st.mean(dq)])
        alla = arms[iface].get("tmpl_hspall")
        if alla:
            ip = [x for x, _ in alla]
            dq = [q for _, q in alla]
            rows.append(["ALL 5 together", None, None, st.mean(ip), max(dq), st.mean(dq)])

        # what each protocol hands you, selecting by ipTM with no DockQ knowledge
        free_sel = max(s.values())[1]
        best_arm = max(per, key=lambda i: st.mean([x for x, _ in per[i]]))
        pp_sel = max(per[best_arm])[1]
        all_sel = max(alla)[1] if alla else None
        print(table(["", "shown DockQ", "shown ipTM", "arm ipTM", "arm best DockQ", "arm mean DockQ"],
                    rows, align="l r r r r r",
                    dec={"shown DockQ": 3, "shown ipTM": 3, "arm ipTM": 3,
                         "arm best DockQ": 3, "arm mean DockQ": 3},
                    title=f"{iface}  seed {want[iface]}",
                    rules=[4]))
        print(f"  selected by ipTM -> free {free_sel:.3f} | per-pose {pp_sel:.3f}"
              + (f" | all-5 {all_sel:.3f}" if all_sel is not None else " | all-5 pending"))
        print()
    if partial:
        print("  still running: " + ", ".join(partial))


if __name__ == "__main__":
    main()
