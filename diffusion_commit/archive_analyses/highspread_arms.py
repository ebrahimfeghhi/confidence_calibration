#!/usr/bin/env python
"""Per-arm detail for the high-spread self-templating sweep: what each arm was shown, what it returned.

One row per arm. "shown" is the pose that arm was templated on, as scored in the free pool; "arm ipTM"
and "arm DockQ" are that arm's own 5 output poses. The all-5 arm has no single shown pose, so its
shown columns carry the spread of the five it averaged.

The question the rows answer: after committing to one pose, does the confidence the model reports
track the quality of THAT pose? Free inference cannot do this -- ipTM is a property of the seed, so
all five poses of a seed inherit nearly the same value (on these seeds the shown-ipTM spread is as low
as 0.010 while DockQ spans 0.83).
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

    rows = []
    for iface in sorted(arms):
        s = shown.get(iface, {})
        order = sorted(arms[iface], key=lambda a: (a == "tmpl_hspall", a))
        for arm in order:
            v = arms[iface][arm]
            ip = [x for x, _ in v]
            dq = [q for _, q in v]
            if arm == "tmpl_hspall":
                sh_dq = f"{min(d for d, _ in s.values()):.2f}-{max(d for d, _ in s.values()):.2f}" if s else ""
                sh_ip = None
                lab = "all 5 poses"
            else:
                i = int(arm[8:])
                sh_dq = f"{s[i][0]:.3f}" if i in s else ""
                sh_ip = s[i][1] if i in s else None
                lab = f"pose {i}"
            rows.append([iface, lab, sh_dq, sh_ip, len(v),
                         st.mean(ip), max(dq), st.mean(dq)])
    print(table(["interface", "templated on", "shown DockQ", "shown ipTM", "poses out",
                 "arm ipTM", "arm best DockQ", "arm mean DockQ"], rows,
                align="l l r r r r r r",
                dec={"shown ipTM": 3, "arm ipTM": 3, "arm best DockQ": 3, "arm mean DockQ": 3}))
    print(f"\n  arms shown: {len(rows)}   (shown DockQ/ipTM are the pose as scored in the free pool)")


if __name__ == "__main__":
    main()
