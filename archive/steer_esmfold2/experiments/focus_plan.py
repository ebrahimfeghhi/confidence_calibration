#!/usr/bin/env python
# focus_plan.py OUT_CSV SOLVE g1 g2 ... -- cx cx cx
# Prints, one line per GPU, the still-unsolved (oracle-max DockQ < SOLVE) complexes assigned
# to it by greedy cost-balance (cost ~ L^2.8). A GPU with no work prints "g:". Empty overall
# (all solved) prints nothing -> controller stops.
import sys, csv, collections
csvp = sys.argv[1]; solve = float(sys.argv[2])
rest = sys.argv[3:]; sep = rest.index("--")
gpus = rest[:sep]; six = rest[sep + 1:]
L = {"9en2": 600, "8vzo": 1006, "8gye": 1216, "8vvl": 707, "8sgn": 868, "9jbq": 573}
best = collections.defaultdict(lambda: -1.0)
try:
    for r in csv.DictReader(open(csvp)):
        if r.get("dockq_status") == "chain_unmapped":
            continue
        a = r["assembly_id"].split("-")[0]
        if a in six:
            try: best[a] = max(best[a], float(r["dockq"]))
            except ValueError: pass
except FileNotFoundError:
    pass
active = [c for c in six if best[c] < solve]
if not active:
    sys.exit(0)  # all solved -> no output
cost = lambda c: (L[c] / 642) ** 2.8
load = {g: 0.0 for g in gpus}; sh = {g: [] for g in gpus}
for c in sorted(active, key=lambda c: -cost(c)):
    g = min(load, key=load.get); sh[g].append(c); load[g] += cost(c)
for g in gpus:
    print(f"{g}:{' '.join(sh[g])}")
