#!/usr/bin/env python
"""randopt_select_analysis.py — THE selection contrast on a weight-perturbation pool (env-aware).

Per complex on the pool: oracle (best DockQ) vs ipTM-selected (DockQ of the max pair_iptm structure)
vs baseline. Answers: does ipTM-selection RECOVER the oracle (works) or land at baseline (fails)?
Set A defaults; override for B/C via env:
  RANDOPT_CX="7txw 7xld ..." RANDOPT_BASE=results/csv_files/randopt_screen_B/baseline.csv \
  RANDOPT_MINE=results/csv_files/randopt_mine_s002_B.csv python randopt_select_analysis.py
"""
import csv, os, collections
import numpy as np

CX = os.environ.get("RANDOPT_CX",
      "8ts0 8ath 7yxu 7zoz 8a44 8w84 7ua2 8dcm 8oxw 7ucx 8vui 8aok 8oni 8q3j 8ivx 8vzo").split()
BASE = os.environ.get("RANDOPT_BASE", "results/csv_files/randopt_screen/baseline.csv")
MINE = os.environ.get("RANDOPT_MINE", "results/csv_files/randopt_mine_s002.csv")
# optional extra pool CSVs (e.g. set-A's 4 screen draws), comma-separated
EXTRA = [p for p in os.environ.get("RANDOPT_EXTRA", "").split(",") if p]

def base_map(p):
    b = collections.defaultdict(float)
    if not os.path.exists(p): return {}
    for r in csv.DictReader(open(p)):
        if r.get("dockq_status") == "chain_unmapped": continue
        try: b[r["assembly_id"][:4]] = max(b[r["assembly_id"][:4]], float(r["dockq"]))
        except (ValueError, KeyError): pass
    return dict(b)

def main():
    base = base_map(BASE)
    pool = collections.defaultdict(list)   # cx -> [(dockq, pair_iptm), ...]
    for path in [MINE] + EXTRA:
        if not os.path.exists(path): continue
        for r in csv.DictReader(open(path)):
            if r.get("dockq_status") == "chain_unmapped": continue
            try: dq = float(r["dockq"]); ip = float(r["pair_iptm"])
            except (ValueError, KeyError): continue
            pool[r["assembly_id"][:4]].append((dq, ip))
    have = [c for c in CX if c in pool and c in base]
    if not have:
        print("no scored pool yet"); return
    print(f"{'complex':>8} {'base':>6} {'iptmSel':>8} {'oracle':>7}")
    orc, sel, bas = [], [], []
    for c in have:
        o = max(d for d, i in pool[c]); s = max(pool[c], key=lambda x: x[1])[0]
        orc.append(o); sel.append(s); bas.append(base[c])
        print(f"{c:>8} {base[c]:6.3f} {s:8.3f} {o:7.3f}")
    orc, sel, bas = np.mean(orc), np.mean(sel), np.mean(bas)
    rec = 100*(sel-bas)/(orc-bas) if orc > bas else 0
    print(f"\n{'MEAN':>8} {bas:6.3f} {sel:8.3f} {orc:7.3f}   ({len(have)} complexes)")
    print(f"ipTM-selection recovers {rec:.0f}% of the oracle gain over baseline "
          f"-> selection {'WORKS' if rec>60 else 'PARTIAL' if rec>25 else 'FAILS'} on this set")

if __name__ == "__main__":
    main()
