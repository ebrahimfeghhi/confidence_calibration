#!/usr/bin/env python
"""severe_selection_at_depth.py — does ipTM selection find the good dock once the pool is DEEP?

For each confident-decoy interface (decoy_structural_analysis.csv), from the deep score pool
(dockq_ost_esmfold2fast_tts_deep.csv), compute at increasing seed depth:
  oracle  = best DockQ available (coverage ceiling)
  sel     = DockQ of the max-pair_iptm structure (what label-free ipTM selection deploys)
A "severe trap" resolves at depth if sel climbs from <0.23 (decoy) to >=0.23 (acceptable+).

  python steer/experiments/severe_selection_at_depth.py
"""
import csv, collections, math

DEEP = "results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"
IFACES = "results/csv_files/decoy_structural_analysis.csv"
DEPTHS = [50, 100, 250, 500, 1024]

def main():
    want = set()
    for r in csv.DictReader(open(IFACES)):
        want.add((r["cx"][:4], r["c1"], r["c2"]))

    # (cx,c1,c2) -> [(seed, dockq, pair_iptm)]
    pool = collections.defaultdict(list)
    for r in csv.DictReader(open(DEEP)):
        key = (r["assembly_id"][:4], r["chain1"], r["chain2"])
        if key not in want:
            key = (r["assembly_id"][:4], r["chain2"], r["chain1"])  # try swapped
            if key not in want: continue
        if not r["seed"].lstrip("-").isdigit(): continue
        try: dq = float(r["dockq"])
        except (ValueError, KeyError): continue
        try: ip = float(r["pair_iptm"])
        except (ValueError, KeyError): ip = float("nan")
        pool[key].append((int(r["seed"]), dq, ip))

    def sel_at(rows, depth):
        sub = [(dq, ip) for s, dq, ip in rows if s < depth and not math.isnan(ip)]
        if not sub: return None, None
        oracle = max(dq for dq, _ in sub)
        sel = max(sub, key=lambda x: x[1])[0]
        return oracle, sel

    print(f"{'interface':>12} {'nseed':>5} {'oracle':>6} "
          + " ".join(f"sel@{d}".rjust(8) for d in DEPTHS) + "  verdict")
    agg = collections.defaultdict(list); resolved=survived=nodata=0
    rows_out=[]
    for key in sorted(pool):
        rows = pool[key]
        nseed = 1 + max(s for s, _, _ in rows)
        oracle_full = max(dq for _, dq, _ in rows)
        sels = []
        for d in DEPTHS:
            o, s = sel_at(rows, d)
            sels.append(s)
            if s is not None: agg[d].append(s)
        # trap at shallow(50) with a reachable good dock; resolved if deepest sel>=0.23
        deep_sel = next((s for s in reversed(sels) if s is not None), None)
        shallow_sel = sels[0]
        if oracle_full < 0.23 or shallow_sel is None:
            verdict = "no-basin" if oracle_full < 0.23 else "?"
            nodata += 1
        elif shallow_sel < 0.23 and deep_sel is not None and deep_sel >= 0.23:
            verdict = "RESOLVED @depth"; resolved += 1
        elif shallow_sel < 0.23:
            verdict = "trap survives"; survived += 1
        else:
            verdict = "ok@50"
        cx,c1,c2 = key
        sstr = " ".join((f"{s:.2f}" if s is not None else "  - ").rjust(8) for s in sels)
        print(f"{cx+' '+c1+'-'+c2:>12} {nseed:>5} {oracle_full:6.2f} {sstr}  {verdict}")

    print("\n--- aggregate ipTM-selected DockQ vs depth (mean over interfaces with data) ---")
    for d in DEPTHS:
        v = agg[d]
        print(f"  depth {d:>4}: mean sel-DockQ = {sum(v)/len(v):.3f}  (n={len(v)})" if v else f"  depth {d}: -")
    print(f"\nsevere@50 traps: RESOLVED at depth = {resolved}, still SURVIVE = {survived}, "
          f"no-basin/other = {nodata}")

if __name__ == "__main__":
    main()
