"""eval_cluster_metrics.py — offline: for every saved greedy clustering (per target, per THR),
score MANY label-free selection rules and report how many of the 7 interfaces each rule recovers
the good cluster on. Nothing here re-clusters; it only reads the tiny assignment CSVs.

Assignment CSVs: results/csv_files/cluster_assign/<cx>_<c1><c2>_thr<THR>.csv
  columns: seed, sample, cluster_id, dockq_gt, pair_iptm   (cluster_id = -1 is noise)

"Recovered" = the rule's picked cluster IS the oracle good cluster (the cluster holding the most
good poses, dockq_gt>0.4). Ground truth (dockq_gt) is used ONLY to (a) label which cluster is the
oracle-good one and (b) judge correctness — never inside the label-free selection rule itself.
"""
import csv, glob, os, re, math, statistics as st
from collections import defaultdict

ASSIGN_DIR = os.environ.get("CLUSTER_OUT", "results/csv_files/cluster_assign")
GOOD_THR = 0.4          # a pose is "good" if dockq_gt > this
MIN_N = 5               # ignore clusters smaller than this when a rule has a size floor

T7 = [("8ts0","A","C"),("8ts0","B","C"),("8a44","A","D"),("7yxu","A","C"),
      ("8a44","A","C"),("7zoz","A","C"),("8vui","B","C")]

def load(fn):
    """-> dict cluster_id -> list of (dockq_gt, pair_iptm), plus flat list. noise (cid<0) dropped."""
    cl = defaultdict(list)
    for r in csv.DictReader(open(fn)):
        cid = int(r["cluster_id"])
        if cid < 0:
            continue
        cl[cid].append((float(r["dockq_gt"]), float(r["pair_iptm"])))
    return cl

def oracle_good_cluster(cl):
    """cluster id holding the most good poses; None if no good poses anywhere."""
    best, bn = None, 0
    for cid, mem in cl.items():
        g = sum(1 for gt, _ in mem if gt > GOOD_THR)
        if g > bn:
            best, bn = cid, g
    return best if bn > 0 else None

# --- label-free selection rules: cl -> picked cluster id ---
def _ipt(mem):  return [ip for _, ip in mem]
def _mass(mem): return len(mem)                       # mass = #poses in cluster

def rule_factory(name):
    def pick(cl, floor=False):
        cand = {c: m for c, m in cl.items() if (not floor or len(m) >= MIN_N)}
        if not cand:
            cand = cl
        def score(m):
            ip = _ipt(m); n = _mass(m)
            if name == "mass":          return n
            if name == "mean":          return st.mean(ip)
            if name == "median":        return st.median(ip)
            if name == "max_pose":      return max(ip)
            if name == "mass*mean":     return n * st.mean(ip)
            if name == "mass*median":   return n * st.median(ip)
            if name == "sqrtmass*mean": return math.sqrt(n) * st.mean(ip)
            if name == "sqrtmass*med":  return math.sqrt(n) * st.median(ip)
            if name == "logmass*mean":  return math.log1p(n) * st.mean(ip)
            if name == "logmass*med":   return math.log1p(n) * st.median(ip)
            raise ValueError(name)
        return max(cand, key=lambda c: score(cand[c]))
    return pick

RULES = ["mass", "mean", "median", "max_pose", "mass*mean", "mass*median",
         "sqrtmass*mean", "sqrtmass*med", "logmass*mean", "logmass*med"]

def main():
    files = glob.glob(os.path.join(ASSIGN_DIR, "*_thr*.csv"))
    # group by THR
    bythr = defaultdict(dict)   # thr -> {(cx,c1,c2): clusters}
    for fn in files:
        m = re.search(r"([0-9a-z]{4})_([A-Za-z])([A-Za-z])_thr([0-9.]+)\.csv$", os.path.basename(fn))
        if not m: continue
        cx, c1, c2, thr = m.group(1), m.group(2), m.group(3), float(m.group(4))
        bythr[thr][(cx, c1, c2)] = load(fn)

    print(f"assignment dir: {ASSIGN_DIR}")
    print(f"THR values found: {sorted(bythr)}\n")

    # results[(rule,floor)][thr] = (k_recovered, detail_string)
    summary = defaultdict(dict)
    for thr in sorted(bythr):
        tg = bythr[thr]
        for rule in RULES:
            for floor in (False, True):
                pickfn = rule_factory(rule)
                got = []
                for t in T7:
                    cl = tg.get(t)
                    if not cl:
                        got.append("?"); continue
                    ocg = oracle_good_cluster(cl)
                    if ocg is None:
                        got.append("x"); continue      # no good basin exists
                    picked = pickfn(cl, floor=floor)
                    got.append("G" if picked == ocg else ".")
                k = got.count("G")
                summary[(rule, floor)][thr] = (k, "".join(got))

    # print a table: rows = rule(+floor), cols = THR, cell = k/7
    thrs = sorted(bythr)
    hdr = "rule".ljust(16) + "floor  " + "".join(f"thr{t:<5.2f}" for t in thrs) + " best"
    print(hdr); print("-" * len(hdr))
    best_overall = (0, None)
    for rule in RULES:
        for floor in (False, True):
            row = summary[(rule, floor)]
            cells = ""
            bestk = 0
            for t in thrs:
                k = row[t][0]; cells += f"  {k}/7   "
                bestk = max(bestk, k)
            print(f"{rule:<16}{str(floor):<7}{cells}  {bestk}/7")
            if bestk > best_overall[0]:
                best_overall = (bestk, (rule, floor))
    print(f"\nBEST: {best_overall[0]}/7 via rule={best_overall[1]}")

    # detail for the best rule: which interfaces at which THR
    rule, floor = best_overall[1]
    print(f"\nper-interface detail for best rule ({rule}, floor={floor}):")
    print("  THR   " + " ".join(f"{'-'.join(t)}" for t in T7))
    for t in thrs:
        k, got = summary[(rule, floor)][t]
        cells = " ".join(f"{g:^7}" for g in got)
        print(f"  {t:<5.2f} {cells}   {k}/7")
    print("\n(G=recovered good cluster, .=picked wrong cluster, x=no good basin, ?=missing run)")

if __name__ == "__main__":
    main()
