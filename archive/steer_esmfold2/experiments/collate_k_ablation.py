"""Collate the 7txw K/BPTT ablation -> one comparison table.

Reads, per experiment tag: repel_{tag}.csv (steer cost), dockq_{tag}.csv (per-(run,sample)
iptm + ab-ag DockQ), decode_cost_{tag}.csv (decode wall-clock + peak VRAM). Reports cost
(steer/decode wall-clock + peak VRAM, analytical trunk-loop FLOP proxy) and quality (DockQ
+ iptm distributions over the runs).

Run: python collate_k_ablation.py [tag1 tag2 ...]
"""
import sys
import csv
from pathlib import Path
from statistics import mean, median

OUT = Path(__file__).resolve().parent

# Analytical FLOP proxy: trunk loop-units (forward + backward) per grad step. The steering
# cost that actually varies by config; decode is a ~constant overhead on top.
LOOPS_PER_STEP = {
    "7txw_k3": 3 + 3,                                  # full bptt through K=3
    "7txw_k6": 6 + 6,                                  # full bptt through K=6
    "7txw_mk369_bptt3": (3 + 6 + 9) + (3 + 3 + 3),     # fwd to {3,6,9}; bwd last 3 each
}


def read_csv(p):
    return list(csv.DictReader(open(p))) if p.exists() else []


def fnum(x):
    try:
        v = float(x)
        return v if v == v else None                  # drop NaN
    except (TypeError, ValueError):
        return None


def main():
    tags = sys.argv[1:] or ["7txw_k3", "7txw_k6", "7txw_mk369_bptt3"]
    hdr = (f"{'experiment':>18} | {'runs':>4} {'steer_s':>8} {'st_GB':>6} | "
           f"{'dec_s':>6} {'dec_GB':>6} | {'loopFLOP':>9} | "
           f"{'dq_mean':>7} {'dq_med':>6} {'dq_best':>7} {'>=.23':>5} {'>=.80':>5} | {'ip_mn':>6} {'ip_bst':>6}")
    print(hdr)
    print("-" * len(hdr))
    for tag in tags:
        steer = read_csv(OUT / f"repel_{tag}.csv")
        dq = read_csv(OUT / f"dockq_{tag}.csv")
        dcost = read_csv(OUT / f"decode_cost_{tag}.csv")
        nruns = len(steer)
        steer_s = sum(fnum(r.get("steer_secs")) or 0 for r in steer)
        steer_gb = max((fnum(r.get("peak_gb")) or 0 for r in steer), default=0)
        dec_s = fnum(dcost[0]["decode_secs"]) if dcost else None
        dec_gb = fnum(dcost[0]["peak_gb"]) if dcost else None

        by_run = {}
        for r in dq:
            by_run.setdefault(r["run"], []).append(r)
        best_dq = [m for m in (max((fnum(x["dockq"]) for x in rows if fnum(x["dockq"]) is not None),
                                   default=None) for rows in by_run.values()) if m is not None]
        best_ip = [m for m in (max((fnum(x["iptm"]) for x in rows if fnum(x["iptm"]) is not None),
                                   default=None) for rows in by_run.values()) if m is not None]
        loopflop = LOOPS_PER_STEP.get(tag, 0) * 5 * nruns      # x M=5 x runs

        def f(x, p="{:.3f}"):
            return p.format(x) if x is not None else "  -"
        dqm = mean(best_dq) if best_dq else None
        dqmed = median(best_dq) if best_dq else None
        dqbest = max(best_dq) if best_dq else None
        n23 = sum(d >= 0.23 for d in best_dq)
        n80 = sum(d >= 0.80 for d in best_dq)
        ipm = mean(best_ip) if best_ip else None
        ipbest = max(best_ip) if best_ip else None
        print(f"{tag:>18} | {nruns:>4} {steer_s:>8.0f} {steer_gb:>6.1f} | "
              f"{f(dec_s, '{:.0f}'):>6} {f(dec_gb, '{:.1f}'):>6} | {loopflop:>9} | "
              f"{f(dqm):>7} {f(dqmed):>6} {f(dqbest):>7} {n23:>5} {n80:>5} | {f(ipm):>6} {f(ipbest):>6}")
    print("\nloopFLOP = trunk loop-units (fwd+bwd) x M x runs — the steering cost that varies by config; "
          "decode is the shared overhead.")


if __name__ == "__main__":
    main()
