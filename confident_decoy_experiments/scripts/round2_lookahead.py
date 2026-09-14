#!/usr/bin/env python
"""One-round lookahead for the generational leaders loop: what would ROUND 2's templates be?

The proposed loop is: 50 free seeds -> leaders-20 -> 5 seeds per template (500 poses) -> leaders-20
on THOSE poses -> repeat. Round 1 already exists on disk: the deployed leader20 arm at seeds 101-105
is exactly 20 templates x 5 seeds x 5 samples = 500 poses. So round 2's template set can be derived
with no GPU inference at all -- only the pose-pose DockQ matrix among those 500 poses, which leaders
needs.

The question it answers: does template quality DRIFT UP between rounds? Round 0's leaders topped out
at donor DockQ 0.436 (and suppressed the 0.529 pose entirely at threshold 0.23). If round 2's best
leader is meaningfully above 0.436 the loop is climbing; if it is at or below, round 2 repeats round 1
and the loop has no engine.

LEADERS = greedy non-maximum suppression down the pair_iptm ranking, dropping any pose within
`thresh` pose-pose DockQ of an already-kept pose. thresh=0.23 is the project standard.

Pose-pose DockQ is computed with scripts/dockq_torch.py (exact DockQ 2.1.3, batched on
GPU), NEVER ligand-RMSD. Model chain names for the 8che_BE interface are B?/E? taken from the arm's
OST output by NAME (native B-E -> model C-E here), not assumed.

DockQ-vs-native is used ONLY to report what each candidate template is worth. Leaders itself sees
only pair_iptm and pose-pose geometry -- no labels.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/round2_lookahead.py [--device cuda:3]
Writes: artifacts/round2_lookahead_8che_BE.json
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/home/ebrahim/tts_abag/confident_decoy_experiments/scripts")
from paths import EXP
from dockq_torch import dockq_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--iface", default="8che_BE")
ap.add_argument("--device", default="cuda:3")
ap.add_argument("--thresh", type=float, default=0.23)
A = ap.parse_args()

IFACE = A.iface
SEEDS = [101, 102, 103, 104, 105]
SAMPLES = range(5)
OUT = f"{EXP}/artifacts/sweep_out/leader20/leader20"
CACHE = f"{EXP}/artifacts/leader20_ost"

# model chain pair for this interface, read from the arm's own OST output BY NAME
c1n, c2n = IFACE.split("_")[1][0], IFACE.split("_")[1][1]
probe = json.load(open(f"{CACHE}/{IFACE}_L01_101_0.json"))
mc = None
for t in probe.get("dockq_interfaces") or []:
    if {t[0], t[1]} == {c1n, c2n}:
        mc = (t[2], t[3]); break
assert mc, f"no {IFACE} interface in the probe OST file"
print(f"{IFACE}: native {c1n}-{c2n} -> model {mc[0]}-{mc[1]}")

items, meta = [], []
for L in range(1, 21):
    name = f"{IFACE}_L{L:02d}"
    for s in SEEDS:
        for k in SAMPLES:
            cif = f"{OUT}/{name}/seed_{s}/predictions/{name}_sample_{k}.cif"
            o = f"{CACHE}/{name}_{s}_{k}.json"
            cf = f"{OUT}/{name}/seed_{s}/predictions/{name}_summary_confidence_sample_{k}.json"
            if not (os.path.exists(cif) and os.path.exists(o) and os.path.exists(cf)):
                continue
            d = json.load(open(o)); M = json.load(open(cf))["chain_pair_iptm"]
            qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
            dq = ip = None
            for i, t in enumerate(ifs):
                if {t[0], t[1]} == {c1n, c2n} and i < len(qs):      # BY NAME
                    dq = qs[i]
                    x, y = ord(t[2]) - 65, ord(t[3]) - 65
                    if 0 <= x < len(M) and 0 <= y < len(M):
                        ip = M[x][y]
                    break
            if dq is None or ip is None:
                continue
            items.append((cif, mc[0], mc[1]))
            meta.append({"leader": L, "seed": s, "sample": k, "iptm": ip, "dockq": dq})

n = len(items)
assert n == 500, f"expected 500 round-1 poses, got {n} -- refusing to run leaders on a partial pool"
print(f"round-1 pool: {n} poses (20 templates x 5 seeds x 5 samples)")
iptm = np.array([m["iptm"] for m in meta])
gt = np.array([m["dockq"] for m in meta])
print(f"  best DockQ in round-1 pool: {gt.max():.3f}   (round-0 leaders topped out at 0.436)")

print(f"computing {n}x{n} pose-pose DockQ on {A.device} ...")
Mp = dockq_matrix(items, device=A.device, verbose=False)
Mp = np.asarray(Mp)
assert Mp.shape == (n, n), Mp.shape


def leaders(thresh):
    order = np.argsort(-iptm)
    kept = []
    for i in order:
        if all(Mp[i, j] < thresh for j in kept):
            kept.append(int(i))
    return kept


out = {"iface": IFACE, "n_round1_poses": n, "round1_best_dockq": float(gt.max()),
       "round0_best_leader_dockq": 0.436, "settings": []}
print(f"\n{'thresh':>7} {'#leaders':>9} {'best DockQ among first N leaders':>34}")
print(f"{'':>7} {'':>9} {'N=10':>10} {'N=20':>10} {'N=30':>10}")
for th in (0.23, 0.30, 0.40, 1.01):
    L = leaders(th)
    b = {N: (float(gt[L[:N]].max()) if L else None) for N in (10, 20, 30)}
    out["settings"].append({"thresh": th, "n_leaders": len(L),
                            "best_dockq_at_N": {str(k): v for k, v in b.items()}})
    print(f"{th:>7.2f} {len(L):>9} {b[10]:>10.3f} {b[20]:>10.3f} {b[30]:>10.3f}")

L23 = leaders(A.thresh)[:20]
out["round2_templates"] = [{"rank": i + 1, "leader_of_round1": meta[j]["leader"],
                            "seed": meta[j]["seed"], "sample": meta[j]["sample"],
                            "iptm": round(float(iptm[j]), 4), "dockq": round(float(gt[j]), 3)}
                           for i, j in enumerate(L23)]
print(f"\nROUND 2's 20 TEMPLATES at thresh {A.thresh} (what the loop would actually run):")
print(f"{'rank':>4} {'from L':>7} {'seed':>5} {'smp':>4} {'ipTM':>7} {'DockQ':>7}")
for r in out["round2_templates"]:
    print(f"{r['rank']:>4} {r['leader_of_round1']:>7} {r['seed']:>5} {r['sample']:>4} "
          f"{r['iptm']:>7.3f} {r['dockq']:>7.3f}")
best2 = max(r["dockq"] for r in out["round2_templates"])
print(f"\nbest template round 2 would hand you: {best2:.3f}")
print(f"best template round 1 had          : 0.436")
print(f"DRIFT: {best2 - 0.436:+.3f}")
out["round2_best_template_dockq"] = best2
out["drift_vs_round0"] = round(best2 - 0.436, 3)

json.dump(out, open(f"{EXP}/artifacts/round2_lookahead_{IFACE}.json", "w"), indent=1)
print(f"\nWROTE {EXP}/artifacts/round2_lookahead_{IFACE}.json")
