#!/usr/bin/env python
"""Score the COMPLETE 2500-seed free pool for 8che, for a budget-matched comparison
against allpose8che10 (250 donors x 10 seeds = 2500 seeds).

The free pool is split across two locations, which is why earlier tallies were short:
  seeds    0- 150  -> /home/ebrahim/Protenix/foldbench_abag/predictions_template/8che/seed_N
                      (101-150 is the canonical 250-pose pool; 0-100 had NEVER been scored)
  seeds  151-2499  -> artifacts/sweep_out/freedeep8che/.../seed_N
Total 2500 seeds x 5 samples = 12500 poses.

SCORING PROTOCOL is the project standard (tts_abag/eval/eval_ost_foldbench.py): one OST call
per pose, full native assembly vs full model, --fault-tolerant --min-pep-length 4
--min-nuc-length 4 --dockq, interface extracted BY NAME (never dockq[0]).

Refuses to report unless all 2500 seeds are present with 5 poses each and every pose scored;
a partial pool silently understates best-of-pool, which is exactly the comparison being made.

Usage: /home/ebrahim/Protenix/.venv/bin/python scripts/score_free_complete.py [--workers 16]
Writes: artifacts/free8che_complete_scores.json
"""
import argparse, json, os, glob, subprocess, sys, statistics as st
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, OST, ENV, GT

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=16)
A = ap.parse_args()

CACHE = f"{EXP}/artifacts/free8che_complete_ost"
os.makedirs(CACHE, exist_ok=True)
IFACES = {"8che_AE": ("A", "E"), "8che_BE": ("B", "E")}
NSAMP = 5
PRED = "/home/ebrahim/Protenix/foldbench_abag/predictions_template/8che"
FD = f"{EXP}/artifacts/sweep_out/freedeep8che"


def cif_and_conf(seed, k):
    """(cif, summary_confidence json) for one free pose, from whichever tree holds it."""
    p = f"{PRED}/seed_{seed}/predictions"
    if os.path.exists(f"{p}/8che_sample_{k}.cif"):
        return f"{p}/8che_sample_{k}.cif", f"{p}/8che_summary_confidence_sample_{k}.json"
    for d in glob.glob(f"{FD}/*/*/seed_{seed}/predictions"):
        c = glob.glob(f"{d}/*_sample_{k}.cif")
        if c:
            base = os.path.basename(c[0])[: -len(f"_sample_{k}.cif")]
            return c[0], f"{d}/{base}_summary_confidence_sample_{k}.json"
    return None, None


# ---- inventory first: refuse to score an incomplete pool -------------------------------
missing = []
for s in range(2500):
    for k in range(NSAMP):
        if cif_and_conf(s, k)[0] is None:
            missing.append((s, k))
if missing:
    seeds = sorted({s for s, _ in missing})
    sys.exit(f"REFUSING: {len(missing)} poses absent across {len(seeds)} seeds, "
             f"e.g. {seeds[:10]} -- the free pool is not 2500x5")
print(f"inventory OK: 2500 seeds x {NSAMP} samples = {2500 * NSAMP} poses present")


def pose(a):
    """{iface: (iptm, dockq)} for one pose, or None."""
    s, k = a
    cif, conf = cif_and_conf(s, k)
    o = f"{CACHE}/{s}_{k}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset", "-c", "0-63", OST, "compare-structures", "-m", cif,
                        "-r", f"{GT}/8che-assembly1.cif", "-o", o, "--fault-tolerant",
                        "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq"],
                       env=ENV, capture_output=True, timeout=900)
    try:
        d = json.load(open(o))
        M = json.load(open(conf))["chain_pair_iptm"]
    except Exception:
        return None
    qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
    got = {}
    for lab, (c1, c2) in IFACES.items():
        for i, t in enumerate(ifs):
            if {t[0], t[1]} == {c1, c2} and i < len(qs):        # BY NAME
                x, y = ord(t[2]) - 65, ord(t[3]) - 65
                if 0 <= x < len(M) and 0 <= y < len(M):
                    got[lab] = (M[x][y], qs[i])
                break
    return got or None


tasks = [(s, k) for s in range(2500) for k in range(NSAMP)]
with ThreadPoolExecutor(A.workers) as ex:
    res = list(ex.map(pose, tasks))

bad = [t for t, r in zip(tasks, res) if r is None]
if bad:
    sys.exit(f"REFUSING: {len(bad)} poses failed to score, e.g. {bad[:10]}")

out = {"n_seeds": 2500, "n_poses": len(tasks)}
for lab in IFACES:
    per = {}
    for (s, k), r in zip(tasks, res):
        per.setdefault(s, []).append(r[lab])
    allp = [x for v in per.values() for x in v]
    sel = max(allp, key=lambda x: x[0])
    best = sorted((max(x[1] for x in v) for v in per.values()), reverse=True)
    out[lab] = {
        "best_dockq": round(max(x[1] for x in allp), 3),
        "n_acceptable": sum(1 for x in allp if x[1] >= 0.23),
        "n_above_06": sum(1 for x in allp if x[1] > 0.6),
        "n_above_085": sum(1 for x in allp if x[1] > 0.85),
        "sel_dockq": round(sel[1], 3), "sel_iptm": round(sel[0], 3),
        "seeds_above_06": sum(1 for b in best if b > 0.6),
        "median_seed_best": round(st.median(best), 3),
        "sd_seed_best": round(st.pstdev(best), 4),
    }
    v = out[lab]
    print(f"\n  {lab}: best {v['best_dockq']:.3f} | poses >0.6 {v['n_above_06']} "
          f">0.85 {v['n_above_085']} | acceptable {v['n_acceptable']}/{len(allp)}")
    print(f"    pooled ipTM selection: DockQ {v['sel_dockq']:.3f} (ipTM {v['sel_iptm']:.3f})")
    print(f"    seeds whose best >0.6: {v['seeds_above_06']}/2500  "
          f"median seed-best {v['median_seed_best']:.3f}")

json.dump(out, open(f"{EXP}/artifacts/free8che_complete_scores.json", "w"), indent=1)
print(f"\nWROTE {EXP}/artifacts/free8che_complete_scores.json")
