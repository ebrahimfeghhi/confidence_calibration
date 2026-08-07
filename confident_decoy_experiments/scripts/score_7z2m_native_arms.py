#!/usr/bin/env python
"""Score 7z2m's corrected interfaces (A-K, B-K) for the three NATIVE-template arms, ASAP.

Those three arms (native50 / abonly / agonly) already have complete 250-pose pools at seeds 101-150,
so they can be answered now without waiting for the criteria arms to finish generating.

PROTOCOL (tts_abag/eval/eval_ost_foldbench.py -- not improvised): ONE ost compare-structures call per
pose, FULL native assembly vs FULL model, --fault-tolerant --min-pep-length 4 --min-nuc-length 4
--dockq, OST's own global chain mapping; then extract the labelled pair BY NAME from dockq_interfaces.
Both A-K and B-K are read from that single call, so they share one mapping and describe the same
predicted Fab. A pose whose labelled pair is absent from the mapping is skipped, never substituted.

Writes to its own file so it cannot race the full scorer's sweep_scores.json.
Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/score_7z2m_native_arms.py
Writes: artifacts/7z2m_native_arms.json
"""
import json, os, subprocess, sys, glob
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import OST, ENV, EXP, GT, SEEDS, SAMPLES, THRESH

PRED = f"{EXP}/predictions"
CACHE = f"{EXP}/artifacts/sweep_ost"; os.makedirs(CACHE, exist_ok=True)
OUTJ = f"{EXP}/artifacts/7z2m_native_arms.json"
NWORK = 8
PAIRS = [("A", "K"), ("B", "K")]
ARMS = {"native50": f"{PRED}/native50_out/7z2m_nativetmpl",
        "abonly":   f"{PRED}/abonly_out/7z2m_abonly",
        "agonly":   f"{PRED}/agonly_out/7z2m_nativetmpl"}


def one(a):
    """All labelled pairs for one pose, from a single protocol-conformant OST call."""
    arm, d, name, s, k = a
    cif = f"{d}/seed_{s}/predictions/{name}_sample_{k}.cif"
    if not os.path.exists(cif):
        return None
    o = f"{CACHE}/{arm}_7z2m_pose_{s}_{k}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset", "-c", "0-31", OST, "compare-structures", "-m", cif,
                        "-r", f"{GT}/7z2m-assembly1.cif", "-o", o,
                        "--fault-tolerant", "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq"],
                       env=ENV, capture_output=True, timeout=900)
    try:
        dd = json.load(open(o))
    except Exception:
        return None
    qs, ifs = dd.get("dockq") or [], dd.get("dockq_interfaces") or []
    try:
        M = json.load(open(f"{d}/seed_{s}/predictions/"
                          f"{name}_summary_confidence_sample_{k}.json"))["chain_pair_iptm"]
    except Exception:
        return None
    out = {}
    for c1, c2 in PAIRS:
        for i, t in enumerate(ifs):
            if {t[0], t[1]} == {c1, c2} and i < len(qs):
                x, y = ord(t[2]) - 65, ord(t[3]) - 65
                pi = M[x][y] if (0 <= x < len(M) and 0 <= y < len(M)) else -1
                out[f"7z2m_{c1}{c2}"] = (pi, qs[i])
                break
    return out or None


res = {}
for arm, d in ARMS.items():
    name = os.path.basename(d)
    jobs = [(arm, d, name, s, k) for s in SEEDS for k in SAMPLES]
    with ThreadPoolExecutor(NWORK) as ex:
        got = [r for r in ex.map(one, jobs) if r]
    res[arm] = {}
    for label in ("7z2m_AK", "7z2m_BK"):
        ps = [g[label] for g in got if label in g]
        if not ps:
            print(f"{arm:9} {label}: NO POSES"); continue
        sel = max(ps, key=lambda x: x[0])
        ok = [p for p in ps if p[1] >= THRESH]
        bad = [p for p in ps if p[1] < THRESH]
        res[arm][label] = {"n": len(ps), "n_ok": len(ok), "top_pair_iptm": round(sel[0], 3),
                           "sel_dockq": round(sel[1], 3), "best_dockq": round(max(p[1] for p in ps), 3),
                           "rescued": sel[1] >= THRESH,
                           "max_iptm_good": round(max([p[0] for p in ok], default=-1), 3),
                           "max_iptm_bad": round(max([p[0] for p in bad], default=-1), 3)}
        v = res[arm][label]
        print(f"{arm:9} {label} n={v['n']:3d} n_ok={v['n_ok']:3d} top_ipTM={v['top_pair_iptm']:.3f} "
              f"sel_DockQ={v['sel_dockq']:.3f} best={v['best_dockq']:.3f} "
              f"{'RESCUED' if v['rescued'] else 'still decoy'}", flush=True)
json.dump(res, open(OUTJ, "w"), indent=1)
print(f"\nWROTE {OUTJ}")
