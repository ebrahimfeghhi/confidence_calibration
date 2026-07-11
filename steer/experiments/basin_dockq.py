"""basin_dockq.py — define the confident-decoy basin by DockQ, per the 7 tight-margin interfaces.
Reference = the top-ipTM pose (the confident decoy that wins selection). Score DockQ(pose, top-ipTM) for
all good docks + a decoy sample via OST. Report: are the good docks INSIDE the confident-decoy basin
(high DockQ to it = docks the same way) or OUTSIDE (low = docks differently)?
"""
import sys, json, csv, os
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "eval"); sys.path.insert(0, ".")
import eval_ost_foldbench as E

DEEP = Path.home()/"esm/foldbench_predictions/esmfold2fast_tts"
TMP = Path("/tmp/claude-1004/-home-ebrahim-tts-abag/ea5db07e-452e-45f8-a6d6-98121d6d1264/scratchpad/basin_ost")
TMP.mkdir(parents=True, exist_ok=True)
T7 = [('8ts0','A','C'),('8ts0','B','C'),('8a44','A','D'),('7yxu','A','C'),('8a44','A','C'),
      ('7zoz','A','C'),('8vui','B','C')]
POSES = defaultdict(list); T7SET = {tuple(t) for t in T7}
for r in csv.DictReader(open("results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv")):
    key = (r['assembly_id'][:4], r['chain1'], r['chain2'])
    if key not in T7SET: continue
    try: POSES[key].append((int(r['seed']), int(r['sample']), float(r['dockq']), float(r['pair_iptm'])))
    except: pass

def worker(args):
    cx, c1, c2, s, k, rs, rk = args
    pose = DEEP/cx/f"seed_{s}"/"predictions"/f"{cx}_sample_{k}.cif"
    ref = DEEP/cx/f"seed_{rs}"/"predictions"/f"{cx}_sample_{rk}.cif"
    if not pose.exists(): return ((s, k), None)
    out = TMP/f"{cx}_{s}_{k}_{os.getpid()}.json"
    try:
        E.run_ost(pose, ref, out)
        if not out.exists(): return ((s, k), None)
        data = json.load(open(out)); dq, st = E.extract_dockq(data, c1, c2, data.get('chain_mapping', {}))
        return ((s, k), dq if st == 'ok' else None)
    except Exception:
        return ((s, k), None)
    finally:
        try: out.unlink()
        except Exception: pass

def process(key):
    cx, c1, c2 = key; a = POSES[key]
    top = max(a, key=lambda x: x[3]); rs, rk = top[0], top[1]
    good = [(s, k) for s, k, dq, ip in a if dq > 0.4][:800]
    dec = [(s, k) for s, k, dq, ip in a if dq < 0.1]
    decs = dec[::max(1, len(dec)//1000)][:1000]
    poses = list(set(good) | set(decs))
    with Pool(40) as p:
        res = dict(p.map(worker, [(cx, c1, c2, s, k, rs, rk) for s, k in poses]))
    gv = np.array([res[sk] for sk in good if res.get(sk) is not None])
    dv = np.array([res[sk] for sk in decs if res.get(sk) is not None])
    return (key, (top[2], gv, dv))

if __name__ == "__main__":
    print(f"{'interface':>11} {'decoyGTdq':>9} | good->decoy: {'med':>5} {'%inside(>.23)':>13} | decoys->decoy: {'%basin(>.5)':>11}")
    for t in T7:
        key, out = process(tuple(t))
        if out is None or out[1] is None or len(out[1]) == 0: print(f"{t[0]+' '+t[1]+'-'+t[2]:>11}  (skip)"); continue
        gtdq, gv, dv = out
        gmed = np.median(gv); ginside = (gv > 0.23).mean()*100
        dbasin = (dv > 0.5).mean()*100 if len(dv) else float('nan')
        print(f"{t[0]+' '+t[1]+'-'+t[2]:>11} {gtdq:>9.2f} |             {gmed:>5.2f} {ginside:>12.0f}% |               {dbasin:>10.0f}%", flush=True)
    print("DONE  (good->decoy med LOW + %inside LOW = good docks are OUTSIDE the confident-decoy basin)", flush=True)
