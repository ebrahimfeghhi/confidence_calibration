"""basin_dockq_full.py — score DockQ of ALL 5120 poses vs the top-ipTM (confident-decoy) reference, per
tight-margin interface. Saves per-pose (dockq_to_GT, dockq_to_topIptm, iptm) so we can FORM the confident-
decoy cluster (dockq_to_topIptm > thresh) and compare it against the good-DockQ cluster (dockq_to_GT>0.4).
"""
import sys, json, csv, os
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
sys.path.insert(0, "eval"); sys.path.insert(0, ".")
import eval_ost_foldbench as E

DEEP = Path.home()/"esm/foldbench_predictions/esmfold2fast_tts"
TMP = Path("/tmp/claude-1004/-home-ebrahim-tts-abag/ea5db07e-452e-45f8-a6d6-98121d6d1264/scratchpad/basin_ost")
TMP.mkdir(parents=True, exist_ok=True)
OUT = "results/csv_files/basin_dockq_full.csv"
T7 = [('8ts0','A','C'),('8ts0','B','C'),('8a44','A','D'),('7yxu','A','C'),('8a44','A','C'),
      ('7zoz','A','C'),('8vui','B','C')]
POSES = defaultdict(dict); T7SET = {tuple(t) for t in T7}
for r in csv.DictReader(open("results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv")):
    key = (r['assembly_id'][:4], r['chain1'], r['chain2'])
    if key not in T7SET: continue
    try: POSES[key][(int(r['seed']), int(r['sample']))] = (float(r['dockq']), float(r['pair_iptm']))
    except: pass

def worker(args):
    cx, c1, c2, s, k, rs, rk = args
    pose = DEEP/cx/f"seed_{s}"/"predictions"/f"{cx}_sample_{k}.cif"
    ref = DEEP/cx/f"seed_{rs}"/"predictions"/f"{cx}_sample_{rk}.cif"
    if not pose.exists(): return ((s, k), None)
    out = TMP/f"{cx}_{c1}{c2}_{s}_{k}_{os.getpid()}.json"
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

if __name__ == "__main__":
    fout = open(OUT, "w", newline=""); W = csv.writer(fout)
    W.writerow(["cx", "c1", "c2", "seed", "sample", "dockq_gt", "dockq_toref", "iptm"]); fout.flush()
    for t in T7:
        key = tuple(t); cx, c1, c2 = key
        a = POSES[key]; rs, rk = max(a, key=lambda sk: a[sk][1])   # top-ipTM (seed,sample)
        poses = sorted(a.keys())
        print(f"{cx} {c1}-{c2}: top-ipTM ref seed {rs} samp {rk} (its GTdq {a[(rs,rk)][0]:.2f}); scoring {len(poses)} poses...", flush=True)
        with Pool(50) as p:
            res = dict(p.map(worker, [(cx, c1, c2, s, k, rs, rk) for s, k in poses]))
        n = 0
        for (s, k) in poses:
            tr = res.get((s, k))
            if tr is None: continue
            gt, ip = a[(s, k)]
            W.writerow([cx, c1, c2, s, k, f"{gt:.4f}", f"{tr:.4f}", f"{ip:.4f}"]); n += 1
        fout.flush(); print(f"  {cx} {c1}-{c2}: wrote {n} poses", flush=True)
    fout.close(); print("DONE", flush=True)
