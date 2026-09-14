"""greedy_leader_fullcompare.py — pose-level greedy-leader dedup on the FULL pool (all 5120 poses) for the
15 Type-2 interfaces, ordered by trunk-ON vs trunk-OFF ipTM, side by side. Uses the full trunk-off run
(results/csv_files/trunkoff_full.csv). Reports the good pose's leader-rank under each ordering.

Test: does trunk-off (raise the good fluke's confidence) + dedup (strip redundant decoys) pull the good
dock into a short leader list, where trunk-on alone could not?
"""
import sys, csv
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from sklearn.preprocessing import StandardScaler

DEEP = Path.home()/"esm/foldbench_predictions/esmfold2fast_tts"
T2 = [('8ux6','A','F'),('7ua2','A','C'),('8vzo','A','C'),('8slb','A','C'),('8av2','A','C'),
      ('8w84','B','D'),('8oni','A','F'),('8oni','B','F'),('8oni','A','E'),('8uky','A','C'),
      ('8dcm','B','C'),('7zoz','A','B'),('8ivx','A','C'),('8q3j','A','B'),('8q3j','A','C')]
AGAB = {}
for r in csv.DictReader(open("results/csv_files/decoy_structural_analysis.csv")):
    if r.get('pred_ag_chain') and r.get('pred_ab_chain'):
        AGAB[(r['cx'], r['c1'], r['c2'])] = (r['pred_ag_chain'], r['pred_ab_chain'])
POSES = defaultdict(dict)  # (cx,c1,c2)->{(s,k):(dq, on, off)}
for r in csv.DictReader(open("results/csv_files/trunkoff_full.csv")):
    key = (r['cx'], r['c1'], r['c2'])
    try: POSES[key][(int(r['seed']), int(r['sample']))] = (float(r['dockq']), float(r['iptm_on']), float(r['iptm_off']))
    except: pass

def fast_reps(cif):
    with open(cif) as f: lines = f.readlines()
    cols = {}; i = 0
    for i, ln in enumerate(lines):
        if ln.startswith('_atom_site.'): cols[ln.strip()] = len(cols)
        elif ln.startswith('ATOM'): break
    C = lambda n: cols['_atom_site.'+n]
    ich, iseq, inm = C('auth_asym_id'), C('label_seq_id'), C('label_atom_id')
    ix, iy, iz = C('Cartn_x'), C('Cartn_y'), C('Cartn_z')
    res = {}; order = []
    for ln in lines[i:]:
        if not ln.startswith('ATOM'): continue
        p = ln.split(); k = (p[ich], p[iseq])
        if k not in res: res[k] = {}; order.append(k)
        res[k][p[inm]] = (float(p[ix]), float(p[iy]), float(p[iz]))
    out = defaultdict(list)
    for k in order:
        a = res[k]; xyz = a.get('CB', a.get('CA'))
        if xyz: out[k[0]].append(np.array(xyz, dtype=np.float32))
    return {c: np.stack(v) for c, v in out.items()}

def fw(args):
    s, k, cx, ag, ab, agi, abi = args
    cif = DEEP/cx/f"seed_{s}"/"predictions"/f"{cx}_sample_{k}.cif"
    if not cif.exists(): return ((s, k), None)
    try:
        R = fast_reps(cif)
        if ag not in R or ab not in R: return ((s, k), None)
        A = R[ag][agi]; B = R[ab][abi]; iu = np.triu_indices(len(A), 1)
        intra = np.linalg.norm(A[:, None, :]-A[None, :, :], axis=-1)[iu]
        cross = np.linalg.norm(A[:, None, :]-B[None, :, :], axis=-1).ravel()
        return ((s, k), np.concatenate([intra, cross]).astype(np.float32))
    except Exception: return ((s, k), None)

def leader_rank(F, order, dq, T):
    leaders = []
    for i in order:
        if leaders:
            d = np.linalg.norm(F[leaders]-F[i], axis=1)
            if d.min() < T: continue
        leaders.append(i)
    for rank, li in enumerate(leaders, 1):
        if dq[li] > 0.4: return rank, len(leaders)
    return None, len(leaders)

def process(key):
    cx, c1, c2 = key
    if key not in AGAB: return (key, None)
    ag, ab = AGAB[key]; poses = sorted(POSES[key].keys())
    ref = None
    for s, k in poses:
        f = DEEP/cx/f"seed_{s}"/"predictions"/f"{cx}_sample_{k}.cif"
        if f.exists(): ref = fast_reps(f); break
    if not ref or ag not in ref or ab not in ref: return (key, None)
    nag, nab = len(ref[ag]), len(ref[ab])
    agi = np.linspace(0, nag-1, min(60, nag)).astype(int); abi = np.linspace(0, nab-1, min(60, nab)).astype(int)
    with Pool(40) as p:
        res = dict(p.map(fw, [(s, k, cx, ag, ab, agi, abi) for s, k in poses]))
    rows = [(sk, res[sk]) for sk in poses if res.get(sk) is not None]
    if sum(1 for sk, _ in rows if POSES[key][sk][0] > 0.4) < 3: return (key, None)
    F = StandardScaler().fit_transform(np.stack([f for _, f in rows]))
    dq = np.array([POSES[key][sk][0] for sk, _ in rows])
    on = np.array([POSES[key][sk][1] for sk, _ in rows]); off = np.array([POSES[key][sk][2] for sk, _ in rows])
    o_on = list(np.argsort(-on)); o_off = list(np.argsort(-off))
    n = len(F); ri = np.random.RandomState(0).randint(0, n, (4000, 2))
    pdists = np.linalg.norm(F[ri[:, 0]]-F[ri[:, 1]], axis=1); pdists = pdists[pdists > 0]
    out = {}
    for pct in [10, 20]:
        T = np.percentile(pdists, pct)
        ron, nl = leader_rank(F, o_on, dq, T); roff, _ = leader_rank(F, o_off, dq, T)
        out[pct] = (ron, roff, nl)
    return (key, out)

if __name__ == "__main__":
    print(f"{'iface':>10} | {'p10: on->off (#lead)':>22} | {'p20: on->off (#lead)':>22}")
    for t in T2:
        key, out = process(tuple(t))
        if out is None: print(f"{t[0]+' '+t[1]+'-'+t[2]:>10} | (skip)", flush=True); continue
        def cell(p): r = out[p]; return f"{str(r[0]):>4} -> {str(r[1]):<4} ({r[2]})"
        print(f"{t[0]+' '+t[1]+'-'+t[2]:>10} | {cell(10):>22} | {cell(20):>22}", flush=True)
    print("cells = goodLeaderRank(trunk-ON) -> goodLeaderRank(trunk-OFF)  (#leaders)")
    print("DONE", flush=True)
