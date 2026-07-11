"""greedy_f1.py — best good-dock cluster F1 from the GREEDY-LEADER clustering (pose-level, full pool,
ipTM-ordered), for the 7 tight-margin interfaces — to compare against the Ward best-F1. Same F1 metric.
"""
import sys, csv
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from sklearn.preprocessing import StandardScaler

DEEP = Path.home()/"esm/foldbench_predictions/esmfold2fast_tts"
T7 = [('8ts0','A','C'),('8ts0','B','C'),('8a44','A','D'),('7yxu','A','C'),('8a44','A','C'),
      ('7zoz','A','C'),('8vui','B','C')]
KS_pct = [2,5,10,20,35]
AGAB = {}
for r in csv.DictReader(open("results/csv_files/decoy_structural_analysis.csv")):
    if r.get('pred_ag_chain') and r.get('pred_ab_chain'):
        AGAB[(r['cx'], r['c1'], r['c2'])] = (r['pred_ag_chain'], r['pred_ab_chain'])
POSES = defaultdict(dict); T7SET = {tuple(t) for t in T7}
for r in csv.DictReader(open("results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv")):
    key = (r['assembly_id'][:4], r['chain1'], r['chain2'])
    if key not in T7SET: continue
    try: POSES[key][(int(r['seed']), int(r['sample']))] = (float(r['dockq']), float(r['pair_iptm']))
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

def leader_assign(F, order, T):
    leaders = []; assign = np.full(len(F), -1)
    for i in order:
        if leaders:
            d = np.linalg.norm(F[leaders]-F[i], axis=1); j = int(d.argmin())
            if d[j] < T: assign[i] = j; continue
        assign[i] = len(leaders); leaders.append(i)
    return assign

def bestf1(lab, good, ngood):
    best = 0.0; bp = bc = 0
    for c in np.unique(lab):
        m = lab == c; inter = int((m & good).sum())
        if inter == 0: continue
        P = inter/m.sum(); Cv = inter/ngood; f1 = 2*P*Cv/(P+Cv)
        if f1 > best: best = f1; bp, bc = P, Cv
    return best, bp, bc

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
    dq = np.array([POSES[key][sk][0] for sk, _ in rows]); ip = np.array([POSES[key][sk][1] for sk, _ in rows])
    good = dq > 0.4; ngood = int(good.sum())
    if ngood < 3: return (key, None)
    F = StandardScaler().fit_transform(np.stack([f for _, f in rows]))
    order = list(np.argsort(-ip))
    n = len(F); ri = np.random.RandomState(0).randint(0, n, (4000, 2))
    pdist = np.linalg.norm(F[ri[:, 0]]-F[ri[:, 1]], axis=1); pdist = pdist[pdist > 0]
    best = (0.0, 0, 0)
    for pct in KS_pct:
        lab = leader_assign(F, order, np.percentile(pdist, pct))
        f1, P, C = bestf1(lab, good, ngood)
        if f1 > best[0]: best = (f1, P, C)
    return (key, (best, ngood))

if __name__ == "__main__":
    print(f"{'interface':>11} {'ngood':>5} | greedy best {'F1':>5} {'purity':>7} {'coverage':>8}")
    for t in T7:
        key, out = process(tuple(t))
        if out is None: print(f"{t[0]+' '+t[1]+'-'+t[2]:>11}  (skip)"); continue
        (f1, P, C), ng = out
        print(f"{t[0]+' '+t[1]+'-'+t[2]:>11} {ng:>5} |        {f1:>5.2f} {P:>7.2f} {C:>8.2f}", flush=True)
    print("DONE", flush=True)
