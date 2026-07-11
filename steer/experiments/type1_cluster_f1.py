"""type1_cluster_f1.py — for the 7 tight-margin interfaces: is there a CLEAR structural cluster of the
good docks? Cluster (Ward on distogram features, full pool), sweep K, and at each K find the cluster that
best matches the ground-truth good docks (DockQ>0.4) by F1 = 2*purity*coverage/(purity+coverage).
Report the best F1 achievable (+ the K, purity, coverage where it occurs). High F1 = clean good-dock cluster.
"""
import sys, csv
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster

DEEP = Path.home()/"esm/foldbench_predictions/esmfold2fast_tts"
T7 = [('8ts0','A','C'),('8ts0','B','C'),('8a44','A','D'),('7yxu','A','C'),('8a44','A','C'),
      ('7zoz','A','C'),('8vui','B','C')]
KS = [2,3,5,8,12,20,30,50,75,100,150,200,300]
AGAB = {}
for r in csv.DictReader(open("results/csv_files/decoy_structural_analysis.csv")):
    if r.get('pred_ag_chain') and r.get('pred_ab_chain'):
        AGAB[(r['cx'], r['c1'], r['c2'])] = (r['pred_ag_chain'], r['pred_ab_chain'])
POSES = defaultdict(dict); T7SET = {tuple(t) for t in T7}
for r in csv.DictReader(open("results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv")):
    key = (r['assembly_id'][:4], r['chain1'], r['chain2'])
    if key not in T7SET: continue
    try: POSES[key][(int(r['seed']), int(r['sample']))] = float(r['dockq'])
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
    dq = np.array([POSES[key][sk] for sk, _ in rows]); good = dq > 0.4; ngood = int(good.sum())
    if ngood < 3: return (key, None)
    F = PCA(n_components=min(30, len(rows)-1), random_state=0).fit_transform(StandardScaler().fit_transform(np.stack([f for _, f in rows])))
    L = linkage(F, method='ward')
    best = (0.0, None, None, None)
    for K in KS:
        lab = fcluster(L, K, criterion='maxclust')
        for c in np.unique(lab):
            m = lab == c; inter = int((m & good).sum())
            if inter == 0: continue
            P = inter/m.sum(); Cv = inter/ngood; f1 = 2*P*Cv/(P+Cv)
            if f1 > best[0]: best = (f1, K, P, Cv)
    return (key, (best, ngood))

if __name__ == "__main__":
    print(f"{'interface':>11} {'ngood':>5} | best {'F1':>5} {'@K':>4} {'purity':>7} {'coverage':>8}")
    for t in T7:
        key, out = process(tuple(t))
        if out is None: print(f"{t[0]+' '+t[1]+'-'+t[2]:>11}  (skip)"); continue
        (f1, K, P, C), ng = out
        print(f"{t[0]+' '+t[1]+'-'+t[2]:>11} {ng:>5} |     {f1:>5.2f} {K:>4} {P:>7.2f} {C:>8.2f}", flush=True)
    print("DONE", flush=True)
