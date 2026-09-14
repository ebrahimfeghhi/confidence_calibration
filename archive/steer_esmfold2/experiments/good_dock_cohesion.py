"""good_dock_cohesion.py — for the 7 tight-margin interfaces: are all the good docks (DockQ>0.4) one tight
structural group, or several distinct sub-poses? Parse ONLY the good docks, cluster them (Ward), sweep
k=1..6, pick k by best silhouette (k=1 if no split is clean). Report #sub-groups, largest-sub-group frac,
silhouette. Low silhouette / one dominant group -> all similar; clean multi-cluster -> structurally diverse.
"""
import sys, csv
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

DEEP = Path.home()/"esm/foldbench_predictions/esmfold2fast_tts"
T7 = [('8ts0','A','C'),('8ts0','B','C'),('8a44','A','D'),('7yxu','A','C'),('8a44','A','C'),
      ('7zoz','A','C'),('8vui','B','C')]
AGAB = {}
for r in csv.DictReader(open("results/csv_files/decoy_structural_analysis.csv")):
    if r.get('pred_ag_chain') and r.get('pred_ab_chain'):
        AGAB[(r['cx'], r['c1'], r['c2'])] = (r['pred_ag_chain'], r['pred_ab_chain'])
GOOD = defaultdict(list)
for r in csv.DictReader(open("results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv")):
    key = (r['assembly_id'][:4], r['chain1'], r['chain2'])
    if key not in {tuple(t) for t in T7}: continue
    try:
        if float(r['dockq']) > 0.4: GOOD[key].append((int(r['seed']), int(r['sample'])))
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
    if not cif.exists(): return None
    try:
        R = fast_reps(cif)
        if ag not in R or ab not in R: return None
        A = R[ag][agi]; B = R[ab][abi]; iu = np.triu_indices(len(A), 1)
        intra = np.linalg.norm(A[:, None, :]-A[None, :, :], axis=-1)[iu]
        cross = np.linalg.norm(A[:, None, :]-B[None, :, :], axis=-1).ravel()
        return np.concatenate([intra, cross]).astype(np.float32)
    except Exception: return None

def process(key):
    cx, c1, c2 = key
    if key not in AGAB: return (key, None)
    ag, ab = AGAB[key]; poses = GOOD[key]
    ref = None
    for s, k in poses:
        f = DEEP/cx/f"seed_{s}"/"predictions"/f"{cx}_sample_{k}.cif"
        if f.exists(): ref = fast_reps(f); break
    if not ref: return (key, None)
    nag, nab = len(ref[ag]), len(ref[ab])
    agi = np.linspace(0, nag-1, min(60, nag)).astype(int); abi = np.linspace(0, nab-1, min(60, nab)).astype(int)
    with Pool(40) as p:
        feats = p.map(fw, [(s, k, cx, ag, ab, agi, abi) for s, k in poses])
    F = np.stack([f for f in feats if f is not None])
    if len(F) < 6: return (key, None)
    F = StandardScaler().fit_transform(F)
    best = (1, -1, 1.0)  # k, silhouette, largest-frac
    for k in range(2, min(7, len(F))):
        lab = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(F)
        try: sil = silhouette_score(F, lab)
        except Exception: continue
        frac = max(np.bincount(lab))/len(lab)
        if sil > best[1]: best = (k, sil, frac)
    return (key, (len(F), best))

if __name__ == "__main__":
    print(f"{'interface':>11} {'nGood':>5} | best split: {'k':>2} {'silhouette':>10} {'largestFrac':>11}")
    for t in T7:
        key, out = process(tuple(t))
        if out is None: print(f"{t[0]+' '+t[1]+'-'+t[2]:>11}  (skip)"); continue
        n, (k, sil, frac) = out
        verdict = "ONE group" if sil < 0.4 else f"{k} sub-groups"
        print(f"{t[0]+' '+t[1]+'-'+t[2]:>11} {n:>5} |            {k:>2} {sil:>10.2f} {frac:>11.2f}   {verdict}", flush=True)
    print("DONE (silhouette<0.4 = no clean split = all similar)", flush=True)
