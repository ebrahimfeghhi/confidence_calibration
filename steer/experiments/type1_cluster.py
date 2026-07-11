"""type1_cluster.py — for the 10 Type-1 confident-decoy interfaces, cluster the decoded structures in
distogram space and show whether the good docks form a clean cluster (viability of the cluster-median
ipTM fix).

Feature (answers the misfolded-antigen worry): per pose, the rep-atom (CB/CA) distogram over the
INTRA-ANTIGEN block (the fold) + the ANTIGEN x ANTIBODY cross-block (the dock), subsampled to ~60
residues/chain. A misfolded antigen has a different intra-antigen block -> it can't hide in a good
docking cluster. Distances only (no Kabsch). Ward + silhouette-selected k.

Per interface: a 2-panel figure (DockQ | trunk-on ipTM) with boxplots by cluster (median/IQR/max) + mean
diamond; clusters ordered by median DockQ, good cluster highlighted, n annotated. Plus a stats CSV.
"""
import sys, csv
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

DEEP = Path.home()/"esm/foldbench_predictions/esmfold2fast_tts"
FIGDIR = Path("results/figures/type1_clusters"); FIGDIR.mkdir(parents=True, exist_ok=True)
STATS = "results/csv_files/type1_cluster_stats.csv"
T1 = [('8a44','A','C'),('8a44','A','D'),('7yxu','A','C'),('8vui','B','C'),('8w84','B','C'),
      ('8aok','A','B'),('8oni','B','E'),('8ts0','A','C'),('8ts0','B','C'),('7zoz','A','C')]

# antigen/antibody chain per interface from curated metadata
AGAB = {}
for r in csv.DictReader(open("results/csv_files/decoy_structural_analysis.csv")):
    if r.get('pred_ag_chain') and r.get('pred_ab_chain'):
        AGAB[(r['cx'], r['c1'], r['c2'])] = (r['pred_ag_chain'], r['pred_ab_chain'])

# deep CSV: per interface -> {(seed,sample): (dockq, trunk-on pair_iptm)}
POSES = defaultdict(dict); T1SET = {tuple(t) for t in T1}
for r in csv.DictReader(open("results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv")):
    key = (r['assembly_id'][:4], r['chain1'], r['chain2'])
    if key not in T1SET: continue
    try: POSES[key][(int(r['seed']), int(r['sample']))] = (float(r['dockq']), float(r['pair_iptm']))
    except: pass

def reps_by_chain(cif):
    """{chain: [(resid, rep_xyz)]} using CB (CA fallback), in file order."""
    d = MMCIF2Dict(str(cif)); grp = d['_atom_site.group_PDB']
    ch = d['_atom_site.auth_asym_id']; seq = d['_atom_site.label_seq_id']; nm = d['_atom_site.label_atom_id']
    X = d['_atom_site.Cartn_x']; Y = d['_atom_site.Cartn_y']; Z = d['_atom_site.Cartn_z']
    res = {}   # (chain,seq) -> {atom: xyz}
    order = []
    for i in range(len(grp)):
        if grp[i] != 'ATOM': continue
        k = (ch[i], seq[i])
        if k not in res: res[k] = {}; order.append(k)
        res[k][nm[i]] = (float(X[i]), float(Y[i]), float(Z[i]))
    out = defaultdict(list)
    for (c, s) in order:
        a = res[(c, s)]
        xyz = a.get('CB', a.get('CA'))
        if xyz is not None: out[c].append(np.array(xyz, dtype=np.float32))
    return {c: np.stack(v) for c, v in out.items()}

_CTX = {}
def feat_worker(args):
    s, k, cx, ag, ab, agi, abi = args
    cif = DEEP/cx/f"seed_{s}"/"predictions"/f"{cx}_sample_{k}.cif"
    if not cif.exists(): return (s, k, None)
    try:
        R = reps_by_chain(cif)
        if ag not in R or ab not in R: return (s, k, None)
        A = R[ag][agi]; Bc = R[ab][abi]
        intra = np.linalg.norm(A[:, None, :] - A[None, :, :], axis=-1)
        iu = np.triu_indices(len(A), 1)
        cross = np.linalg.norm(A[:, None, :] - Bc[None, :, :], axis=-1)
        return (s, k, np.concatenate([intra[iu], cross.ravel()]).astype(np.float32))
    except Exception:
        return (s, k, None)

def pick(key):
    return sorted(POSES[key].keys())      # FULL POOL — every (seed,sample) for the interface

def process(key):
    cx, c1, c2 = key
    if key not in AGAB: return (key, "no ag/ab meta")
    ag, ab = AGAB[key]
    poses = pick(key)
    # subsample indices from a reference cif
    ref = None
    for s, k in poses:
        f = DEEP/cx/f"seed_{s}"/"predictions"/f"{cx}_sample_{k}.cif"
        if f.exists(): ref = reps_by_chain(f); break
    if not ref or ag not in ref or ab not in ref: return (key, "no ref/chains")
    nag, nab = len(ref[ag]), len(ref[ab])
    agi = np.linspace(0, nag-1, min(60, nag)).astype(int)
    abi = np.linspace(0, nab-1, min(60, nab)).astype(int)
    with Pool(20) as p:
        res = p.map(feat_worker, [(s, k, cx, ag, ab, agi, abi) for s, k in poses])
    rows = [(s, k, f) for s, k, f in res if f is not None]
    if len(rows) < 30: return (key, f"too few features ({len(rows)})")
    F = np.stack([f for _, _, f in rows]); dq = np.array([POSES[key][(s, k)][0] for s, k, _ in rows])
    ip = np.array([POSES[key][(s, k)][1] for s, k, _ in rows])
    Z = PCA(n_components=min(30, F.shape[0]-1, F.shape[1])).fit_transform(StandardScaler().fit_transform(F))
    best = None
    for kk in range(2, 9):
        if kk >= len(Z): break
        lab = AgglomerativeClustering(n_clusters=kk, linkage='ward').fit_predict(Z)
        try: sil = silhouette_score(Z, lab)
        except Exception: continue
        if best is None or sil > best[0]: best = (sil, kk, lab)
    if best is None: return (key, "no clustering")
    sil, kk, lab = best
    # good cluster = highest fraction DockQ>0.4
    fr = {c: (dq[lab == c] > 0.4).mean() for c in range(kk)}
    gc = max(fr, key=fr.get); ngood = int((dq > 0.4).sum())
    purity = fr[gc]; coverage = (dq[lab == gc] > 0.4).sum() / max(1, ngood)
    # order clusters by median DockQ
    order = sorted(range(kk), key=lambda c: np.median(dq[lab == c]))
    # ---- plot ----
    fig, (axd, axi) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, vals, ylab in [(axd, dq, "DockQ"), (axi, ip, "trunk-on pair_ipTM")]:
        data = [vals[lab == c] for c in order]
        bp = ax.boxplot(data, showfliers=True, widths=0.6, patch_artist=True)
        for pi, c in enumerate(order):
            col = "#2ca02c" if c == gc else "#c7c7c7"
            bp['boxes'][pi].set(facecolor=col, alpha=0.6)
            ax.scatter([pi+1]*np.sum(lab == c), vals[lab == c], s=6, color=col, alpha=0.25, zorder=1)
            ax.scatter([pi+1], [vals[lab == c].mean()], marker='D', color='black', s=30, zorder=3)
            ax.text(pi+1, ax.get_ylim()[0], f"n={np.sum(lab==c)}", ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(1, kk+1)); ax.set_xticklabels([f"C{c}" for c in order])
        ax.set_ylabel(ylab); ax.set_xlabel("cluster (ordered by median DockQ)")
    axd.axhline(0.4, ls='--', color='red', lw=0.8, alpha=0.6)
    fig.suptitle(f"{cx} {c1}-{c2}  |  k={kk} (sil={sil:.2f})  |  good cluster C{gc}: "
                 f"purity={purity:.2f} coverage={coverage:.2f}  (green = good cluster; diamond = mean)")
    fig.tight_layout()
    fig.savefig(FIGDIR/f"{cx}_{c1}{c2}.png", dpi=110); plt.close(fig)
    # ---- stats rows ----
    srows = []
    for c in order:
        d = dq[lab == c]; p = ip[lab == c]
        srows.append([cx, f"{c1}-{c2}", c, int((lab == c).sum()), "good" if c == gc else "",
                      f"{d.max():.3f}", f"{np.median(d):.3f}", f"{d.mean():.3f}", f"{d.std():.3f}",
                      f"{p.max():.3f}", f"{np.median(p):.3f}", f"{p.mean():.3f}", f"{p.std():.3f}"])
    return (key, dict(k=kk, sil=sil, gc=gc, purity=purity, coverage=coverage, n=len(rows), srows=srows))

if __name__ == "__main__":
    fs = open(STATS, "w", newline=""); WS = csv.writer(fs)
    WS.writerow(["cx","iface","cluster","n","tag","dq_max","dq_med","dq_mean","dq_std",
                 "ip_max","ip_med","ip_mean","ip_std"]); fs.flush()
    print(f"{'iface':>12} {'k':>2} {'sil':>5} {'purity':>6} {'cover':>6} {'n':>5}")
    for t in T1:
        key = tuple(t); res = process(key)[1]
        if isinstance(res, str):
            print(f"{t[0]+' '+t[1]+'-'+t[2]:>12}  SKIP: {res}", flush=True); continue
        for r in res['srows']: WS.writerow(r)
        fs.flush()
        print(f"{t[0]+' '+t[1]+'-'+t[2]:>12} {res['k']:>2} {res['sil']:>5.2f} "
              f"{res['purity']:>6.2f} {res['coverage']:>6.2f} {res['n']:>5}", flush=True)
    fs.close()
    print("DONE", flush=True)
