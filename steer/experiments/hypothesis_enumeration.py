"""hypothesis_enumeration.py — label-free selection on confident-decoy interfaces by enumerating
distinct confident hypotheses from the seed x seed DockQ matrix.

Reproduces the first real finding (2026-07-10). Reads the matrices built by matrix_iptm.py
(results/iptm_matrix/matrix_<cx>_<c1><c2>.npy + _index.csv). No ground truth is used to SELECT;
dockq_gt is read only to SCORE whether a pick is a good dock (>0.4).

Two methods:
  1. FIRST-DISTINCT: descend seeds by ipTM; H1 = most confident. H2 = first seed with
     DockQ(H1, seed) < TAU_DISTINCT (the top confident structure that disagrees with H1).
     On confident-decoy traps H2 is the good dock. -> 7/7 here, robust over TAU in [0.1,0.4].
  2. CLUSTER-CONFIDENCE: anchor two clusters on H1 and H2 (nearest reference, min-similarity floor
     TAU), and ask whether the good hypothesis H2's cluster is MORE confident (pooled ipTM) than
     the decoy cluster H1. median beats mean; mass HURTS (good cluster is usually smaller).
     -> H2 wins median on 4/7 (3/7 statistically significant); mass*median only 2-3/7.

Mann-Whitney U is implemented inline (no scipy dependency).
"""
import csv, math, sys
import numpy as np

MATDIR = "results/iptm_matrix"
T7 = [("8ts0","A","C"),("8ts0","B","C"),("8a44","A","D"),("7yxu","A","C"),
      ("8a44","A","C"),("7zoz","A","C"),("8vui","B","C")]
GOOD = 0.4              # DockQ above this = a good dock
TAU_DISTINCT = 0.23     # "genuinely a different interface" for finding H2

def load(cx, c1, c2):
    tag = f"{cx}_{c1}{c2}"
    D = np.load(f"{MATDIR}/matrix_{tag}.npy").astype(np.float64)
    idx = list(csv.DictReader(open(f"{MATDIR}/matrix_{tag}_index.csv")))
    ip = np.array([float(r['iptm']) for r in idx])
    gt = np.array([float(r['dockq_gt']) for r in idx])
    seeds = [r['seed'] for r in idx]
    return D, ip, gt, seeds

def find_H1_H2(D, ip, tau=TAU_DISTINCT):
    order = np.argsort(-ip); H1 = order[0]
    H2 = next((j for j in order[1:] if D[H1][j] < tau), None)
    return int(H1), (int(H2) if H2 is not None else None)

def mwu(a, b):
    """Two-sided Mann-Whitney U -> (p, Cliff's delta). delta>0 => a stochastically higher."""
    a = np.asarray(a, float); b = np.asarray(b, float); n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0: return float('nan'), float('nan')
    allv = np.concatenate([a, b]); order = allv.argsort(); vals = allv[order]
    sr = np.empty(len(allv)); i = 0; N = len(allv)
    while i < N:
        j = i
        while j + 1 < N and vals[j + 1] == vals[i]: j += 1
        sr[i:j + 1] = (i + 1 + j + 1) / 2.0; i = j + 1
    rank = np.empty(N); rank[order] = sr
    R1 = rank[:n1].sum(); U1 = R1 - n1 * (n1 + 1) / 2.0
    _, c = np.unique(allv, return_counts=True); tie = (c**3 - c).sum()
    denom = (n1 + n2) * (n1 + n2 - 1)
    sig = math.sqrt((n1 * n2 / 12.0) * ((n1 + n2 + 1) - (tie / denom if denom else 0)))
    if sig == 0: return float('nan'), float('nan')
    z = (U1 - n1 * n2 / 2.0) / sig
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return p, 2 * U1 / (n1 * n2) - 1

def method_first_distinct(taus=(0.1, 0.15, 0.23, 0.3, 0.4)):
    print("=== METHOD 1: first distinct hypothesis (pick H2) ===")
    for tau in taus:
        res = []
        for cx, c1, c2 in T7:
            D, ip, gt, _ = load(cx, c1, c2)
            _, H2 = find_H1_H2(D, ip, tau)
            res.append('G' if (H2 is not None and gt[H2] > GOOD) else '.')
        print(f"  TAU_distinct={tau:<4}: {res.count('G')}/7  " +
              " ".join(f"{'-'.join(t)}:{r}" for t, r in zip(T7, res)))

def method_cluster_confidence(tau=0.5):
    print(f"\n=== METHOD 2: cluster-confidence (H1 vs H2 clusters), floor TAU={tau} ===")
    print(f"  {'interface':10}{'H1n':>5}{'H1med':>7}{'H2n':>5}{'H2med':>7}{'H2good':>7}"
          f"  {'medWin':>7}{'Cliffδ':>8}{'MWUp':>9}  massXmedWin")
    tally = dict(median=0, median_sig=0, massmed=0)
    for cx, c1, c2 in T7:
        D, ip, gt, _ = load(cx, c1, c2)
        H1, H2 = find_H1_H2(D, ip)
        dH1, dH2 = D[H1], D[H2]
        inH1 = (dH1 >= tau) & (dH1 >= dH2); inH2 = (dH2 >= tau) & (dH2 > dH1)
        a, b = ip[inH1], ip[inH2]
        if len(a) == 0 or len(b) == 0:
            print(f"  {cx} {c1}-{c2}: empty cluster"); continue
        md1, md2 = np.median(a), np.median(b)
        p, cl = mwu(a, b)
        medwin = md2 > md1
        sig = medwin and (p < 0.05) and abs(cl) > 0.15
        mmwin = len(b) * md2 > len(a) * md1
        tally['median'] += medwin; tally['median_sig'] += sig; tally['massmed'] += mmwin
        print(f"  {cx} {c1}-{c2:2}{len(a):>5}{md1:>7.3f}{len(b):>5}{md2:>7.3f}{(gt[inH2]>GOOD).mean():>7.0%}"
              f"  {'H2' if medwin else 'H1':>7}{cl:>8.3f}{p:>9.1e}  {'H2' if mmwin else 'H1'}")
    print(f"  => H2 wins: median={tally['median']}/7 (significant={tally['median_sig']}/7)  "
          f"mass*median={tally['massmed']}/7")

if __name__ == "__main__":
    tau = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    method_first_distinct()
    method_cluster_confidence(tau)
