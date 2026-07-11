"""hypothesis_enumeration.py — label-free structure selection on confident-decoy interfaces.

ONE method, two stages. (Do not think of these as rival methods: stage 1 decides WHAT the
candidate answers are, stage 2 decides WHICH one to believe. They are measured by different
metrics — recall and precision — and both are needed.)

  STAGE 1 - ENUMERATE. Greedy non-maximum suppression on the ipTM ranking, with DockQ as the
    overlap measure. Descend seeds by ipTM; a seed becomes a new hypothesis H_k only if it is
    structurally distinct (DockQ < tau) from EVERY hypothesis already accepted. This compresses
    ~1025 seeds into a handful of competing structural hypotheses.
    Metric: RECALL@H — is a good dock (dockq_gt > 0.4) among the top-H hypotheses?

  STAGE 2 - SCORE. Assign every seed to its nearest anchor (highest DockQ, and >= tau), then
    score each hypothesis by POOLING ipTM over its cluster. Pooling is a variance-reduced
    confidence estimate: it asks "is this hypothesis CONSISTENTLY confident?" rather than
    "did one pose get a lucky high ipTM?".
    Metric: PRECISION@1 — is the top-SCORED hypothesis a good dock?

tau is the single parameter, used for BOTH the distinctness cutoff and the cluster-assignment
floor. No minimum-cluster rule (good docks can be genuinely rare — 6 seeds of 1025 on 8a44).

Reads the matrices from matrix_iptm.py (results/iptm_matrix/matrix_<cx>_<c1><c2>.npy +
_index.csv). dockq_gt is NEVER used to select — only to score whether a pick is good.

Mann-Whitney U is implemented inline (no scipy in the venv).

Usage:  python steer/experiments/hypothesis_enumeration.py [tau]
"""
import csv, math, sys
import numpy as np

MATDIR = "results/iptm_matrix"
T7 = [("8ts0", "A", "C"), ("8ts0", "B", "C"), ("8a44", "A", "D"), ("7yxu", "A", "C"),
      ("8a44", "A", "C"), ("7zoz", "A", "C"), ("8vui", "B", "C")]
GOOD = 0.4        # dockq_gt above this = a good dock
TAU = 0.23        # the single parameter: distinctness cutoff AND cluster-assignment floor
MAX_H = 5         # how many hypotheses to enumerate


def load(cx, c1, c2):
    tag = f"{cx}_{c1}{c2}"
    D = np.load(f"{MATDIR}/matrix_{tag}.npy").astype(np.float64)
    idx = list(csv.DictReader(open(f"{MATDIR}/matrix_{tag}_index.csv")))
    ip = np.array([float(r["iptm"]) for r in idx])
    gt = np.array([float(r["dockq_gt"]) for r in idx])
    return D, ip, gt


# ---------------------------------------------------------------- stage 1: enumerate
def enumerate_hypotheses(D, ip, tau=TAU, max_h=MAX_H):
    """Greedy NMS on the ipTM ranking. Returns hypothesis row-indices, most confident first.

    H_k is accepted only if DockQ(H_i, H_k) < tau for EVERY already-accepted H_i — i.e. it is a
    genuinely new structural mode, not a near-duplicate of anything we have already proposed.
    """
    hyps = []
    for j in np.argsort(-ip):                       # descend by confidence
        j = int(j)
        if all(D[h][j] < tau for h in hyps):        # distinct from ALL accepted hypotheses
            hyps.append(j)
            if len(hyps) == max_h:
                break
    return hyps


# ---------------------------------------------------------------- stage 2: assign + score
def assign(D, hyps, tau=TAU):
    """Assign each seed to its nearest anchor. Returns a list of member-index arrays.

    A seed joins the hypothesis it is most similar to, but only if that similarity clears the
    tau floor — seeds close to nothing stay unassigned (noise). Same tau as enumeration.
    """
    sim = np.stack([D[h] for h in hyps])            # (n_hyp, n_seed)
    best = sim.argmax(axis=0)
    ok = sim.max(axis=0) >= tau
    return [np.where(ok & (best == k))[0] for k in range(len(hyps))]


def score(ip, members, rule="median"):
    """Pooled confidence of a hypothesis = ipTM aggregated over its cluster."""
    if len(members) == 0:
        return -np.inf
    v = ip[members]
    if rule == "median":      return float(np.median(v))
    if rule == "mean":        return float(np.mean(v))
    if rule == "mass_median": return float(len(v) * np.median(v))
    if rule == "top":         return float(np.max(v))   # ~ naive per-pose ipTM (no pooling)
    raise ValueError(rule)


# ---------------------------------------------------------------- stats
def mwu(a, b):
    """Two-sided Mann-Whitney U -> (p, Cliff's delta). delta > 0 => a stochastically higher."""
    a = np.asarray(a, float); b = np.asarray(b, float); n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    allv = np.concatenate([a, b]); order = allv.argsort(); vals = allv[order]
    sr = np.empty(len(allv)); i = 0; N = len(allv)
    while i < N:
        j = i
        while j + 1 < N and vals[j + 1] == vals[i]:
            j += 1
        sr[i:j + 1] = (i + 1 + j + 1) / 2.0; i = j + 1
    rank = np.empty(N); rank[order] = sr
    R1 = rank[:n1].sum(); U1 = R1 - n1 * (n1 + 1) / 2.0
    _, c = np.unique(allv, return_counts=True); tie = (c ** 3 - c).sum()
    denom = (n1 + n2) * (n1 + n2 - 1)
    sig = math.sqrt((n1 * n2 / 12.0) * ((n1 + n2 + 1) - (tie / denom if denom else 0)))
    if sig == 0:
        return float("nan"), float("nan")
    z = (U1 - n1 * n2 / 2.0) / sig
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return p, 2 * U1 / (n1 * n2) - 1


# ---------------------------------------------------------------- reports
RULES = ["top", "mean", "median", "mass_median"]


def report(tau=TAU):
    print(f"\n{'='*100}\nSINGLE METHOD, ONE tau = {tau}  |  enumerate (NMS) -> assign -> pool ipTM\n{'='*100}")

    print("\n--- STAGE 1: ENUMERATION.  Metric = recall@H (is a good dock among the top-H "
          "hypotheses?) ---")
    print(f"  {'interface':12}{'#hypotheses':>12}{'ipTM-rank of each hypothesis':>32}"
          f"{'depth of first good dock':>26}")
    depths = []
    for cx, c1, c2 in T7:
        D, ip, gt = load(cx, c1, c2)
        hyps = enumerate_hypotheses(D, ip, tau)
        ranks = {int(r): k + 1 for k, r in enumerate(np.argsort(-ip))}   # row -> ipTM rank
        good = [k + 1 for k, h in enumerate(hyps) if gt[h] > GOOD]       # 1-based hypothesis depth
        d = good[0] if good else None
        depths.append(d)
        print(f"  {cx} {c1}-{c2:<6}{len(hyps):>12}{str([ranks[h] for h in hyps]):>32}"
              f"{(str(d) if d else 'NOT FOUND'):>26}")
    print()
    for H in (1, 2, 3):
        hit = sum(1 for d in depths if d is not None and d <= H)
        print(f"  recall@H={H}: {hit}/7 interfaces have a good dock in the top-{H} hypotheses")

    print("\n--- STAGE 2: CONFIDENCE.  Metric = precision@1 (is the top-SCORED hypothesis good?) ---")
    print("  Pooling rules: 'top' = max ipTM in cluster (~no pooling, the naive baseline);")
    print("                 'mean'/'median' = pooled cluster confidence; 'mass_median' = size x median")
    print(f"\n  {'interface':12}" + "".join(f"{r:>14}" for r in RULES))
    tally = {r: 0 for r in RULES}
    for cx, c1, c2 in T7:
        D, ip, gt = load(cx, c1, c2)
        hyps = enumerate_hypotheses(D, ip, tau)
        mem = assign(D, hyps, tau)
        row = f"  {cx} {c1}-{c2:<6}"
        for r in RULES:
            s = [score(ip, m, r) for m in mem]
            pick = hyps[int(np.argmax(s))]
            ok = gt[pick] > GOOD
            tally[r] += ok
            row += f"{('GOOD ' if ok else 'decoy') + f' {gt[pick]:.2f}':>14}"
        print(row)
    print(f"\n  {'precision@1':12}" + "".join(f"{str(tally[r]) + '/7':>14}" for r in RULES))

    print("\n--- Is the good hypothesis' pooled confidence SIGNIFICANTLY higher than the decoy's? ---")
    print(f"  {'interface':12}{'good hyp size':>15}{'good median ipTM':>18}"
          f"{'decoy hyp size':>16}{'decoy median ipTM':>19}{'Cliff delta':>13}{'MWU p':>10}")
    nsig = 0
    for cx, c1, c2 in T7:
        D, ip, gt = load(cx, c1, c2)
        hyps = enumerate_hypotheses(D, ip, tau)
        mem = assign(D, hyps, tau)
        gi = [k for k, h in enumerate(hyps) if gt[h] > GOOD]
        di = [k for k, h in enumerate(hyps) if gt[h] <= GOOD]
        if not gi or not di:
            print(f"  {cx} {c1}-{c2:<6}  (no good/decoy pair among hypotheses)"); continue
        g, d = mem[gi[0]], mem[di[0]]                 # top good hyp vs top decoy hyp
        if len(g) == 0 or len(d) == 0:
            print(f"  {cx} {c1}-{c2:<6}  (empty cluster)"); continue
        p, delta = mwu(ip[g], ip[d])
        sig = (p < 0.05) and (np.median(ip[g]) > np.median(ip[d]))
        nsig += sig
        print(f"  {cx} {c1}-{c2:<6}{len(g):>15}{np.median(ip[g]):>18.3f}"
              f"{len(d):>16}{np.median(ip[d]):>19.3f}{delta:>13.3f}{p:>10.1e}"
              f"{'  <-- good sig. more confident' if sig else ''}")
    print(f"\n  good hypothesis significantly more confident than decoy on {nsig}/7 interfaces")


def tau_sweep():
    print(f"\n{'='*100}\nTAU SENSITIVITY (the single parameter)\n{'='*100}")
    print(f"  {'tau':>6}{'#hyps (mean)':>14}{'recall@1':>11}{'recall@2':>11}{'recall@3':>11}"
          f"{'precision@1 (median pooling)':>30}")
    for tau in (0.10, 0.15, 0.23, 0.30, 0.40, 0.50, 0.60):
        depths, nh, prec = [], [], 0
        for cx, c1, c2 in T7:
            D, ip, gt = load(cx, c1, c2)
            hyps = enumerate_hypotheses(D, ip, tau)
            nh.append(len(hyps))
            good = [k + 1 for k, h in enumerate(hyps) if gt[h] > GOOD]
            depths.append(good[0] if good else None)
            mem = assign(D, hyps, tau)
            s = [score(ip, m, "median") for m in mem]
            prec += gt[hyps[int(np.argmax(s))]] > GOOD
        r = lambda H: sum(1 for d in depths if d is not None and d <= H)
        print(f"  {tau:>6.2f}{np.mean(nh):>14.1f}{str(r(1)) + '/7':>11}{str(r(2)) + '/7':>11}"
              f"{str(r(3)) + '/7':>11}{str(prec) + '/7':>30}")


if __name__ == "__main__":
    tau = float(sys.argv[1]) if len(sys.argv) > 1 else TAU
    report(tau)
    tau_sweep()
