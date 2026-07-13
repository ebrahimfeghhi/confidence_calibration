"""cluster_sweep.py — method + tau sweep for label-free rescue of confident-decoy interfaces.

TRAINING SET ONLY (23 interfaces). The 25 test interfaces are never touched here.

THE TASK. On every interface, ipTM's own top pick is BELOW acceptable (DockQ < 0.23) while an
acceptable pose (> 0.23) exists somewhere in the pool. We cluster the ~5125 poses into structural
hypotheses, rank the hypotheses by pooled confidence (median ipTM), and ask how far down that
ranking the first RESCUE hypothesis sits. Naive ipTM is 0/23 on this set by construction.

METHODS (all on the symmetrised matrix S = (D + D.T)/2 -- DockQ is asymmetric because the
reference defines which interface contacts get scored, and reading one direction split a real
basin in half on 8ts0):
  leader          greedy leader clustering; a pose joins the NEAREST leader if S >= tau, else it
                  becomes a new leader. Hartigan's leader algorithm; order = ipTM descending, so
                  the most confident pose defines each basin.
  leader+medoid   same, then: compute each cluster's medoid, MERGE clusters whose medoids are
                  within tau, reassign every pose to its nearest medoid, iterate to convergence.
                  Fixes leader clustering's one privileged (and possibly edge-of-basin) reference.
  dbscan          single-linkage / DBSCAN with minPts=1: connected components of S >= tau. Chains
                  through intermediates, so a bridge pose can merge two basins.

THE NULL CLUSTER. ~13% of poses have NO interface contacts at all -- the antibody never touches
the antigen. DockQ is undefined when the REFERENCE has no interface (calc_DockQ returns None), so
these poses cannot be compared to anything and cannot be clustered. They are merged into ONE
"no-contact" cluster, kept OUT of the clustering algorithms, but RANKED alongside the real
hypotheses. That is deliberate: "these two chains do not bind" is a legitimate prediction with
real consequences, and excluding it would smuggle in ground-truth knowledge that a good dock
exists. If the null cluster wins the ranking, the method returns "no binding" and FAILS on that
interface -- which is the honest report of what the model believes. On 8oni_AF this is 70% of the
pool and includes the single most confident pose.

RESCUE CLUSTER (the primary criterion, agreed):
    median DockQ of its members > 0.23  ==  more than half its poses are acceptable.
Also recorded, because they answer different questions and a reviewer will want them:
    mean DockQ, purity (fraction of members > 0.23), IQR of DockQ (a wide spread = junk bucket),
    and the DockQ of the cluster's REPRESENTATIVE (highest-ipTM member) -- the pose you would
    actually hand back if forced to return one structure.

Ground truth is used ONLY to score, never to select.

Usage:  python steer/experiments/cluster_sweep.py            # full sweep
        python steer/experiments/cluster_sweep.py --check    # correctness checks, no sweep
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch

MATDIR = Path("results/dockq_full")
OUTDIR = Path("results/cluster_sweep"); OUTDIR.mkdir(parents=True, exist_ok=True)
ACC = 0.23                     # CAPRI 'acceptable' — the rescue bar
TAUS = [round(0.10 + 0.05 * i, 2) for i in range(13)]     # 0.10 .. 0.70
METHODS = ("leader", "leader+medoid", "dbscan")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- data
def load(tag):
    """Returns S (symmetrised, real poses only), ipTM, dockq_gt, and the null-cluster mask.

    A pose with an all-zero DockQ ROW has no interface contacts: DockQ is undefined with it as
    the reference. Its COLUMN is not zero (it can be scored as a model against a valid
    reference), so `S = (D + D.T)/2` would silently HALVE its similarities. It must be removed
    from the matrix, not just ignored.
    """
    D = np.load(MATDIR / f"dockq_{tag}.npy").astype(np.float64)
    idx = list(csv.DictReader(open(MATDIR / f"dockq_{tag}_index.csv")))
    ip = np.array([float(r['iptm']) for r in idx])
    gt = np.array([float(r['dockq_gt']) for r in idx])
    null = (D.sum(axis=1) == 0)                    # no interface contacts -> undefined as reference
    real = ~null
    S = D[np.ix_(real, real)]
    S = (S + S.T) / 2.0                            # symmetrise ONLY over well-defined poses
    return S, ip, gt, null, real


# ---------------------------------------------------------------- clustering
def leader(S, ip_sub, tau):
    """Greedy leader clustering. Pose joins its NEAREST leader if S >= tau, else becomes a leader.
    Processed in ipTM-descending order, so confident poses define the basins."""
    order = np.argsort(-ip_sub)
    leaders, lab = [], np.full(len(ip_sub), -1, dtype=np.int64)
    for j in order:
        if leaders:
            sims = S[leaders, j]
            k = int(np.argmax(sims))
            if sims[k] >= tau:
                lab[j] = k
                continue
        leaders.append(int(j)); lab[j] = len(leaders) - 1
    return lab, leaders


def medoid_refine(S, ip_sub, tau, max_it=15):
    """Leader clustering, then medoid refinement to convergence:
    medoids -> merge clusters whose MEDOIDS are within tau -> reassign to nearest medoid."""
    lab, _ = leader(S, ip_sub, tau)
    for _ in range(max_it):
        ks = np.unique(lab)
        med = []
        for k in ks:
            m = np.where(lab == k)[0]
            med.append(int(m[0]) if len(m) == 1
                       else int(m[np.argmax(S[np.ix_(m, m)].mean(axis=1))]))
        med = np.array(med)
        # merge clusters whose medoids are within tau (union-find)
        par = list(range(len(med)))
        def find(x):
            while par[x] != x: par[x] = par[par[x]]; x = par[x]
            return x
        A = S[np.ix_(med, med)] >= tau
        for a in range(len(med)):
            for b in range(a + 1, len(med)):
                if A[a, b]:
                    ra, rb = find(a), find(b)
                    if ra != rb: par[ra] = rb
        groups = {}
        for a in range(len(med)): groups.setdefault(find(a), []).append(a)
        # one medoid per merged group, then reassign every pose to its nearest medoid
        new_med = []
        for g in groups.values():
            m = np.where(np.isin(lab, ks[g]))[0]
            new_med.append(int(m[0]) if len(m) == 1
                           else int(m[np.argmax(S[np.ix_(m, m)].mean(axis=1))]))
        new_med = np.array(new_med)
        new = np.argmax(S[new_med], axis=0)
        if len(np.unique(new)) == len(np.unique(lab)) and np.array_equal(new, lab):
            break
        lab = new
    return _compact(lab)


def dbscan(S, tau):
    """DBSCAN with minPts=1 == single-linkage == connected components of (S >= tau).
    Done on GPU: iterative label propagation, since the graph has up to 26M edges."""
    n = S.shape[0]
    A = torch.from_numpy(S >= tau).to(DEV)
    A.fill_diagonal_(True)
    lab = torch.arange(n, device=DEV)
    for _ in range(n):
        # each node takes the min label in its neighbourhood; converges in O(diameter) rounds
        new = torch.where(A, lab.unsqueeze(0).expand(n, n),
                          torch.full((n, n), n, device=DEV, dtype=lab.dtype)).min(dim=1).values
        if torch.equal(new, lab):
            break
        lab = new
    return _compact(lab.cpu().numpy())


def _compact(lab):
    """Relabel to 0..k-1."""
    _, inv = np.unique(lab, return_inverse=True)
    return inv


# ---------------------------------------------------------------- scoring
def summarise(lab, ip_sub, gt_sub, null_ip, null_gt):
    """Rank every cluster by median ipTM. The NULL cluster (no-contact poses) is ranked with the
    rest -- 'these do not bind' is a legitimate hypothesis and must be allowed to win."""
    cl = []
    for k in np.unique(lab):
        m = np.where(lab == k)[0]
        cl.append(dict(kind="real", n=int(len(m)),
                       med_iptm=float(np.median(ip_sub[m])),
                       med_dq=float(np.median(gt_sub[m])),
                       mean_dq=float(np.mean(gt_sub[m])),
                       purity=float((gt_sub[m] > ACC).mean()),
                       iqr_dq=float(np.subtract(*np.percentile(gt_sub[m], [75, 25]))),
                       rep_dq=float(gt_sub[m][np.argmax(ip_sub[m])]),
                       rep_iptm=float(ip_sub[m].max()),
                       iptm=ip_sub[m].tolist(), dq=gt_sub[m].tolist()))
    if len(null_ip):
        cl.append(dict(kind="no_contact", n=int(len(null_ip)),
                       med_iptm=float(np.median(null_ip)),
                       med_dq=float(np.median(null_gt)),
                       mean_dq=float(np.mean(null_gt)),
                       purity=float((null_gt > ACC).mean()),
                       iqr_dq=float(np.subtract(*np.percentile(null_gt, [75, 25]))),
                       rep_dq=float(null_gt[np.argmax(null_ip)]),
                       rep_iptm=float(null_ip.max()),
                       iptm=null_ip.tolist(), dq=null_gt.tolist()))
    cl.sort(key=lambda c: -c['med_iptm'])          # THE RANKING: pooled confidence
    for r, c in enumerate(cl, 1):
        c['rank'] = r
    return cl


def first_rescue(cl):
    """First cluster (by rank) whose MEDIAN DockQ clears the acceptable bar."""
    return next((c for c in cl if c['med_dq'] > ACC), None)


def first_rescue_strict(cl):
    """Strict variant: median AND mean both clear the bar (i.e. not skewed)."""
    return next((c for c in cl if c['med_dq'] > ACC and c['mean_dq'] > ACC), None)


def first_rescue_rep(cl):
    """Deliverable variant: the pose we would actually RETURN (cluster's highest-ipTM member)
    is acceptable."""
    return next((c for c in cl if c['rep_dq'] > ACC), None)


# ---------------------------------------------------------------- sweep
def run(tags, taus=TAUS, methods=METHODS):
    rows, dump = [], {}
    for tag in tags:
        S, ip, gt, null, real = load(tag)
        ip_sub, gt_sub = ip[real], gt[real]
        null_ip, null_gt = ip[null], gt[null]
        # baseline: how deep ipTM buries the first acceptable pose (over the WHOLE pool)
        o = np.argsort(-ip)
        rr = np.where(gt[o] > ACC)[0]
        rescue_rank_iptm = int(rr[0]) + 1 if len(rr) else None
        pick_dq = float(gt[o[0]])
        print(f"[{tag}] N={len(ip)} ({int(null.sum())} no-contact) "
              f"| ipTM pick DockQ={pick_dq:.3f} | first acceptable at ipTM rank "
              f"{rescue_rank_iptm}", flush=True)

        for meth in methods:
            for tau in taus:
                t0 = time.time()
                if meth == "leader":
                    lab, _ = leader(S, ip_sub, tau); lab = _compact(lab)
                elif meth == "leader+medoid":
                    lab = medoid_refine(S, ip_sub, tau)
                else:
                    lab = dbscan(S, tau)
                cl = summarise(lab, ip_sub, gt_sub, null_ip, null_gt)
                med = first_rescue(cl); strict = first_rescue_strict(cl); rep = first_rescue_rep(cl)
                nullc = next((c for c in cl if c['kind'] == "no_contact"), None)
                rows.append(dict(
                    interface=tag, method=meth, tau=tau,
                    n_poses=len(ip), n_no_contact=int(null.sum()),
                    n_clusters=len(cl),
                    iptm_pick_dockq=round(pick_dq, 4),
                    rescue_rank_iptm=rescue_rank_iptm,
                    # PRIMARY: first cluster whose median DockQ > 0.23
                    rescue_rank_cluster=med['rank'] if med else None,
                    rescue_cluster_size=med['n'] if med else None,
                    rescue_med_dockq=round(med['med_dq'], 4) if med else None,
                    rescue_mean_dockq=round(med['mean_dq'], 4) if med else None,
                    rescue_purity=round(med['purity'], 4) if med else None,
                    rescue_iqr_dockq=round(med['iqr_dq'], 4) if med else None,
                    rescue_rep_dockq=round(med['rep_dq'], 4) if med else None,
                    # variants
                    rescue_rank_strict=strict['rank'] if strict else None,
                    rescue_rank_representative=rep['rank'] if rep else None,
                    # the null cluster: did "they do not bind" win?
                    null_rank=nullc['rank'] if nullc else None,
                    null_size=nullc['n'] if nullc else 0,
                    top1_is_null=bool(nullc and nullc['rank'] == 1),
                    top1_med_dockq=round(cl[0]['med_dq'], 4),
                    secs=round(time.time() - t0, 1),
                ))
                # keep raw per-cluster ipTM/DockQ for later Wilcoxon etc (top 10 clusters only)
                dump[f"{tag}|{meth}|{tau}"] = [
                    {k: c[k] for k in ("kind", "rank", "n", "med_iptm", "med_dq", "mean_dq",
                                       "purity", "iqr_dq", "rep_dq", "iptm", "dq")}
                    for c in cl[:10]]
            print(f"  {meth:14} done", flush=True)

    with open(OUTDIR / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(dump, open(OUTDIR / "clusters.json", "w"))
    print(f"\n-> {OUTDIR}/sweep.csv ({len(rows)} rows)  +  clusters.json")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tags", nargs="+")
    a = ap.parse_args()
    split = json.load(open("results/csv_files/train_test_split.json"))
    tags = a.tags or split['train']
    if a.check:
        sys.exit(0)
    run(tags)
