"""cluster_sweep.py — method x tau sweep for label-free rescue of confident-decoy interfaces.

TRAINING SET ONLY (22 interfaces). The 25 test interfaces are never touched.

THE TASK. On every interface ipTM's own top pick is BELOW acceptable (DockQ < 0.23) while an
acceptable pose (> 0.23) exists in the pool. We cluster the ~5125 poses into structural
hypotheses, rank the hypotheses by pooled confidence, and ask how far down that ranking the first
RESCUE hypothesis sits. Naive ipTM is 0/22 on this set by construction.

METHODS (all on the symmetrised matrix S = (D + D.T)/2 -- DockQ is asymmetric because the
REFERENCE defines which interface contacts get scored):
  leader          greedy leader clustering. A pose joins its NEAREST leader if S >= tau, else it
                  becomes a new leader. Hartigan's leader algorithm; poses are processed in
                  ipTM-descending order so the most confident pose defines each basin. The pass is
                  inherently SEQUENTIAL (each pose depends on the leaders found before it), so it
                  runs on CPU; everything else is on GPU.
  leader+medoid   then: compute each cluster's medoid, MERGE clusters whose medoids are within
                  tau, reassign every pose to its nearest medoid, iterate to convergence. Fixes
                  leader clustering's one privileged (and possibly edge-of-basin) reference.
  dbscan          DBSCAN with minPts=1 == single-linkage == connected components of (S >= tau).
                  Chains through intermediates, so one bridge pose can merge two basins.

THE NULL CLUSTER. Some poses have NO interface contacts. DockQ is undefined when the REFERENCE
has none (calc_DockQ returns None), so they cannot be compared to anything and cannot be
clustered. They are merged into ONE "no-contact" cluster, kept OUT of the clustering algorithms
but RANKED alongside the real hypotheses. Deliberate: "these two chains do not bind" is a
legitimate prediction with real consequences, and excluding it would smuggle in ground-truth
knowledge that a good dock exists. If the null cluster wins, the method returns "no binding" and
FAILS on that interface -- the honest report of what the model believes.

SCORING RULES (the clustering does not depend on these, only the RANKING does -- so all five are
computed from one clustering, for free):
  median | mean | max (~ naive ipTM, the baseline) | size x median | size x mean

RESCUE CLUSTER (primary): median DockQ of its members > 0.23, i.e. more than half its poses are
acceptable. Also recorded: mean, purity (fraction > 0.23), IQR (a wide spread = junk bucket), and
the DockQ of the REPRESENTATIVE (highest-ipTM member) -- the pose we would actually hand back.

Ground truth is used ONLY to score, never to select.

Usage:  python steer/experiments/cluster_sweep.py                  # full sweep (train only)
        python steer/experiments/cluster_sweep.py --tags 8ts0_AC   # one interface
"""
import argparse, csv, json, resource, sys, time
from pathlib import Path
import numpy as np
import torch

MATDIR = Path("results/dockq_full")
OUTDIR = Path("results/cluster_sweep"); OUTDIR.mkdir(parents=True, exist_ok=True)
ACC = 0.23
TAUS = [round(0.10 + 0.05 * i, 2) for i in range(13)]        # 0.10 .. 0.70
METHODS = ("leader", "leader+medoid", "dbscan")
RULES = ("median", "mean", "max", "size_x_median", "size_x_mean")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- data
def load(tag):
    """S (symmetrised, REAL poses only), ipTM, dockq_gt, null mask, real mask.

    A pose with an all-zero DockQ ROW has no interface contacts -- undefined as a reference. Its
    COLUMN is NOT zero (it can be scored as a model against a valid reference), so (D + D.T)/2
    would silently HALVE its similarities. It must be REMOVED from the matrix, not just ignored.
    """
    D = np.load(MATDIR / f"dockq_{tag}.npy").astype(np.float32)
    idx = list(csv.DictReader(open(MATDIR / f"dockq_{tag}_index.csv")))
    ip = np.array([float(r['iptm']) for r in idx], dtype=np.float64)
    gt = np.array([float(r['dockq_gt']) for r in idx], dtype=np.float64)
    null = (D.sum(axis=1) == 0)
    real = ~null
    S = D[np.ix_(real, real)]
    S = (S + S.T) / 2.0
    return torch.from_numpy(S).to(DEV), ip, gt, null, real


# ---------------------------------------------------------------- clustering
def leader(S, ip_sub, tau):
    """Greedy leader clustering. SEQUENTIAL by construction -> CPU. Each pose joins its nearest
    leader if S >= tau, else becomes a new leader. Order = ipTM descending."""
    Sc = S.cpu().numpy()
    order = np.argsort(-ip_sub)
    lead, lab = [], np.full(len(ip_sub), -1, dtype=np.int64)
    for j in order:
        if lead:
            sims = Sc[lead, j]
            k = int(np.argmax(sims))
            if sims[k] >= tau:
                lab[j] = k; continue
        lead.append(int(j)); lab[j] = len(lead) - 1
    return lab


def medoid_refine(S, ip_sub, tau, max_it=15):
    """Leader clustering, then medoid refinement to convergence (GPU).
    medoids -> merge clusters whose MEDOIDS are within tau -> reassign to nearest medoid."""
    lab = torch.from_numpy(leader(S, ip_sub, tau)).to(DEV)
    n = S.shape[0]
    for _ in range(max_it):
        ks = torch.unique(lab)
        # medoid of each cluster = member with the highest MEAN similarity to the rest (GPU matmul)
        onehot = (lab.unsqueeze(0) == ks.unsqueeze(1)).float()          # [k, n]
        sizes = onehot.sum(1).clamp(min=1)
        meansim = (onehot @ S) / sizes.unsqueeze(1)                     # [k, n] mean sim to cluster
        masked = torch.where(onehot.bool(), meansim, torch.full_like(meansim, -1.0))
        med = masked.argmax(dim=1)                                      # [k] medoid index per cluster
        # merge clusters whose MEDOIDS are within tau (connected components over the medoid graph)
        A = (S[med][:, med] >= tau)
        A.fill_diagonal_(True)
        g = torch.arange(len(med), device=DEV)
        for _ in range(len(med)):
            new = torch.where(A, g.unsqueeze(0).expand(len(med), -1),
                              torch.full_like(A, len(med), dtype=g.dtype)).min(dim=1).values
            if torch.equal(new, g): break
            g = new
        # one medoid per merged group, then reassign every pose to its nearest medoid
        groups = torch.unique(g)
        gm = (g.unsqueeze(0) == groups.unsqueeze(1)).float()            # [G, k]
        memb = gm @ onehot                                              # [G, n] membership
        gsz = memb.sum(1).clamp(min=1)
        gmean = (memb @ S) / gsz.unsqueeze(1)
        gmask = torch.where(memb.bool(), gmean, torch.full_like(gmean, -1.0))
        gmed = gmask.argmax(dim=1)
        new_lab = S[gmed].argmax(dim=0)
        if torch.equal(new_lab, lab): break
        lab = new_lab
    return _compact(lab.cpu().numpy())


def dbscan(S, tau):
    """DBSCAN minPts=1 == connected components of (S >= tau). GPU label propagation."""
    n = S.shape[0]
    A = (S >= tau)
    A.fill_diagonal_(True)
    lab = torch.arange(n, device=DEV)
    BIG = torch.full((n, n), n, device=DEV, dtype=lab.dtype)
    for _ in range(n):
        new = torch.where(A, lab.unsqueeze(0).expand(n, n), BIG).min(dim=1).values
        if torch.equal(new, lab): break
        lab = new
    return _compact(lab.cpu().numpy())


def _compact(lab):
    _, inv = np.unique(lab, return_inverse=True)
    return inv


# ---------------------------------------------------------------- scoring
def cluster_stats(lab, ip_sub, gt_sub, null_ip, null_gt):
    """One dict per cluster. The NULL cluster is included -- it must be allowed to win."""
    cl = []
    for k in np.unique(lab):
        m = np.where(lab == k)[0]
        cl.append(_stats("real", ip_sub[m], gt_sub[m]))
    if len(null_ip):
        cl.append(_stats("no_contact", null_ip, null_gt))
    return cl


def _stats(kind, ip, gt):
    return dict(kind=kind, n=int(len(ip)),
                med_iptm=float(np.median(ip)), mean_iptm=float(np.mean(ip)),
                max_iptm=float(np.max(ip)),
                med_dq=float(np.median(gt)), mean_dq=float(np.mean(gt)),
                purity=float((gt > ACC).mean()),
                iqr_dq=float(np.subtract(*np.percentile(gt, [75, 25]))),
                rep_dq=float(gt[np.argmax(ip)]),
                iptm=ip, dq=gt)


def score(c, rule):
    if rule == "median":        return c['med_iptm']
    if rule == "mean":          return c['mean_iptm']
    if rule == "max":           return c['max_iptm']            # ~ naive ipTM
    if rule == "size_x_median": return c['n'] * c['med_iptm']
    if rule == "size_x_mean":   return c['n'] * c['mean_iptm']
    raise ValueError(rule)


def rank_and_find(cl, rule):
    """Rank clusters by `rule`; return (rank of first rescue cluster, that cluster, null rank)."""
    order = sorted(range(len(cl)), key=lambda i: -score(cl[i], rule))
    resc_rank = resc = None
    null_rank = None
    for r, i in enumerate(order, 1):
        if cl[i]['kind'] == "no_contact" and null_rank is None:
            null_rank = r
        if resc is None and cl[i]['med_dq'] > ACC and cl[i]['kind'] == "real":
            resc_rank, resc = r, cl[i]
    return resc_rank, resc, null_rank, cl[order[0]]


# ---------------------------------------------------------------- sweep
def peak_cpu_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # KB -> GB


def run(tags):
    rows, dump = [], {}
    for tag in tags:
        S, ip, gt, null, real = load(tag)
        ip_sub, gt_sub = ip[real], gt[real]
        null_ip, null_gt = ip[null], gt[null]
        o = np.argsort(-ip)
        ok = (gt > ACC) & (~null)                     # genuinely acceptable (must HAVE contacts)
        rescue_rank_iptm = int(np.where(ok[o])[0][0]) + 1 if ok.any() else None
        print(f"[{tag}] N={len(ip)} ({int(null.sum())} no-contact) | "
              f"first acceptable at ipTM rank {rescue_rank_iptm}", flush=True)

        for meth in METHODS:
            for tau in TAUS:
                if DEV == "cuda": torch.cuda.reset_peak_memory_stats()
                cpu0 = peak_cpu_gb(); t0 = time.perf_counter()
                if meth == "leader":         lab = _compact(leader(S, ip_sub, tau))
                elif meth == "leader+medoid": lab = medoid_refine(S, ip_sub, tau)
                else:                         lab = dbscan(S, tau)
                secs = time.perf_counter() - t0
                vram = torch.cuda.max_memory_allocated() / 1e9 if DEV == "cuda" else 0.0
                cpu = max(peak_cpu_gb(), cpu0)

                cl = cluster_stats(lab, ip_sub, gt_sub, null_ip, null_gt)
                row = dict(interface=tag, method=meth, tau=tau,
                           n_poses=len(ip), n_no_contact=int(null.sum()),
                           n_clusters=len(cl),
                           rescue_rank_iptm=rescue_rank_iptm,
                           secs=round(secs, 3), peak_vram_gb=round(vram, 3),
                           peak_cpu_gb=round(cpu, 2))
                for rule in RULES:
                    rr, rc, nr, top = rank_and_find(cl, rule)
                    row[f"rank_{rule}"] = rr
                    row[f"size_{rule}"] = rc['n'] if rc else None
                    row[f"null_rank_{rule}"] = nr
                    row[f"top1_is_null_{rule}"] = bool(nr == 1)
                    if rule == "median":                       # full detail for the primary rule
                        row.update(rescue_med_dockq=round(rc['med_dq'], 4) if rc else None,
                                   rescue_mean_dockq=round(rc['mean_dq'], 4) if rc else None,
                                   rescue_purity=round(rc['purity'], 4) if rc else None,
                                   rescue_iqr_dockq=round(rc['iqr_dq'], 4) if rc else None,
                                   rescue_rep_dockq=round(rc['rep_dq'], 4) if rc else None,
                                   top1_med_dockq=round(top['med_dq'], 4))
                rows.append(row)
                # raw per-cluster ipTM/DockQ (top 10 by median ipTM) for later stats tests
                top10 = sorted(cl, key=lambda c: -c['med_iptm'])[:10]
                dump[f"{tag}|{meth}|{tau}"] = [
                    {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in c.items()} for c in top10]
            print(f"  {meth:14} done "
                  f"({sum(r['secs'] for r in rows if r['interface']==tag and r['method']==meth):.0f}s)",
                  flush=True)
        del S
        if DEV == "cuda": torch.cuda.empty_cache()

    with open(OUTDIR / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(dump, open(OUTDIR / "clusters.json", "w"))
    print(f"\n-> {OUTDIR}/sweep.csv ({len(rows)} rows) + clusters.json")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+")
    a = ap.parse_args()
    split = json.load(open("results/csv_files/train_test_split.json"))
    run(a.tags or split['train'])
