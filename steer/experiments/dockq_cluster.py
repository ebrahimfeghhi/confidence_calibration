"""dockq_cluster.py — label-free DockQ greedy-leader clustering + mass x median-ipTM selection.

Implements the user's algorithm (no ground truth used to form/select clusters):
  1. Order poses by pair_iptm (desc).
  2. Greedy-leader: highest-iptm leftover seeds a cluster; gather everything with DockQ >= THR to it;
     remove; repeat until leftover < MIN_CLUST (noise).
  3. Medoid cleanup: per cluster, medoid = member with highest avg DockQ to a random <=MEDOID_SAMPLE
     subsample of clustermates (O(m*30) instead of O(m^2)); then reassign EVERY pose to the nearest
     medoid (real DockQ), keeping it only if within THR of some medoid.
  4. Score each cluster: mass = #distinct seeds, median pair_iptm, representative = highest-iptm member.
  5. Pick the hypothesis = argmax(mass * median_iptm).

dockq_gt is read ONLY for evaluation (never used to form/select). DockQ via the DockQ package
(validated == OST to 5e-4, ~100x faster), interface c1-c2 only, no_align (identical pred sequences).
"""
import sys, csv, os, random, threading
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
from multiprocessing import Pool
import numpy as np

# --- patch DockQ's mmCIF parser for occupancy-less esmfold2 cifs ---
import DockQ.parsers as _P
_o = _P.MMCIF2Dict
def _pt(fn):
    d = _o(fn)
    if '_atom_site.occupancy' not in d:
        d['_atom_site.occupancy'] = ['1.00'] * len(d['_atom_site.id'])
    return d
_P.MMCIF2Dict = _pt
from DockQ.DockQ import load_PDB, run_on_all_native_interfaces

DEEP = Path.home() / "esm/foldbench_predictions/esmfold2fast_tts"
CSVIN = "results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"
THR = float(os.environ.get("THR", 0.23))   # greedy merge threshold (sweepable overnight)
MIN_CLUST = 5
MEDOID_SAMPLE = 30
NPROC = int(os.environ.get("NPROC", 32))     # override with NPROC=N; shared box, keep memory×workers bounded
random.seed(0)

# --- memory watchdog: shared multi-user server, never make it unresponsive ---
# The 2026-07-09 incident: unbounded _load cache x N workers grew RSS until it ate ~all RAM.
# Every WATCH_SEC seconds, check TOTAL SYSTEM memory usage:
#   >= FLAG_PCT  -> print a loud FLAG line (also written to a sentinel file for external alerting)
#   >= KILL_PCT  -> print ALERT, terminate our whole job (protect the shared box)
GB = 1024 ** 3
WATCH_SEC = float(os.environ.get("WATCH_SEC", 30))
FLAG_PCT = float(os.environ.get("FLAG_PCT", 80))    # system mem used% at which to flag
KILL_PCT = float(os.environ.get("KILL_PCT", 90))    # system mem used% at which to kill our job
FLAG_FILE = os.environ.get("MEM_FLAG_FILE", "steer/experiments/MEM_FLAG")  # touched when flagged
_stop_watch = threading.Event()

def _sys_mem():
    """Return (used_bytes, total_bytes, used_pct) for system RAM, using MemAvailable."""
    tot = avail = None
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                tot = int(line.split()[1]) * 1024
            elif line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) * 1024
            if tot is not None and avail is not None:
                break
    used = tot - avail
    return used, tot, 100.0 * used / tot

def _memwatch(pool):
    flagged = False
    while not _stop_watch.wait(WATCH_SEC):
        try:
            used, tot, pct = _sys_mem()
        except Exception:
            continue
        print(f"  [memwatch] system mem {pct:.1f}% used ({used/GB:.0f}/{tot/GB:.0f} GB)  "
              f"[flag@{FLAG_PCT:.0f}% kill@{KILL_PCT:.0f}%]", flush=True)
        if pct >= KILL_PCT:
            print(f"\n!!!!! [memwatch] MEMORY KILL — system at {pct:.1f}% (>= {KILL_PCT:.0f}%). "
                  f"Terminating our job NOW to protect the shared server.\n", flush=True)
            try:
                pool.terminate()
            except Exception:
                pass
            os._exit(3)
        elif pct >= FLAG_PCT and not flagged:
            flagged = True   # flag once per crossing, don't spam
            print(f"\n>>>>> [memwatch] MEMORY FLAG — system at {pct:.1f}% (>= {FLAG_PCT:.0f}%). "
                  f"Watching closely; will kill our job at {KILL_PCT:.0f}%.\n", flush=True)
            try:
                with open(FLAG_FILE, "w") as f:
                    f.write(f"{pct:.1f}% used ({used/GB:.0f}/{tot/GB:.0f} GB)\n")
            except Exception:
                pass
        elif pct < FLAG_PCT:
            flagged = False  # reset so a later re-crossing flags again

CX = None; _CM = None; _C1 = None; _C2 = None

def cif(sk):
    s, k = sk
    return str(DEEP / CX / f"seed_{s}" / "predictions" / f"{CX}_sample_{k}.cif")

@lru_cache(maxsize=256)   # BOUNDED — was maxsize=None (unbounded), which x NPROC workers ate all RAM (2026-07-09)
def _load(path):
    return load_PDB(path)

def _init(cx, cm, c1, c2):
    global CX, _CM, _C1, _C2
    CX = cx; _CM = cm; _C1 = c1; _C2 = c2

def _dq(args):
    """DockQ of interface c1-c2 : how well `pose` (model) reproduces `ref` (native)."""
    ref_path, pose_path = args
    try:
        n = _load(ref_path); m = _load(pose_path)
        res, _ = run_on_all_native_interfaces(m, n, chain_map=_CM, no_align=True)
    except Exception:
        return 0.0
    for kk, v in res.items():
        if set(kk) == {_C1, _C2}:
            return float(v['DockQ'])
    return 0.0

def main(cx, c1, c2):
    global CX
    CX = cx
    P = {}
    for r in csv.DictReader(open(CSVIN)):
        if (r['assembly_id'][:4], r['chain1'], r['chain2']) != (cx, c1, c2):
            continue
        try:
            P[(int(r['seed']), int(r['sample']))] = (float(r['dockq']), float(r['pair_iptm']))
        except Exception:
            pass
    poses = [sk for sk in P if Path(cif(sk)).exists()]
    print(f"[{cx} {c1}-{c2}] {len(poses)} poses", flush=True)
    st = load_PDB(cif(poses[0])); chains = [ch.id for ch in st]; CM = {c: c for c in chains}
    pool = Pool(NPROC, initializer=_init, initargs=(cx, CM, c1, c2))
    print(f"  NPROC={NPROC}  mem-watchdog: every {WATCH_SEC:.0f}s, flag@{FLAG_PCT:.0f}% kill@{KILL_PCT:.0f}% system mem",
          flush=True)
    threading.Thread(target=_memwatch, args=(pool,), daemon=True).start()

    # 1-2. greedy-leader
    import time
    MAX_IT = int(os.environ.get("MAX_IT", 100000))   # cap tail iterations (tiny clusters); big basins seed early
    leftover = set(poses); clusters = []
    it = 0; t0 = time.time()
    while len(leftover) >= MIN_CLUST and it < MAX_IT:
        it += 1
        seed = max(leftover, key=lambda sk: P[sk][1])
        lst = list(leftover)
        dqs = pool.map(_dq, [(cif(seed), cif(x)) for x in lst])
        members = [x for x, d in zip(lst, dqs) if d >= THR]
        if seed not in members:
            members.append(seed)
        clusters.append(members); leftover -= set(members)
        print(f"  [greedy it={it}] seed={seed} took_this_round={len(members)} "
              f"leftover={len(leftover)} elapsed={time.time()-t0:.1f}s", flush=True)
    print(f"  greedy: {len(clusters)} clusters, {len(leftover)} noise, {time.time()-t0:.1f}s total", flush=True)

    # optional: dump greedy pose->cluster assignments (tiny CSV) so all selection metrics can be
    # evaluated offline without re-clustering. SKIP_MEDOID short-circuits the expensive medoid+reassign
    # step — for a THR/metric sweep the greedy partition is what we study.
    cluster_out = os.environ.get("CLUSTER_OUT")
    if cluster_out:
        from pathlib import Path as _P
        d = _P(cluster_out); d.mkdir(parents=True, exist_ok=True)
        fn = d / f"{cx}_{c1}{c2}_thr{THR:.2f}.csv"
        with open(fn, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["seed", "sample", "cluster_id", "dockq_gt", "pair_iptm"])
            for cid, mem in enumerate(clusters):
                for (s, k) in mem:
                    gt, ip = P[(s, k)]; w.writerow([s, k, cid, f"{gt:.4f}", f"{ip:.4f}"])
            for (s, k) in leftover:
                gt, ip = P[(s, k)]; w.writerow([s, k, -1, f"{gt:.4f}", f"{ip:.4f}"])
        print(f"  saved greedy assignments -> {fn}", flush=True)
        if os.environ.get("SKIP_MEDOID"):
            _stop_watch.set(); pool.close(); pool.join()
            print(f"  SKIP_MEDOID set — done (greedy-only) for {cx} {c1}-{c2} @ THR={THR:.2f}", flush=True)
            return None, None

    # 3a. medoid per cluster (subsampled)
    medoids = []
    for cl in clusters:
        if len(cl) == 1:
            medoids.append(cl[0]); continue
        cand = cl if len(cl) <= MEDOID_SAMPLE else random.sample(cl, MEDOID_SAMPLE)
        dqs = pool.map(_dq, [(cif(c), cif(mb)) for mb in cl for c in cand])
        arr = np.array(dqs).reshape(len(cl), len(cand))
        medoids.append(cl[int(arr.mean(1).argmax())])

    # 3b. reassign every pose to nearest medoid (within THR)
    allp = list(poses)
    D = np.array(pool.map(_dq, [(cif(med), cif(x)) for med in medoids for x in allp])
                 ).reshape(len(medoids), len(allp))
    assign = D.argmax(0); best = D.max(0)
    newc = defaultdict(list)
    for i, x in enumerate(allp):
        if best[i] >= THR:
            newc[int(assign[i])].append(x)
    _stop_watch.set()   # DockQ work done — stop the watchdog before scoring/eval
    pool.close()

    # 4-5. score + pick
    res = []
    for cid, mem in newc.items():
        mass = len({s for s, k in mem})
        ipts = np.array([P[sk][1] for sk in mem]); gts = np.array([P[sk][0] for sk in mem])
        rep = max(mem, key=lambda sk: P[sk][1])
        res.append(dict(cid=cid, n=len(mem), mass=mass, medip=float(np.median(ipts)),
                        score=mass * float(np.median(ipts)), rep=rep, rep_gt=P[rep][0],
                        rep_ip=P[rep][1], maxgt=float(gts.max()), fracgood=float((gts > 0.4).mean())))
    res.sort(key=lambda r: -r['score'])

    # evaluation (uses dockq_gt)
    tot_good = sum(1 for sk in poses if P[sk][0] > 0.4)
    good_cl = max(res, key=lambda r: r['fracgood'] * r['n']) if res else None
    picked = res[0]
    top_ip = max(poses, key=lambda sk: P[sk][1]); base_gt = P[top_ip][0]

    print(f"  {'cid':>4}{'n':>6}{'mass':>5}{'medIp':>7}{'score':>7}{'repGT':>7}{'repIp':>7}{'maxGT':>7}{'fracG':>7}")
    for r in res[:8]:
        star = ' <-PICK' if r is picked else (' <-GOOD' if r is good_cl else '')
        print(f"  {r['cid']:>4}{r['n']:>6}{r['mass']:>5}{r['medip']:>7.3f}{r['score']:>7.1f}"
              f"{r['rep_gt']:>7.3f}{r['rep_ip']:>7.3f}{r['maxgt']:>7.3f}{r['fracgood']:>7.2f}{star}", flush=True)
    print(f"  >> baseline top-iptm pick GT={base_gt:.3f} | method pick rep GT={picked['rep_gt']:.3f} "
          f"(fracgood={picked['fracgood']:.2f}) | good-cluster rank by score = "
          f"{[i for i,r in enumerate(res) if r is good_cl][0] if good_cl else -1}", flush=True)

    with open("results/csv_files/dockq_cluster_summary.csv", "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([cx, c1, c2, len(poses), tot_good, len(res),
                    f"{base_gt:.3f}", f"{picked['rep_gt']:.3f}", f"{picked['fracgood']:.2f}",
                    picked['mass'], f"{picked['medip']:.3f}",
                    ([i for i, r in enumerate(res) if r is good_cl][0] if good_cl else -1),
                    f"{good_cl['fracgood']:.2f}" if good_cl else "", good_cl['n'] if good_cl else 0])
    return picked, base_gt

if __name__ == "__main__":
    cx, c1, c2 = (sys.argv[1], sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("8ts0", "A", "C")
    main(cx, c1, c2)
