"""leader_discovery.py — phase 1 of the Option-B validation on held-out confident-decoy interfaces.

Discovers the LEADER UNION per interface: for each tau in the grid, run greedy leader discovery
(cap K) over the full Option-B pool (all 5 diffusion samples per seed, N ~ 5125), and take the
union of leaders across all tau. Those leaders become the columns of the "rectangle" that phase 2
fills in, which is what makes tau sweepable offline afterwards.

Why a rectangle and not a matrix: leader clustering only ever needs DockQ(leader, pose) -- it never
compares two ordinary poses. So we need N x |union|, not N x N. At N=5125 that is ~0.5-3M DockQ
runs instead of 26M.

Discovery early-exits: a pose is NOT a leader as soon as ANY existing leader is within tau of it,
so we stop at the first hit. (Assignment, in phase 2, needs the NEAREST leader and cannot early
-exit -- but that is the rectangle's job, and the rectangle covers it.)

DockQ is asymmetric (the reference defines which interface contacts are scored), so every pose-pair
needs BOTH directions and we use the mean. Reading one direction only is what split 8ts0's good
basin in two.

Memory: shared box. Bounded structure cache + 30s system-memory watchdog (flag 80%, kill 90%).

Usage:  python steer/experiments/leader_discovery.py            # all held-out interfaces
        python steer/experiments/leader_discovery.py 8ux6_AF    # one interface
"""
import sys, os, csv, json, time, threading
from pathlib import Path
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
DECOY = "results/csv_files/decoy_structural_analysis.csv"
OUTDIR = Path(os.environ.get("OUT", "results/leader_rect"))
OUTDIR.mkdir(parents=True, exist_ok=True)

TAUS = [0.23, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
K = int(os.environ.get("K", 100))                 # max leaders per tau
NPROC = int(os.environ.get("NPROC", 32))
CACHE_MAX = int(os.environ.get("CACHE_MAX", 96))  # small: discovery streams, it does not need all
GOOD = 0.4      # 'good dock' bar (reporting only)
ACC  = 0.23     # CAPRI 'acceptable' bar — this is what DEFINES the trap class

T7 = {("8ts0","A","C"),("8ts0","B","C"),("8a44","A","D"),("7yxu","A","C"),
      ("8a44","A","C"),("7zoz","A","C"),("8vui","B","C")}

# ---------------- memory watchdog (never make the shared box unresponsive) ----------------
GB = 1024 ** 3
WATCH_SEC, FLAG_PCT, KILL_PCT = 30.0, 80.0, 90.0
_stop = threading.Event()

def _sys_mem():
    tot = avail = None
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):       tot = int(line.split()[1]) * 1024
            elif line.startswith("MemAvailable:"): avail = int(line.split()[1]) * 1024
            if tot and avail: break
    return tot - avail, tot, 100.0 * (tot - avail) / tot

def _memwatch(pool):
    while not _stop.wait(WATCH_SEC):
        try: used, tot, pct = _sys_mem()
        except Exception: continue
        print(f"  [memwatch] system {pct:.1f}% ({used/GB:.0f}/{tot/GB:.0f} GB)", flush=True)
        if pct >= KILL_PCT:
            print(f"\n!!!!! [memwatch] KILL — system {pct:.1f}% >= {KILL_PCT}%. Terminating.\n",
                  flush=True)
            try: pool.terminate()
            except Exception: pass
            os._exit(3)
        elif pct >= FLAG_PCT:
            print(f"\n>>>>> [memwatch] FLAG — system {pct:.1f}% >= {FLAG_PCT}%\n", flush=True)

# ---------------- per-worker DockQ ----------------
_PATHS = _CM = _C1 = _C2 = None

@lru_cache(maxsize=CACHE_MAX)
def _load(p):
    return load_PDB(p)

def _init(paths, cm, c1, c2):
    global _PATHS, _CM, _C1, _C2
    _PATHS, _CM, _C1, _C2 = paths, cm, c1, c2

def _pair(ij):
    """Symmetrized DockQ between poses i and j: mean of both reference directions.
    DockQ(ref=i, target=j) != DockQ(ref=j, target=i) because the REFERENCE defines which
    interface contacts get scored. Reading only one direction is a real bug (see 8ts0)."""
    i, j = ij
    try:
        a, b = _load(_PATHS[i]), _load(_PATHS[j])
        vals = []
        for ref, tgt in ((a, b), (b, a)):
            res, _ = run_on_all_native_interfaces(tgt, ref, chain_map=_CM, no_align=True)
            v = next((float(x['DockQ']) for k, x in res.items()
                      if set(k) == {_C1, _C2}), 0.0)
            vals.append(v)
        return float(np.mean(vals))
    except Exception:
        return 0.0

# ---------------- pool ----------------
def build_pool(cx, c1, c2):
    """Option B: EVERY diffusion sample of every seed. No best-ipTM-per-seed pre-filter --
    that pre-filter uses argmax ipTM, the exact operation we have proven is untrustworthy."""
    rows = []
    for r in csv.DictReader(open(CSVIN)):
        if (r['assembly_id'][:4], r['chain1'], r['chain2']) != (cx, c1, c2):
            continue
        try:
            s, k = r['seed'], int(r['sample'])
            ip, gt = float(r['pair_iptm']), float(r['dockq'])
        except Exception:
            continue
        sub = "baseline_seed" if s == "baseline_seed" else f"seed_{s}"
        p = DEEP / cx / sub / "predictions" / f"{cx}_sample_{k}.cif"
        if p.exists():
            rows.append((str(p), ip, gt, s, k))
    rows.sort(key=lambda r: -r[1])                # ipTM descending: the discovery order
    return rows

def discover(cx, c1, c2):
    tag = f"{cx}_{c1}{c2}"
    out = OUTDIR / f"leaders_{tag}.json"
    if out.exists():
        print(f"[{tag}] exists, skipping", flush=True); return

    rows = build_pool(cx, c1, c2)
    N = len(rows)
    if N < 500:
        print(f"[{tag}] only {N} poses — SKIP", flush=True); return
    paths = tuple(r[0] for r in rows)
    ip = np.array([r[1] for r in rows]); gt = np.array([r[2] for r in rows])
    ngood = int((gt > GOOD).sum())
    print(f"[{tag}] N={N} poses (Option B, all samples), {ngood} good docks "
          f"({100*ngood/N:.2f}%)", flush=True)

    st = load_PDB(paths[0]); CM = {ch.id: ch.id for ch in st}
    pool = Pool(NPROC, initializer=_init, initargs=(paths, CM, c1, c2))
    threading.Thread(target=_memwatch, args=(pool,), daemon=True).start()

    cache = {}                                    # (leader, pose) -> symmetrized DockQ, shared across taus
    BATCH = max(NPROC, 64)

    def fill(pairs):
        """Compute any uncached (leader, pose) pairs in ONE parallel map."""
        need = [p for p in dict.fromkeys(pairs) if p not in cache]
        if not need:
            return
        for p, v in zip(need, pool.map(_pair, need, chunksize=1)):
            cache[p] = v

    t0 = time.time()
    per_tau = {}
    for tau in TAUS:
        L, i = [], 0
        while i < N and len(L) < K:
            if not L:
                L.append(i); i += 1; continue
            # SPECULATIVE BATCH: the next BATCH poses all need DockQ against the CURRENT leaders.
            # Compute them in one map (BATCH x |L| tasks) so every worker stays busy, then walk
            # them sequentially against the cache. If one becomes a new leader, the poses after it
            # simply need topping up against that single new leader -- same result, ~30x better
            # utilisation than mapping |L| tasks at a time.
            nxt = list(range(i, min(i + BATCH, N)))
            fill([(h, j) for j in nxt for h in L])
            for j in nxt:
                if len(L) >= K:
                    break
                fill([(h, j) for h in L])          # top-up vs leaders added inside this batch
                if max(cache[(h, j)] for h in L) < tau:
                    L.append(j)
                i = j + 1
        per_tau[tau] = L
        print(f"  [{tag}] tau={tau:<5} -> {len(L):>3} leaders  "
              f"(scanned {i}/{N}, DockQ pairs {len(cache):,}, {time.time()-t0:.0f}s)", flush=True)

    _stop.set(); pool.close(); pool.join(); _stop.clear()
    union = sorted({i for L in per_tau.values() for i in L})
    print(f"  [{tag}] LEADER UNION = {len(union)}   DockQ pair-calls = {len(cache):,}   "
          f"{time.time()-t0:.0f}s", flush=True)

    with open(out, "w") as f:
        json.dump({
            "interface": tag, "N": N, "ngood": ngood, "K": K, "taus": TAUS,
            "union": union, "union_size": len(union),
            "leaders_per_tau": {str(t): L for t, L in per_tau.items()},
            "dockq_calls": len(cache), "seconds": round(time.time() - t0, 1),
            "pool": [{"path": r[0], "iptm": r[1], "dockq_gt": r[2], "seed": r[3], "sample": r[4]}
                     for r in rows],
        }, f)
    print(f"  [{tag}] saved -> {out}", flush=True)

def held_out(include_t7=False):
    """The confident-decoy class, CAPRI bar throughout:
         ipTM's pick is BELOW acceptable (DockQ < 0.23), but an acceptable pose (> 0.23) EXISTS.

    An earlier version required the best pose to clear 0.4, which wrongly discarded 9 real traps
    whose only recoverable pose sits in 0.23-0.40 — recovering an incorrect pick to an acceptable
    one is still a win. Excluded now: only interfaces where ipTM already picks an acceptable pose
    (9fww, 9g2g) — those are 'improve a mediocre pick', a different problem.
    -> 48 interfaces: 7 dev (T7) + 41 held-out.
    """
    from collections import defaultdict
    pool = defaultdict(list)
    for r in csv.DictReader(open(CSVIN)):
        k = (r['assembly_id'][:4], r['chain1'], r['chain2'])
        try: pool[k].append((float(r['pair_iptm']), float(r['dockq'])))
        except Exception: pass
    out = []
    for r in csv.DictReader(open(DECOY)):
        k = (r['cx'], r['c1'], r['c2'])
        p = pool.get(k, [])
        if len(p) < 500: continue
        if k in T7 and not include_t7: continue
        p.sort(key=lambda x: -x[0])
        if p[0][1] < ACC and max(d for _, d in p) > ACC:
            out.append(k)
    return out

if __name__ == "__main__":
    todo = held_out()
    if len(sys.argv) > 1:
        want = set(sys.argv[1:])
        todo = [k for k in todo if f"{k[0]}_{k[1]}{k[2]}" in want or k[0] in want]
    print(f"=== leader discovery: {len(todo)} interfaces | K={K} | taus={TAUS} | "
          f"NPROC={NPROC} ===", flush=True)
    for cx, c1, c2 in todo:
        discover(cx, c1, c2)
    print("ALL DONE", flush=True)
