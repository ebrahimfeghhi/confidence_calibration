"""matrix_iptm.py — soft, threshold-free "consensus ipTM" via a DockQ-weighted average.

For each of the 7 interfaces (Option A: one representative pose per seed = the best-ipTM diffusion
sample of that seed):
  1. Build the full ASYMMETRIC seed x seed DockQ matrix D, where D[i][j] = DockQ(reference=i, target=j)
     (row i = seed i as the native/reference, columns = every seed scored against it). Diagonal
     D[i][i] = DockQ(i,i) computed explicitly (~1.0), INCLUDED.
  2. For each seed i, weight every seed j's ipTM by structural closeness D[i][j]:
        S[i] = sum_j D[i][j] * ipTM[j]                 (weighted SUM  — encodes basin mass)
        A[i] = sum_j D[i][j] * ipTM[j] / sum_j D[i][j] (weighted AVG  — mass-normalized)
  3. Save D (.npy) + row index + per-seed scores, and report whether argmax S / argmax A is a good dock.

dockq_gt is carried only for EVALUATION (never used to compute S/A). DockQ via the DockQ package,
interface c1-c2, no_align. Memory-safe: bounded per-worker structure cache + 30s system-mem watchdog.
"""
import sys, csv, os, threading, time
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
OUTDIR = Path(os.environ.get("MATRIX_OUT", "results/iptm_matrix"))
OUTDIR.mkdir(parents=True, exist_ok=True)
NPROC = int(os.environ.get("NPROC", 16))
CACHE_MAX = int(os.environ.get("CACHE_MAX", 1200))   # big enough to hold all ~1025 structs/worker
T7 = [("8ts0","A","C"),("8ts0","B","C"),("8a44","A","D"),("7yxu","A","C"),
      ("8a44","A","C"),("7zoz","A","C"),("8vui","B","C")]

# --- memory watchdog (shared box; never make it unresponsive) ---
GB = 1024 ** 3
WATCH_SEC = float(os.environ.get("WATCH_SEC", 30))
FLAG_PCT = float(os.environ.get("FLAG_PCT", 80))
KILL_PCT = float(os.environ.get("KILL_PCT", 90))
_stop_watch = threading.Event()
def _sys_mem():
    tot = avail = None
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"): tot = int(line.split()[1]) * 1024
            elif line.startswith("MemAvailable:"): avail = int(line.split()[1]) * 1024
            if tot and avail: break
    return (tot - avail), tot, 100.0 * (tot - avail) / tot
def _memwatch(pool):
    flagged = False
    while not _stop_watch.wait(WATCH_SEC):
        try: used, tot, pct = _sys_mem()
        except Exception: continue
        print(f"  [memwatch] system mem {pct:.1f}% used ({used/GB:.0f}/{tot/GB:.0f} GB)", flush=True)
        if pct >= KILL_PCT:
            print(f"\n!!!!! [memwatch] MEMORY KILL — system {pct:.1f}% >= {KILL_PCT:.0f}%. Terminating.\n", flush=True)
            try: pool.terminate()
            except Exception: pass
            os._exit(3)
        elif pct >= FLAG_PCT and not flagged:
            flagged = True
            print(f"\n>>>>> [memwatch] MEMORY FLAG — system {pct:.1f}% >= {FLAG_PCT:.0f}%\n", flush=True)
        elif pct < FLAG_PCT:
            flagged = False

# --- per-worker DockQ ---
_UNITS = None; _CM = None; _C1 = None; _C2 = None
@lru_cache(maxsize=CACHE_MAX)
def _load(path):
    return load_PDB(path)
def _init(units, cm, c1, c2):
    global _UNITS, _CM, _C1, _C2
    _UNITS = units; _CM = cm; _C1 = c1; _C2 = c2
def _cell(ij):
    """DockQ(reference = unit i, target = unit j) = how well pose j reproduces native i."""
    i, j = ij
    ref, pose = _UNITS[i], _UNITS[j]
    try:
        n = _load(ref); m = _load(pose)
        res, _ = run_on_all_native_interfaces(m, n, chain_map=_CM, no_align=True)
    except Exception:
        return 0.0
    for kk, v in res.items():
        if set(kk) == {_C1, _C2}:
            return float(v['DockQ'])
    return 0.0

def cif(cx, s, k):
    # baseline_seed lives in its own dir; numeric TTS seeds live in seed_<N>
    sub = "baseline_seed" if str(s) == "baseline_seed" else f"seed_{s}"
    return str(DEEP / cx / sub / "predictions" / f"{cx}_sample_{k}.cif")

def _seed_sortkey(s):
    # numeric seeds first (in order), baseline_seed last
    return (1, 0) if s == "baseline_seed" else (0, int(s))

def one_interface(cx, c1, c2):
    tag = f"{cx}_{c1}{c2}"
    npy = OUTDIR / f"matrix_{tag}.npy"
    if npy.exists():
        print(f"[{tag}] matrix exists, skipping", flush=True); return
    # load poses, pick best-ipTM sample per seed (Option A). seeds kept as strings so the
    # baseline_seed (clean mask0/drop0 reference fold) is INCLUDED as its own row.
    per_seed = {}   # seed(str) -> (sample, dockq_gt, iptm)
    for r in csv.DictReader(open(CSVIN)):
        if (r['assembly_id'][:4], r['chain1'], r['chain2']) != (cx, c1, c2):
            continue
        s = r['seed']
        try:
            k = int(r['sample']); gt, ip = float(r['dockq']), float(r['pair_iptm'])
        except Exception:
            continue
        if not Path(cif(cx, s, k)).exists():
            continue
        if s not in per_seed or ip > per_seed[s][2]:
            per_seed[s] = (k, gt, ip)
    seeds = sorted(per_seed, key=_seed_sortkey)
    units = tuple(cif(cx, s, per_seed[s][0]) for s in seeds)
    gt_vec = np.array([per_seed[s][1] for s in seeds])
    ip_vec = np.array([per_seed[s][2] for s in seeds])
    N = len(seeds)
    print(f"[{tag}] {N} seeds (best-ipTM sample each); building {N}x{N} DockQ matrix...", flush=True)

    st = load_PDB(units[0]); CM = {ch.id: ch.id for ch in st}
    pool = Pool(NPROC, initializer=_init, initargs=(units, CM, c1, c2))
    print(f"  NPROC={NPROC} CACHE_MAX={CACHE_MAX} watchdog flag@{FLAG_PCT:.0f}%/kill@{KILL_PCT:.0f}%", flush=True)
    threading.Thread(target=_memwatch, args=(pool,), daemon=True).start()

    D = np.zeros((N, N), dtype=np.float32)
    t0 = time.time(); BLK = 64
    for i0 in range(0, N, BLK):
        i1 = min(i0 + BLK, N)
        tasks = [(i, j) for i in range(i0, i1) for j in range(N)]
        vals = pool.map(_cell, tasks, chunksize=64)
        D[i0:i1, :] = np.array(vals, dtype=np.float32).reshape(i1 - i0, N)
        print(f"  rows {i1}/{N}  elapsed={time.time()-t0:.0f}s", flush=True)
    _stop_watch.set(); pool.close(); pool.join()
    print(f"  matrix done in {time.time()-t0:.0f}s", flush=True)
    _stop_watch.clear()

    # save matrix + index
    np.save(npy, D)
    with open(OUTDIR / f"matrix_{tag}_index.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["row", "seed", "sample", "iptm", "dockq_gt"])
        for r_i, s in enumerate(seeds):
            k, gt, ip = per_seed[s]; w.writerow([r_i, s, k, f"{ip:.4f}", f"{gt:.4f}"])

    # weighted scores (diagonal included since D[i][i] is in the row)
    S = D @ ip_vec                      # weighted sum
    W = D.sum(axis=1)                    # mass of coefficients per row
    A = S / np.where(W > 0, W, 1.0)      # weighted average
    with open(OUTDIR / f"scores_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["seed", "sample", "iptm", "dockq_gt", "wsum", "wavg"])
        for idx, s in enumerate(seeds):
            k, gt, ip = per_seed[s]
            w.writerow([s, k, f"{ip:.4f}", f"{gt:.4f}", f"{S[idx]:.4f}", f"{A[idx]:.6f}"])

    # evaluation (dockq_gt only used here)
    def report(name, vec):
        i = int(np.argmax(vec))
        return f"{name}: pick seed={seeds[i]} score={vec[i]:.3f} dockq_gt={gt_vec[i]:.3f} {'GOOD' if gt_vec[i]>0.4 else 'decoy'}"
    base_i = int(np.argmax(ip_vec))
    ngood = int((gt_vec > 0.4).sum())
    print(f"  [{tag}] ngood={ngood}/{N} | baseline top-ipTM gt={gt_vec[base_i]:.3f} "
          f"{'GOOD' if gt_vec[base_i]>0.4 else 'decoy'}", flush=True)
    print(f"  {report('weighted-SUM ', S)}", flush=True)
    print(f"  {report('weighted-AVG ', A)}", flush=True)
    # append one-line summary to a shared results file
    with open(OUTDIR / "matrix_iptm_summary.csv", "a", newline="") as f:
        w = csv.writer(f)
        si, ai = int(np.argmax(S)), int(np.argmax(A))
        w.writerow([cx, c1, c2, N, ngood,
                    f"{gt_vec[base_i]:.3f}", f"{gt_vec[si]:.3f}", f"{gt_vec[ai]:.3f}",
                    int(gt_vec[base_i] > 0.4), int(gt_vec[si] > 0.4), int(gt_vec[ai] > 0.4)])

def main():
    which = sys.argv[1:] if len(sys.argv) > 1 else None
    for (cx, c1, c2) in T7:
        if which and f"{cx}_{c1}{c2}" not in which and cx not in which:
            continue
        one_interface(cx, c1, c2)
    print("ALL DONE", flush=True)

if __name__ == "__main__":
    main()
