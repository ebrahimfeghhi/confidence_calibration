# Label-free DockQ clustering — algorithm + handoff

_Last updated: 2026-07-09. Written after a server crash killed the previous session mid-run,
with no captured log — this doc exists so a future session (or a future crash) doesn't lose
the thread again._

## The idea

Can we pick the "good" (high-DockQ) structure out of a pool of decoy poses **without ground
truth**, using only pairwise structural agreement + `pair_iptm`? Test case: `8ts0` interface
A-C, which is a near-tie by earlier back-of-envelope estimate (good basin mass ~1171, mean ipTM
~0.49 vs. decoy basin's ~0.50) — a good stress test for the selection rule.

## Algorithm (`steer/experiments/dockq_cluster.py`)

Ground truth (`dockq_gt`) is read **only for evaluation**, never used to form or select clusters.

1. **Greedy-leader clustering** on real DockQ (interface-restricted, `no_align=True`, threshold
   `THR=0.23`): take the highest-`pair_iptm` leftover pose as a seed, gather everything with
   DockQ ≥ THR to it into a cluster, remove, repeat until leftover < `MIN_CLUST=5` (treated as
   noise).
2. **Medoid cleanup, using a shared anchor set** (per user correction — the anchor sample must
   be the *same* ~30 poses for every candidate in a cluster, not fresh samples per candidate,
   or the medoid pick is noisy): per cluster, medoid = member with highest avg DockQ to a
   random ≤`MEDOID_SAMPLE=30` subsample of clustermates (shared across all candidates in that
   cluster). Then reassign **every** pose to its nearest medoid (real DockQ), keeping it only if
   within `THR` of some medoid.
3. **Score + pick**: per final cluster, `mass` = # distinct seeds, `medip` = median `pair_iptm`,
   `score = mass * medip`. Hypothesis = argmax(score). Representative = highest-ipTM member.

DockQ via the `DockQ` pip package (validated == OST to 5e-4, ~100x faster than OST), interface
c1-c2 only. `_dq()` in the script monkeypatches `DockQ.parsers.MMCIF2Dict` to inject a dummy
occupancy column, since esmfold2fast cifs don't write one and DockQ's parser requires it.

## Agreed but NOT YET implemented: distogram-nominated medoid

Step 2's medoid computation currently does a full real-DockQ sweep (`cluster_size × 30` DockQ
calls per cluster) — this is the expensive part and the most likely cause of the long runtime
we're debugging. The agreed fix (not yet coded): use **distogram distance** to nominate a medoid
candidate near-for-free (member closest to the cluster's mean distogram), then confirm only that
one nominee with a real DockQ call. This would cut medoid cost from ~O(cluster_size × 30) DockQ
calls to ~O(cluster_size) cheap distogram-distance calls + 1 DockQ call per cluster.
No distogram-distance helper exists yet in this file; `cluster_basins.py` has adjacent
feature-extraction code (via `forward_grad.py`) but nothing directly reusable — would need to be
written from scratch (load each pose's saved distogram / recompute it, mean-pool per cluster,
nearest-to-mean nomination).

## Memory incident + fixes (2026-07-09)

The prior run took the shared server to near-OOM and disrupted other users (confirmed by Michael
Yuan: "it wasn't CPU usage… it was just using basically all of the memory"). The box never
rebooted (73-day uptime) — it was memory pressure / swap, not a hardware crash; the earlier
session just dropped.

**Root cause:** `_load` used `@lru_cache(maxsize=None)` — an *unbounded* cache — inside every one
of the 64 pool workers. Over a run each worker gradually cached all ~5120 distinct structures
(~5.3 MB each in RAM, measured), heading toward ~1.7 TB worst-case across 64 workers → filled the
1 TB box. Workers only ever *compute* one DockQ at a time; the leak was what they *retained*.

**Fixes applied to `dockq_cluster.py`:**
1. **Bounded cache** — `@lru_cache(maxsize=256)`. Caps each worker at 256 retained structures
   (~1.3 GB); worst-case total ≈ 256 × 5.3 MB × NPROC (≈42 GB @ 32, ≈84 GB @ 64). This is THE fix.
2. **`NPROC` is now an env var** (`NPROC=N`, default 32). Worker count is a memory multiplier.
3. **Memory watchdog thread** — every `WATCH_SEC=30`s reads `/proc/meminfo` and logs system used%.
   At `FLAG_PCT=80%` it prints a loud `MEMORY FLAG` line + touches a `MEM_FLAG` sentinel file; at
   `KILL_PCT=90%` it prints `MEMORY KILL`, terminates the pool, and exits. All thresholds are
   env-overridable. This is the backstop for any *future* unbounded growth the cache bound can't
   foresee.

Guardrail for all shared-box jobs: check `free -h`/`nvidia-smi`/`uptime` first, bound caches,
estimate worst-case memory = per-worker peak × workers, redirect output to a file, and never make
the box unresponsive for other users.

## Runtime findings from the 2026-07-09 run (8ts0 A-C, 5120 poses)

Added per-iteration logging to the greedy-leader loop (previously it only printed **after** the
entire while-loop finished, so a stalled/slow run looked identical to a hung one — this was the
actual cause of "zero output for 40+ minutes" in the prior session, not necessarily an infinite
loop).

- Greedy-leader phase: **33.6s total**, 54 clusters + 4 noise poses. Two dominant clusters
  (3771 and 1181 members) formed in the first two iterations; the remaining ~52 iterations mop
  up small (1-6 member) clusters from the 168 leftover poses. This phase is *not* the
  bottleneck.
- Medoid phase (step 2): still running as of this doc (~7 min elapsed and counting) — this
  is presumably where the earlier 40+ minute stall happened, and is the strongest argument for
  landing the distogram-nomination speedup.

## How to resume after a crash

```bash
cd ~/tts_abag && source .venv/bin/activate
nohup python steer/experiments/dockq_cluster.py 8ts0 A C \
  > steer/experiments/dockq_cluster_8ts0.out 2>&1 &
```

**Always redirect to a file** (`nohup ... > file 2>&1 &` or `tee`) — the prior crash lost all
run output because it was only going to an interactive terminal. Check progress with
`tail -f steer/experiments/dockq_cluster_8ts0.out`; per-iteration greedy progress now prints
live. Output CSV (once a run completes): `results/csv_files/dockq_cluster_summary.csv`.

Next steps once this run completes: confirm whether mass×median-ipTM picks the good cluster
on 8ts0 A-C (near-tie case). If it does, implement distogram-nomination for medoid (see above)
and run all 7 targets. If the good cluster isn't top-ranked, we need a tie-breaker beyond
mass×median-ipTM.
