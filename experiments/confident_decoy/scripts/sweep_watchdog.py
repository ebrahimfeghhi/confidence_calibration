#!/usr/bin/env python
"""Health watchdog for the 7-run template sweep. Wakes every 30 min, checks, and self-heals.

Checks and actions each cycle:
  1. SCHEDULER ALIVE  - if the driver/scheduler died with jobs left, relaunch run_sweep_p34.sh.
  2. ORPHANS          - inference processes that are NOT children of the live scheduler hold GPU memory
                        and can OOM the real jobs (7z2m needs ~92 GB of a 97 GB card). Kill them.
                        A killed job loses nothing: the scheduler resumes at seed granularity.
  3. STALL            - if no new seed has been written anywhere for 90 min while jobs are running,
                        something is wedged; report it (do NOT auto-kill, a 7z2m seed legitimately
                        takes ~10 min and a whole job ~8 h).
  4. OOM              - scan job logs for CUDA OOM and report which job.
  5. RESOURCES        - measured CPU cores (budget 64) and GPUs in use (must stay within 0-3).

Only NOTABLE lines are printed with an ALERT/ACTION prefix so a Monitor can filter on them; a routine
cycle prints one OK line. Never uses a bare pkill pattern -- an earlier incident had pkill match the
watchdog's own command line and kill it.

Usage: setsid nohup .../python scripts/sweep_watchdog.py > artifacts/decoy_logs/watchdog.log 2>&1 &
"""
import argparse, json, os, subprocess, sys, time, glob

EXP = "/home/ebrahim/confident_decoy_experiments"
LOG = f"{EXP}/artifacts/decoy_logs"
# Parameterised so one watchdog serves any run: the health logic (dead scheduler, true orphans,
# stalls, OOM, CPU/GPU budget) is identical, only the job list, output root and driver differ.
_ap = argparse.ArgumentParser()
_ap.add_argument("--jobs", default=f"{EXP}/artifacts/sweep_jobs.json")
_ap.add_argument("--out", default=f"{EXP}/artifacts/sweep_out")
_ap.add_argument("--driver", default="run_sweep_p34.sh")
_A = _ap.parse_args()
SEEDS = list(range(101, 151))
INTERVAL = 1800          # 30 min
STALL_LIMIT = 90 * 60    # 90 min with zero new seeds = wedged
CORE_BUDGET = 100        # HARD cap (64 is only a soft target)
ALLOWED_GPUS = {0, 1, 2, 3}


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def procs():
    """[(pid, ppid, argv)] read straight from /proc.

    NOT pgrep: `pgrep -f <pat>` run through a shell also matches the shell that is running it, so it
    can return a transient PID instead of the real process. On 2026-08-01 that made sched_pid() return
    a short-lived shell, so every legitimate inference child looked like an orphan and the watchdog
    killed four running jobs. Read /proc directly and match argv exactly."""
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            argv = open(f"/proc/{d}/cmdline", "rb").read().decode().split("\x00")
            ppid = int(open(f"/proc/{d}/stat").read().split()[3])
        except Exception:
            continue
        out.append((int(d), ppid, argv))
    return out


def sched_pid():
    for pid, _, argv in procs():
        if len(argv) > 1 and argv[1].endswith("scripts/sweep_sched.py"):
            return pid
    return None


def driver_alive():
    return any(any(a.endswith(_A.driver) for a in argv) for _, _, argv in procs())


def seeds_done():
    try:
        jobs = json.load(open(_A.jobs))
    except Exception:
        return 0, 0
    tot = ndone = 0
    for j in jobs:
        d = f"{_A.out}/{j['run']}/{j['name']}"
        n = sum(1 for s in SEEDS
                if os.path.exists(f"{d}/seed_{s}/predictions/{j['name']}_summary_confidence_sample_4.json"))
        tot += n
        ndone += (n >= len(SEEDS))
    return tot, ndone


def kill_orphans(pid):
    """TRUE orphans only: inference processes re-parented to init (ppid == 1).

    A live job is always a direct child of the scheduler, so ppid==1 is the only safe signature.
    The previous version killed anything whose ppid != sched_pid(), which destroyed running jobs the
    moment sched_pid() was wrong. Requires a live scheduler, so nothing is killed during a handover."""
    if pid is None:
        return []
    killed = []
    for cpid, ppid, argv in procs():
        if ppid == 1 and len(argv) > 1 and "runner/inference.py" in argv[1]:
            try:
                os.kill(cpid, 9); killed.append(str(cpid))
            except Exception:
                pass
    return killed


def scored():
    try:
        s = json.load(open(f"{EXP}/artifacts/sweep_scores.json"))
        return sum(len(v) for v in s.values())
    except Exception:
        return 0


NTOT = len(json.load(open(_A.jobs)))
last_seeds, last_change = None, time.time()
prev_cotenant = set()      # only alert when co-tenancy CHANGES; it is handled by the scheduler's
                           # memory check, so repeating it every cycle is pure noise
print(f"[{time.strftime('%H:%M')}] watchdog start (every {INTERVAL//60} min)", flush=True)

while True:
    now = time.strftime("%H:%M")
    # GPU index<->uuid and which cards are MINE. Computed first: both the co-tenant check and the
    # out-of-range check need it. Flagging any busy card (rather than only mine) gave a false alert
    # when a co-worker started on GPU4.
    uuid2idx = {}
    for l in sh("nvidia-smi --query-gpu=index,uuid --format=csv,noheader").splitlines():
        i, u = [x.strip() for x in l.split(",")]
        uuid2idx[u] = int(i)
    apps = []
    for l in sh("nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader").splitlines():
        parts = [x.strip() for x in l.split(",")]
        if len(parts) == 3:
            apps.append((uuid2idx.get(parts[0]), parts[1], sh(f"ps -o user= -p {parts[1]}").strip(), parts[2]))
    mine = {i for i, _, o, _ in apps if o == "ebrahim" and i is not None}
    pid = sched_pid()
    tot, ndone = seeds_done()
    notes = []

    if last_seeds is None or tot > last_seeds:
        last_seeds, last_change = tot, time.time()
    stalled = time.time() - last_change

    # 1. scheduler / driver health
    if pid is None:
        if ndone >= NTOT:
            print(f"[{now}] OK  phase 3 finished ({ndone}/108 jobs); scoring phase owns the pipeline now",
                  flush=True)
        elif not driver_alive():
            print(f"[{now}] ACTION scheduler AND driver dead with {ndone}/{NTOT} done -> relaunching",
                  flush=True)
            subprocess.Popen(["setsid", "nohup", f"{EXP}/scripts/{_A.driver}"], cwd=EXP,
                             stdout=open(f"{LOG}/run_sweep_p34.log", "a"),
                             stderr=subprocess.STDOUT, start_new_session=True)
        else:
            notes.append("scheduler gone but driver alive (between phases)")
    else:
        # 2. orphans
        k = kill_orphans(pid)
        if k:
            print(f"[{now}] ACTION killed {len(k)} orphan inference proc(s) {k} "
                  f"(GPU-memory / OOM risk; seeds are preserved)", flush=True)
        # 3. stall
        if stalled > STALL_LIMIT:
            print(f"[{now}] ALERT no new seed for {stalled/60:.0f} min with scheduler alive "
                  f"({ndone}/{NTOT} jobs, {tot} seeds) -- check artifacts/sweep_logs/", flush=True)

    # 3b. co-tenant on one of my GPUs -- 7z2m needs ~92 GB of a 97 GB card, so a neighbour holding
    #     tens of GB will OOM it. Report so it can be rescheduled rather than failing silently.
    cot = {(idx, owner) for idx, _, owner, _ in apps
           if owner and owner != "ebrahim" and idx in ALLOWED_GPUS}
    if cot != prev_cotenant:
        for idx, owner in sorted(cot - prev_cotenant):
            mem = next((m for i, _, o, m in apps if i == idx and o == owner), "?")
            print(f"[{now}] ALERT GPU{idx} now shared with {owner} ({mem}) -- big jobs will be "
                  f"routed elsewhere by the scheduler's memory check", flush=True)
        for idx, owner in sorted(prev_cotenant - cot):
            print(f"[{now}] ACTION GPU{idx} freed by {owner} -- big jobs can use it again", flush=True)
        prev_cotenant = cot

    # 3c. jobs that "finished" with ZERO output. A per-seed data error (e.g. a template a3m the
    #     parser rejects) makes inference exit 0 having produced nothing, so the scheduler records
    #     DONE and neither the stall nor the OOM check fires. 8im0 sat broken for 8 h this way.
    try:
        zero = [j["pdb"] for j in json.load(open(_A.jobs))
                if not glob.glob(f"{_A.out}/{j['run']}/{j['name']}/seed_*")
                and glob.glob(f"{EXP}/artifacts/*_logs/*{j['pdb']}*.log")]
        if zero:
            print(f"[{now}] ALERT job(s) ran but produced ZERO seeds: {zero} -- inspect their logs",
                  flush=True)
    except Exception:
        pass

    # 4. OOM in job logs
    oom = sh(f"grep -l 'CUDA out of memory' {EXP}/artifacts/sweep_logs/*.log 2>/dev/null")
    if oom:
        print(f"[{now}] ALERT CUDA OOM in: {[os.path.basename(x) for x in oom.split()][:4]}", flush=True)

    # 5. resources
    # INSTANTANEOUS cpu via top. `ps pcpu` averages over each process's whole lifetime and badly
    # overstates short-lived OST processes -- it read 113 cores when the true figure was ~80.
    cores = sh("top -bn2 -d 2 | grep -A300 'PID USER' | "
               "awk '$1 ~ /^[0-9]+$/ && $2 == \"ebrahim\" {c+=$9} END{printf \"%.0f\", c/100}'") or "0"
    if int(cores) > CORE_BUDGET:
        print(f"[{now}] ALERT measured {cores} cores > budget {CORE_BUDGET}", flush=True)
    if mine - ALLOWED_GPUS:
        print(f"[{now}] ALERT GPUs in use outside 0-3: {sorted(mine - ALLOWED_GPUS)}", flush=True)

    if not any(x.startswith(("ALERT", "ACTION")) for x in notes):
        print(f"[{now}] OK  jobs {ndone}/{NTOT}, seeds {tot}/{NTOT*50}, cores {cores}, "
              f"gpus {sorted(mine)}, scored {scored()}"
              + (f" | {'; '.join(notes)}" if notes else ""), flush=True)
    time.sleep(INTERVAL)
