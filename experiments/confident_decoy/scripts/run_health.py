#!/usr/bin/env python
"""Pre-launch gate and periodic health check for any templated arm. Run BEFORE and DURING every arm.

Written after `leader20` shipped THREE bugs in one afternoon, every one of which was detectable before
a GPU was touched:

  1. the scheduler ignored job["seeds"] and used a hardcoded 50   -> 10x the intended compute
  2. template ids were 5 chars (g0001); Protenix requires exactly 4  -> "Found 0 templates", a
     silently UNTEMPLATED run. Also the true cause of the void topdockq/topiptm/sposefv arms, whose
     4-digit counters overflowed past q999
  3. obj["name"] was never set, so every template of a pdb wrote into ONE directory, overwriting each
     other, while the scheduler logged 0/5 seeds for jobs that had run

Each failed silently and produced plausible output. That is the whole point: an arm can look healthy
and be measuring nothing.

MODES
  --preflight   checks a jobs file BEFORE launch. Exit 1 blocks the launch.
  --watch       loops every 30 min while an arm runs, alerting on anything that means the arm is not
                doing what it claims. Emits one line per event, so it can front a Monitor.

Usage:
  python scripts/run_health.py --preflight --jobs artifacts/leader20_jobs.json --seeds 5
  python scripts/run_health.py --watch --jobs artifacts/leader20_jobs.json \
      --out artifacts/sweep_out/leader20 --logs artifacts/leader20_logs --seeds 5
"""
import argparse, glob, json, os, re, subprocess, sys, time, collections

ap = argparse.ArgumentParser()
ap.add_argument("--preflight", action="store_true")
ap.add_argument("--watch", action="store_true")
ap.add_argument("--jobs", required=True)
ap.add_argument("--out", default=None)
ap.add_argument("--logs", default=None)
ap.add_argument("--seeds", type=int, default=None, help="expected seeds per job")
ap.add_argument("--interval", type=int, default=1800)
ap.add_argument("--max-cores", type=int, default=100)
A = ap.parse_args()

jobs = json.load(open(A.jobs))
MMCIF = "/home/ebrahim/mmcif"


def preflight():
    """Everything checkable without a GPU. Exit 1 = do not launch."""
    bad = collections.Counter(); ex = []

    def flag(key, detail):
        bad[key] += 1
        if len(ex) < 12: ex.append(f"{key}: {detail}")

    names = collections.Counter()
    for j in jobs:
        # --- the job record itself ---
        for k in ("run", "pdb", "json", "name"):
            if k not in j: flag("job missing field", f"{j.get('name','?')} lacks {k}")
        names[j.get("name")] += 1
        if A.seeds is not None and len(j.get("seeds") or []) != A.seeds:
            flag("wrong seed count", f"{j.get('name')} has {len(j.get('seeds') or [])}, expected {A.seeds}")
        if not os.path.isabs(j.get("json", "")): flag("relative json path", j.get("json"))
        if not os.path.exists(j.get("json", "")): flag("json missing", j.get("json")); continue

        o = json.load(open(j["json"])); o = o[0] if isinstance(o, list) else o
        # --- BUG 3: Protenix names output dirs and cif files after obj["name"] ---
        if o.get("name") != j["name"]:
            flag("json name != job name (outputs will collide)",
                 f"{j['name']} vs json {o.get('name')!r}")
        # --- templates ---
        n_t = 0
        for e in o.get("sequences", []):
            p = e.get("proteinChain", {}).get("templatesPath")
            if not p: continue
            n_t += 1
            if not os.path.exists(p): flag("a3m missing", p); continue
            L = open(p).read().split("\n")
            if len(L) < 4: flag("a3m too short", p); continue
            m = re.match(r">(\w+)_(\S+)/1-(\d+)", L[2])
            if not m: flag("malformed hit header", p); continue
            tid = m.group(1)
            # --- BUG 2: Protenix requires a 4-char PDB-style template id ---
            if len(tid) != 4:
                flag("template id not 4 chars (Protenix will reject -> Found 0 templates)",
                     f"{p} -> '{tid}'")
            if not os.path.exists(f"{MMCIF}/{tid}.cif"): flag("tid not registered", f"{p} -> {tid}")
            if len(L[1]) != len(L[3]): flag("query/hit length mismatch", p)
            if L[1] == L[3]: flag("hit == query (read as MSA row, not a template)", p)
        if n_t == 0: flag("job templates NO chains", j["name"])

    for nm, c in names.items():
        if c > 1: flag("duplicate job name (outputs collide)", f"{nm} x{c}")

    print(f"PREFLIGHT {A.jobs}: {len(jobs)} jobs, {len(names)} unique names")
    if bad:
        print("FAILURES:")
        for k, c in bad.most_common(): print(f"  {c:4d}  {k}")
        print("examples:")
        for e in ex: print("   " + e)
        return 1
    print("  all checks passed -- safe to launch")
    return 0


def cores():
    try:
        out = subprocess.run(["top", "-bn2", "-d", "2"], capture_output=True, text=True).stdout
        tot = 0.0
        for ln in out.split("PID USER")[-1].split("\n"):
            f = ln.split()
            if len(f) > 8 and f[1] == "ebrahim":
                try: tot += float(f[8])
                except ValueError: pass
        return tot / 100
    except Exception:
        return -1


def watch():
    prev_done = -1
    stalls = 0
    while True:
        # templates loading
        n1 = n0 = 0
        for f in glob.glob(f"{A.logs}/*.log"):
            t = open(f, errors="ignore").read()
            n1 += t.count("Found 1 templates"); n0 += t.count("Found 0 templates")
        # seeds written
        done = 0; wrong_name = 0
        want = {j["name"] for j in jobs}
        for j in jobs:
            d = f"{A.out}/{j['run']}/{j['name']}"
            done += len(glob.glob(f"{d}/seed_*/predictions/{j['name']}_summary_confidence_sample_4.json"))
        # "ERR" is Protenix's own error directory (runner/inference.py sets dump_dir/ERR), not a
        # stray output. Its NAME is expected; its CONTENTS are not -- anything in it means jobs failed.
        stray = [os.path.basename(d) for d in glob.glob(f"{A.out}/{jobs[0]['run']}/*")
                 if os.path.basename(d) not in want and os.path.basename(d) != "ERR"]
        err_dir = f"{A.out}/{jobs[0]['run']}/ERR"
        n_err = len(glob.glob(f"{err_dir}/*")) if os.path.isdir(err_dir) else 0
        c = cores()
        alive = sum(1 for p in glob.glob("/proc/[0-9]*")
                    if "inference.py" in (open(f"{p}/cmdline", errors="ignore").read()
                                          if os.path.exists(f"{p}/cmdline") else ""))
        total = sum(len(j.get("seeds") or []) for j in jobs)
        msgs = []
        if n1 == 0 and n0 > 0:
            msgs.append(f"ALERT templates NEVER loading ({n0} null) -- UNTEMPLATED run")
        elif n0 > 0.05 * max(n0 + n1, 1):
            msgs.append(f"ALERT {n0}/{n0+n1} chain-loads found NO template")
        if stray:
            msgs.append(f"ALERT {len(stray)} output dirs not matching any job name "
                        f"(name collision?): {stray[:3]}")
        if n_err:
            msgs.append(f"ALERT {n_err} entries in Protenix's ERR dir -- jobs are failing")
        if c > A.max_cores:
            msgs.append(f"ALERT {c:.0f} cores in use (cap {A.max_cores})")
        if done == prev_done and alive > 0:
            stalls += 1
            if stalls >= 2: msgs.append(f"ALERT no new seeds for {2*A.interval//60} min at {done}/{total}")
        else:
            stalls = 0
        if alive == 0 and done < total:
            msgs.append(f"ALERT no inference processes but only {done}/{total} seeds done")
        prev_done = done
        stamp = time.strftime("%H:%M")
        if msgs:
            for m in msgs: print(f"[{stamp}] {m}", flush=True)
        else:
            print(f"[{stamp}] OK seeds {done}/{total}, templates {n1} loaded / {n0} null, "
                  f"{alive} procs, {c:.0f} cores", flush=True)
        if alive == 0 and done >= total:
            print(f"[{stamp}] COMPLETE {done}/{total} seeds", flush=True); break
        time.sleep(A.interval)


sys.exit(preflight() if A.preflight else (watch() or 0))
