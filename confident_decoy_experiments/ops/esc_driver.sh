#!/bin/bash
# Escalation driver. After every scoring pass:
#   - interface within TOL of its 50-seed MSA-subsampling target  -> done, never extended
#   - interface more than TOL below                                -> +5 more seeds, up to MAX_SEEDS
# Extending = rewriting that arm's jobs.json with 5 additional seeds; the workers pick the arm up
# again automatically because it now has fewer poses than it wants.
# Runs every 5 min. Stops when nothing is left to extend.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
LOG=$REPO/ops/esc_driver.log
say(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$LOG"; }
say "escalation driver started (tol 0.10, +5 seeds per round, cap 50 seeds, every 5 min)"
while true; do
    $PY - <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, os, glob
EXP="/home/ebrahim/tts_abag/confident_decoy_experiments"
TOL, STEP, MAX_SEEDS = 0.10, 5, 50
man=json.load(open(f"{EXP}/artifacts/wnoise_manifest.json"))
try: st=json.load(open(f"{EXP}/artifacts/esc_status.json"))
except Exception:
    print("  no esc_status.json yet -- waiting for the first scoring pass"); raise SystemExit
ext=done=pend=busy=0
for arm,meta in sorted(man.items()):
    if not arm.startswith(("esc_","e52_","e25_","e1_")): continue
    jf=f"{EXP}/artifacts/{arm}_jobs.json"
    jobs=json.load(open(jf)); seeds=jobs[0]["seeds"]
    have=len(glob.glob(f"{EXP}/artifacts/sweep_out/{arm}/**/*_sample_*.cif", recursive=True))
    if have < len(seeds)*5:            # still generating -- don't touch its job file
        busy+=1; continue
    s=st.get(arm)
    if not s or s.get("status")=="pending": pend+=1; continue
    if s["selected"] >= s["target"]-TOL: done+=1; continue
    if len(seeds) >= MAX_SEEDS:
        done+=1; continue              # hit the cap, stop extending
    new=list(range(101, 101+len(seeds)+STEP))
    for j in jobs: j["seeds"]=new
    json.dump(jobs, open(jf,"w"), indent=1)
    print(f"  EXTEND {s['interface']:<12} sel {s['selected']:.3f} vs target {s['target']:.3f} "
          f"-> {len(seeds)} to {len(new)} seeds")
    ext+=1
print(f"  extended {ext} | settled {done} | generating {busy} | unscored {pend}")
PYEOF
    sleep 300
done
