#!/usr/bin/env python
"""Auto-extend the sigma ladder when the curve is still climbing at an end.

Rule: if the end rung BEATS its inward neighbour by more than EPS, the optimum is plausibly outside
the measured range, so add the next rung outward. Stop when it flattens or reverses.

Two guards, both load-bearing:
  - MIN_IFACE: never act on a handful of complete curves. Early-finishing interfaces are the small
    fast ones, and they are not representative of the mean.
  - EPS: a bare > comparison would chase sampling noise off the end of the ladder indefinitely.

sigma=0 is the control, not a rung -- the low end is extended from the smallest NONZERO sigma, since
there is nothing below zero.
"""
import json, os, sys

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"; A = f"{EXP}/artifacts"
# Kill switch, checked before anything else. The watcher loop is already running, so editing
# dose_watch.sh would not reliably take effect -- a flag file the script itself reads does.
if os.path.exists(f"{A}/.dose_no_extend"):
    # Ladder extension stays off. Instead, once every sigma=0 arm is complete, deepen sigma=0 ONLY,
    # +10 seeds a round, hard cap 50 per interface. Nothing else is ever re-queued.
    import glob as _g
    S0_CAP, S0_STEP = 50, 10
    man = json.load(open(f"{A}/wnoise_manifest.json"))
    arms = [l.strip() for l in open(f"{A}/dose_queue.txt") if l.strip()
            and man.get(l.strip(), {}).get("sigma") == 0.0]
    pend = tot = 0
    for a in arms:
        jf = f"{A}/{a}_jobs.json"
        if not os.path.exists(jf): continue
        want = sum(len(j["seeds"]) for j in json.load(open(jf))) * 5
        tot += 1
        if len(_g.glob(f"{A}/sweep_out/{a}/**/*_sample_*.cif", recursive=True)) < want: pend += 1
    if pend:
        print(f"  sigma=0 sweep: {tot-pend}/{tot} arms complete -- not deepening yet"); sys.exit()
    at_cap = grown = 0
    for a in arms:
        jf = f"{A}/{a}_jobs.json"
        jobs = json.load(open(jf)); n = len(jobs[0]["seeds"])
        if n >= S0_CAP: at_cap += 1; continue
        new = list(range(101, 101 + min(n + S0_STEP, S0_CAP)))
        for j in jobs: j["seeds"] = new
        json.dump(jobs, open(jf, "w"), indent=1); grown += 1
    if grown:
        print(f"  DEEPEN sigma=0 -> {grown} arms +{S0_STEP} seeds (cap {S0_CAP}); {at_cap} already at cap")
    else:
        print(f"  sigma=0 complete and all {at_cap} arms at the {S0_CAP}-seed cap -- nothing left to do")
    sys.exit()
JD = "/home/ebrahim/Protenix/foldbench_abag/jsons"
MIN_IFACE, EPS = 12, 0.010
LOW_LADDER = [5e-4, 1e-4, 5e-5]
HIGH_LADDER = [5e-1, 1.0, 5.0]

try:
    C = json.load(open(f"{A}/dose_curve.json"))
except Exception:
    print("  no dose_curve.json yet"); sys.exit()
curve = {float(k): v for k, v in C["curve"].items()}
if not curve:
    print("  curve empty"); sys.exit()
n = next(iter(curve.values()))["n"]
if n < MIN_IFACE:
    print(f"  only {n} complete curves (< {MIN_IFACE}) -- not extending yet"); sys.exit()

ALL = json.load(open(f"{A}/abag_all_101_150.json"))
man = json.load(open(f"{A}/wnoise_manifest.json"))
S = json.load(open(f"{A}/dose_sample.json"))
sample, SEEDS = S["sample"], S["seeds"]
have = sorted(curve)
nz = [s for s in have if s > 0]


def tag(s):
    return "s" + f"{s:g}".replace("-0", "").replace("-", "").replace(".", "p").replace("+", "")


def add(sig):
    if any(abs(m.get("sigma", -1) - sig) < 1e-12 for k, m in man.items() if k.startswith("dose_")):
        print(f"  sigma {sig:g} already present"); return 0
    q = []
    for iface in sample:
        arm = f"dose_{tag(sig)}_{iface}"; pdb = ALL[iface]["pdb"]
        json.dump([{"run": arm, "pdb": pdb, "json": f"{JD}/{pdb}-final-updated.json",
                    "name": pdb, "seeds": SEEDS}], open(f"{A}/{arm}_jobs.json", "w"), indent=1)
        man[arm] = {"sigma": sig, "scope": "msa+pair", "interface": iface,
                    "target": ALL[iface]["sel_dockq"]}
        q.append(arm)
    json.dump(man, open(f"{A}/wnoise_manifest.json", "w"), indent=1)
    with open(f"{A}/dose_queue.txt", "a") as f:
        f.write("\n".join(q) + "\n")
    print(f"  EXTEND -> added sigma {sig:g} ({len(q)} arms, {len(q)*len(SEEDS)} seeds)")
    return 1


added = 0
if len(nz) >= 2:
    lo, lo2 = nz[0], nz[1]
    d = curve[lo]["sel_mean"] - curve[lo2]["sel_mean"]
    print(f"  low end  sigma {lo:g} {curve[lo]['sel_mean']:.3f} vs {lo2:g} "
          f"{curve[lo2]['sel_mean']:.3f}  (delta {d:+.3f})")
    if d > EPS:
        nxt = [s for s in LOW_LADDER if s < lo]
        if nxt: added += add(nxt[0])
        else: print("  low ladder exhausted")

    hi, hi2 = have[-1], have[-2]
    d = curve[hi]["sel_mean"] - curve[hi2]["sel_mean"]
    print(f"  high end sigma {hi:g} {curve[hi]['sel_mean']:.3f} vs {hi2:g} "
          f"{curve[hi2]['sel_mean']:.3f}  (delta {d:+.3f})")
    if d > EPS:
        nxt = [s for s in HIGH_LADDER if s > hi]
        if nxt: added += add(nxt[0])
        else: print("  high ladder exhausted")
if not added:
    print("  no ladder extension this round")
    # Nothing left to widen and nothing left to run -> DEEPEN instead: +10 seeds on every arm, so the
    # GPUs keep buying precision on the existing curve rather than idling overnight.
    import glob as _g
    arms = [a for a in man if a.startswith("dose_")]
    pend = 0
    for a in arms:
        jf = f"{A}/{a}_jobs.json"
        if not os.path.exists(jf): continue
        want = sum(len(j["seeds"]) for j in json.load(open(jf))) * 5
        if len(_g.glob(f"{A}/sweep_out/{a}/**/*_sample_*.cif", recursive=True)) < want:
            pend += 1
    if pend:
        print(f"  {pend} arms still generating -- not deepening yet")
    else:
        for a in arms:
            jf = f"{A}/{a}_jobs.json"
            jobs = json.load(open(jf))
            new = list(range(101, 101 + len(jobs[0]["seeds"]) + 10))
            for j in jobs: j["seeds"] = new
            json.dump(jobs, open(jf, "w"), indent=1)
        print(f"  DEEPEN -> every arm {len(arms)} +10 seeds (now {len(new)} each)")
