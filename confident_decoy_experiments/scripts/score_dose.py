#!/usr/bin/env python
"""Score the dose-response sweep and print the sigma curve.

Per arm (one interface at one sigma): pool every pose, take the top-pair_iptm pose as the SELECTED
DockQ, and the best pose as the ORACLE. Reporting both matters -- a flat selected curve under a
rising oracle is an ipTM selection failure, not an absence of effect, and the two are indistinguishable
from selected DockQ alone.

The curve is averaged over interfaces with a COMPLETE set of rungs only. Averaging over whatever
happens to be finished would confound sigma with which interfaces got there first, and the fast
interfaces are the small ones.

Writes artifacts/dose_status.json (per arm) and artifacts/dose_curve.json (the aggregate).
"""
import glob, json, os, subprocess, sys, statistics
sys.path.insert(0, "/home/ebrahim/tts_abag/confident_decoy_experiments/scripts")
from decoy_set import OST, ENV, GT
EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
ALL = json.load(open(f"{EXP}/artifacts/abag_all_101_150.json"))
man = json.load(open(f"{EXP}/artifacts/wnoise_manifest.json"))
ACC = 0.23


def score(arm, pdb, c1, c2):
    res = {}
    for cif in sorted(glob.glob(f"{EXP}/artifacts/sweep_out/{arm}/**/*_sample_*.cif", recursive=True)):
        name = os.path.basename(cif)[:-4]; seed = int(cif.split("/seed_")[1].split("/")[0])
        o = f"{EXP}/artifacts/{arm}_ost/{seed}_{name}.json"
        os.makedirs(os.path.dirname(o), exist_ok=True)
        if not (os.path.exists(o) and os.path.getsize(o)):
            subprocess.run(["taskset", "-c", "0-47", OST, "compare-structures", "-m", cif, "-r",
                            f"{GT}/{pdb}-assembly1.cif", "-o", o, "--fault-tolerant",
                            "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq"],
                           env=ENV, capture_output=True, timeout=1800)
        try: d = json.load(open(o))
        except Exception: continue
        qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
        conf = os.path.join(os.path.dirname(cif),
                            name.replace("_sample_", "_summary_confidence_sample_") + ".json")
        try: M = json.load(open(conf))["chain_pair_iptm"]
        except Exception: continue
        for i, t in enumerate(ifs):
            if {t[0], t[1]} == {c1, c2} and i < len(qs):
                res.setdefault(seed, []).append(
                    (qs[i], M[ord(t[2][0]) - 65][ord(t[3][0]) - 65])); break
    return res


out = {}
for arm, meta in sorted(man.items()):
    if not arm.startswith("dose_"): continue
    lab = meta["interface"]; pdb = ALL[lab]["pdb"]
    c1, c2 = ALL[lab]["chains"].split("-", 1)
    R = score(arm, pdb, c1, c2)
    p = [x for v in R.values() for x in v]
    if not p:
        out[arm] = {"interface": lab, "sigma": meta["sigma"], "poses": 0, "status": "pending"}
        continue
    out[arm] = {"interface": lab, "sigma": meta["sigma"], "poses": len(p), "seeds": len(R),
                "selected": max(p, key=lambda x: x[1])[0], "oracle": max(q for q, _ in p),
                "frac": sum(1 for q, _ in p if q >= ACC) / len(p),
                "target": meta["target"], "status": "scored"}
json.dump(out, open(f"{EXP}/artifacts/dose_status.json", "w"), indent=1)

# ---- aggregate: complete-curve interfaces only ------------------------------------------------
sigmas = sorted({v["sigma"] for v in out.values()})
by = {}
for v in out.values():
    if v["status"] == "scored":
        by.setdefault(v["interface"], {})[v["sigma"]] = v
complete = sorted(k for k, d in by.items() if len(d) == len(sigmas))
print(f"sigmas: {sigmas}")
print(f"interfaces with a complete curve: {len(complete)} / {len(by)} started\n")
curve = {}
if complete:
    print(f"{'sigma':>10}{'sel_mean':>10}{'sel_med':>9}{'orc_mean':>10}{'frac_acc':>10}{'n_iface':>9}")
    print("-" * 58)
    for s in sigmas:
        sel = [by[i][s]["selected"] for i in complete]
        orc = [by[i][s]["oracle"] for i in complete]
        fr = [by[i][s]["frac"] for i in complete]
        curve[str(s)] = {"sel_mean": statistics.mean(sel), "sel_median": statistics.median(sel),
                         "oracle_mean": statistics.mean(orc), "frac_mean": statistics.mean(fr),
                         "n": len(complete)}
        print(f"{s:>10.4g}{statistics.mean(sel):>10.3f}{statistics.median(sel):>9.3f}"
              f"{statistics.mean(orc):>10.3f}{statistics.mean(fr):>10.3f}{len(complete):>9}")
json.dump({"curve": curve, "complete": complete, "sigmas": sigmas},
          open(f"{EXP}/artifacts/dose_curve.json", "w"), indent=1)
n_sc = sum(1 for v in out.values() if v["status"] == "scored")
print(f"\narms scored {n_sc}/{len(out)}")
