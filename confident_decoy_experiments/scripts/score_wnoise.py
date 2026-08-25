#!/usr/bin/env python
"""Score the weight-noise sweep: does perturbing weights substitute for MSA subsampling?

All conditions: donor 137_3 template, mask ON, dropout OFF, MSA subsampling OFF (full depth).
Weight noise is therefore the only source of trunk variation.
Reference points for this donor + seed set: subsampled 0.942 | full MSA, no noise 0.089.
Scores whatever is finished; incomplete conditions are reported with their pose count.
"""
import glob, json, os, subprocess, sys
sys.path.insert(0, "scripts")
from decoy_set import OST, ENV, GT
IF = {"8che_AE": ("A", "E"), "8che_BE": ("B", "E")}
man = json.load(open("artifacts/wnoise_manifest.json"))

def score(arm):
    res = {}
    for cif in sorted(glob.glob(os.path.abspath(f"artifacts/sweep_out/{arm}") + "/**/*_sample_*.cif",
                                recursive=True)):
        name = os.path.basename(cif)[:-4]; seed = int(cif.split("/seed_")[1].split("/")[0])
        o = os.path.abspath(f"artifacts/{arm}_ost/{seed}_{name}.json")
        os.makedirs(os.path.dirname(o), exist_ok=True)
        if not (os.path.exists(o) and os.path.getsize(o)):
            subprocess.run(["taskset", "-c", "0-31", OST, "compare-structures", "-m", cif, "-r",
                            f"{GT}/8che-assembly1.cif", "-o", o, "--fault-tolerant",
                            "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq"],
                           env=ENV, capture_output=True, timeout=1800)
        try: d = json.load(open(o))
        except Exception: continue
        qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
        conf = os.path.join(os.path.dirname(cif),
                            name.replace("_sample_", "_summary_confidence_sample_") + ".json")
        M = json.load(open(conf))["chain_pair_iptm"]
        for lab, (c1, c2) in IF.items():
            for i, t in enumerate(ifs):
                if {t[0], t[1]} == {c1, c2} and i < len(qs):
                    res.setdefault(lab, {}).setdefault(seed, []).append(
                        (qs[i], M[ord(t[2]) - 65][ord(t[3]) - 65])); break
    return res

for exp, scope in [("exp1", "MSA module only"), ("exp2", "MSA module + Pairformer")]:
    print(f"\n=== {exp.upper()} : {scope} — 8che, donor 137_3, dropout OFF, full MSA ===")
    print(f"{'sigma':>10}{'poses':>7}{'AE sel':>9}{'AE oracle':>11}{'AE frac':>9}"
          f"{'BE sel':>9}{'BE oracle':>11}")
    for run in sorted(r for r in man if exp in r):
        R = score(run); sig = man[run]["sigma"]
        n = sum(len(v) for lab in R for v in R[lab].values()) // max(len(R), 1)
        cells = []
        for lab in ("8che_AE", "8che_BE"):
            p = [x for v in R.get(lab, {}).values() for x in v]
            if not p: cells.append(None); continue
            cells.append((max(p, key=lambda x: x[1])[0], max(q for q, _ in p),
                          sum(1 for q, _ in p if q >= 0.23) / len(p)))
        ae, be = cells
        if ae is None: print(f"{sig:>10.2e}{n:>7}   (no scores yet)"); continue
        print(f"{sig:>10.2e}{n:>7}{ae[0]:>9.3f}{ae[1]:>11.3f}{ae[2]:>9.2f}"
              f"{(be[0] if be else float('nan')):>9.3f}{(be[1] if be else float('nan')):>11.3f}")
print("\nreference (same donor+seeds): MSA subsampled 0.942 | full MSA no noise 0.089")
