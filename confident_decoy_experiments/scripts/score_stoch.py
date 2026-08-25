#!/usr/bin/env python
"""Score the dropout x MSA-subsampling cells for 8che donor 137_3 (all mask ON, same 10 seeds)."""
import glob, json, os, subprocess, sys
sys.path.insert(0, "scripts")
from decoy_set import OST, ENV, GT
IF = {"8che_AE": ("A", "E"), "8che_BE": ("B", "E")}
SEEDS = [137, 108, 130, 250, 332, 832, 1036, 1040, 1074, 1076]

def score(arm):
    res = {}
    root = os.path.abspath(f"artifacts/sweep_out/{arm}")
    for cif in sorted(glob.glob(root + "/**/*_sample_*.cif", recursive=True)):
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

def merge(*rs):
    out = {}
    for r in rs:
        for lab, byseed in r.items():
            for s, v in byseed.items(): out.setdefault(lab, {}).setdefault(s, []).extend(v)
    return out

CELLS = [("dropout 0.4 + subsampled  (stock)", score("rep137_3_maskon")),
         ("dropout 0.4 + FULL MSA",            score("drop_fullmsa")),
         ("dropout 0   + subsampled",          score("nodrop_subsample")),
         ("dropout 0   + FULL MSA",            merge(score("nodrop_fullmsa10"),
                                                     score("nodrop_fullmsa10b")))]
for lab in ["8che_AE", "8che_BE"]:
    print(f"\n=== {lab} : donor 137_3 (DockQ 0.398), mask ON, same 10 seeds ===")
    print(f"{'cell':<34}{'selected':>10}{'oracle':>9}{'frac>=0.23':>12}{'seeds>=0.23':>13}{'n':>5}")
    for name, R in CELLS:
        r = R.get(lab, {})
        p = [x for v in r.values() for x in v]
        if not p: print(f"{name:<34}   (no scores yet)"); continue
        ns = sum(1 for v in r.values() if max(q for q, _ in v) >= 0.23)
        print(f"{name:<34}{max(p,key=lambda x:x[1])[0]:>10.3f}{max(q for q,_ in p):>9.3f}"
              f"{sum(1 for q,_ in p if q>=0.23)/len(p):>12.2f}{ns:>9}/{len(r):<3}{len(p):>5}")
    print("  per-seed max:")
    for name, R in CELLS:
        r = R.get(lab, {})
        if not r: continue
        row = "  ".join(f"{s}:{max(q for q,_ in r[s]):.2f}" for s in SEEDS if s in r)
        print(f"    {name:<34}{row}")
