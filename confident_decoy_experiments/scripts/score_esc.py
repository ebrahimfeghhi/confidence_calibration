#!/usr/bin/env python
"""Score the escalation sweep: sigma=1e-2 per-cycle weight noise vs the 50-seed MSA-subsampling result.

For every esc_* arm in the manifest, pools all poses generated so far and reports the top-ipTM
selection against that interface's target (its 50-seed sel_dockq). MATCH = within 0.10 of target.
Writes artifacts/esc_status.json for the escalation driver to read.
"""
import glob, json, os, subprocess, sys
sys.path.insert(0, "/home/ebrahim/tts_abag/confident_decoy_experiments/scripts")
from decoy_set import OST, ENV, GT
EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
ALL = json.load(open(f"{EXP}/artifacts/abag_all_101_150.json"))
man = json.load(open(f"{EXP}/artifacts/wnoise_manifest.json"))
TOL = 0.10

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

out, rows = {}, []
for arm, meta in sorted(man.items()):
    if not arm.startswith(("esc_","e52_","e25_","e1_")): continue
    lab = meta["interface"]; pdb = ALL[lab]["pdb"]
    c1, c2 = ALL[lab]["chains"].split("-", 1)
    R = score(arm, pdb, c1, c2)
    p = [x for v in R.values() for x in v]
    if not p:
        out[arm] = {"interface": lab, "poses": 0, "seeds": 0, "status": "pending"}
        continue
    sel = max(p, key=lambda x: x[1])[0]; orc = max(q for q, _ in p)
    tgt = meta["target"]
    st = "MATCH" if sel >= tgt - TOL else "below"
    out[arm] = {"interface": lab, "poses": len(p), "seeds": len(R), "selected": sel,
                "oracle": orc, "frac": sum(1 for q, _ in p if q >= 0.23) / len(p),
                "target": tgt, "status": st}
    rows.append((lab, len(p), len(R), sel, orc, out[arm]["frac"], tgt, st))
json.dump(out, open(f"{EXP}/artifacts/esc_status.json", "w"), indent=1)
rows.sort(key=lambda r: (r[7] != "MATCH", -(r[3] - r[6])))
print(f"{'interface':<12}{'poses':>6}{'seeds':>6}{'selected':>10}{'oracle':>9}{'frac':>7}"
      f"{'target':>9}{'  status'}")
print("-" * 72)
for r in rows:
    print(f"{r[0]:<12}{r[1]:>6}{r[2]:>6}{r[3]:>10.3f}{r[4]:>9.3f}{r[5]:>7.2f}{r[6]:>9.3f}  {r[7]}")
n_m = sum(1 for r in rows if r[7] == "MATCH")
print(f"\nscored {len(rows)}/43   MATCH {n_m}   below {len(rows)-n_m}   "
      f"pending {sum(1 for v in out.values() if v['status']=='pending')}")
