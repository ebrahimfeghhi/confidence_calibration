#!/usr/bin/env python
"""Score the sigma=1e-2 (msa+pair) runs on interfaces that 50 seeds rescued.
Config: dropout OFF, MSA subsampling OFF, natural hmmsearch templates, NO donor.
Reference: the 1-seed and 50-seed selections from abag_all_101_150.json."""
import glob, json, os, subprocess, sys
sys.path.insert(0, "scripts")
from decoy_set import OST, ENV, GT
ALL = json.load(open("artifacts/abag_all_101_150.json"))
ONE = {"8vyl_CF": 0.013, "8pmz_AB": 0.025, "7txw_AC": 0.027, "8c5h_AB": 0.014, "8jeo_AB": 0.079}
PICK = [("8vyl_CF", "8vyl"), ("8pmz_AB", "8pmz"), ("7txw_AC", "7txw"),
        ("8c5h_AB", "8c5h"), ("8jeo_AB", "8jeo")]
print(f"{'interface':<11}{'poses':>6}{'noise sel':>11}{'oracle':>9}{'frac':>7} | "
      f"{'1 seed':>8}{'50 seeds':>10}{'n_acc':>7}")
print("-" * 74)
for lab, pdb in PICK:
    arm = f"wn5_{pdb}"; c1, c2 = ALL[lab]["chains"].split("-", 1); res = {}
    for cif in sorted(glob.glob(os.path.abspath(f"artifacts/sweep_out/{arm}") + "/**/*_sample_*.cif",
                                recursive=True)):
        name = os.path.basename(cif)[:-4]; seed = int(cif.split("/seed_")[1].split("/")[0])
        o = os.path.abspath(f"artifacts/{arm}_ost/{seed}_{name}.json")
        os.makedirs(os.path.dirname(o), exist_ok=True)
        if not (os.path.exists(o) and os.path.getsize(o)):
            subprocess.run(["taskset", "-c", "0-31", OST, "compare-structures", "-m", cif, "-r",
                            f"{GT}/{pdb}-assembly1.cif", "-o", o, "--fault-tolerant",
                            "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq"],
                           env=ENV, capture_output=True, timeout=1800)
        try: d = json.load(open(o))
        except Exception: continue
        qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
        conf = os.path.join(os.path.dirname(cif),
                            name.replace("_sample_", "_summary_confidence_sample_") + ".json")
        M = json.load(open(conf))["chain_pair_iptm"]
        for i, t in enumerate(ifs):
            if {t[0], t[1]} == {c1, c2} and i < len(qs):
                res.setdefault(seed, []).append(
                    (qs[i], M[ord(t[2][0]) - 65][ord(t[3][0]) - 65])); break
    p = [x for v in res.values() for x in v]; v = ALL[lab]
    if not p:
        print(f"{lab:<11}{0:>6}   (pending)"); continue
    print(f"{lab:<11}{len(p):>6}{max(p, key=lambda x: x[1])[0]:>11.3f}"
          f"{max(q for q, _ in p):>9.3f}{sum(1 for q, _ in p if q >= 0.23)/len(p):>7.2f} | "
          f"{ONE[lab]:>8.3f}{v['sel_dockq']:>10.3f}{v['n_acceptable']:>7}")
