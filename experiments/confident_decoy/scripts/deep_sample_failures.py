#!/usr/bin/env python
"""Are the 8 native-template failures genuinely coverage-starved, or just under-sampled?

Seven of the eight interfaces that native-fold templating did NOT rescue had only 1-6 acceptable poses
out of 250 in the free run (0.4-2.4%). At that rate the difference between "1 acceptable pose" and
"0 acceptable poses" is a coin flip -- and it is exactly the difference between being classified a
confident decoy and a generation failure. Their whole classification may be a sampling artifact.

Free-run predictions already exist for seeds 0-150 (and 0-296 for 8vui); only 101-150 were ever scored,
because the project standardised on that range after the seed-range trap. This scores the FULL free
range for those interfaces -- 755 poses instead of 250, no GPU needed -- and asks whether the acceptable
fraction holds up.

NOTE ON COMPARABILITY: this deepens the FREE arm only. The native-templated arm still has 50 seeds, so
these numbers must not be compared directly against the native results; they answer a question about
the free run alone (is coverage really ~0?). Extending the native arm to match is separate GPU work.

SCORING PROTOCOL: unchanged (tts_abag/eval/eval_ost_foldbench.py) -- one OST call per pose on the full
native assembly vs the full model, labelled interface extracted by name, isolated-pair fallback if the
pair is absent.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/deep_sample_failures.py
Writes: artifacts/deep_sample_failures.json
"""
import json, os, subprocess, sys, collections
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import OST, ENV, GT, EXP, PRED, SAMPLES, THRESH

SEEDS = range(0, 151)                     # everything on disk for these pdbs
CACHE = f"{EXP}/artifacts/deep_sample_ost"; os.makedirs(CACHE, exist_ok=True)
OUTJ = f"{EXP}/artifacts/deep_sample_failures.json"
NWORK = 16

scores = json.load(open(f"{EXP}/artifacts/native_arm_scores.json"))
FAILS = {k: v for k, v in scores.items()
         if v["free_status"] == "confident_decoy" and not v["rescued"]}
by_pdb = collections.defaultdict(list)
import csv
for r in csv.DictReader(open("/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv")):
    pdb = r["pdb_id"].split("-")[0]
    c1, c2 = r["interface_chain_id_1"], r["interface_chain_id_2"]
    if f"{pdb}_{c1}{c2}" in FAILS:
        by_pdb[pdb].append((c1, c2))


def run_pose(a):
    pdb, pairs, s, k = a
    cif = f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif"
    if not os.path.exists(cif):
        return {}
    o = f"{CACHE}/{pdb}_{s}_{k}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset", "-c", "0-71", OST, "compare-structures", "-m", cif,
                        "-r", f"{GT}/{pdb}-assembly1.cif", "-o", o, "--fault-tolerant",
                        "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq"],
                       env=ENV, capture_output=True, timeout=900)
    try:
        d = json.load(open(o))
    except Exception:
        return {}
    qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
    try:
        M = json.load(open(f"{PRED}/{pdb}/seed_{s}/predictions/"
                          f"{pdb}_summary_confidence_sample_{k}.json"))["chain_pair_iptm"]
    except Exception:
        return {}
    out = {}
    for c1, c2 in pairs:
        for i, t in enumerate(ifs):
            if {t[0], t[1]} == {c1, c2} and i < len(qs):
                x, y = ord(t[2]) - 65, ord(t[3]) - 65
                out[(c1, c2)] = (M[x][y] if (0 <= x < len(M) and 0 <= y < len(M)) else -1, qs[i])
                break
    return out


res = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}
for pdb, pairs in sorted(by_pdb.items()):
    if all(f"{pdb}_{c1}{c2}" in res for c1, c2 in pairs):
        continue
    jobs = [(pdb, pairs, s, k) for s in SEEDS for k in SAMPLES]
    with ThreadPoolExecutor(NWORK) as ex:
        got = list(ex.map(run_pose, jobs))
    for c1, c2 in pairs:
        lab = f"{pdb}_{c1}{c2}"
        poses = [g[(c1, c2)] for g in got if (c1, c2) in g]
        if not poses:
            continue
        sel = max(poses, key=lambda x: x[0])
        ok = [p for p in poses if p[1] >= THRESH]
        prev = FAILS[lab]
        res[lab] = {"pdb": pdb, "n_poses_full": len(poses), "n_acceptable_full": len(ok),
                    "frac_full": round(len(ok) / len(poses), 4),
                    "n_poses_101_150": 250, "n_acceptable_101_150": prev["free_n_acceptable"],
                    "frac_101_150": round(prev["free_n_acceptable"] / 250, 4),
                    "sel_dockq_full": round(sel[1], 3), "top_iptm_full": round(sel[0], 3),
                    "best_dockq_full": round(max(p[1] for p in poses), 3),
                    "best_dockq_101_150": prev["free_best_dockq"],
                    "still_decoy_full": (sel[1] < THRESH and len(ok) > 0),
                    "no_acceptable_full": len(ok) == 0}
        v = res[lab]
        print(f"{lab:11} seeds 101-150: {v['n_acceptable_101_150']:3d}/250 ({v['frac_101_150']:.1%})"
              f"   ->  FULL: {v['n_acceptable_full']:3d}/{v['n_poses_full']} ({v['frac_full']:.1%})"
              f"   best {v['best_dockq_101_150']:.3f}->{v['best_dockq_full']:.3f}"
              f"   {'NO ACCEPTABLE POSE' if v['no_acceptable_full'] else 'still a decoy'}", flush=True)
    json.dump(res, open(OUTJ, "w"), indent=1)

print(f"\n{sum(1 for v in res.values() if v['no_acceptable_full'])}/{len(res)} have ZERO acceptable poses "
      f"over the full free range -> they are generation failures, not selection failures")
print(f"WROTE {OUTJ}")
