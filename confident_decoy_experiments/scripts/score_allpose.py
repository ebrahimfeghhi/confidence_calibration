#!/usr/bin/env python
"""Score an allpose arm: 250 donor templates x 5 samples, on BOTH 8che interfaces.

One row per (donor, output pose): the donor's own properties (its DockQ on each interface, and which
seed/sample produced it) alongside what templating on it produced. That is the response surface --
every candidate donor in the pool, not a criterion's pick.

Per donor it reports, for each interface separately:
  donor_dockq      the donor pose's own DockQ on that interface (the candidate's quality)
  out_max/median   DockQ of the 5 poses that donor produced
  out_sel          DockQ of the highest-ipTM of those 5  (what selection would return per donor)
  frac_acceptable  of the 5

SCORING PROTOCOL is the project standard (tts_abag/eval/eval_ost_foldbench.py): one OST call per
pose, full native assembly vs full model, --fault-tolerant --min-pep-length 4 --min-nuc-length 4
--dockq, and the interface extracted BY NAME (never dockq[0]).

Refuses to report if any template loaded 0 templates -- a null arm must not be scored (three arms were
scored and believed while silently untemplated).

Usage: /home/ebrahim/Protenix/.venv/bin/python scripts/score_allpose.py --run allpose8che [--workers 16]
Writes: artifacts/<run>_scores.json
"""
import argparse, json, os, glob, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, OST, ENV, GT, SAMPLES, THRESH

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="allpose8che")
ap.add_argument("--workers", type=int, default=16)
A = ap.parse_args()

RUN = A.run
OUT = f"{EXP}/artifacts/sweep_out/{RUN}/{RUN}"
CACHE = f"{EXP}/artifacts/{RUN}_ost"; os.makedirs(CACHE, exist_ok=True)
jobs = json.load(open(f"{EXP}/artifacts/{RUN}_jobs.json"))
IFACES = {"8che_AE": ("A", "E"), "8che_BE": ("B", "E")}

nulls = len([1 for f in glob.glob(f"{EXP}/artifacts/{RUN}_logs/*.log")
             for l in open(f, errors="ignore") if "Found 0 templates" in l])
if nulls:
    sys.exit(f"REFUSING TO SCORE: {nulls} chain-loads reported 'Found 0 templates' -- arm is null")


def pose(a):
    """{iface: (iptm, dockq)} for one output pose, or None."""
    j, s, k = a
    name = j["name"]
    cif = f"{OUT}/{name}/seed_{s}/predictions/{name}_sample_{k}.cif"
    if not os.path.exists(cif):
        return None
    o = f"{CACHE}/{name}_{s}_{k}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset", "-c", "0-63", OST, "compare-structures", "-m", cif,
                        "-r", f"{GT}/8che-assembly1.cif", "-o", o, "--fault-tolerant",
                        "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq"],
                       env=ENV, capture_output=True, timeout=900)
    try:
        d = json.load(open(o))
        M = json.load(open(f"{OUT}/{name}/seed_{s}/predictions/"
                           f"{name}_summary_confidence_sample_{k}.json"))["chain_pair_iptm"]
    except Exception:
        return None
    qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
    got = {}
    for lab, (c1, c2) in IFACES.items():
        for i, t in enumerate(ifs):
            if {t[0], t[1]} == {c1, c2} and i < len(qs):        # BY NAME
                x, y = ord(t[2]) - 65, ord(t[3]) - 65
                if 0 <= x < len(M) and 0 <= y < len(M):
                    got[lab] = (M[x][y], qs[i])
                break
    return got or None


tasks = [(j, s, k) for j in jobs for s in j["seeds"] for k in SAMPLES]
with ThreadPoolExecutor(A.workers) as ex:
    res = list(ex.map(pose, tasks))

byjob, byseed = {}, {}
for (j, s, k), r in zip(tasks, res):
    if r:
        byjob.setdefault(j["name"], []).append(r)
        byseed.setdefault(j["name"], {}).setdefault(s, []).append(r)

# the donor's own DockQ on each interface, from the free-run cache
donor_dq = {}
for lab, (c1, c2) in IFACES.items():
    for f in glob.glob(f"{EXP}/artifacts/decoy_set_ost/{lab}_*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        s, k = os.path.basename(f)[:-5].split("_")[-2:]
        qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
        for i, t in enumerate(ifs):
            if {t[0], t[1]} == {c1, c2} and i < len(qs):
                donor_dq.setdefault((int(s), int(k)), {})[lab] = qs[i]
                break

rows = []
for j in jobs:
    ps = byjob.get(j["name"], [])
    if not ps:
        continue
    sd = byseed.get(j["name"], {})
    row = {"name": j["name"], "donor_seed": j["donor_seed"], "donor_sample": j["donor_sample"],
           "run_seed": j["seeds"][0], "seeds": list(j["seeds"]),
           "n_seeds_done": len(sd), "n_seeds_declared": len(j["seeds"]), "n_out": len(ps)}
    for lab in IFACES:
        v = [p[lab] for p in ps if lab in p]
        if not v:
            continue
        dq = sorted(x[1] for x in v)
        # per-seed: each seed's own 5 poses -> its best, and what ipTM would pick within that seed
        per = {}
        for s, pl in sorted(sd.items()):
            w = [p[lab] for p in pl if lab in p]
            if w:
                per[str(s)] = {"max": round(max(x[1] for x in w), 3),
                               "sel": round(max(w, key=lambda x: x[0])[1], 3),
                               "sel_iptm": round(max(x[0] for x in w), 3), "n": len(w)}
        row[lab] = {
            "donor_dockq": round(donor_dq.get((j["donor_seed"], j["donor_sample"]), {}).get(lab, float("nan")), 3),
            "out_max": round(max(dq), 3), "out_median": round(dq[len(dq) // 2], 3),
            "out_sel": round(max(v, key=lambda x: x[0])[1], 3),
            "sel_iptm": round(max(x[0] for x in v), 3),
            "frac_acceptable": round(sum(1 for x in dq if x >= THRESH) / len(dq), 3),
            "per_seed": per}
    rows.append(row)

json.dump({"run": RUN, "n_donors": len(rows), "n_poses": sum(r["n_out"] for r in rows),
           "rows": rows}, open(f"{EXP}/artifacts/{RUN}_scores.json", "w"), indent=1)
full = sum(1 for r in rows if r["n_seeds_done"] == r["n_seeds_declared"])
print(f"{RUN}: {len(rows)} donors, {sum(r['n_out'] for r in rows)} poses scored, 0 null templates")
print(f"  seed completeness: {full}/{len(rows)} donors have all {rows[0]['n_seeds_declared']} seeds "
      f"(PARTIAL RUN -- numbers below are provisional)" if full < len(rows) else
      f"  seed completeness: all {len(rows)} donors complete")
for lab in IFACES:
    v = [r[lab] for r in rows if lab in r]
    print(f"\n  {lab}:  best out_max over all donors {max(x['out_max'] for x in v):.3f} | "
          f"donors whose 5 poses contain an acceptable one: "
          f"{sum(1 for x in v if x['frac_acceptable'] > 0)}/{len(v)}")
    print(f"    pooled selection (highest ipTM over ALL {sum(r['n_out'] for r in rows)} poses): "
          f"{max(v, key=lambda x: x['sel_iptm'])['out_sel']:.3f} "
          f"(ipTM {max(x['sel_iptm'] for x in v):.3f})")
print(f"\nWROTE {EXP}/artifacts/{RUN}_scores.json")
