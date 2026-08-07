#!/usr/bin/env python
"""Do templated outputs sit NEAR the template, or do they move past it toward native?

The hypothesis under test: templating localizes the search around the template's distribution, so a
good template means heavy sampling around a good region. That predicts outputs cluster near the
template. But dockqdonor's donor is DockQ 0.529 while its outputs are ~0.950 -- outputs are far BETTER
than the template, which a pure "sample near the template" account does not obviously give.

This measures, per donor, three quantities on the SAME scale (DockQ):
  donor_vs_native    how good the template pose itself is
  output_vs_native   how good the poses it produced are
  output_vs_donor    POSE-POSE DockQ between each output and its own template = how localized the
                     search stayed. High => outputs are in the template's basin.

Three distinguishable outcomes:
  outputs cluster AT the donor      -> output_vs_native ~= donor_vs_native, output_vs_donor high
  outputs localize but IMPROVE      -> output_vs_native > donor_vs_native, output_vs_donor high
  outputs ignore the donor          -> output_vs_donor low

Pose-pose DockQ uses the same OST protocol with the DONOR POSE as reference instead of the native.

CHAIN-NAMING TRAP (this bit me once and silently dropped 87% of the data): when both structures are
predicted poses, OST reports dockq_interfaces in MODEL chain naming, not native naming, and enumerates
every chain pairing. Filtering those records on the native names {B,E} matched a handful of unrelated
pairings that happen to be lettered B and E. The correct route is to read the MODEL chain names for
the interface out of the model-vs-native comparison (already cached), then require BOTH the model and
reference chain pairs in the pose-pose record to equal that pair. Asserted, never assumed.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/localization.py [--run allpose8cheseed]
Writes: artifacts/localization_<run>.json
"""
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, PRED, OST, ENV, SAMPLES, THRESH

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="allpose8cheseed")
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--iface", default="8che_BE")
A = ap.parse_args()
PDB = "8che"
c1, c2 = A.iface.split("_")[1][0], A.iface.split("_")[1][1]
OUT = f"{EXP}/artifacts/sweep_out/{A.run}/{A.run}"
CACHE = f"{EXP}/artifacts/loc_{A.run}_ost"; os.makedirs(CACHE, exist_ok=True)
jobs = json.load(open(f"{EXP}/artifacts/{A.run}_jobs.json"))
scores = json.load(open(f"{EXP}/artifacts/{A.run}_scores.json"))["rows"]
byname = {r["name"]: r for r in scores}


def model_chains(name, s, k):
    """The MODEL chain names for this interface, read from the cached model-vs-native comparison."""
    f = f"{EXP}/artifacts/{A.run}_ost/{name}_{s}_{k}.json"
    try:
        d = json.load(open(f))
    except Exception:
        return None
    for t in (d.get("dockq_interfaces") or []):
        if {t[0], t[1]} == {c1, c2}:          # native names here -- this file IS vs the native
            return frozenset((t[2], t[3]))
    return None


def pp(a):
    """pose-pose DockQ between one output and its OWN donor pose (donor as reference)."""
    j, s, k = a
    name = j["name"]
    mdl = f"{OUT}/{name}/seed_{s}/predictions/{name}_sample_{k}.cif"
    ref = f"{PRED}/{PDB}/seed_{j['donor_seed']}/predictions/{PDB}_sample_{j['donor_sample']}.cif"
    if not (os.path.exists(mdl) and os.path.exists(ref)):
        return None
    o = f"{CACHE}/{name}_{s}_{k}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset", "-c", "0-63", OST, "compare-structures", "-m", mdl, "-r", ref,
                        "-o", o, "--fault-tolerant", "--min-pep-length", "4",
                        "--min-nuc-length", "4", "--dockq"], env=ENV, capture_output=True, timeout=900)
    try:
        d = json.load(open(o))
    except Exception:
        return None
    mc = model_chains(name, s, k)
    if mc is None:
        return None
    qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
    for i, t in enumerate(ifs):
        # BOTH sides are predicted poses -> both chain pairs are in MODEL naming
        if frozenset((t[2], t[3])) == mc and frozenset((t[0], t[1])) == mc and i < len(qs):
            return (name, qs[i])
    return None


tasks = [(j, j["seeds"][0], k) for j in jobs for k in SAMPLES]
with ThreadPoolExecutor(A.workers) as ex:
    res = [r for r in ex.map(pp, tasks) if r]
agg = {}
for name, v in res:
    agg.setdefault(name, []).append(v)

rows = []
for j in jobs:
    n = j["name"]
    if n not in agg or n not in byname or A.iface not in byname[n]:
        continue
    r = byname[n][A.iface]
    rows.append({"name": n, "donor_seed": j["donor_seed"], "donor_sample": j["donor_sample"],
                 "donor_vs_native": r["donor_dockq"],
                 "output_vs_native_max": r["out_max"], "output_vs_native_median": r["out_median"],
                 "output_vs_donor_median": round(float(np.median(agg[n])), 3),
                 "output_vs_donor_max": round(float(max(agg[n])), 3), "n": len(agg[n])})

D = np.array([r["donor_vs_native"] for r in rows])
Om = np.array([r["output_vs_native_median"] for r in rows])
S = np.array([r["output_vs_donor_median"] for r in rows])
m = ~np.isnan(D)
print(f"{A.run}, {A.iface}: {len(rows)} donors, {len(res)} output-vs-donor comparisons\n")
print(f"  output-vs-donor pose-pose DockQ (localization): median {np.median(S):.3f}, "
      f"IQR {np.percentile(S,25):.3f}-{np.percentile(S,75):.3f}")
print(f"  outputs in the donor's BASIN (pose-pose >= 0.23): "
      f"{int((S>=THRESH).sum())}/{len(S)} donors (by median output)")
print(f"  outputs NEAR-IDENTICAL to donor (pose-pose >= 0.8): {int((S>=0.8).sum())}/{len(S)}")
print(f"\n  donor_vs_native   median {np.nanmedian(D):.3f}")
print(f"  output_vs_native  median {np.median(Om):.3f}   (per-donor median pose)")
print(f"  outputs BETTER than their donor: {int((Om[m]>D[m]).sum())}/{int(m.sum())}")
from scipy.stats import spearmanr
print(f"\n  Spearman(donor_vs_native, output_vs_native_median) = {spearmanr(D[m],Om[m]).statistic:+.3f}")
print(f"  Spearman(output_vs_donor,  output_vs_native_median) = {spearmanr(S,Om).statistic:+.3f}")
top = sorted(rows, key=lambda r: -r["output_vs_native_max"])[:6]
print(f"\n  {'donor':16} {'donor vs nat':>12} {'out vs nat (med)':>17} {'out vs DONOR (med)':>19}")
for r in top:
    print(f"  seed{r['donor_seed']}/s{r['donor_sample']:<9} {r['donor_vs_native']:12.3f} "
          f"{r['output_vs_native_median']:17.3f} {r['output_vs_donor_median']:19.3f}")
json.dump({"run": A.run, "iface": A.iface, "rows": rows},
          open(f"{EXP}/artifacts/localization_{A.run}.json", "w"), indent=1)
print(f"\nWROTE {EXP}/artifacts/localization_{A.run}.json")
