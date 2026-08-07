#!/usr/bin/env python
"""Per-TEMPLATE response table for the leader20 arm: does a template's DockQ response separate
GOOD donors from DECOY donors, using no native template at any point?

THE QUESTION. Templating on a good pose raises max DockQ (dockqdonor: median +0.046, 19/25 up);
templating on the decoy's own fold lowers it (topiptm26: median -0.222, 7/25 up). If that asymmetry
holds per template, then "how much did this template's runs improve" is a LABEL-FREE signal for which
candidate pose is right -- you template each candidate and keep the one that responds. leader20 is
already exactly this experiment: 20 candidate templates per interface, each run separately (5 seeds),
each with a KNOWN donor quality (the leader's own DockQ vs native) that the signal never sees.

UNIT OF ANALYSIS = one template run (interface x leader rank), NOT one arm. ~454 observations.

TWO POSE-COUNT TRAPS, both handled:
  1. A leader template has 5 seeds x 5 samples = 25 poses; the free run has 250. max() over 25 is
     lower than max() over 250 BY CONSTRUCTION, so `best_dockq - free_best_dockq` is biased negative
     and would fake the very asymmetry we are testing. Fixed with `free_best_matched`: the median of
     max(DockQ) over 2000 random 25-pose subsamples of the free pool (rng seeded, reproducible).
  2. frac_acceptable is a RATE and is pose-count safe, so it is the primary outcome.

DONOR LABEL is the leader's own DockQ vs native (`dockq_gt` in leader20_index.json), thresholded at
0.23. This is the oracle label the response signal is being tested against -- it is used ONLY to
evaluate, never to compute the signal.

SEPARATION is measured WITHIN interface (AUC of response vs donor label over that interface's
leaders), because interfaces differ enormously in difficulty and pooling across them would mostly
measure which interfaces are easy. Interfaces whose leaders are all-good or all-bad have no AUC and
are reported as skipped, not silently dropped.

Reads only cached OST results (artifacts/leader20_ost/, artifacts/decoy_set_ost/) -- no new OST calls,
no GPU, safe to run alongside a live arm.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/leader_response.py [--md]
Writes: artifacts/leader_response.json
"""
import argparse, json, os, sys, statistics as st
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, SAMPLES, THRESH

ap = argparse.ArgumentParser()
ap.add_argument("--md", action="store_true")
ap.add_argument("--boot", type=int, default=2000, help="subsamples for the pose-count-matched baseline")
A = ap.parse_args()

LOST = f"{EXP}/artifacts/leader20_ost"
FOST = f"{EXP}/artifacts/decoy_set_ost"
POUT = f"{EXP}/artifacts/sweep_out/leader20/leader20"
index = json.load(open(f"{EXP}/artifacts/leader20_index.json"))
jobs = json.load(open(f"{EXP}/artifacts/leader20_jobs.json"))


def dockq_of(path, lab, want_chains=False):
    """DockQ for interface `lab` out of one cached OST json, matched BY NAME.

    With want_chains, also returns the MODEL chain letters OST paired to that interface, so ipTM can
    be indexed by the same identification rather than by a max over the chain_pair_iptm matrix (which
    picks the wrong pair on 3-chain complexes).
    """
    try:
        d = json.load(open(path))
    except Exception:
        return (None, None) if want_chains else None
    parts = lab.split("_", 1)[1]
    c1, c2 = (parts[0], parts[1:]) if len(parts) > 2 else (parts[0], parts[1])
    qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
    for i, t in enumerate(ifs):
        if {t[0], t[1]} == {c1, c2} and i < len(qs):
            return (qs[i], (t[2], t[3])) if want_chains else qs[i]
    return (None, None) if want_chains else None


def free_poses(lab):
    """Every free-run pose DockQ for this interface (expect 250 = 50 seeds x 5 samples).

    The free-run OST results are spread over THREE caches with two naming schemes: decoy_set_ost and
    extra_set_ost key on the INTERFACE label, abag_all_ost keys on the PDB alone (one json holding all
    of that pdb's interfaces). Looking in only one silently dropped 13 of 26 interfaces on the first
    run of this script. dockq_of matches the interface BY NAME, so the pdb-keyed cache is safe to use.
    """
    pdb = lab.split("_")[0]
    for pat in (f"{FOST}/{lab}_%d_%d.json",
                f"{EXP}/artifacts/extra_set_ost/{lab}_%d_%d.json",
                f"{EXP}/artifacts/abag_all_ost/{pdb}_%d_%d.json"):
        out = []
        for s in range(101, 151):
            for k in SAMPLES:
                v = dockq_of(pat % (s, k), lab)
                if v is not None:
                    out.append(v)
        if len(out) >= 200:                 # a real pool, not a handful of stragglers
            return out, pat.rsplit("/", 2)[-2]
    return [], None


def iptm_of(name, s, k):
    try:
        M = json.load(open(f"{POUT}/{name}/seed_{s}/predictions/"
                           f"{name}_summary_confidence_sample_{k}.json"))["chain_pair_iptm"]
    except Exception:
        return None
    return M


def sel_dockq(rows):
    """DockQ of the highest-ipTM pose among (iptm, dockq) rows; None if no ipTM was readable."""
    ok = [r for r in rows if r[0] is not None]
    return max(ok, key=lambda r: r[0])[1] if ok else None


by_iface = {}
for j in jobs:
    by_iface.setdefault(j["interface"], []).append(j)

rng = np.random.default_rng(0)
rows, per_iface = [], {}

for lab, js in sorted(by_iface.items()):
    fp, src = free_poses(lab)
    if not fp:
        print(f"{lab:11} NO cached free poses in any of the 3 caches -- SKIPPED", flush=True)
        continue
    if len(fp) != 250:
        print(f"{lab:11} free pool is {len(fp)}/250 poses (cache {src}) -- rates still valid, noted",
              flush=True)
    fa = np.asarray(fp)
    free_frac = float((fa >= THRESH).mean())
    free_best = float(fa.max())
    L = {x["rank"]: x for x in index[lab]["leaders"]}

    recs = []
    for j in sorted(js, key=lambda x: x["leader_rank"]):
        r = j["leader_rank"]
        pr = []
        for s in j["seeds"]:
            for k in SAMPLES:
                d, mc = dockq_of(f"{LOST}/{lab}_L{r:02d}_{s}_{k}.json", lab, want_chains=True)
                if d is None:
                    continue
                M, ip = iptm_of(j["name"], s, k), None
                if M and mc:                                   # index ipTM by the SAME chain pair
                    x, y = ord(mc[0]) - 65, ord(mc[1]) - 65
                    if 0 <= x < len(M) and 0 <= y < len(M):
                        ip = M[x][y]
                pr.append((ip, d))
        if not pr:
            continue
        da = np.asarray([p[1] for p in pr])
        n = len(da)
        # pose-count-matched free baseline: median max over `n`-pose subsamples of the free pool
        if n < len(fa):
            idx = rng.integers(0, len(fa), size=(A.boot, n))
            fb_matched = float(np.median(fa[idx].max(axis=1)))
        else:
            fb_matched = free_best
        lead = L.get(r, {})
        recs.append({
            "interface": lab, "leader_rank": r, "n_poses": n,
            "donor_dockq": lead.get("dockq_gt"), "donor_acceptable": bool(lead.get("acceptable")),
            "donor_iptm": lead.get("iptm"),
            "frac_acceptable": round(float((da >= THRESH).mean()), 4),
            "best_dockq": round(float(da.max()), 3),
            "median_dockq": round(float(np.median(da)), 3),
            "sel_dockq": (lambda v: round(v, 3) if v is not None else None)(sel_dockq(pr)),
            "free_frac_acceptable": round(free_frac, 4),
            "free_best_dockq": round(free_best, 3),
            "free_best_matched": round(fb_matched, 3),
            "d_frac": round(float((da >= THRESH).mean()) - free_frac, 4),
            "d_best_matched": round(float(da.max()) - fb_matched, 3),
            "d_best_raw": round(float(da.max()) - free_best, 3),
        })
    if not recs:
        continue
    rows += recs
    per_iface[lab] = recs

# ---- within-interface separation: does the response rank good donors above bad ones? ----
def auc(pos, neg):
    """P(random positive scores above random negative), ties = 0.5."""
    if not pos or not neg:
        return None
    w = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return w / (len(pos) * len(neg))


sep, skipped = {}, []
for m in ("d_frac", "d_best_matched", "median_dockq", "frac_acceptable"):
    vals = []
    for lab, recs in per_iface.items():
        pos = [r[m] for r in recs if r["donor_acceptable"]]
        neg = [r[m] for r in recs if not r["donor_acceptable"]]
        a = auc(pos, neg)
        if a is None:
            if m == "d_frac":
                skipped.append((lab, len(pos), len(neg)))
            continue
        vals.append((lab, a))
    sep[m] = {"per_interface": dict(vals), "n_interfaces": len(vals),
              "median_auc": round(st.median([v for _, v in vals]), 3) if vals else None,
              "mean_auc": round(st.mean([v for _, v in vals]), 3) if vals else None,
              "auc_above_half": sum(1 for _, v in vals if v > 0.5)}

good = [r for r in rows if r["donor_acceptable"]]
bad = [r for r in rows if not r["donor_acceptable"]]
out = {
    "n_template_runs": len(rows), "n_interfaces": len(per_iface),
    "n_good_donors": len(good), "n_bad_donors": len(bad),
    "pooled": {m: {"good_median": round(st.median([r[m] for r in good]), 4) if good else None,
                   "bad_median": round(st.median([r[m] for r in bad]), 4) if bad else None}
               for m in ("d_frac", "d_best_matched", "d_best_raw", "median_dockq")},
    "separation_within_interface": sep,
    "interfaces_skipped_no_contrast": skipped,
    "rows": rows,
}
json.dump(out, open(f"{EXP}/artifacts/leader_response.json", "w"), indent=1)

print(f"{len(rows)} template runs over {len(per_iface)} interfaces "
      f"({len(good)} good donors, {len(bad)} decoy/bad donors)\n")
print(f"{'response metric':22} {'good donors':>12} {'bad donors':>11} | {'median within-iface AUC':>23} {'n ifaces':>8}")
for m in ("d_frac", "d_best_matched", "median_dockq"):
    p = out["pooled"][m]; s = sep[m]
    print(f"{m:22} {str(p['good_median']):>12} {str(p['bad_median']):>11} | "
          f"{str(s['median_auc']):>23} {s['n_interfaces']:>8}")
print(f"\n  AUC 0.5 = response carries no information about donor quality; 1.0 = perfect separation.")
print(f"  d_best_raw (POSE-COUNT BIASED, shown only to expose the trap): "
      f"good {out['pooled']['d_best_raw']['good_median']}, bad {out['pooled']['d_best_raw']['bad_median']}")
if skipped:
    print(f"\n  {len(skipped)} interfaces had no good/bad contrast among their leaders (no AUC): "
          f"{', '.join(l for l, _, _ in skipped[:8])}{' ...' if len(skipped) > 8 else ''}")
print(f"\nWROTE {EXP}/artifacts/leader_response.json")
