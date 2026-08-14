#!/usr/bin/env python
"""Does a donor's FOLD quality predict the poses it produces? (all 250 donors of an allpose arm)

WHY. Donor DockQ barely predicts output quality (r = 0.12/0.16 on 8che where templating WINS, vs
0.47/0.42 on 8slb where it FAILS -- see RESULTS.md). That is expected from the code: multichain_mask
(Protenix pairformer.py:1013,1062) zeroes every cross-chain template feature, so a donor can only
transmit INTRA-CHAIN geometry, never a binding arrangement. The quantity that could actually matter is
therefore how well the donor is FOLDED -- globally, and specifically at the residues that do the
binding.

MEASURES, per donor pose (the free-run pose that became the template):
  global_lddt    residue-weighted mean of per-chain lDDT over all protein chains
  paratope_lddt  mean per-residue lDDT over the antibody residues contacting the antigen in the CRYSTAL
  epitope_lddt   same over the antigen residues contacting the antibody
  mean_pe        mean(paratope_lddt, epitope_lddt)
each correlated against that donor's out_max (best DockQ among the 50 poses it produced).

CHAINS ARE SCORED IN ISOLATION. Scoring a chain inside the intact complex lets inter-chain contacts
dominate and just re-measures docking -- on one 8fas pose, isolated chain A scored lDDT 0.933 and the
same chain in-complex scored 0.294. Isolation is what makes this a fold measurement.

Reuses the verified helpers in donor_iface_lddt_exact.py (chain_lddt: OST --lddt --local-lddt,
single-chain model vs single-chain native matched by sequence; iface_idx_pair: 5 A heavy-atom contact
residues from the crystal).

EVERY VALUE IS CACHED so this never has to be recomputed: per-pose OST output under
artifacts/donor_iface_exact/, and the tidy per-donor table at artifacts/<run>_donor_foldq.json
(re-running skips donors already present).

Usage:
  .../python scripts/donor_fold_quality_allpose.py --run allpose9jbq10 --pdb 9jbq \
      --ifaces 9jbq_AC:A:C --workers 8
Writes: artifacts/<run>_donor_foldq.json
"""
import argparse, json, os, sys, statistics as st
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from donor_iface_lddt_exact import chain_lddt, iface_idx_pair
from decoy_set import EXP
from fold_quality import nat_res

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="allpose arm, e.g. allpose9jbq10")
ap.add_argument("--pdb", required=True)
ap.add_argument("--ifaces", required=True,
                help="comma-separated label:antibodyChain:antigenChain (ORDER MATTERS: "
                     "paratope is computed on the FIRST chain, epitope on the SECOND)")
ap.add_argument("--workers", type=int, default=8)
A = ap.parse_args()

ART = f"{EXP}/artifacts"
OUTJ = f"{ART}/{A.run}_donor_foldq.json"
scores = json.load(open(f"{ART}/{A.run}_scores.json"))["rows"]

IF = []
for spec in A.ifaces.split(","):
    lab, ab, ag = spec.split(":")
    IF.append((lab, ab, ag))

# all protein chains of the native, for the global (whole-fold) measure
ALL_CH = []
for lab, ab, ag in IF:
    for c in (ab, ag):
        if c not in ALL_CH:
            ALL_CH.append(c)

# interface residue sets, from the CRYSTAL, fixed for every donor
idxs, nres = {}, {}
for lab, ab, ag in IF:
    idxs[(lab, "paratope")] = iface_idx_pair(A.pdb, ab, ag)
    idxs[(lab, "epitope")] = iface_idx_pair(A.pdb, ag, ab)
    nres[lab] = (len(idxs[(lab, "paratope")]), len(idxs[(lab, "epitope")]))
    print(f"{lab}: paratope {nres[lab][0]} residues on chain {ab} | "
          f"epitope {nres[lab][1]} residues on chain {ag}", flush=True)

prev = {}
if os.path.exists(OUTJ):
    prev = {(r["donor_seed"], r["donor_sample"]): r for r in json.load(open(OUTJ))["donors"]}
    print(f"resuming: {len(prev)} donors already computed", flush=True)

chain_len = {c: len(nat_res(A.pdb, c)) for c in ALL_CH}


def one(row):
    s, k = int(row["donor_seed"]), int(row["donor_sample"])
    if (s, k) in prev:
        return prev[(s, k)]
    rec = {"donor_seed": s, "donor_sample": k, "name": row["name"], "chains": {}}
    # Resolve the residue set PER (interface, chain): a chain can play different roles in different
    # interfaces (8slb chain A is the antigen for BOTH interfaces; B and C are separate antibody
    # chains), so keying off IF[0] would score a chain against another chain's residue indices.
    # The OST cache tag is keyed by CHAIN only, so the expensive call runs once per chain and the
    # second lookup just averages a different residue subset from the same cached output.
    for lab, ab, ag in IF:
        for c, role in ((ab, "paratope"), (ag, "epitope")):
            g, il = chain_lddt(A.pdb, c, s, k, idxs[(lab, role)], f"{A.run}_{s}_{k}_{c}")
            rec["chains"].setdefault(c, {})["whole"] = g
            rec[f"{lab}_{role}_lddt"] = il
        p, e = rec.get(f"{lab}_paratope_lddt"), rec.get(f"{lab}_epitope_lddt")
        rec[f"{lab}_mean_pe"] = (st.mean([p, e]) if (p is not None and e is not None) else None)
    tot = sum(chain_len[c] for c in ALL_CH if rec["chains"].get(c, {}).get("whole") is not None)
    rec["global_lddt"] = (sum(rec["chains"][c]["whole"] * chain_len[c] for c in ALL_CH
                              if rec["chains"].get(c, {}).get("whole") is not None) / tot) if tot else None
    return rec


with ThreadPoolExecutor(A.workers) as ex:
    donors = list(ex.map(one, scores))
json.dump({"run": A.run, "pdb": A.pdb, "ifaces": [l for l, _, _ in IF],
           "n_iface_res": nres, "donors": donors}, open(OUTJ, "w"), indent=1)
print(f"\nWROTE {OUTJ}  ({len(donors)} donors)")


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3 or x[m].std() == 0 or y[m].std() == 0:
        return float("nan"), int(m.sum())
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())


byname = {r["name"]: r for r in donors}
for lab, ab, ag in IF:
    out_max = [r[lab]["out_max"] if lab in r else np.nan for r in scores]
    dq = [r[lab]["donor_dockq"] if lab in r else np.nan for r in scores]
    print(f"\n=== {lab} ===   (n donors = {len(scores)})")
    print(f"  {'predictor':<22}{'r vs out_max':>14}{'n':>6}")
    for key, label in [("global_lddt", "global lDDT"),
                       (f"{lab}_paratope_lddt", "paratope lDDT"),
                       (f"{lab}_epitope_lddt", "epitope lDDT"),
                       (f"{lab}_mean_pe", "mean(para,epi) lDDT")]:
        v = [byname.get(r["name"], {}).get(key, np.nan) for r in scores]
        v = [np.nan if x is None else x for x in v]
        r_, n_ = corr(v, out_max)
        print(f"  {label:<22}{r_:>14.3f}{n_:>6}")
    r_, n_ = corr([np.nan if x is None else x for x in dq], out_max)
    print(f"  {'donor DockQ (ref)':<22}{r_:>14.3f}{n_:>6}")
