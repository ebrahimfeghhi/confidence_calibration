#!/usr/bin/env python
"""Can chain_gPDE select accurate folds where pLDDT cannot?

pLDDT is per-residue and local ("is this residue's environment right?"). PDE is pairwise and
distance-based ("is the distance between these two residues right?"), contact-probability weighted:
    gpde = sum(pde * contact_probs) / sum(contact_probs)
`chain_gpde` is the intra-chain version, i.e. a fold-quality estimate, and it is already saved in every
summary_confidence JSON. It is a DIFFERENT head from pLDDT and can in principle penalise the failure
pLDDT is blind to: locally clean geometry with a wrong long-range arrangement.

gPDE is an ERROR, so lower is better -- selection takes the argmin.

LIMITATION: only the chain-level scalar is saved, not the per-residue PDE matrix, so this can only be
tested as a WHOLE-CHAIN criterion (the analogue of global pLDDT), never an interface-local one. Testing
an interface-local PDE would need inference re-run dumping `token_pair_pde`.

Compares, per chain, over the 250 free-run poses:
    - correlation of each criterion with true global and interface lDDT
    - REGRET: interface lDDT given up by selecting with that criterion vs the best fold available
against the pLDDT baselines already in artifacts/fold_selection_table.json.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/gpde_fold_selection.py
Writes: artifacts/gpde_fold_selection.json
"""
import json, os, sys, difflib, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gemmi
from decoy_set import EXP, PRED, SEEDS, SAMPLES
from fold_quality import TLC, nat_res, kind

tbl = json.load(open(f"{EXP}/artifacts/fold_selection_table.json"))
dec = json.load(open(f"{EXP}/artifacts/decoy_set_101_150.json"))
DPDB = {dec["per_interface"][l]["pdb"] for l in dec["decoys"]}
GPDB = {e["pdb"] for e in json.load(open(f"{EXP}/artifacts/safety15.json"))}


def chain_index_map(pdb, native_chains_wanted):
    """native chain -> index into the chain_gpde list, via sequence matching on one representative pose.

    Chain order is consistent across poses of the same pdb, so this is done once per pdb."""
    for s in SEEDS:
        cif = f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_0.cif"
        if os.path.exists(cif):
            break
    else:
        return {}
    st = gemmi.read_structure(cif); st.setup_entities()
    mchains = [(i, "".join(TLC[r.name] for r in c if r.name in TLC)) for i, c in enumerate(st[0])]
    mchains = [(i, s_) for i, s_ in mchains if s_]
    out = {}
    for nc in native_chains_wanted:
        nseq = "".join(TLC[r.name] for r in nat_res(pdb, nc))
        best = max(mchains, key=lambda x: difflib.SequenceMatcher(None, x[1], nseq,
                                                                 autojunk=False).ratio())
        out[nc] = best[0]
    return out


rows = []
for pdb, rec in sorted(tbl.items()):
    want = [r["native_chain"] for r in rec.values()]
    idx = chain_index_map(pdb, want)
    # chain_gpde per (seed, sample)
    gp = {}
    for s in SEEDS:
        for k in SAMPLES:
            f = f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_summary_confidence_sample_{k}.json"
            if not os.path.exists(f):
                continue
            try:
                gp[(s, k)] = json.load(open(f)).get("chain_gpde")
            except Exception:
                pass
    for q, r in rec.items():
        nc = r["native_chain"]
        ci = idx.get(nc)
        if ci is None:
            continue
        P = []
        for p in r["poses"]:
            g = gp.get((p[0], p[1]))
            if g is None or ci >= len(g) or any(x is None for x in p[2:]):
                continue
            P.append((p[2], p[3], p[4], p[5], g[ci]))     # gl, il, gp, ip, gpde
        if len(P) < 50:
            continue
        A = np.array(P)
        gl, il, gplddt, iplddt, gpde = A[:, 0], A[:, 1], A[:, 2], A[:, 3], A[:, 4]
        best_il = il.max()
        rows.append(dict(
            pdb=pdb, chain=nc, n=len(P),
            group="decoy" if pdb in DPDB else ("good" if pdb in GPDB else "?"),
            role=kind("".join(TLC[x.name] for x in nat_res(pdb, nc))),
            # gPDE is an error: negate so "higher is better" for all criteria
            r_gpde_vs_iface=float(np.corrcoef(-gpde, il)[0, 1]),
            r_gpde_vs_global=float(np.corrcoef(-gpde, gl)[0, 1]),
            r_gplddt_vs_global=float(np.corrcoef(gplddt, gl)[0, 1]),
            r_iplddt_vs_iface=float(np.corrcoef(iplddt, il)[0, 1]),
            regret_gpde=float(best_il - il[int(np.argmin(gpde))]),
            regret_gplddt=float(best_il - il[int(np.argmax(gplddt))]),
            regret_iplddt=float(best_il - il[int(np.argmax(iplddt))]),
        ))

json.dump(rows, open(f"{EXP}/artifacts/gpde_fold_selection.json", "w"), indent=1)


def show(name, sel):
    v = [x for x in rows if sel(x)]
    if not v:
        return
    med = lambda k: statistics.median(x[k] for x in v)
    print(f"  {name:30} n={len(v):3d} | corr with true fold accuracy:  "
          f"gPDE {med('r_gpde_vs_iface'):+.3f}   global pLDDT {med('r_gplddt_vs_global'):+.3f}   "
          f"iface pLDDT {med('r_iplddt_vs_iface'):+.3f}")
    print(f"  {'':30}        | interface-lDDT regret:        "
          f"gPDE {med('regret_gpde'):.3f}   global pLDDT {med('regret_gplddt'):.3f}   "
          f"iface pLDDT {med('regret_iplddt'):.3f}")


print("Can gPDE pick better folds than pLDDT? (median over chains; regret = interface lDDT lost vs oracle)\n")
show("ALL chains", lambda x: True)
show("DECOY interfaces", lambda x: x["group"] == "decoy")
show("GOOD interfaces", lambda x: x["group"] == "good")
show("antibody chains in DECOYS", lambda x: x["group"] == "decoy" and x["role"] in ("HEAVY", "LIGHT"))
print(f"\nWROTE {EXP}/artifacts/gpde_fold_selection.json")
