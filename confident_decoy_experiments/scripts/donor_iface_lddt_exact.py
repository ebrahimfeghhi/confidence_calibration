#!/usr/bin/env python
"""EXACT per-interface lDDT of each dockqdonor donor pose (no union approximation).

WHY: donor_fold_vs_rescue.py showed that among interfaces where a rescue is achievable (native
succeeds), the donors dockqdonor FAILS on are worse folded at the binding site -- median interface
lDDT 0.539 vs 0.727 -- while whole-chain lDDT separates them not at all (0.857 vs 0.865). But that used
fold_selection_table.json, whose `iface_lddt` is UNION-based: for a chain sitting in two labelled
interfaces it scores residues contacting EITHER partner. Testing 8yxi_AC then includes chain C
residues that only touch B, diluting the exact quantity under test. 5 of the 9 native-only interfaces
are affected, so the reported spearman of +0.257 is a floor, not an estimate.

This recomputes the donor's binding-site lDDT restricted to the residues of THE INTERFACE UNDER TEST:
per-residue lDDT from OST (`--local-lddt`), single-chain model vs single-chain native matched by
sequence, then averaged over exactly the residues of that chain within 5 A of ITS partner.

CPU/OST only -- does not touch the GPU arms.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/donor_iface_lddt_exact.py
Writes: artifacts/donor_iface_lddt_exact.json
"""
import json, os, subprocess, sys, difflib, statistics as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gemmi
from decoy_set import OST, ENV, PRED, EXP, native_chains
from fold_quality import TLC, nat_res, nat_pdb

W = f"{EXP}/artifacts/donor_iface_exact"; os.makedirs(W, exist_ok=True)
OUTJ = f"{EXP}/artifacts/donor_iface_lddt_exact.json"

don = json.load(open(f"{EXP}/artifacts/dockq_arm_donors.json"))
dq = json.load(open(f"{EXP}/artifacts/dockqdonor_scores.json"))
nat = json.load(open(f"{EXP}/artifacts/native_arm_scores.json"))


def iface_idx_pair(pdb, ch, partner):
    """0-based indices into nat_res(pdb,ch) contacting THIS partner only."""
    D = native_chains(pdb)
    XA, IA = D[ch]; XB, _ = D[partner]
    order = {v: i for i, v in enumerate(sorted(set(IA), key=lambda x: int(x)))}
    m = np.linalg.norm(XA[:, None, :] - XB[None, :, :], axis=-1) < 5.0
    return {order[IA[i]] for i in np.where(m.any(1))[0]}


def chain_lddt(pdb, ch, seed, sample, idx, tag):
    """(whole-chain lDDT, lDDT over `idx` residues) for one chain of one pose."""
    seq = "".join(TLC[r.name] for r in nat_res(pdb, ch))
    ref = nat_pdb(pdb, ch)
    cif = f"{PRED}/{pdb}/seed_{seed}/predictions/{pdb}_sample_{sample}.cif"
    if not os.path.exists(cif):
        return None, None
    st_ = gemmi.read_structure(cif); st_.setup_entities()
    best, br = None, -1
    for c in st_[0]:
        rs = [r for r in c if r.name in TLC]
        if not rs: continue
        r_ = difflib.SequenceMatcher(None, "".join(TLC[x.name] for x in rs), seq,
                                     autojunk=False).ratio()
        if r_ > br: br, best = r_, rs
    if best is None:
        return None, None
    mp = f"{W}/m_{tag}.pdb"
    ns = gemmi.Structure(); ns.add_model(gemmi.Model("1")); cc = gemmi.Chain("A")
    for r in best: cc.add_residue(r)
    ns[0].add_chain(cc); ns.setup_entities(); ns.write_pdb(mp)
    o = f"{W}/ld_{tag}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset", "-c", "0-63", OST, "compare-structures", "-m", mp, "-r", ref,
                        "-o", o, "--fault-tolerant", "--min-pep-length", "4",
                        "--lddt", "--local-lddt"], env=ENV, capture_output=True, timeout=600)
    try:
        d = json.load(open(o))
    except Exception:
        return None, None
    g = d.get("lddt")
    ll = d.get("local_lddt") or {}
    aln = (d.get("aln") or [None])[0]
    if not aln:
        return g, None
    p = aln.split("\n"); ri = mi = 0; per = {}
    for a, b in zip(p[1], p[3]):
        if a != "-" and b != "-":
            v = ll.get(f"A.{mi + 1}.")
            if v is not None: per[ri] = v
        if a != "-": ri += 1
        if b != "-": mi += 1
    vals = [per[i] for i in idx if i in per]
    return g, (st.mean(vals) if vals else None)


# Guarded so this module can be IMPORTED for its verified chain_lddt()/iface_idx_pair() helpers
# (see donor_fold_quality_allpose.py) without executing the dockqdonor sweep below.
def spear(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v)
        for p_, i in enumerate(o): r[i] = p_ + 1
        return r
    rx, ry = rk(x), rk(y); mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0



def _main():
  rows = []
  print(f"{'interface':11} {'group':16} {'chain':6} {'whole':>7} {'iface(exact)':>13} {'n_res':>6}")
  for lab in sorted(set(dq) & set(don)):
      d = don[lab]; pdb = d["pdb"]; c1, c2 = d["interface"]
      grp = ("both" if dq[lab]["rescued"] and nat[lab]["rescued"] else
             "native_only" if nat[lab]["rescued"] else
             "dockqdonor_only" if dq[lab]["rescued"] else "neither")
      per_chain = {}
      for ch, partner in ((c1, c2), (c2, c1)):
          try:
              idx = iface_idx_pair(pdb, ch, partner)
          except Exception as e:
              print(f"{lab:11} {ch}: iface idx failed ({e})"); continue
          if not idx:
              continue
          g, il = chain_lddt(pdb, ch, d["seed"], d["sample"], idx, f"{lab}_{ch}")
          if il is None:
              continue
          per_chain[ch] = {"whole": round(g, 3) if g else None,
                           "iface_exact": round(il, 3), "n_iface_res": len(idx)}
          print(f"{lab:11} {grp:16} {ch:6} {g if g else 0:7.3f} {il:13.3f} {len(idx):6d}", flush=True)
      if per_chain:
          rows.append({"interface": lab, "pdb": pdb, "group": grp,
                       "donor_seed": d["seed"], "donor_sample": d["sample"],
                       "donor_dockq": d["donor_dockq"], "chains": per_chain,
                       "min_iface_exact": round(min(v["iface_exact"] for v in per_chain.values()), 3),
                       "mean_iface_exact": round(st.mean(v["iface_exact"] for v in per_chain.values()), 3),
                       "dockqdonor_rescued": dq[lab]["rescued"], "native_rescued": nat[lab]["rescued"]})
          json.dump(rows, open(OUTJ, "w"), indent=1)

  print("\ngroup medians, EXACT per-interface lDDT (min across the interface's two chains):")
  for g in ("both", "native_only", "dockqdonor_only", "neither"):
      v = [r for r in rows if r["group"] == g]
      if v:
          print(f"  {g:16} n={len(v):2d}  {st.median(r['min_iface_exact'] for r in v):.3f}")

  able = [r for r in rows if r["native_rescued"]]
  if able:
      b = [r for r in able if r["dockqdonor_rescued"]]; f = [r for r in able if not r["dockqdonor_rescued"]]
      print(f"\nRestricted to the {len(able)} native-rescuable interfaces:")
      print(f"  dockqdonor rescues  n={len(b):2d}  exact iface lDDT {st.median(r['min_iface_exact'] for r in b):.3f}")
      print(f"  dockqdonor fails    n={len(f):2d}  exact iface lDDT {st.median(r['min_iface_exact'] for r in f):.3f}")
      y = [1.0 if r["dockqdonor_rescued"] else 0.0 for r in able]
      print(f"  spearman(exact iface lDDT, rescued | native-rescuable) = "
            f"{spear([r['min_iface_exact'] for r in able], y):+.3f}   (union-based was +0.257)")
  print(f"\nWROTE {OUTJ}")


if __name__ == "__main__":
    _main()
