#!/usr/bin/env python
"""Re-examine the generation failures using INTERFACE-LOCAL fold quality, not whole-chain.

An earlier comparison concluded that the 4 generation failures native templating rescued had worse
folds (median whole-chain lDDT 0.885) than the 11 it did not move (0.931) -- i.e. templating helps
where there is fold error to correct. That used WHOLE-CHAIN lDDT.

8oxw then showed why that is unsafe: its heavy chain scores 0.917 whole-chain but 0.711 over the 14
paratope residues. Averaging across 225 residues hid a large error in the only region that matters,
and native templating there produced a 9x increase in acceptable poses. The same masking could apply
to the "unmoved" group -- their folds may be fine globally and poor at the paratope, which would
invalidate the earlier conclusion.

This recomputes both groups interface-locally: for each chain, lDDT restricted to the residues within
5 A of the partner in the native complex, alongside the whole-chain value for comparison.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/genfail_iface_folds.py
Writes: artifacts/genfail_iface_folds.json
"""
import csv, json, os, subprocess, sys, difflib, statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemmi
from decoy_set import OST, ENV, PRED, EXP
from fold_quality import TLC, nat_res, nat_pdb, iface_idx, kind

W = f"{EXP}/artifacts/genfail_iface"; os.makedirs(W, exist_ok=True)
OUTJ = f"{EXP}/artifacts/genfail_iface_folds.json"
SEEDS = range(101, 116)          # 15 poses per chain; enough for a median, cheap


def measure(pdb, ch, other):
    """(median whole-chain lDDT, median interface-local lDDT, n interface residues)."""
    seq = "".join(TLC[r.name] for r in nat_res(pdb, ch))
    ref = nat_pdb(pdb, ch)
    idx = iface_idx(pdb, ch, other)
    g, l = [], []
    for s in SEEDS:
        cif = f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_0.cif"
        if not os.path.exists(cif):
            continue
        st = gemmi.read_structure(cif); st.setup_entities()
        best, br = None, -1
        for c in st[0]:
            rs = [r for r in c if r.name in TLC]
            if not rs: continue
            r_ = difflib.SequenceMatcher(None, "".join(TLC[x.name] for x in rs), seq,
                                         autojunk=False).ratio()
            if r_ > br: br, best = r_, rs
        if best is None:
            continue
        mp = f"{W}/m_{pdb}_{ch}_{s}.pdb"
        ns = gemmi.Structure(); ns.add_model(gemmi.Model("1")); cc = gemmi.Chain("A")
        for r in best: cc.add_residue(r)
        ns[0].add_chain(cc); ns.setup_entities(); ns.write_pdb(mp)
        o = f"{W}/ld_{pdb}_{ch}_{s}.json"
        if not (os.path.exists(o) and os.path.getsize(o)):
            subprocess.run(["taskset", "-c", "0-63", OST, "compare-structures", "-m", mp, "-r", ref,
                            "-o", o, "--fault-tolerant", "--min-pep-length", "4",
                            "--lddt", "--local-lddt"], env=ENV, capture_output=True, timeout=600)
        try:
            d = json.load(open(o))
        except Exception:
            continue
        if d.get("lddt"): g.append(d["lddt"])
        ll = d.get("local_lddt") or {}
        aln = (d.get("aln") or [None])[0]
        if not aln: continue
        p = aln.split("\n"); ri = mi = 0; per = {}
        for a, b in zip(p[1], p[3]):
            if a != "-" and b != "-":
                v = ll.get(f"A.{mi + 1}.")
                if v is not None: per[ri] = v
            if a != "-": ri += 1
            if b != "-": mi += 1
        vals = [per[i] for i in idx if i in per]
        if vals: l.append(statistics.mean(vals))
    return (statistics.median(g) if g else None,
            statistics.median(l) if l else None, len(idx))


scores = json.load(open(f"{EXP}/artifacts/native_arm_scores.json"))
gf = {k: v for k, v in scores.items() if v["free_status"] == "no_acceptable_pose"}
pairs = {}
for r in csv.DictReader(open("/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv")):
    pdb = r["pdb_id"].split("-")[0]
    lab = f"{pdb}_{r['interface_chain_id_1']}{r['interface_chain_id_2']}"
    if lab in gf:
        pairs[lab] = (pdb, r["interface_chain_id_1"], r["interface_chain_id_2"])

rows = []
print(f"{'interface':11} {'resc':>5} {'chain':6} {'role':8} {'whole':>7} {'iface':>7} {'drop':>7} {'n_res':>6}")
for lab, (pdb, c1, c2) in sorted(pairs.items()):
    resc = gf[lab]["rescued"]
    for ch, other in ((c1, c2), (c2, c1)):
        try:
            seq = "".join(TLC[r.name] for r in nat_res(pdb, ch))
        except Exception:
            continue
        role = kind(seq)
        gl, il, n = measure(pdb, ch, other)
        if gl is None or il is None:
            continue
        rows.append(dict(interface=lab, rescued=resc, chain=ch, role=role,
                         whole=round(gl, 3), iface=round(il, 3), drop=round(gl - il, 3), n_res=n))
        print(f"{lab:11} {'YES' if resc else '-':>5} {ch:6} {role:8} {gl:7.3f} {il:7.3f} "
              f"{gl - il:+7.3f} {n:6d}", flush=True)

json.dump(rows, open(OUTJ, "w"), indent=1)
print()
for lab, sel in (("RESCUED by folds", True), ("unmoved by folds", False)):
    for role in ("HEAVY", "LIGHT", "ANTIGEN"):
        v = [r for r in rows if r["rescued"] == sel and r["role"] == role]
        if not v: continue
        print(f"  {lab:18} {role:8} n={len(v):2d}  whole {statistics.median(r['whole'] for r in v):.3f}"
              f"   interface {statistics.median(r['iface'] for r in v):.3f}"
              f"   drop {statistics.median(r['drop'] for r in v):+.3f}")
print(f"\nWROTE {OUTJ}")
