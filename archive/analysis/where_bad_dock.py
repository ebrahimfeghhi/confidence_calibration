"""Where do the bad seeds dock the antigen on 7txw?

Antibody chains A,B have identical residue numbering across GT and all
predictions, so we can compare interface (paratope) residues directly by number
- no superposition. For each seed we report:
  * antigen interface size  (is there a real interface, or floating?)
  * antibody (A,B) residues contacted, and whether they sit in the variable
    domain (CDR/paratope, low resi) or the constant domain (high resi)
  * overlap of the antibody paratope with the NATIVE paratope.
"""
from pathlib import Path
import numpy as np

PRED = Path("/home/ebrahim/esm/foldbench_predictions/esmfold2fast_tts/7txw")
GT = Path("/home/ebrahim/FoldBench/examples/ground_truths/7txw-assembly1.cif")
GOOD = ["seed_10", "seed_14", "seed_19", "seed_33"]
BAD = ["seed_0", "seed_1", "seed_5", "seed_20", "seed_40", "baseline_seed"]
CUT = 8.0  # CA-CA contact cutoff (coarse interface proxy)


def parse(path):
    cols = {}; idx = 0; inl = False; out = {}
    for line in open(path):
        s = line.strip()
        if s.startswith("_atom_site."):
            cols[s.split(".", 1)[1]] = idx; idx += 1; inl = True; continue
        if inl and s.startswith(("ATOM", "HETATM")):
            p = s.split()
            if p[cols["label_atom_id"]] != "CA":
                continue
            ch = p[cols["label_asym_id"]]
            rid = int(p[cols["auth_seq_id"]])
            xyz = np.array([float(p[cols["Cartn_x"]]), float(p[cols["Cartn_y"]]),
                            float(p[cols["Cartn_z"]])])
            out.setdefault(ch, {})[rid] = xyz
        elif inl and s.startswith("#"):
            inl = False
    return out


def interface(st, ab, ag="C", cut=CUT):
    """Return (antibody resi set contacted, antigen resi set contacted)."""
    abk = list(st[ab]); abv = np.array([st[ab][k] for k in abk])
    ab_hit, ag_hit = set(), set()
    for rk, v in st[ag].items():
        d = np.sqrt(((abv - v) ** 2).sum(1))
        m = d < cut
        if m.any():
            ag_hit.add(rk)
            for i in np.where(m)[0]:
                ab_hit.add(abk[i])
    return ab_hit, ag_hit


def summ(resset):
    if not resset:
        return "  none"
    r = sorted(resset)
    return f"n={len(r):2d} range {min(r):3d}-{max(r):3d} median {int(np.median(r)):3d}"


gt = parse(GT)
# native paratope (per antibody chain) from GT itself
nat = {}
for ch in ("A", "B"):
    ab_hit, ag_hit = interface(gt, ch)
    nat[ch] = ab_hit
print("NATIVE paratope (GT):")
for ch in ("A", "B"):
    print(f"  chain {ch}: {summ(nat[ch])}")

print(f"\n{'seed':14s} {'antigen-iface':>13s}  {'chainA paratope':<34s} {'chainB paratope':<34s} {'A-overlap w/native':>18s}")
for grp, seeds in (("GOOD", GOOD), ("BAD", BAD)):
    print(f"-- {grp} --")
    for seed in seeds:
        f = PRED / seed / "predictions" / "7txw_sample_0.cif"
        if not f.exists():
            continue
        st = parse(f)
        aA, gA = interface(st, "A")
        aB, gB = interface(st, "B")
        ag_total = gA | gB
        ov = len(aA & nat["A"])
        print(f"{seed:14s} {len(ag_total):>13d}  {summ(aA):<34s} {summ(aB):<34s} {ov:>10d}/{len(nat['A']):d}")
