"""Why is the B2 basin confidently wrong on 7txw?

B2 (upper-blue loop16 cluster) has interface ipTM ~= the correct basin (0.50 vs 0.54)
but DockQ ~0. This script asks WHAT the confidence head is rewarding, per basin:

  - antigen fold quality      : mean pLDDT of chain C  (well-folded antigen?)
  - interface certainty       : mean PAE over antibody<->antigen CONTACT pairs (low=confident)
  - interface size            : # antigen residues within 8A of antibody (big buried iface?)
  - paratope location         : median antibody resi contacted + overlap w/ NATIVE paratope
                                (low resi = CDR/variable/correct; high = framework decoy)
  - epitope correctness       : overlap of contacted antigen residues w/ NATIVE epitope

If B2 = high antigen pLDDT + low interface PAE + large interface but LOW native overlap,
the head is rewarding a clean, well-packed, self-consistent but WRONG-epitope dock.

sample_0 per seed (geometry + its matching plddt/pae). numpy only.
"""
from pathlib import Path
import numpy as np

R = Path("/home/ebrahim/esm/foldbench_predictions/7txw_reps/7txw")
GT = Path("/home/ebrahim/FoldBench/examples/ground_truths/7txw-assembly1.cif")
SP = {"A": slice(0, 219), "B": slice(219, 438), "C": slice(438, 609)}
AB = slice(0, 438)
CUT = 8.0
BASINS = {
    "G":  [10, 14, 33, 39],
    "B1": [1, 2, 3, 4, 5, 7, 8, 9, 11, 16, 17, 18, 19, 20, 21, 22, 23, 26, 28, 29, 30,
           31, 32, 34, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49, 52, 55, 56, 57, 58, 59],
    "B2": [0, 6, 12, 13, 15, 24, 27, 35, 43, 47, 50, 53, 54],
}


def parse(path):
    """CA: ordered xyz [N,3], chain list, auth resi list."""
    cols, idx, inl = {}, 0, False
    xyz, ch, rid = [], [], []
    for line in open(path):
        s = line.strip()
        if s.startswith("_atom_site."):
            cols[s.split(".", 1)[1]] = idx; idx += 1; inl = True; continue
        if inl and s.startswith(("ATOM", "HETATM")):
            p = s.split()
            if p[cols["label_atom_id"]] != "CA":
                continue
            ch.append(p[cols["label_asym_id"]])
            rid.append(int(p[cols["auth_seq_id"]]))
            xyz.append((float(p[cols["Cartn_x"]]), float(p[cols["Cartn_y"]]),
                        float(p[cols["Cartn_z"]])))
        elif inl and s.startswith("#"):
            inl = False
    return np.asarray(xyz, np.float32), np.array(ch), np.array(rid)


def native_interface():
    xyz, ch, rid = parse(GT)
    print("GT chains:", {c: int((ch == c).sum()) for c in sorted(set(ch))})
    ab = np.isin(ch, ["A", "B"]); ag = ch == "C"
    d = np.linalg.norm(xyz[ab][:, None] - xyz[ag][None], axis=2)
    m = d < CUT
    nat_para = set(zip(ch[ab][m.any(1)].tolist(), rid[ab][m.any(1)].tolist()))
    nat_epi = set(rid[ag][m.any(0)].tolist())
    return nat_para, nat_epi


def seed_stats(seed, nat_para, nat_epi):
    f = R / f"seed_{seed}" / "predictions" / "7txw_sample_0.cif"
    xyz, ch, rid = parse(f)
    plddt = np.load(R / f"seed_{seed}" / "predictions" / "7txw_plddt_sample_0.npy")
    pae = np.load(R / f"seed_{seed}" / "predictions" / "7txw_pae_sample_0.npy")
    abm = np.arange(438); agm = np.arange(438, 609)
    d = np.linalg.norm(xyz[abm][:, None] - xyz[agm][None], axis=2)
    m = d < CUT
    ab_hit = abm[m.any(1)]; ag_hit = agm[m.any(0)]
    para = set(zip(ch[ab_hit].tolist(), rid[ab_hit].tolist()))
    epi = set(rid[ag_hit].tolist())
    iface_pae = pae[np.ix_(ab_hit, ag_hit)].mean() if len(ab_hit) and len(ag_hit) else np.nan
    return dict(
        ag_plddt=float(plddt[438:].mean()),
        iface_pae=float(iface_pae),
        iface_n=int(len(ag_hit)),
        para_med=float(np.median(rid[ab_hit])) if len(ab_hit) else np.nan,
        para_ov=len(para & nat_para) / max(len(nat_para), 1),
        epi_ov=len(epi & nat_epi) / max(len(nat_epi), 1),
    )


def main():
    nat_para, nat_epi = native_interface()
    print(f"native paratope n={len(nat_para)}  native epitope n={len(nat_epi)} "
          f"(resi {sorted(nat_epi)[:3]}..{sorted(nat_epi)[-3:]})\n")
    keys = ["ag_plddt", "iface_pae", "iface_n", "para_med", "para_ov", "epi_ov"]
    hdr = {"ag_plddt": "agC_pLDDT", "iface_pae": "iface_PAE", "iface_n": "iface_sz",
           "para_med": "para_medR", "para_ov": "para_ov%", "epi_ov": "epi_ov%"}
    print(f"{'basin':5s} {'n':>3s} " + " ".join(f"{hdr[k]:>10s}" for k in keys))
    for nm, mem in BASINS.items():
        rows = [seed_stats(s, nat_para, nat_epi) for s in mem]
        agg = {k: np.nanmean([r[k] for r in rows]) for k in keys}
        line = " ".join(
            (f"{agg[k]*100:10.1f}" if k.endswith("_ov") else f"{agg[k]:10.2f}")
            for k in keys)
        print(f"{nm:5s} {len(mem):3d} {line}")
    print("\nlow iface_PAE + high agC_pLDDT + large iface_sz but low para_ov%/epi_ov% "
          "=> confident, well-packed, WRONG-epitope dock")


if __name__ == "__main__":
    main()
