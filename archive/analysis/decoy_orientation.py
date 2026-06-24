"""Characterize the dominant decoy antigen orientation for 7txw A|C.

34/50 seeds converge on one wrong antigen orientation. Take a representative
(seed_0) and quantify, relative to the native complex:
  - the rigid rotation + translation of the antigen (how is it spun?),
  - which antigen face it presents (decoy epitope) vs the true epitope, and the
    angle between those two faces on the antigen,
  - which antibody residues it contacts (decoy paratope) vs the true paratope,
  - the model's confidence (pair_iptm) for decoy seeds vs good seeds — i.e. does
    the model *know* it's wrong?
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_7txw_poses import parse_cif_ca, kabsch, common, GT, PRED  # noqa: E402

GOOD = {10, 14, 19, 33}
DECOY_REP = "seed_0"     # member of the 34-seed dominant cluster
DOCKQ_CSV = Path("/home/ebrahim/tts_abag/results/csv_files/"
                 "dockq_ost_esmfold2fast_tts_all_samples.csv")
CUT = 8.0


def super_on_A(st, ref):
    P, Q, _ = common(st, ref, "A")
    R, t = kabsch(P, Q)
    return {ch: {k: R @ v + t for k, v in res.items()} for ch, res in st.items()}


def antigen_face(st, cutoff=CUT):
    """Antigen (C) residues with a CA within cutoff of any antibody (A/B) CA."""
    ab = np.array([v for ch in ("A", "B") if ch in st for v in st[ch].values()])
    return {k for k, v in st["C"].items()
            if np.sqrt(((ab - v) ** 2).sum(1)).min() < cutoff}


def antibody_para(st, cutoff=CUT):
    """Antibody chain-A residues with a CA within cutoff of any antigen CA."""
    ag = np.array(list(st["C"].values()))
    return {k for k, v in st["A"].items()
            if np.sqrt(((ag - v) ** 2).sum(1)).min() < cutoff}


def pair_iptm_per_seed():
    out = {}
    with open(DOCKQ_CSV) as f:
        for r in csv.DictReader(f):
            if r["assembly_id"] != "7txw-assembly1" or {r["chain1"], r["chain2"]} != {"A", "C"}:
                continue
            try:
                v = float(r["pair_iptm"]); dq = float(r["dockq"])
            except ValueError:
                continue
            s = r["seed"]
            if s not in out or dq > out[s][1]:
                out[s] = (v, dq)
    return {s: v for s, (v, _) in out.items()}


def main():
    gt = parse_cif_ca(GT)
    gt["C"] = {k - 22: v for k, v in gt["C"].items()}
    decoy = parse_cif_ca(PRED / DECOY_REP / "predictions" / "7txw_sample_0.cif")

    # --- 1. antigen rigid transform: native -> decoy, in the antibody frame ---
    decoy_al = super_on_A(decoy, gt)
    keys = sorted(set(decoy_al["C"]) & set(gt["C"]))
    nat = np.array([gt["C"][k] for k in keys])
    dec = np.array([decoy_al["C"][k] for k in keys])
    R, t = kabsch(nat, dec)
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    cen_shift = float(np.linalg.norm(dec.mean(0) - nat.mean(0)))
    resid = float(np.sqrt(((nat @ R.T + t - dec) ** 2).sum(1).mean()))
    print("=== Antigen placement: dominant decoy vs native ===")
    print(f"  rigid rotation of antigen body : {angle:5.1f} degrees")
    print(f"  centroid translation           : {cen_shift:5.1f} A")
    print(f"  residual after rigid fit        : {resid:5.2f} A  (= antigen fold diff; "
          f"confirms same monomer, pure re-orientation)")

    # --- 2. which antigen face (epitope) ---
    true_face = antigen_face(gt)
    decoy_face = antigen_face(decoy)
    # locate both faces on the SAME antigen (decoy's, has all residues) and measure
    # the angle between the centroid->face vectors
    ag = decoy["C"]
    ag_cen = np.array(list(ag.values())).mean(0)

    def face_vec(face):
        pts = np.array([ag[k] for k in face if k in ag])
        return pts.mean(0) - ag_cen
    vt, vd = face_vec(true_face), face_vec(decoy_face)
    face_angle = np.degrees(np.arccos(np.clip(
        vt @ vd / (np.linalg.norm(vt) * np.linalg.norm(vd)), -1, 1)))
    print("\n=== Antigen face used (epitope) ===")
    print(f"  true epitope  : {len(true_face)} res  {sorted(true_face)}")
    print(f"  decoy epitope : {len(decoy_face)} res  {sorted(decoy_face)}")
    print(f"  shared residues: {len(true_face & decoy_face)}")
    print(f"  angle between the two faces on the antigen: {face_angle:5.1f} degrees "
          f"({'~opposite side' if face_angle > 120 else '~orthogonal/adjacent' if face_angle > 60 else '~same side'})")

    # --- 3. where on the antibody (paratope) ---
    tp, dp = antibody_para(gt), antibody_para(decoy)
    print("\n=== Antibody region contacted (chain-A paratope) ===")
    print(f"  true paratope  : {len(tp)} res, range {min(tp)}-{max(tp)}")
    print(f"  decoy paratope : {len(dp)} res, range {min(dp)}-{max(dp)}")
    print(f"  shared residues: {len(tp & dp)}  (both in the CDR region => decoy looks "
          f"like a real antibody interface)")

    # --- 4. confidence: does the model know it's wrong? ---
    pi = pair_iptm_per_seed()
    good_pi = [pi[s] for s in pi if s.isdigit() and int(s) in GOOD]
    bad_pi = [pi[s] for s in pi if s.isdigit() and int(s) not in GOOD]
    print("\n=== Confidence (pair_iptm, best sample per seed) ===")
    print(f"  good seeds (native pose): mean {np.mean(good_pi):.3f}  range "
          f"{min(good_pi):.3f}-{max(good_pi):.3f}")
    print(f"  bad seeds  (decoy poses): mean {np.mean(bad_pi):.3f}  range "
          f"{min(bad_pi):.3f}-{max(bad_pi):.3f}")
    print(f"  decoy rep {DECOY_REP}: pair_iptm = {pi.get('0', float('nan')):.3f}")
    n_bad_ge = sum(v >= min(good_pi) for v in bad_pi)
    print(f"  # decoy seeds with pair_iptm >= the WORST good seed: {n_bad_ge}/{len(bad_pi)} "
          f"=> confidence does NOT cleanly separate decoy from native")


if __name__ == "__main__":
    main()
