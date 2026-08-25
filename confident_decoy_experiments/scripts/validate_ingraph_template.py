#!/usr/bin/env python
"""Validate an IN-GRAPH template featurisation against Protenix's own offline pipeline.

WHY. To feed diffusion output back as a template mid-forward-pass, the coordinates must be turned into
the features the TemplateEmbedder actually reads: template_distogram, template_unit_vector,
template_pseudo_beta_mask, template_backbone_frame_mask, template_aatype. If that conversion is
subtly wrong -- residue order, atom slot, aatype encoding -- the model still runs and still emits
plausible structures, it is just conditioned on meaningless geometry. Nothing downstream would reveal
it. So the conversion is checked against the trusted path before any science run.

METHOD. Take a structure Protenix has ALREADY featurised offline (dumped via TMPL_DUMP_INFER), build
the same features here from the structure's coordinates, and compare tensors elementwise.

NOT via template_atom_positions: the offline dict's atom_positions/atom_mask do not hold plain dense
coordinates (mask is all-True, 24/24 slots on every token, coords span only +-39 A), so filling that
buffer is not the route. The three geometry functions are called directly instead -- they are the same
functions the offline path uses, so a match here means the conversion is correct on the quantity the
model actually consumes.

Usage: .../python scripts/validate_ingraph_template.py --offline <npz> --cif <donor.cif>
"""
import argparse, collections, os, sys

sys.path.insert(0, "/home/ebrahim/Protenix")
import gemmi
import numpy as np
from protenix.data.constants import DENSE_ATOM, STD_RESIDUES_WITH_GAP
from protenix.data.template.template_utils import TemplateFeatures
from protenix.data.template.template_featurizer import Templates

ap = argparse.ArgumentParser()
ap.add_argument("--offline", required=True)
ap.add_argument("--cif", required=True)
A = ap.parse_args()


def dense_from_cif(path):
    """(aatype [N], dense_pos [N,24,3], dense_mask [N,24]) in DENSE_ATOM slot order.

    Atoms outside the canonical list (only OXT on C-terminal residues here) are dropped -- DENSE_ATOM
    has no slot for them.
    """
    b = gemmi.cif.read(path).sole_block()
    cols = [list(b.find_values("_atom_site." + x)) for x in
            ("label_asym_id", "label_seq_id", "label_comp_id", "label_atom_id",
             "Cartn_x", "Cartn_y", "Cartn_z", "group_PDB")]
    res = collections.OrderedDict()
    for ch, seq, comp, at, x, y, z, g in zip(*cols):
        if g != "ATOM":
            continue
        res.setdefault((ch, int(seq)), (comp, []))[1].append(
            (at.strip('"'), float(x), float(y), float(z)))
    n = len(res)
    aat = np.zeros(n, dtype=np.int32)
    pos = np.zeros((n, 24, 3), dtype=np.float32)
    msk = np.zeros((n, 24), dtype=np.float32)
    dropped = collections.Counter()
    for i, ((ch, sq), (comp, ats)) in enumerate(res.items()):
        aat[i] = STD_RESIDUES_WITH_GAP.get(comp, STD_RESIDUES_WITH_GAP.get("UNK", 20))
        names = list(DENSE_ATOM.get(comp, ()))
        for an, x, y, z in ats:
            if an in names:
                j = names.index(an)
                pos[i, j] = (x, y, z); msk[i, j] = 1.0
            else:
                dropped[an] += 1
    return aat, pos, msk, dict(dropped)


off = np.load(A.offline)
o_aat = off["template_aatype"][0]
o_dg, o_uv = off["template_distogram"][0], off["template_unit_vector"][0]
o_pb, o_bb = off["template_pseudo_beta_mask"][0], off["template_backbone_frame_mask"][0]

aat, pos, msk, dropped = dense_from_cif(A.cif)
print(f"parsed {len(aat)} residues from cif; atoms dropped (no DENSE_ATOM slot): {dropped}")
print(f"offline tokens: {len(o_aat)}")

# --- check 1: aatype. Also verifies residue ORDER matches the offline token order --------------
if len(aat) != len(o_aat):
    sys.exit(f"FAIL length mismatch: mine {len(aat)} vs offline {len(o_aat)}")
same = int((aat == o_aat).sum())
print(f"\n[1] aatype match: {same}/{len(aat)} ({100*same/len(aat):.1f}%)")
if same != len(aat):
    bad = np.where(aat != o_aat)[0][:8]
    print(f"    first mismatches at tokens {bad.tolist()}: mine {aat[bad].tolist()} vs offline {o_aat[bad].tolist()}")
    sys.exit("FAIL: aatype differs -- residue ordering or encoding is wrong; stopping")

# --- check 2: the geometry features, via Protenix's own functions ------------------------------
cfg = Templates._DGRAM_CONFIG
pb_pos, pb_mask = TemplateFeatures.pseudo_beta_fn(aat, pos * msk[..., None], msk)
pb_mask_2d = pb_mask[:, None] * pb_mask[None, :]
dg = TemplateFeatures.dgram_from_positions(pb_pos, config=cfg) * pb_mask_2d[..., None]
uv, bb_2d = TemplateFeatures.compute_template_unit_vector(aat, pos * msk[..., None], msk)
uv = uv * bb_2d[..., None]

for name, mine, ref in [("template_distogram", dg, o_dg),
                        ("template_unit_vector", uv, o_uv),
                        ("template_pseudo_beta_mask", pb_mask_2d, o_pb),
                        ("template_backbone_frame_mask", bb_2d, o_bb)]:
    mine = np.asarray(mine, dtype=np.float32); ref = np.asarray(ref, dtype=np.float32)
    if mine.shape != ref.shape:
        print(f"[2] {name:<30} SHAPE MISMATCH mine {mine.shape} vs offline {ref.shape}"); continue
    md = float(np.abs(mine - ref).max())
    frac = float((np.abs(mine - ref) > 1e-4).mean())
    print(f"[2] {name:<30} max|diff| {md:.6g}   frac differing {frac:.4%}   "
          f"{'MATCH' if md < 1e-4 else 'MISMATCH'}")
