#!/usr/bin/env python
"""THE single source of truth for "which chains is this interface, in the model and in the GT".

WHY THIS EXISTS. FoldBench's targets/interface_antibody_antigen.csv gives `antibody_chain` /
`antigen_chain` as REFERENCE (ground-truth) chain names. Several scripts here assumed they were MODEL
chain names and mapped them through OST's chain_mapping in the wrong role. Where the mapping is the
identity (53 of 104 complexes) the two readings agree and nothing looks wrong; where it is not, the
measurement silently lands on different chains. On 8gy5 the labels B|C are GT light-chain|antigen, but
model chains B and C are heavy-chain and light-chain -- so the "epitope" measured there was an
antibody-internal contact. 72 of 161 interfaces (45%) were affected.

Verified against all 161 interfaces of the 104-complex standard set: the label pair appears as a
REFERENCE pair in OST's `dockq_interfaces` for 158, and the other 3 (8b7h, 8fxv) differ only by OST's
"-2" suffix on duplicated reference chains. It appears as a MODEL pair only for the 105 identity-mapped
ones, i.e. never uniquely.

DO NOT re-derive chains from `chain_mapping` in callers. Every artifacts/abag_all_ost/<pdb>_<s>_<k>.json
carries `dockq_interfaces`, whose rows are [ref1, ref2, mdl1, mdl2] -- both sides of the correspondence
DockQ was actually computed with, for THAT pose. Multi-copy assemblies get a different row per pose, so
resolve per pose. Take both sides from the same row and there is nothing left to assume.
"""
import functools
import glob
import json
import os
import re

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
OST_DIR = f"{EXP}/artifacts/abag_all_ost"


def _base(ch):
    """OST suffixes duplicated reference chains ('A' -> 'A-2'); FoldBench labels carry no suffix."""
    return re.sub(r"-\d+$", "", ch)


@functools.lru_cache(maxsize=None)
def _load(pdb, seed, smp):
    f = f"{OST_DIR}/{pdb}_{seed}_{smp}.json"
    if not os.path.exists(f):
        raise FileNotFoundError(f"no OST json for {pdb} seed {seed} sample {smp}: {f}")
    return json.load(open(f))


def resolve(pdb, seed, smp, ab_label, ag_label):
    """-> dict(ref_ab, ref_ag, mdl_ab, mdl_ag, dockq, fnat, irmsd, lrmsd) for ONE pose.

    ab_label/ag_label are FoldBench REFERENCE chain names. Raises if the interface is not present in
    this pose's OST output -- a missing interface is a real condition to handle, never a silent None.
    """
    d = _load(pdb, seed, smp)
    rows = d.get("dockq_interfaces") or []
    want = {ab_label, ag_label}
    for i, row in enumerate(rows):
        if len(row) < 4:
            continue
        r1, r2, m1, m2 = row[0], row[1], row[2], row[3]
        if {_base(r1), _base(r2)} != want:
            continue
        # keep antibody/antigen roles straight: the row order is not guaranteed to match the labels
        if _base(r1) == ab_label:
            ref_ab, ref_ag, mdl_ab, mdl_ag = r1, r2, m1, m2
        else:
            ref_ab, ref_ag, mdl_ab, mdl_ag = r2, r1, m2, m1

        def at(key):
            v = d.get(key)
            return v[i] if isinstance(v, list) and i < len(v) else None

        return dict(ref_ab=ref_ab, ref_ag=ref_ag, mdl_ab=mdl_ab, mdl_ag=mdl_ag,
                    dockq=at("dockq"), fnat=at("fnat"), irmsd=at("irmsd"), lrmsd=at("lrmsd"))
    raise KeyError(f"{pdb} {ab_label}|{ag_label} not in dockq_interfaces for seed {seed} sample {smp}; "
                   f"have {[(r[0], r[1]) for r in rows]}")


def first_pose_with(pdb, ab_label, ag_label, limit=20):
    """Resolve using the first cached pose that has this interface -- for setup work (e.g. defining the
    native epitope) that needs GT chain names before iterating poses."""
    for f in sorted(glob.glob(f"{OST_DIR}/{pdb}_*.json"))[:limit]:
        base = os.path.basename(f)[:-5]
        try:
            _, seed, smp = base.rsplit("_", 2)
            return resolve(pdb, seed, smp, ab_label, ag_label), seed, smp
        except (KeyError, FileNotFoundError, ValueError):
            continue
    raise KeyError(f"{pdb} {ab_label}|{ag_label}: no cached OST pose contains this interface")


if __name__ == "__main__":
    # self-check on the case that exposed the bug
    r, s, k = first_pose_with("8gy5", "B", "C")
    print("8gy5 B|C ->", r, f"(from seed {s} sample {k})")
    assert (r["mdl_ab"], r["mdl_ag"]) == ("D", "F"), r
    print("OK")
