#!/usr/bin/env python
"""Build a single-entry template job from 9x1h's BEST pose, with real inter-chain geometry.

WHY. We now hold free-run trunk dumps for a good-basin seed (300038, DockQ 0.615) and for the
confident decoy (300018, DockQ 0.019, ipTM 0.700). Templating on the good pose and dumping THAT
trunk state gives a third point: what the pair rep looks like when the model is handed the right
answer, versus when it finds it on its own, versus when it is confidently wrong.

INTER-CHAIN GEOMETRY MUST SURVIVE, which needs two env flags at run time (set by the runner script):
  PROTENIX_MULTICHAIN_MASK=0   template cross-chain features are otherwise zeroed -- hardcoded on in
                               training, no config knob -- so the model would see only intra-chain folds
  PROTENIX_TMPL_KEEP_FRAME=1   _zero_center_positions centres each chain on its OWN centroid, which
                               superimposes every chain of the template at the origin and destroys the
                               dock. Measured on a real donor: chain A centroid (9.10,-14.21,7.12) -> (0,0,0)
Without both, this experiment silently becomes "template the folds, discard the pose".

PROTOMER CHAINS ARE MEASURED, NEVER ASSUMED. 9x1h is a 2-copy assembly (A,B light / C,D heavy /
E,F antigen) and which copies form the docked triple varies per pose -- across 250 poses the split was
A/C/F 79, B/D/E 70, A/C/E 52, B/D/F 49. Matching donor chains to query entities by sequence would pair
a Fab with the antigen copy it is NOT bound to: harmless with the cross-chain mask on, fatal with it off.
Here the triple is chosen by total heavy-atom contacts (5 A) over its three pairs.

TRAPS GUARDED (each has silently corrupted an arm in this project):
  - template id must be exactly 4 chars, else Protenix logs "Found 0 templates" and runs UNTEMPLATED
    while still emitting plausible poses
  - the a3m hit must DIFFER from the query or it is parsed as an MSA row, not a template
  - donor cifs are used RAW: a gemmi round-trip makes Protenix report "No protein chains found"

Usage: build_bestpose_template.py --out artifacts/tmplbest [--pose <cif>]
Writes <out>/job.json, <out>/a3m/*.a3m, <out>/chosen.json
"""
import argparse
import glob
import itertools
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemmi  # noqa: E402
from cifconv import convert  # noqa: E402

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
CACHE_F = "/home/ebrahim/common/release_date_cache.json"
GROUPS = {"light": ("A", "B"), "heavy": ("C", "D"), "antigen": ("E", "F")}
CUTOFF = 5.0
TLC = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
       'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
       'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def seqof(ch):
    return "".join(TLC[r.name] for r in ch if r.name in TLC)


def best_pose():
    """Highest CD DockQ across every scored pose from today's hitting arms."""
    best = (None, -1.0)
    for arm, jd in (("dump50", f"{EXP}/artifacts/dump50/pxm_json"),
                    ("faithful50", f"{EXP}/artifacts/faithful50/pxm_json")):
        for f in glob.glob(f"{jd}/*.json"):
            m = re.match(r"(\d+)_(\d+)\.json$", os.path.basename(f))
            if not m:
                continue
            try:
                J = json.load(open(f))
            except Exception:
                continue
            v = (J.get("interface") or {}).get("C,D") or (J.get("interface") or {}).get("D,C")
            if not v or v.get("dockq") is None:
                continue
            d = float(v["dockq"])
            if d > best[1]:
                cif = None
                for g in (0, 1, 2, 6):
                    c = (f"{EXP}/artifacts/{arm}/g{g}/9x1h/seed_{m.group(1)}/predictions/"
                         f"9x1h_sample_{m.group(2)}.cif")
                    if os.path.exists(c):
                        cif = c
                        break
                if cif:
                    best = ((arm, int(m.group(1)), int(m.group(2)), cif), d)
    return best


def protomer(cif):
    """(light, heavy, antigen) triple with the most total heavy-atom contacts."""
    st = gemmi.read_structure(cif)
    st.setup_entities()
    st.remove_ligands_and_waters()
    pos = {}
    for c in st[0]:
        if len(c) < 20:
            continue
        pos[c.name] = np.array([[a.pos.x, a.pos.y, a.pos.z] for r in c for a in r
                                if a.element.name != "H"])

    def contacts(a, b):
        if a not in pos or b not in pos:
            return 0
        d = np.linalg.norm(pos[a][:, None, :] - pos[b][None, :, :], axis=-1)
        return int((d.min(axis=1) < CUTOFF).sum())

    best = None
    for L, H, Ag in itertools.product(*GROUPS.values()):
        lh, la, ha = contacts(L, H), contacts(L, Ag), contacts(H, Ag)
        tot = lh + la + ha
        if best is None or tot > best[-1]:
            best = (L, H, Ag, lh, la, ha, tot)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{EXP}/artifacts/tmplbest")
    ap.add_argument("--tid", default="z9x1")
    ap.add_argument("--pose", default=None)
    A = ap.parse_args()
    assert len(A.tid) == 4, "template id must be exactly 4 chars"
    os.makedirs(f"{A.out}/a3m", exist_ok=True)

    if A.pose:
        cif, dq, arm, seed, smp = A.pose, float("nan"), "manual", -1, -1
    else:
        (arm, seed, smp, cif), dq = best_pose()
        if cif is None:
            sys.exit("no scored poses found")
    L, H, Ag, lh, la, ha, tot = protomer(cif)
    print(f"best pose: {arm} seed {seed} sample {smp}  CD DockQ {dq:.3f}")
    print(f"  {cif}")
    print(f"  protomer: light={L} heavy={H} antigen={Ag}  contacts lh={lh} la={la} ha={ha}")
    if lh < 50 or (la + ha) < 20:
        sys.exit("chosen pose is not a assembled, docked protomer -- refusing to template it")

    cache = json.load(open(CACHE_F))
    cache.setdefault(A.tid, {"release_date": "2000-01-01"})
    open(f"/home/ebrahim/mmcif/{A.tid}.cif", "w").write(convert(open(cif).read(), A.tid))
    json.dump(cache, open(CACHE_F, "w"))

    base = json.load(open(f"{P}/jsons/9x1h-final-updated.json"))
    obj = json.loads(json.dumps(base[0] if isinstance(base, list) else base))
    obj["name"] = "9x1h_tmplbest"
    st = gemmi.read_structure(cif)
    st.setup_entities()
    have = {c.name: seqof(c) for c in st[0] if len(seqof(c)) >= 4}
    chains = (L, H, Ag)
    pi = 0
    for e in obj["sequences"]:
        if "proteinChain" not in e:
            continue
        q = e["proteinChain"]["sequence"]
        nm = chains[pi]
        if nm not in have:
            sys.exit(f"chain {nm} absent from {cif}")
        hs = list(q)
        p1, p2 = len(q) // 3, (len(q) // 2 + 1 + pi) % len(q)
        if p2 == p1:
            p2 = (p2 + 1) % len(q)
        for p in (p1, p2):
            hs[p] = "A" if hs[p] != "A" else "G"
        hit = "".join(hs)
        assert hit != q, "template hit identical to query -> parsed as an MSA row, not a template"
        a3m = f"{A.out}/a3m/q{pi}.a3m"
        open(a3m, "w").write(
            f">query\n{q}\n>{A.tid}_{nm}/1-{len(q)} [subseq from] mol:protein "
            f"length:{len(q)}  bestpose\n{hit}\n")
        e["proteinChain"]["templatesPath"] = a3m
        pi += 1
    assert pi == 3, f"{pi} templated chains, expected 3"
    json.dump([obj], open(f"{A.out}/job.json", "w"))
    json.dump(dict(arm=arm, seed=seed, sample=smp, dockq=dq, cif=cif, tid=A.tid,
                   light=L, heavy=H, antigen=Ag), open(f"{A.out}/chosen.json", "w"), indent=1)
    print(f"wrote {A.out}/job.json  (template id {A.tid})")


if __name__ == "__main__":
    main()
