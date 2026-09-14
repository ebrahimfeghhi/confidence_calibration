#!/usr/bin/env python
"""Template 9x1h on SEVERAL of its own poses at once, with real inter-chain geometry.

For recursive self-templating where a whole round's output is fed back, not just its best pose. That
removes selection entirely: a round's 5 samples are mixed quality (a real free seed gave 0.683, 0.562,
0.031, 0.033, 0.023), and handing all of them back asks whether the ensemble converges on the good
poses or is dragged by the bad ones.

PER-DONOR PROTOMER DETECTION IS REQUIRED, not optional. 9x1h is a 2-copy assembly (A,B light /
C,D heavy / E,F antigen) and which copies form the docked triple varies POSE BY POSE -- across 250
poses the split was A/C/F 79, B/D/E 70, A/C/E 52, B/D/F 49. Matching donor chains to query entities by
sequence would pair a Fab with the antigen copy it is not bound to. Harmless with the cross-chain mask
on; fatal with it off, which is the regime this runs in. So each donor's triple is measured by contacts.

TWO DEDUP TRAPS, both of which have silently reduced a multi-template arm to one template in this
project:
  1. a3m level -- hits are deduped by SEQUENCE, so every donor needs a DIFFERENT mutated hit string.
     Positions are offset by donor index here, and the script asserts all hits are distinct.
  2. feature level -- deduped by the sequence read from the mmcif, which collapses N conformations of
     the same molecule into one. Defeated only by PROTENIX_TMPL_NODEDUP=1 at run time.
Also PROTENIX_MAX_TEMPLATES must be >= the donor count, since the trained default is 4.

Template ids must be exactly 4 chars or Protenix logs "Found 0 templates" and runs UNTEMPLATED while
still emitting plausible poses.

Usage:
  build_multipose_template.py --out artifacts/mchain/r1 --tid-prefix a --poses p1.cif p2.cif ...
Writes <out>/job.json, <out>/a3m/*.a3m, <out>/chosen.json
"""
import argparse
import itertools
import json
import os
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


def protomer(cif):
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--tid-prefix", required=True, help="1 char; tids become <p>NNN")
    ap.add_argument("--poses", nargs="+", required=True)
    ap.add_argument("--tids", nargs="+", default=None,
                    help="explicit 4-char template ids, one per pose. REQUIRED at scale: the "
                         "prefix+index scheme gives only 5 ids per prefix and collides across "
                         "concurrent chains.")
    ap.add_argument("--no-cache-write", action="store_true",
                    help="do not rewrite release_date_cache.json. MANDATORY when several workers "
                         "run at once -- it is a 228k-entry file that every job on this box reads, "
                         "and concurrent rewrites corrupt it. Pre-register ids with prereg_tids.py.")
    A = ap.parse_args()
    assert len(A.tid_prefix) == 1 or A.tids
    os.makedirs(f"{A.out}/a3m", exist_ok=True)

    cache = json.load(open(CACHE_F))
    base = json.load(open(f"{P}/jsons/9x1h-final-updated.json"))
    obj = json.loads(json.dumps(base[0] if isinstance(base, list) else base))
    obj["name"] = "9x1h_mchain"

    donors = []
    for d, cif in enumerate(A.poses):
        L, H, Ag, lh, la, ha, tot = protomer(cif)
        if lh < 50 or (la + ha) < 20:
            print(f"   SKIP donor {d}: not an assembled docked protomer (lh={lh} la={la} ha={ha})")
            continue
        tid = A.tids[d] if A.tids else f"{A.tid_prefix}{d:03d}"
        assert len(tid) == 4
        cache.setdefault(tid, {"release_date": "2000-01-01"})
        open(f"/home/ebrahim/mmcif/{tid}.cif", "w").write(convert(open(cif).read(), tid))
        donors.append(dict(idx=d, cif=cif, tid=tid, chains=(L, H, Ag), lh=lh, la=la, ha=ha))
        print(f"   donor {d}: {tid}  {L}/{H}/{Ag}  lh={lh} la={la} ha={ha}")
    if not donors:
        sys.exit("no usable donors")
    if not A.no_cache_write:
        json.dump(cache, open(CACHE_F, "w"))
    else:
        missing = [d["tid"] for d in donors if d["tid"] not in cache]
        if missing:
            sys.exit(f"tids not pre-registered in the cache: {missing[:5]}")

    pi = 0
    for e in obj["sequences"]:
        if "proteinChain" not in e:
            continue
        q = e["proteinChain"]["sequence"]
        Lq = len(q)
        lines = [">query", q]
        hits = set()
        for k, dn in enumerate(donors):
            st = gemmi.read_structure(dn["cif"])
            st.setup_entities()
            have = {c.name: seqof(c) for c in st[0] if len(seqof(c)) >= 4}
            nm = dn["chains"][pi]
            if nm not in have:
                sys.exit(f"{dn['cif']}: chain {nm} missing")
            hs = list(q)
            # offset by DONOR index so no two donors produce the same hit string; the a3m dedups
            # on sequence and would otherwise keep only the first
            for p in ((Lq // 3 + 5 * k) % Lq, (Lq // 2 + 1 + pi + 7 * k) % Lq):
                hs[p] = "A" if hs[p] != "A" else "G"
            hit = "".join(hs)
            assert hit != q, "hit identical to query -> parsed as an MSA row"
            assert hit not in hits, f"duplicate hit string for donor {k} -> a3m would dedup it away"
            hits.add(hit)
            lines += [f">{dn['tid']}_{nm}/1-{Lq} [subseq from] mol:protein length:{Lq}  mchain", hit]
        a3m = f"{A.out}/a3m/q{pi}.a3m"
        open(a3m, "w").write("\n".join(lines) + "\n")
        e["proteinChain"]["templatesPath"] = a3m
        pi += 1
    assert pi == 3, f"{pi} templated chains, expected 3"
    json.dump([obj], open(f"{A.out}/job.json", "w"))
    json.dump(donors, open(f"{A.out}/chosen.json", "w"), indent=1)
    print(f"   wrote {A.out}/job.json with {len(donors)} templates per chain")


if __name__ == "__main__":
    main()
