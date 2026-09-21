#!/usr/bin/env python
"""Check every template namespace BEFORE spending GPU time on it.

WHY. Templates are built offline, so everything that went wrong in this experiment was knowable
without a single fold: donors templated from the wrong protomer (1053 of 2920), a namespace missing
its release-date file, ids colliding in a shared 4-character space. Each was found by a fold failing
hours later, or worse, by a fold NOT failing -- Protenix logs "Found 5 templates" and runs happily on
a template for the wrong chains.

WHAT IS CHECKED, cheapest first:

  ids        every donor's tid is unique across the whole namespace, has a cif, and has a release-date
             entry. Ids are 4 chars because Protenix reads them as pdb codes, and a collision means one
             fold silently overwrites a template another is using.
  chains     the chains the builder chose CONTAIN the chains chain_map.py says that pose uses for this
             interface. The pose was ranked on one chain pair; templating another measures something
             else.
  hits       every a3m hit names a chain that exists in its cif with residues, and the hits are
             mutually distinct -- identical hit strings get deduped away and a 5-template arm quietly
             becomes a 1-template arm.

NOT CHECKED HERE: that featurisation yields a non-zero atom mask. That needs the model's own config
construction, and a hand-rolled version of it produced all-zero tensors that I briefly mistook for a
real finding. It is verified once, on one real fold, rather than approximated 584 times.

    preflight.py --ns <namespace> --queue queue_top5.tsv
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gemmi  # noqa: E402
import metrics  # noqa: E402
from build_templates import pose_key  # noqa: E402
from fmt import table  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", required=True)
    ap.add_argument("--queue", default="queue_top5.tsv")
    ap.add_argument("--show", type=int, default=6)
    A = ap.parse_args()

    maps, roles = metrics.load_chain_maps(), metrics.load_roles()
    rd = json.load(open(f"{A.ns}/release_dates.json")) if os.path.exists(
        f"{A.ns}/release_dates.json") else {}
    cifs = {os.path.basename(p)[:-4] for p in glob.glob(f"{A.ns}/mmcif/*.cif")}

    want = collections.OrderedDict()
    for line in open(A.queue):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 6:
            want.setdefault(f"{f[2]}_{f[3]}", (f[2], f[5].split()))

    fail = collections.Counter()
    ex = collections.defaultdict(list)
    seen_tids = {}
    chain_cache = {}

    for key, (iface, poses) in want.items():
        jd = f"{A.ns}/jobs/{key}"
        if not os.path.exists(f"{jd}/job.json"):
            fail["namespace missing"] += 1
            ex["namespace missing"].append(key)
            continue
        chosen = json.load(open(f"{jd}/chosen.json"))
        if len(chosen) != len(poses):
            fail["donor count wrong"] += 1
            ex["donor count wrong"].append(f"{key}: {len(chosen)}/{len(poses)}")
        ab_ref, ag_ref = roles[iface]
        for d in chosen:
            t = d["tid"]
            if len(t) != 4:
                fail["tid not 4 chars"] += 1
            if t in seen_tids and seen_tids[t] != key:
                fail["tid COLLISION"] += 1
                ex["tid COLLISION"].append(f"{t}: {seen_tids[t]} vs {key}")
            seen_tids[t] = key
            if t not in cifs:
                fail["tid has no cif"] += 1
            if t not in rd:
                fail["tid missing release date"] += 1
            # the chains the pose actually uses for THIS interface
            pdb, seed, samp = pose_key(d["cif"])
            m = maps.get((pdb, seed, samp))
            if not m:
                fail["pose unmapped"] += 1
                continue
            need = {m[ab_ref], m[ag_ref]}
            if not need <= set(d["chains"]):
                fail["WRONG CHAINS"] += 1
                ex["WRONG CHAINS"].append(
                    f"{key} {os.path.basename(d['cif'])}: need {'/'.join(sorted(need))}, "
                    f"got {'/'.join(d['chains'])}")
            # the a3m hit must name a chain that exists in the cif
            f_cif = f"{A.ns}/mmcif/{t}.cif"
            if f_cif not in chain_cache and os.path.exists(f_cif):
                st = gemmi.read_structure(f_cif)
                st.setup_entities()
                chain_cache[f_cif] = {c.name: len(c) for c in st[0]}
            have = chain_cache.get(f_cif, {})
            for ch in d["chains"]:
                if have.get(ch, 0) < 1:
                    fail["cif lacks a chosen chain"] += 1
        # a3m hits distinct, and each names a real chain
        for a3m in glob.glob(f"{jd}/a3m/*.a3m"):
            body = open(a3m).read().split("\n")
            names = [l for l in body if l.startswith(">") and "selftmpl" in l]
            seqs = [body[i + 1] for i, l in enumerate(body)
                    if l.startswith(">") and "selftmpl" in l and i + 1 < len(body)]
            if len(set(seqs)) != len(seqs):
                fail["duplicate a3m hit (would dedup)"] += 1
                ex["duplicate a3m hit (would dedup)"].append(os.path.basename(a3m) + " in " + key)
            for nm in names:
                mm = re.match(r">(\w{4})_(\w+)/", nm)
                if not mm:
                    fail["a3m hit name malformed"] += 1
                    continue
                t, ch = mm.groups()
                f_cif = f"{A.ns}/mmcif/{t}.cif"
                if chain_cache.get(f_cif, {}).get(ch, 0) < 1:
                    fail["a3m names a chain absent from its cif"] += 1
                    ex["a3m names a chain absent from its cif"].append(f"{key}: {nm[:40]}")

    print(table(["check", "failures"],
                [[k, v] for k, v in fail.most_common()] or [["all checks passed", 0]],
                align="l r"))
    for k, v in ex.items():
        print(f"\n  {k}:")
        for e in v[:A.show]:
            print(f"    {e}")
    print(f"\n  namespaces: {len(want)}   distinct tids: {len(seen_tids)}   "
          f"cifs: {len(cifs)}   release-date entries: {len(rd)}")
    return 1 if sum(fail.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
