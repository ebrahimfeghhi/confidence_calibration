#!/usr/bin/env python
"""Pre-register every template id the 50-seed chain run will use, in ONE cache write.

WHY THIS EXISTS. build_multipose_template.py normally appends its ids to
/home/ebrahim/common/release_date_cache.json and rewrites the whole 228k-entry file. With four
workers running concurrently, two rewrites can interleave and truncate it -- and every job on this
box reads that cache, so a corrupted copy breaks the PXM sweep too, silently and hours later.
So all ids are registered once here, up front, and the workers run with --no-cache-write.

ID SCHEME: 'w' + base36(seed_index*50 + round*5 + donor), zero-padded to 3 chars.
  50 seeds x 10 rounds x 5 donors = 2500 ids, against 36^3 = 46656 slots.
  A leading LETTER is safe: real PDB ids always start with a digit, so these cannot shadow a real
  entry -- which matters because 9x1h's own a3m lists 9849 real template hits.

Writes a manifest so the workers derive ids the same way rather than re-deriving the convention.

Usage: prereg_tids.py --seeds 300000-300049 --rounds 10 --donors 5
"""
import argparse
import json
import os

CACHE_F = "/home/ebrahim/common/release_date_cache.json"
EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
ALPHA = "0123456789abcdefghijklmnopqrstuvwxyz"


def b36(n, width=3):
    s = ""
    while n:
        s = ALPHA[n % 36] + s
        n //= 36
    return s.rjust(width, "0")


def tid(seed_idx, rnd, donor, prefix="w"):
    return prefix + b36(seed_idx * 50 + rnd * 5 + donor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="300000-300049")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--donors", type=int, default=5)
    ap.add_argument("--prefix", default="w", help="1 leading letter; must differ between runs or "
                    "two runs will write different structures to the same mmcif id")
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    lo, hi = (int(x) for x in A.seeds.split("-"))
    seeds = list(range(lo, hi + 1))

    ids = {}
    for si, s in enumerate(seeds):
        for r in range(1, A.rounds + 1):
            for d in range(A.donors):
                ids[f"{s}_{r}_{d}"] = tid(si, r, d, A.prefix)
    assert len(set(ids.values())) == len(ids), "tid collision -- scheme is wrong"

    cache = json.load(open(CACHE_F))
    n_before = len(cache)
    clash = [t for t in set(ids.values()) if t in cache]
    for t in set(ids.values()):
        cache.setdefault(t, {"release_date": "2000-01-01"})
    tmp = CACHE_F + ".tmp"
    json.dump(cache, open(tmp, "w"))
    os.replace(tmp, CACHE_F)            # atomic; a torn write here would break every job on the box

    os.makedirs(f"{EXP}/artifacts/mchain50", exist_ok=True)
    outp = A.out or f"{EXP}/artifacts/mchain50/tids.json"
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(ids, open(outp, "w"))
    print(f"registered {len(set(ids.values()))} tids  ({len(ids)} slots: "
          f"{len(seeds)} seeds x {A.rounds} rounds x {A.donors} donors)")
    print(f"cache {n_before} -> {len(cache)} entries;  {len(clash)} already present")
    print(f"example: seed {seeds[0]} round 1 donor 0 -> {ids[f'{seeds[0]}_1_0']}")
    print(f"         seed {seeds[-1]} round {A.rounds} donor 4 -> {ids[f'{seeds[-1]}_{A.rounds}_4']}")


if __name__ == "__main__":
    main()
