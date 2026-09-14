#!/usr/bin/env python
"""Build ONE template per pose so all 250 poses of an interface can be RESCORED in a few model loads.

THE IDEA. ipTM as normally produced is ~100% a SEED property (87% of its variance is between seeds
across PXM, ~100% on 9x1h): the five samples of a seed score alike no matter how different their
structures are, because the confidence head is dominated by z_trunk, which carries that seed's MSA
subsample. So the ipTM attached to a pose says little about the pose.

RESCORING removes that confound. Hand each pose BACK as a template with its full inter-chain geometry,
run a SHORT forward pass, and read the ipTM that comes out. Every pose is then evaluated from the same
kind of starting point -- its own geometry -- rather than from whatever MSA draw produced it. The
readout is label-free, so if the acceptable poses rank top it is a usable selector, not an oracle.

WHY SHORT RECYCLING IS THE POINT. At N_cycle=10 the model has time to drift off the template back to
whatever it prefers: templating the decoy gave ipTM 0.724 and templating a good pose 0.460, which is
the model's PREFERENCE reasserting itself rather than an evaluation of the input. At N_cycle=1 it can
barely move, so the score should reflect the geometry as given. N=3 brackets how fast the drift sets
in.

ONE JSON, MANY ENTRIES. A separate process per pose would spend ~1 minute per model load -- four hours
of pure loading for 250 poses. The input json is a list, so all entries go in one file (sharded across
GPUs) and the checkpoint loads once per shard. Each entry needs a UNIQUE name or the outputs collide.

CHAINS COME FROM artifacts/9x1h_protomers.csv, measured per pose, never assumed. 9x1h is a 2-copy
assembly and the docked triple varies across poses (A/C/F 79, B/D/E 70, A/C/E 52, B/D/F 49 of 250), so
matching donor chains to query entities by sequence -- all copies share a sequence -- would pair a Fab
with the antigen copy it is not bound to. Harmless with the cross-chain mask on, fatal with it off.

TRAPS GUARDED (each has silently corrupted an arm in this project):
  - template ids MUST be exactly 4 chars, else Protenix logs 'Found 0 templates' and runs UNTEMPLATED
    while still emitting plausible poses; ids are checked for availability against /home/ebrahim/mmcif
  - the a3m hit must DIFFER from the query or it is parsed as an MSA row, not a template
  - obj["name"] unique per entry
  - donor cifs are used RAW: a gemmi round-trip makes Protenix report 'No protein chains found'

Usage:
  .../python scripts/build_rescore_templates.py --pdb 9x1h --run rs1 --tid-prefix a --shards 4
Writes artifacts/rescore/<run>/shard{i}.json and artifacts/rescore/<run>/index.csv
"""
import argparse
import csv
import difflib
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemmi  # noqa: E402
from cifconv import convert  # noqa: E402

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
CACHE_F = "/home/ebrahim/common/release_date_cache.json"
TLC = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
       'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
       'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def seqof(ch):
    return "".join(TLC[r.name] for r in ch if r.name in TLC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default="9x1h")
    ap.add_argument("--run", required=True)
    ap.add_argument("--tid-prefix", required=True)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--protomers", default=f"{EXP}/artifacts/9x1h_protomers.csv")
    A = ap.parse_args()
    assert len(A.tid_prefix) == 1

    OUT = f"{EXP}/artifacts/rescore/{A.run}"
    A3M = f"{OUT}/a3m"
    os.makedirs(A3M, exist_ok=True)
    cache = json.load(open(CACHE_F))
    TAKEN = {os.path.basename(p)[:-4] for p in glob.glob("/home/ebrahim/mmcif/*.cif")}
    alpha = "0123456789abcdefghijklmnopqrstuvwxyz"
    pool = (f"{A.tid_prefix}{a}{b}{c}" for a in alpha for b in alpha for c in alpha)

    def new_tid():
        for t in pool:
            if t not in TAKEN:
                TAKEN.add(t)
                return t
        sys.exit("no free tids")

    proto = {(int(r["seed"]), int(r["sample"])): (r["light"], r["heavy"], r["antigen"])
             for r in csv.DictReader(open(A.protomers)) if r["ok"] == "1"}
    base = json.load(open(f"{P}/jsons/{A.pdb}-final-updated.json"))
    base = base[0] if isinstance(base, list) else base

    entries, index = [], []
    for (seed, smp), chains in sorted(proto.items()):
        cif = (f"{P}/out/{A.pdb}/{A.pdb}/seed_{seed}/predictions/{A.pdb}_sample_{smp}.cif")
        if not os.path.exists(cif):
            continue
        tid = new_tid()
        assert len(tid) == 4
        st = gemmi.read_structure(cif)
        st.setup_entities()
        have = {c.name: seqof(c) for c in st[0] if len(seqof(c)) >= 4}
        cache.setdefault(tid, {"release_date": "2000-01-01"})
        open(f"/home/ebrahim/mmcif/{tid}.cif", "w").write(convert(open(cif).read(), tid))

        obj = json.loads(json.dumps(base))
        name = f"{A.pdb}_{A.run}_{seed}_{smp}"
        obj["name"] = name
        pi, nhit = 0, 0
        for e in obj["sequences"]:
            if "proteinChain" not in e:
                continue
            q = e["proteinChain"]["sequence"]
            L = len(q)
            nm = chains[pi]
            if nm not in have:
                sys.exit(f"{cif}: no chain {nm}")
            r = difflib.SequenceMatcher(None, have[nm], q, autojunk=False).ratio()
            if r < 0.9:
                sys.exit(f"{cif}: chain {nm} matches entity {pi} at only {r:.2f}")
            hs = list(q)
            p1, p2 = L // 3, (L // 2 + 1 + pi) % L
            if p2 == p1:
                p2 = (p2 + 1) % L
            for p in (p1, p2):
                hs[p] = "A" if hs[p] != "A" else "G"
            hit = "".join(hs)
            assert hit != q, "hit identical to query -> parsed as MSA row"
            a3mp = f"{A3M}/{seed}_{smp}_q{pi}.a3m"
            open(a3mp, "w").write(
                f">query\n{q}\n>{tid}_{nm}/1-{L} [subseq from] mol:protein length:{L}  rescore\n"
                f"{hit}\n")
            e["proteinChain"]["templatesPath"] = a3mp
            nhit += 1
            pi += 1
        assert nhit == 3, f"{name}: {nhit} templated chains, expected 3"
        entries.append(obj)
        index.append(dict(name=name, seed=seed, sample=smp, tid=tid,
                          light=chains[0], heavy=chains[1], antigen=chains[2]))

    json.dump(cache, open(CACHE_F, "w"))
    assert len({e["name"] for e in entries}) == len(entries), "duplicate entry names"
    n = len(entries)
    per = (n + A.shards - 1) // A.shards
    for i in range(A.shards):
        chunk = entries[i * per:(i + 1) * per]
        if chunk:
            json.dump(chunk, open(f"{OUT}/shard{i}.json", "w"))
    with open(f"{OUT}/index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index[0]))
        w.writeheader()
        w.writerows(index)
    print(f"built {n} entries -> {A.shards} shards in {OUT}")
    for i in range(A.shards):
        f = f"{OUT}/shard{i}.json"
        if os.path.exists(f):
            print(f"   shard{i}: {len(json.load(open(f)))} entries")


if __name__ == "__main__":
    main()
