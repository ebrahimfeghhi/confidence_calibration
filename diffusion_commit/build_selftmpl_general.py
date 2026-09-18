#!/usr/bin/env python
"""Build a multi-pose self-templating job for ANY PXM-2025 complex, not just 9x1h.

Templates a query on several of its own predicted poses at once, with real inter-chain geometry
(cross-chain mask off, per-chain centring off at run time).

WHY A GENERAL VERSION. build_multipose_template.py hardcodes 9x1h's layout -- light (A,B), heavy
(C,D), antigen (E,F). Of the 60 complexes in this run, 20 are multi-copy assemblies with 2-6 protein
entities in various arrangements, so the chain choice has to be derived per complex.

THE CHAIN PROBLEM. Protenix takes ONE template hit per protein ENTITY, and applies it to every copy
of that entity. On a multi-copy assembly the donor has several chains matching each entity, and they
are NOT interchangeable: on 9x1h the two Fabs bind different antigen copies, so pairing chain A with
antigen E instead of F hands the model an antibody and an antigen that are not in contact. With the
cross-chain mask ON that is harmless; with it OFF -- which is the whole point here -- it is fatal.
So candidate chains are enumerated per entity and the COMBINATION with the most mutual heavy-atom
contact (5 A) is chosen, which recovers the docked protomer without assuming its composition.

DEDUP TRAPS, both of which have silently reduced multi-template arms to one template in this project:
  a3m level     -- hits dedup by SEQUENCE, so every donor gets a different mutated hit string, and
                   the script asserts they are distinct.
  feature level -- dedups on the mmcif sequence; defeated only by PROTENIX_TMPL_NODEDUP=1 at run time.
PROTENIX_MAX_TEMPLATES must also be >= the donor count (trained default is 4).

Template ids must be exactly 4 chars or Protenix logs "Found 0 templates" and runs UNTEMPLATED while
still emitting plausible poses.

Usage:
  build_selftmpl_general.py --pdb 9e7o --out <dir> --tids t001 t002 ... --poses p0.cif ... [--no-cache-write]
"""
import argparse
import difflib
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# cifconv is a SHARED helper (~30 build_* scripts import it) and stays where they can all see it.
# This file was moved here out of confident_decoy_experiments/scripts/, so its own directory no longer
# contains cifconv -- hence the explicit path. Copying the module here instead would start exactly the
# kind of drift that left 14 diverging copies of the OST call in this project.
sys.path.insert(0, "/home/ebrahim/tts_abag/confident_decoy_experiments/scripts")
import gemmi  # noqa: E402
from cifconv import convert  # noqa: E402

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
# The base job json to clone. Defaults to the pxm2025 pool; SELFTMPL_JOBS overrides it so the same
# builder serves the FoldBench pool (both name the file <pdb>-final-updated.json).
P = f"{EXP}/artifacts/pxm2025"
JOBS = os.environ.get("SELFTMPL_JOBS", f"{P}/jsons")
# PRIVATE NAMESPACE PER RUN. Template ids must be 4 chars (Protenix treats them as pdb ids), which is
# only 36^4 ~ 1.68M values -- and the shared /home/ebrahim/mmcif already holds >10k of them, so by the
# birthday bound a collision in a global namespace is near-certain, not unlikely. No hashing scheme
# fixes that. Instead each run gets its own template dir and its own release-date file via these two
# env vars: the namespace then holds only that run's handful of ids, collisions are impossible by
# construction, and nothing mutates a file another process has already cached.
MMCIF_DIR = os.environ.get("SELFTMPL_MMCIF_DIR", "/home/ebrahim/mmcif")
CACHE_F = os.environ.get("SELFTMPL_RELEASE_DATES", "/home/ebrahim/common/release_date_cache.json")
CUT = 5.0
TLC = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
       'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
       'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def seqof(ch):
    return "".join(TLC[r.name] for r in ch if r.name in TLC)


def pick_chains(cif, ent_seqs):
    """One donor chain per query entity, choosing the most mutually-contacting combination.

    Returns (chains, contacts) or (None, reason)."""
    st = gemmi.read_structure(cif)
    st.setup_entities()
    st.remove_ligands_and_waters()
    chains = {c.name: seqof(c) for c in st[0] if len(seqof(c)) >= 4}
    if not chains:
        return None, "no protein chains"
    pos = {}
    for c in st[0]:
        if c.name in chains:
            pos[c.name] = np.array([[a.pos.x, a.pos.y, a.pos.z] for r in c for a in r
                                    if a.element.name != "H"])
    # candidates per entity, by sequence identity
    cands = []
    for q in ent_seqs:
        cs = [nm for nm, s in chains.items()
              if difflib.SequenceMatcher(None, s, q, autojunk=False).ratio() >= 0.9]
        if not cs:
            return None, f"no chain matches an entity of length {len(q)}"
        cands.append(cs)
    n_comb = int(np.prod([len(c) for c in cands]))
    if n_comb > 4096:
        return None, f"too many chain combinations ({n_comb})"

    def contacts(a, b):
        d = np.linalg.norm(pos[a][:, None, :] - pos[b][None, :, :], axis=-1)
        return int((d.min(axis=1) < CUT).sum())

    best, best_score = None, -1
    cache = {}
    for combo in itertools.product(*cands):
        if len(set(combo)) != len(combo):      # one chain cannot serve two entities
            continue
        tot = 0
        for a, b in itertools.combinations(sorted(set(combo)), 2):
            if (a, b) not in cache:
                cache[(a, b)] = contacts(a, b)
            tot += cache[(a, b)]
        if tot > best_score:
            best, best_score = combo, tot
    if best is None:
        return None, "no valid chain assignment"
    return list(best), best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--poses", nargs="+", required=True)
    ap.add_argument("--tids", nargs="+", required=True)
    ap.add_argument("--no-cache-write", action="store_true")
    ap.add_argument("--prepend-real", default="",
                    help="real_tmpl_ids.py output: put the FREE arm's real templates BEFORE the "
                         "self-poses, so template_id 0..k-1 are real and k.. are self")
    A = ap.parse_args()
    assert len(A.tids) >= len(A.poses), "need one template id per pose"
    os.makedirs(f"{A.out}/a3m", exist_ok=True)

    base = json.load(open(f"{JOBS}/{A.pdb}-final-updated.json"))
    obj = json.loads(json.dumps(base[0] if isinstance(base, list) else base))
    obj["name"] = f"{A.pdb}_st"
    ent_seqs = [e["proteinChain"]["sequence"] for e in obj["sequences"] if "proteinChain" in e]
    if not ent_seqs:
        sys.exit("no protein entities")

    os.makedirs(MMCIF_DIR, exist_ok=True)
    cache = json.load(open(CACHE_F)) if os.path.exists(CACHE_F) else {}
    donors = []
    for d, cif in enumerate(A.poses):
        chains, info = pick_chains(cif, ent_seqs)
        if chains is None:
            print(f"   SKIP donor {d}: {info}")
            continue
        tid = A.tids[d]
        assert len(tid) == 4, f"tid {tid} must be 4 chars"
        if A.no_cache_write and tid not in cache:
            sys.exit(f"tid {tid} not pre-registered")
        cache.setdefault(tid, {"release_date": "2000-01-01"})
        # write-then-rename so a concurrent featuriser never sees a half-written cif
        _txt = convert(open(cif).read(), tid)
        _tmp = f"{MMCIF_DIR}/.{tid}.tmp"
        with open(_tmp, "w") as _fh:
            _fh.write(_txt)
        os.replace(_tmp, f"{MMCIF_DIR}/{tid}.cif")
        donors.append(dict(idx=d, cif=cif, tid=tid, chains=chains, contacts=info))
        print(f"   donor {d}: {tid}  chains {'/'.join(chains)}  contacts {info}")
    if not donors:
        sys.exit("no usable donors")
    if not A.no_cache_write:
        # ATOMIC, LOCKED, AND RE-READ UNDER THE LOCK. Two workers building templates at once were
        # racing here: this is a read-modify-write on a file they share, and `open(...,"w")` truncates
        # it, so the other worker could read a partial file and lose its tids. Re-reading inside the
        # lock also stops one worker's entries being clobbered by the other's stale copy. Symptom was
        # "templates consumed [0]" on arms whose templates had just been written.
        import fcntl, tempfile
        os.makedirs(os.path.dirname(CACHE_F), exist_ok=True)
        lock_f = CACHE_F + ".lock"
        with open(lock_f, "w") as _lk:
            fcntl.flock(_lk, fcntl.LOCK_EX)
            try:
                merged = json.load(open(CACHE_F)) if os.path.exists(CACHE_F) else {}
            except Exception:
                merged = {}
            merged.update(cache)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(CACHE_F))
            with os.fdopen(fd, "w") as fh:
                json.dump(merged, fh)
            os.replace(tmp, CACHE_F)          # atomic: readers see old or new, never partial
            fcntl.flock(_lk, fcntl.LOCK_UN)

    pi = 0
    for e in obj["sequences"]:
        if "proteinChain" not in e:
            continue
        q = e["proteinChain"]["sequence"]
        Lq = len(q)
        lines, hits = [">query", q], set()

        # --prepend-real: restore the FREE arm's real templates ahead of the self-poses. Written
        # verbatim from the original hmmsearch a3m so that PROTENIX_TMPL_NODEDUP=1 (which run2 needs
        # to keep the 5 near-identical self-poses distinct) cannot change WHICH real templates
        # survive -- only these 4 rows are in the file, so there is nothing left to dedup away.
        # The a3m lists several subsequence ranges per pdb chain and the selected domain name does
        # not record which; the FIRST occurrence is taken, matching the parser's own file order.
        nreal = 0
        if A.prepend_real:
            spec = {r["sequence"]: r for r in json.load(open(A.prepend_real))}
            r = spec.get(q)
            if r is None:
                sys.exit(f"--prepend-real has no entry for entity of length {Lq}")
            body = open(r["templates_path"]).read().split("\n")
            for want in r["names"]:
                for li in range(0, len(body) - 1):
                    if body[li].startswith(f">{want}/"):
                        lines += [body[li], body[li + 1]]
                        hits.add(body[li + 1])
                        nreal += 1
                        break
                else:
                    sys.exit(f"real template {want} not found in {r['templates_path']}")
            print(f"   entity {pi}: prepended {nreal} real templates {r['names']}")
        for k, dn in enumerate(donors):
            hs = list(q)
            for p in ((Lq // 3 + 5 * k) % Lq, (Lq // 2 + 1 + pi + 7 * k) % Lq):
                hs[p] = "A" if hs[p] != "A" else "G"
            hit = "".join(hs)
            assert hit != q, "hit identical to query -> parsed as an MSA row"
            assert hit not in hits, f"duplicate hit for donor {k} -> a3m would dedup it"
            hits.add(hit)
            lines += [f">{dn['tid']}_{dn['chains'][pi]}/1-{Lq} [subseq from] mol:protein "
                      f"length:{Lq}  selftmpl", hit]
        a3m = f"{A.out}/a3m/q{pi}.a3m"
        open(a3m, "w").write("\n".join(lines) + "\n")
        e["proteinChain"]["templatesPath"] = a3m
        pi += 1
    json.dump([obj], open(f"{A.out}/job.json", "w"))
    json.dump(donors, open(f"{A.out}/chosen.json", "w"), indent=1)
    print(f"   wrote {A.out}/job.json: {len(donors)} templates x {pi} entities")


if __name__ == "__main__":
    main()
