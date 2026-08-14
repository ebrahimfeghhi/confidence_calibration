#!/usr/bin/env python
"""Build an allpose arm for ANY pdb: template from every one of its 250 free poses, N seeds each.

This is build_allpose_templates.py + build_allpose10.py generalised past the hardcoded 8che, so the
250x10 design can be replicated on a second interface. First use: 8bf4, to test whether the sweep
reliably finds a good donor on an interface OTHER than the one the method was developed on.

DESIGN (identical to allpose8che10):
  250 jobs, one per free-run pose (seeds 101-150 x samples 0-4). Each job templates ALL protein
  chains of the complex from that single pose, so the only thing varying across jobs is which pose
  donated the folds. N_sample=5.
  SEEDS PER DONOR = the donor's OWN seed + (N-1) drawn at random per donor (rng seeded). Own seed is
  included so no donor is judged purely on foreign seeds; the other N-1 are random PER DONOR so one
  unlucky seed cannot depress every donor at once (which is exactly what sank allpose8che at seed 101).

TEMPLATES CARRY NO DOCKING. multichain_mask zeroes every cross-chain template feature
(pairformer.py:1062-1076), so a donor transmits only its intra-chain backbone conformations.

TRAPS THIS BUILDER GUARDS (each one has burned a previous arm):
  - 4-CHARACTER template ids. Protenix raises `Invalid hit name format` on anything else and logs
    `Found 0 templates`, silently producing an UNTEMPLATED run that still emits plausible poses.
    Three arms were scored and believed for a week this way. tids are <prefix>NNN, asserted 4 chars
    and asserted absent from /home/ebrahim/mmcif.
  - obj["name"] unique per job, else every template writes into one output directory.
  - label_asym_id for native chain identification (native_seqs), never auth_asym_id -- they differ
    for 8bf4 specifically.
  - the a3m hit must differ from the query (2 mutated residues), else Protenix reads it as an MSA row
    rather than a template.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/build_allpose_generic.py \
            --pdb 8bf4 --run allpose8bf410 --tid-prefix b --seeds 10
Writes: artifacts/sweep_tmpl/<run>/*.json, artifacts/sweep_a3m/<run>_*.a3m,
        artifacts/<run>_jobs.json, artifacts/<run>_donors.json
"""
import argparse, json, os, sys, glob, difflib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gemmi
from cifconv import convert
from decoy_set import EXP, PRED, SEEDS, SAMPLES
from build_dockq_templates import native_seqs, best_chain, mut

ap = argparse.ArgumentParser()
ap.add_argument("--pdb", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--tid-prefix", required=True, help="1 char; tids become <p>000-<p>249 (4 chars)")
ap.add_argument("--tid-start", type=int, default=0,
                help="offset for the numeric part; lets one prefix serve two arms (see above)")
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--pool-lo", type=int, default=101)
ap.add_argument("--pool-hi", type=int, default=1100)
ap.add_argument("--rng", type=int, default=0)
A = ap.parse_args()

assert len(A.tid_prefix) == 1, "tid prefix must be exactly 1 character (tids must total 4)"
# Only 8 first-characters remain unused in $HOME/mmcif (a g l q s t x 0), fewer than the complexes
# left to run. --tid-start offsets the numbering so one prefix serves two arms without collision:
# a000-a249 for one, a250-a499 for the next. Still exactly 4 characters, which Protenix requires.
PDB, RUN = A.pdb, A.run
JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
TMPL = f"{EXP}/artifacts/sweep_tmpl/{RUN}"
A3M = f"{EXP}/artifacts/sweep_a3m"
CACHE_F = "/home/ebrahim/common/release_date_cache.json"
os.makedirs(TMPL, exist_ok=True); os.makedirs(A3M, exist_ok=True)

cache = json.load(open(CACHE_F))
TAKEN = {os.path.basename(p)[:-4] for p in glob.glob("/home/ebrahim/mmcif/*.cif")}
base = json.load(open(f"{JDIR}/{PDB}-final-updated.json"))
natseq = native_seqs(PDB)
rng = np.random.default_rng(A.rng)
pool = np.arange(A.pool_lo, A.pool_hi + 1)

jobs, donors, skipped = [], {}, []
for n, (s, k) in enumerate([(s, k) for s in SEEDS for k in SAMPLES]):
    cif = f"{PRED}/{PDB}/seed_{s}/predictions/{PDB}_sample_{k}.cif"
    if not os.path.exists(cif):
        skipped.append(f"seed{s}/s{k}: cif missing"); continue

    tid = f"{A.tid_prefix}{A.tid_start + n:03d}"          # 4 chars EXACTLY -- see docstring
    assert len(tid) == 4, f"template id {tid!r} is not 4 characters"
    assert tid not in TAKEN, f"template id {tid} already registered in /home/ebrahim/mmcif"
    TAKEN.add(tid)
    cache.setdefault(tid, {"release_date": "2000-01-01"})
    open(f"/home/ebrahim/mmcif/{tid}.cif", "w").write(convert(open(cif).read(), tid))

    obj = json.loads(json.dumps(base[0] if isinstance(base, list) else base))
    qseqs = [e["proteinChain"]["sequence"] for e in obj["sequences"] if "proteinChain" in e]
    st = gemmi.read_structure(cif); st.setup_entities()

    pi, chains, ok = 0, {}, True
    for e in obj["sequences"]:
        if "proteinChain" not in e:
            continue
        q = qseqs[pi]
        nc = max(natseq, key=lambda c: difflib.SequenceMatcher(None, natseq[c], q,
                                                               autojunk=False).ratio())
        mc = best_chain(st, q)
        if mc is None:
            skipped.append(f"seed{s}/s{k} q{pi}: no model chain matched"); ok = False; break
        L = len(q); hs = list(q)
        for pos in (L // 3, 2 * L // 3):                # hit MUST differ from query
            hs[pos] = mut(hs[pos])
        a3mp = f"{A3M}/{RUN}_{s}_{k}_q{pi}.a3m"
        open(a3mp, "w").write(
            f">query\n{q}\n>{tid}_{mc.name}/1-{L} [subseq from] mol:protein length:{L}  {RUN}\n"
            f"{''.join(hs)}\n")
        e["proteinChain"]["templatesPath"] = a3mp
        chains[f"q{pi}"] = nc
        pi += 1
    if not ok:
        continue

    name = f"{PDB}_allpose_{s}_{k}"                     # unique per job -- see docstring
    obj["name"] = name
    jp = f"{TMPL}/{name}.json"
    json.dump([obj], open(jp, "w"))

    cand = pool[pool != s]
    extra = rng.choice(cand, size=A.seeds - 1, replace=False)
    seedlist = [int(s)] + sorted(int(x) for x in extra)  # own seed ALWAYS included, first
    jobs.append({"run": RUN, "pdb": PDB, "interface": f"{PDB}_donor_{s}_{k}",
                 "json": jp, "name": name, "seeds": seedlist, "set": "decoy",
                 "donor_seed": s, "donor_sample": k, "own_seed_included": True})
    donors[name] = {"pdb": PDB, "donor_seed": s, "donor_sample": k, "tid": tid,
                    "n_chains_templated": len(chains), "chains": chains}

json.dump(cache, open(CACHE_F, "w"))
json.dump(jobs, open(f"{EXP}/artifacts/{RUN}_jobs.json", "w"), indent=1)
json.dump(donors, open(f"{EXP}/artifacts/{RUN}_donors.json", "w"), indent=1)

assert len(jobs) == len({j["name"] for j in jobs}), "duplicate job name -- outputs would collide"
assert len({d["tid"] for d in donors.values()}) == len(donors), "duplicate template id"
assert all(len(set(j["seeds"])) == A.seeds for j in jobs), "seed count / duplicate seed problem"
assert all(j["donor_seed"] in j["seeds"] for j in jobs), "a donor's own seed is missing"
nch = {d["n_chains_templated"] for d in donors.values()}
assert len(nch) == 1, f"jobs templated different numbers of chains: {nch}"
tot = len(jobs) * A.seeds
print(f"BUILT {len(jobs)} donor jobs x {A.seeds} seeds = {tot} seeds ({tot * 5} poses)")
print(f"   all templating {nch.pop()} chains; tids {A.tid_prefix}{A.tid_start:03d}-{A.tid_prefix}{A.tid_start+len(jobs)-1:03d}")
print(f"   seeds per donor = own + {A.seeds-1} random from {A.pool_lo}-{A.pool_hi} (rng={A.rng})")
print(f"   example: {jobs[0]['name']} donor_seed {jobs[0]['donor_seed']} -> {jobs[0]['seeds']}")
if skipped:
    print(f"SKIPPED {len(skipped)}:")
    for x in skipped[:10]:
        print("  " + x)
