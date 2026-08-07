#!/usr/bin/env python
"""Build the `allpose8che` arm: template 8che from EVERY ONE of its 250 free-run poses, 1 pose out.

WHY. Every template arm so far picked ONE donor by some criterion (best DockQ, top ipTM, best
paratope RMSD, min-max RMSD, the 20 ipTM leaders) and asked whether that criterion picks well. This
removes the criterion entirely: run all 250 candidate donors and observe the full response surface.
On 8che the arms diverge maximally -- dockqdonor reaches DockQ 0.958 on 8che_BE while leader20 at 50
seeds/template (5000 poses) tops out at 0.671 -- and the two donors differ by only 0.093 donor DockQ
(0.529 vs 0.436) and ~0.36 A paratope RMSD. 250 donors spanning the whole pool is the measurement
that says which donor property, if any, tracks the response.

DESIGN. 250 jobs = one per free-run pose (seeds 101-150 x samples 0-4). Each job templates ALL THREE
chains of 8che (A, B, E) from that single pose, so the ONLY thing varying across jobs is which pose
donated the folds. 1 seed (101) at N_sample=5 => 5 output poses per template, 1250 poses total. Five
samples per donor gives each donor its own within-donor spread, so a donor's response is a small
distribution rather than a single draw -- necessary to tell a real donor effect from sampling noise,
which a 1-pose-per-donor design cannot do.

COVERS BOTH INTERFACES. 8che_AE and 8che_BE are both canonical confident decoys and share the same
predictions, so one run scores both -- no separate arm needed.

TEMPLATES CARRY NO DOCKING. multichain_mask zeroes every cross-chain template feature
(pairformer.py:1062-1076), so a donor transmits only its intra-chain backbone conformations, never
its binding arrangement.

TRAPS THIS BUILDER GUARDS (each one has burned a previous arm):
  - 4-CHARACTER template ids. Protenix raises `Invalid hit name format` on anything else and logs
    `Found 0 templates`, silently producing an UNTEMPLATED run that still emits plausible poses. Three
    arms (topdockq/topiptm/sposefv) were scored and believed for a week this way. tids here are
    p000-p249, asserted 4 chars and asserted not already present in /home/ebrahim/mmcif.
  - obj["name"] unique per job. leader20 initially left it unset, so all 20 templates of a pdb wrote
    into one output directory.
  - label_asym_id for native chain identification (native_seqs), never auth_asym_id -- they differ for
    8che specifically.
  - the a3m hit must differ from the query (2 mutated residues), else Protenix reads it as an MSA row
    rather than a template.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/build_allpose_templates.py
Writes: artifacts/sweep_tmpl/allpose8che/*.json, artifacts/sweep_a3m/allpose8che_*.a3m,
        artifacts/allpose8che_jobs.json, artifacts/allpose8che_donors.json
"""
import json, os, sys, glob, difflib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemmi
from cifconv import convert
from decoy_set import EXP, PRED, SEEDS, SAMPLES
from build_dockq_templates import TLC, native_seqs, best_chain, mut

PDB = "8che"
JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
TMPL = f"{EXP}/artifacts/sweep_tmpl/allpose8che"
A3M = f"{EXP}/artifacts/sweep_a3m"
CACHE_F = "/home/ebrahim/common/release_date_cache.json"
os.makedirs(TMPL, exist_ok=True); os.makedirs(A3M, exist_ok=True)

cache = json.load(open(CACHE_F))
TAKEN = {os.path.basename(p)[:-4] for p in glob.glob("/home/ebrahim/mmcif/*.cif")}

base = json.load(open(f"{JDIR}/{PDB}-final-updated.json"))
natseq = native_seqs(PDB)
jobs, donors, skipped = [], {}, []

for n, (s, k) in enumerate([(s, k) for s in SEEDS for k in SAMPLES]):
    cif = f"{PRED}/{PDB}/seed_{s}/predictions/{PDB}_sample_{k}.cif"
    if not os.path.exists(cif):
        skipped.append(f"seed{s}/s{k}: cif missing"); continue

    tid = f"p{n:03d}"                                   # 4 chars EXACTLY -- see docstring
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
        a3mp = f"{A3M}/allpose8che_{s}_{k}_q{pi}.a3m"
        open(a3mp, "w").write(
            f">query\n{q}\n>{tid}_{mc.name}/1-{L} [subseq from] mol:protein length:{L}  allpose8che\n"
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
    jobs.append({"run": "allpose8che", "pdb": PDB, "interface": f"{PDB}_donor_{s}_{k}",
                 "json": jp, "name": name, "seeds": [101], "set": "decoy",
                 "donor_seed": s, "donor_sample": k})
    donors[name] = {"pdb": PDB, "donor_seed": s, "donor_sample": k, "tid": tid,
                    "n_chains_templated": len(chains), "chains": chains}

json.dump(cache, open(CACHE_F, "w"))
json.dump(jobs, open(f"{EXP}/artifacts/allpose8che_jobs.json", "w"), indent=1)
json.dump(donors, open(f"{EXP}/artifacts/allpose8che_donors.json", "w"), indent=1)

assert len(jobs) == len({j["name"] for j in jobs}), "duplicate job name -- outputs would collide"
assert len({d["tid"] for d in donors.values()}) == len(donors), "duplicate template id"
nch = {d["n_chains_templated"] for d in donors.values()}
assert nch == {3}, f"expected all 3 chains templated per job, got {nch}"
print(f"BUILT {len(jobs)} jobs, {len(donors)} unique template ids, all templating "
      f"{nch.pop()} chains -> artifacts/allpose8che_jobs.json")
if skipped:
    print(f"SKIPPED {len(skipped)}:")
    for x in skipped[:10]:
        print("  " + x)
