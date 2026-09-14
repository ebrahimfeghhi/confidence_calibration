#!/usr/bin/env python
"""Stage 2 builder: template each complex on its OWN 5 MSA-free stage-1 poses.

Stage 1 folded each complex with the MSA cut to its query row only, giving 5 diffusion samples with
no co-evolutionary evidence behind them. Here those 5 become the ONLY templates for a 50-seed stock
run (subsampling on, dropout on), so the question is whether MSA-free conformations, fed back, move
what the stock run produces.

NATURAL TEMPLATES ARE DROPPED. The base json points each chain at prep/<pdb>/msa/*/hmmsearch.a3m;
we overwrite templatesPath with an a3m containing only our 5 hits. Keeping the natural ones would
dilute: TemplateEmbedder averages (u = sum / num_templates, pairformer.py), so 4 natural + 5 self
gives each self-pose 1/9 of the signal instead of 1/5.

multichain_mask is left ON (the default), so a template transmits each chain's own backbone and
NOTHING about how the chains are arranged. This is a fold perturbation, not a docking one.

TRAPS (each has burned a previous arm; all are asserted here, not hoped for):
  - template ids must be EXACTLY 4 characters. Anything else -> `Invalid hit name format`, Protenix
    logs `Found 0 templates` and silently runs UNTEMPLATED while still emitting plausible poses.
  - ids must not collide with $HOME/mmcif, or a previous arm's structure is silently substituted.
  - obj["name"] must be unique per job, else jobs overwrite each other's output directory.
  - the a3m hit sequence MUST differ from the query, else it is parsed as an MSA row, not a template,
    AND from every other hit, else de-duplication by hit sequence collapses 5 templates to 1.
  - hit order is what selects: a3m hits carry sum_probs=None (template_parser.py), so the sort key is
    0.0 for all of them and the stable sort preserves file order.

Usage: .../python scripts/build_selfpose_tmpl.py --run selftm --tid-prefix z
Writes: artifacts/sweep_tmpl/<run>/*.json, artifacts/sweep_a3m/<run>_*.a3m, artifacts/<run>_jobs.json
"""
import argparse, difflib, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemmi
from cifconv import convert
from decoy_set import EXP
from build_dockq_templates import native_seqs, best_chain, mut

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="selftm")
ap.add_argument("--tid-prefix", default="z", help="1 char; ids become <p>NNN (4 chars total)")
ap.add_argument("--tid-start", type=int, default=0,
                help="offset for the numeric part, so one prefix can serve two waves without "
                     "colliding (only '8' is still free in ~/mmcif)")
ap.add_argument("--seeds", type=int, default=50)
A = ap.parse_args()
assert len(A.tid_prefix) == 1, "tid prefix must be exactly 1 character"

ART = f"{EXP}/artifacts"
JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
TMPL = f"{ART}/sweep_tmpl/{A.run}"; A3M = f"{ART}/sweep_a3m"
CACHE_F = "/home/ebrahim/common/release_date_cache.json"
os.makedirs(TMPL, exist_ok=True); os.makedirs(A3M, exist_ok=True)

cache = json.load(open(CACHE_F))
TAKEN = {os.path.basename(p)[:-4] for p in glob.glob("/home/ebrahim/mmcif/*.cif")}
picked = json.load(open(f"{ART}/allnoise_split.json"))["picked"]

jobs, meta, skipped = [], {}, []
tid_n = 0
for pdb in picked:
    poses = sorted(glob.glob(f"{ART}/sweep_out/s1_{pdb}/**/*_sample_*.cif", recursive=True))
    if len(poses) < 5:
        skipped.append(f"{pdb}: only {len(poses)} stage-1 poses"); continue
    poses = poses[:5]

    # register each pose as its own 4-char template id
    tids = []
    for cif in poses:
        tid = f"{A.tid_prefix}{A.tid_start + tid_n:03d}"; tid_n += 1
        assert len(tid) == 4, f"template id {tid!r} is not 4 characters"
        assert tid not in TAKEN, f"template id {tid} already in /home/ebrahim/mmcif"
        TAKEN.add(tid)
        cache.setdefault(tid, {"release_date": "2000-01-01"})
        open(f"/home/ebrahim/mmcif/{tid}.cif", "w").write(convert(open(cif).read(), tid))
        tids.append((tid, cif))

    base = json.load(open(f"{JDIR}/{pdb}-final-updated.json"))
    obj = json.loads(json.dumps(base[0] if isinstance(base, list) else base))
    natseq = native_seqs(pdb)
    qseqs = [e["proteinChain"]["sequence"] for e in obj["sequences"] if "proteinChain" in e]

    # ONE ARM PER DONOR. Putting all 5 poses in a single run does not work: after structures are
    # loaded, get_templates de-duplicates on res.features["template_sequence"] -- the sequence read
    # from the mmcif, not the a3m hit line -- and all 5 poses are the same molecule, so they collapse
    # to 1 ("Found 1 templates"). Mutating the a3m hit only defeats the earlier, a3m-level dedup.
    # With one template per run the dedup never fires, and it also avoids the averaging dilution
    # (TemplateEmbedder does u = sum/num_templates) and tells us WHICH donor helped.
    for donor_idx, (tid, cif) in enumerate(tids):
        obj = json.loads(json.dumps(base[0] if isinstance(base, list) else base))
        qseqs = [e["proteinChain"]["sequence"] for e in obj["sequences"] if "proteinChain" in e]
        pi, ok = 0, True
        for e in obj["sequences"]:
            if "proteinChain" not in e:
                continue
            q = qseqs[pi]; L = len(q)
            st = gemmi.read_structure(cif); st.setup_entities()
            mc = best_chain(st, q)
            if mc is None:
                skipped.append(f"{pdb} q{pi} {tid}: no model chain matched"); ok = False; break
            hs = list(q)
            for pos in (L // 3, 2 * L // 3):        # hit MUST differ from the query
                hs[pos] = mut(hs[pos])
            a3mp = f"{A3M}/{A.run}_{pdb}_d{donor_idx}_q{pi}.a3m"
            open(a3mp, "w").write(
                f">query\n{q}\n"
                f">{tid}_{mc.name}/1-{L} [subseq from] mol:protein length:{L}  {A.run}\n"
                f"{''.join(hs)}\n")
            e["proteinChain"]["templatesPath"] = a3mp
            pi += 1
        if not ok:
            continue
        name = f"{pdb}_selftm_d{donor_idx}"
        obj["name"] = name
        jp = f"{TMPL}/{name}.json"
        json.dump([obj], open(jp, "w"))
        arm = f"{A.run}_{pdb}_d{donor_idx}"
        jobs.append({"run": arm, "pdb": pdb, "json": jp, "name": name,
                     "seeds": list(range(101, 101 + A.seeds)), "donor": donor_idx, "tid": tid})
        meta[arm] = {"pdb": pdb, "donor": donor_idx, "tid": tid, "n_chains": pi}

    _unused_pi, _unused_ok, n_hits = 0, True, {}
    for e in []:
        if "proteinChain" not in e:
            continue
        q = qseqs[pi]; L = len(q)
        lines = [f">query\n{q}"]
        for hit_idx, (tid, cif) in enumerate(tids):
            st = gemmi.read_structure(cif); st.setup_entities()
            mc = best_chain(st, q)
            if mc is None:
                skipped.append(f"{pdb} q{pi} {tid}: no model chain matched"); ok = False; break
            hs = list(q)
            # Each hit must differ from the query AND from every other hit: the parser de-duplicates
            # by hit sequence (template_utils.py), so 5 identically-mutated hits collapse to 1 and
            # the run silently uses a single template -- observed as "Found 1 templates".
            for pos in (L // 7 + hit_idx, 3 * L // 7 + hit_idx, 5 * L // 7 + hit_idx):
                if pos < L:
                    hs[pos] = mut(hs[pos])
            lines.append(f">{tid}_{mc.name}/1-{L} [subseq from] mol:protein length:{L}  {A.run}\n"
                         f"{''.join(hs)}")
        if not ok: break
        a3mp = f"{A3M}/{A.run}_{pdb}_q{pi}.a3m"
        open(a3mp, "w").write("\n".join(lines) + "\n")
        e["proteinChain"]["templatesPath"] = a3mp   # natural hmmsearch templates dropped
        n_hits[f"q{pi}"] = len(tids)
        pi += 1

json.dump(cache, open(CACHE_F, "w"))
json.dump(jobs, open(f"{ART}/{A.run}_jobs.json", "w"), indent=1)
json.dump(meta, open(f"{ART}/{A.run}_meta.json", "w"), indent=1)
open(f"{ART}/{A.run}_queue.txt", "w").write("\n".join(j["run"] for j in jobs) + "\n")
for j in jobs:
    json.dump([j], open(f"{ART}/{j['run']}_jobs.json", "w"), indent=1)

print(f"built {len(jobs)} arms ({len(jobs)//5} complexes x 5 donors) x {A.seeds} seeds "
      f"= {len(jobs)*A.seeds} seeds total")
print(f"  template ids {A.tid_prefix}{A.tid_start:03d}-{A.tid_prefix}{A.tid_start+tid_n-1:03d}")
if skipped:
    print(f"  SKIPPED {len(skipped)}:")
    for s in skipped: print(f"    {s}")
