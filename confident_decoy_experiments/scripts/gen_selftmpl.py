#!/usr/bin/env python
"""Generational self-templating: each generation is templated on the previous generation's poses.

TEN INDEPENDENT LINEAGES, five generations deep, over the free run's own seed range 101-150:

    V1  seeds 101-110   REUSED from the existing free run (natural hmmsearch templates)
    V2  seeds 111-120   seed 111 <- seed 101's 5 poses, seed 112 <- seed 102, ...
    V3  seeds 121-130   seed 121 <- seed 111's 5 poses, ...
    V4  seeds 131-140
    V5  seeds 141-150

5 generations x 10 lineages x 5 diffusion samples = 250 poses, matching the free run exactly -- same
count, same seed labels -- so the two arms are directly comparable. Lineage L never mixes with
lineage L': ten separate chains, each conditioned only on its own parent. V1 costs nothing because
those poses are already on disk and already scored (--v1-fresh regenerates them instead).

V2 ONWARD GET NO NATURAL TEMPLATES. Only the parent's 5 poses. An earlier self-templating attempt fed
the self-template alongside the model's usual hmmsearch hits, where TemplateEmbedder's mean over
templates (u = sum/num_templates) diluted it to 1/(1+natural). Here the self-poses are the entire
template set.

MULTICHAIN MASK IS ON (Protenix's default, and the deliberate choice here). The template featuriser
therefore zeroes every cross-chain pair, so what feeds back between generations is each chain's
INTRA-CHAIN FOLD and not the dock -- the model does not see where it previously placed the antigen.
This tests whether conformational feedback alone changes anything. --mask-off carries the dock as
well, and is the obvious follow-up if the mask-on arm shows any effect.

TWO SILENT FAILURE MODES, both asserted rather than assumed (learned the hard way):
  - a3m-level dedup collapses N poses of one molecule to ONE template, since they share a sequence.
    Defeated by mutating a different residue per hit; the run then logs "Found 5 templates" not
    "Found 1". Checked from the log every run.
  - PROTENIX_MAX_TEMPLATES caps the kept set. Set to exactly 5 with candidates well above it.

Usage: .../python scripts/gen_selftmpl.py --pdb 8qf4 --gpu 3
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemmi
from cifconv import convert
from decoy_set import EXP
from build_dockq_templates import best_chain, mut

PROTENIX = "/home/ebrahim/Protenix"
JDIR = f"{PROTENIX}/foldbench_abag/jsons"
FREE = f"{PROTENIX}/foldbench_abag/predictions_template"
PY = f"{PROTENIX}/.venv/bin/python"

ap = argparse.ArgumentParser()
ap.add_argument("--pdb", required=True)
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--generations", type=int, default=5)
ap.add_argument("--lineages", type=int, default=10)
ap.add_argument("--samples", type=int, default=5, help="diffusion samples per seed")
ap.add_argument("--n-step", type=int, default=200)
ap.add_argument("--mask-off", action="store_true",
                help="PROTENIX_MULTICHAIN_MASK=0, letting the self-template carry the DOCK. "
                     "Default is mask ON (Protenix's own default), where cross-chain pairs are "
                     "zeroed and the template feeds back intra-chain fold ONLY.")
ap.add_argument("--v1-fresh", action="store_true",
                help="generate V1 instead of reusing the free run's seeds 101-110, which are "
                     "already on disk and already scored")
ap.add_argument("--tag", default="")
A = ap.parse_args()

TAG = A.tag or f"{A.pdb}_gen{'_maskoff' if A.mask_off else ''}"
OUT = f"{EXP}/artifacts/selftmpl/{TAG}"
A3M = f"{OUT}/a3m"
os.makedirs(A3M, exist_ok=True)

base = json.load(open(f"{JDIR}/{A.pdb}-final-updated.json"))
base = base[0] if isinstance(base, list) else base
QSEQS = [e["proteinChain"]["sequence"] for e in base["sequences"] if "proteinChain" in e]

# Template ids MUST already exist in /home/ebrahim/common/release_date_cache.json. The featurizer
# silently drops any template whose id it cannot date -- it logs "Found 0 templates" and the run
# otherwise looks completely normal, so every generation would quietly become a plain free run. An
# earlier version of this script used "9<gpu>NN", which is NOT registered, and produced exactly that.
# 8500-8799 are all registered (release_date 2000-01-01, so they pass any cutoff); each GPU takes a
# disjoint block of 10 so concurrent workers cannot overwrite each other's mmcif.
TID_LO = 8500 + A.gpu * 10
assert A.samples <= 10, "per-GPU tid block is 10 wide"


SEED0 = 101   # the free run's seed range, so V1 can reuse its existing poses verbatim


def seed_of(gen, lineage):
    """V1 lineage L is seed 101+L; V2 is 111+L; ... V5 is 141+L. Seeds 101-150 overall, matching the
    free run's range exactly so the two arms are labelled the same way."""
    return SEED0 + gen * A.lineages + lineage


def poses_of(gen, lineage):
    """The cifs a given (generation, lineage) produced, in sample order."""
    if gen == 0 and not A.v1_fresh:
        s = SEED0 + lineage
        return sorted(glob.glob(f"{FREE}/{A.pdb}/seed_{s}/predictions/{A.pdb}_sample_*.cif"))
    s = seed_of(gen, lineage)
    return sorted(glob.glob(f"{OUT}/decoded/{TAG}_g{gen}_l{lineage}/seed_{s}/predictions/*_sample_*.cif"))


def ok_templates(gen, lineage):
    """True only if this (gen, lineage) log shows the expected template count. Guards resume against
    re-adopting output from a run where template injection silently failed."""
    log = f"{OUT}/logs/g{gen}_l{lineage}.log"
    if not os.path.exists(log):
        return False
    found = {l.split("Found ")[1].split()[0]
             for l in open(log, errors="ignore") if "templates for sequence" in l}
    return bool(found) and all(int(f) >= A.samples for f in found)


def build_a3m(parent_cifs, gen, lineage):
    """One a3m per protein chain: the query, then one hit per parent pose.

    Each hit carries mutations at positions unique to its index. Without that, Protenix's a3m-level
    dedup on hit_sequence collapses all 5 to a single template and the run looks entirely normal
    while conditioning on one pose.
    """
    paths, tids = [], []
    for j, cif in enumerate(parent_cifs):
        tid = f"{TID_LO + j:04d}"
        assert len(tid) == 4, tid
        open(f"/home/ebrahim/mmcif/{tid}.cif", "w").write(convert(open(cif).read(), tid))
        tids.append(tid)
    for pi, q in enumerate(QSEQS):
        L = len(q)
        lines = [f">query\n{q}"]
        for j, (tid, cif) in enumerate(zip(tids, parent_cifs)):
            st = gemmi.read_structure(cif)
            st.setup_entities()
            mc = best_chain(st, q)
            if mc is None:
                continue
            hs = list(q)
            for pos in (L // 7 + j, 3 * L // 7 + j, 5 * L // 7 + j):
                if pos < L:
                    hs[pos] = mut(hs[pos])
            lines.append(f">{tid}_{mc.name}/1-{L} [subseq from] mol:protein length:{L}  {TAG}\n"
                         f"{''.join(hs)}")
        p = f"{A3M}/g{gen}_l{lineage}_q{pi}.a3m"
        open(p, "w").write("\n".join(lines) + "\n")
        paths.append(p)
    return paths, tids


def run(gen, lineage, a3m_paths):
    """One inference run. a3m_paths=None -> V1, natural templates, no self-conditioning."""
    seed = seed_of(gen, lineage)
    obj = json.loads(json.dumps(base))
    if a3m_paths is not None:
        pi = 0
        for e in obj["sequences"]:
            if "proteinChain" not in e:
                continue
            e["proteinChain"]["templatesPath"] = a3m_paths[pi]
            pi += 1
    obj["name"] = f"{TAG}_g{gen}_l{lineage}"
    jp = f"{OUT}/{obj['name']}.json"
    json.dump([obj], open(jp, "w"))

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(A.gpu), LAYERNORM_TYPE="openfold")
    if a3m_paths is not None:
        env.update(PROTENIX_TMPL_NODEDUP="1",
                   PROTENIX_MAX_TEMPLATES=str(A.samples),
                   PROTENIX_MAX_TEMPLATE_CANDIDATES=str(A.samples + 8))
        if A.mask_off:
            env["PROTENIX_MULTICHAIN_MASK"] = "0"   # let the template carry the DOCK

    log = f"{OUT}/logs/g{gen}_l{lineage}.log"
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as fh:
        subprocess.run([PY, "runner/inference.py", "--model_name", "protenix-v2",
                        "--load_checkpoint_dir", "/home/ebrahim/checkpoint",
                        "--input_json_path", jp, "--dump_dir", f"{OUT}/decoded",
                        "--seeds", str(seed),
                        "--sample_diffusion.N_sample", str(A.samples),
                        "--sample_diffusion.N_step", str(A.n_step),
                        "--use_template", "true"],
                       cwd=PROTENIX, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=14400)

    found = sorted({l.split("Found ")[1].split()[0]
                    for l in open(log, errors="ignore") if "templates for sequence" in l})
    n = len(poses_of(gen, lineage))
    return found, n


print(f"{A.pdb}: {A.generations} generations x {A.lineages} lineages x {A.samples} samples "
      f"= {A.generations * A.lineages * A.samples} poses | multichain_mask="
      f"{'OFF (dock carried)' if A.mask_off else 'ON (intra-chain fold only)'}", flush=True)

manifest = []
for gen in range(A.generations):
    for lin in range(A.lineages):
        # Resume must verify the templates LANDED, not merely that poses exist. A run that found 0
        # templates still writes its 5 poses, and an earlier version of this loop treated those as
        # done -- silently preserving free-run output in a self-templating arm forever.
        if len(poses_of(gen, lin)) >= A.samples and (gen == 0 or ok_templates(gen, lin)):
            print(f"  g{gen} l{lin}: already done, skipping", flush=True)
            continue
        if gen == 0:
            if not A.v1_fresh:
                print(f"  g0 l{lin}: reusing free seeds {int(A.v1_from_free)+lin}", flush=True)
                continue
            found, n = run(0, lin, None)
        else:
            parents = poses_of(gen - 1, lin)
            if len(parents) < A.samples:
                print(f"  g{gen} l{lin}: parent has {len(parents)} poses, need {A.samples} -- SKIP",
                      flush=True)
                continue
            a3m, tids = build_a3m(parents, gen, lin)
            found, n = run(gen, lin, a3m)
            # the a3m trick worked only if the model actually kept all 5
            if found and any(int(f) < A.samples for f in found):
                print(f"  *** g{gen} l{lin}: only {found} templates kept, expected {A.samples} "
                      f"-- dedup or cap is dropping poses ***", flush=True)
        print(f"  g{gen} l{lin}: templates_found={found} poses={n}", flush=True)
        manifest.append({"gen": gen, "lineage": lin, "seed": seed_of(gen, lin),
                         "templates_found": found, "n_poses": n})
        json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)

tot = sum(len(poses_of(g, l)) for g in range(A.generations) for l in range(A.lineages))
print(f"\n{A.pdb}: {tot} poses -> {OUT}", flush=True)
