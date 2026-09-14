#!/usr/bin/env python
"""Iterative self-templating: fold, add the pose to the template set, refold, repeat.

One complex, N iterations. Each iteration runs stock inference (MSA subsampling ON, dropout ON) with
1 seed and 1 diffusion sample, then appends that pose to the template list for the next iteration.
Templates ACCUMULATE, so iteration k is templated on all k-1 previous poses plus the natural
hmmsearch hits.

multichain_mask is left ON (the default), so each accumulated pose contributes its intra-chain
backbone and nothing about the dock.

TWO THINGS THAT SILENTLY BREAK THIS, both guarded here:
  - de-duplication on template_sequence collapses N conformations of one molecule to a single
    template, since they share a sequence. PROTENIX_TMPL_NODEDUP=1 disables it. Without this the run
    logs "Found 1 templates" and looks completely normal.
  - PROTENIX_MAX_TEMPLATES caps how many are kept; it must grow with the accumulating set or later
    iterations silently drop the newest poses.
Both are asserted from the run log each iteration rather than assumed.

Templates are AVERAGED by TemplateEmbedder (u = sum/num_templates), so iteration k's newest pose
contributes 1/(k+natural). The loop therefore gets stiffer as it goes -- expected, not a bug.

Usage: .../python scripts/iter_selftmpl.py --pdb 8bf4 --gpu 0 --iters 50 --tid-prefix 7 --tid-start 0
"""
import argparse, glob, json, os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemmi
from cifconv import convert
from decoy_set import EXP
from build_dockq_templates import best_chain, mut

ap = argparse.ArgumentParser()
ap.add_argument("--pdb", required=True)
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--iters", type=int, default=50)
ap.add_argument("--tid-prefix", required=True, help="1 char")
ap.add_argument("--tid-start", type=int, required=True)
ap.add_argument("--run", default="iter")
A = ap.parse_args()
assert len(A.tid_prefix) == 1

ART = f"{EXP}/artifacts"
JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
RUN = f"{A.run}_{A.pdb}"
TMPL = f"{ART}/sweep_tmpl/{RUN}"; A3M = f"{ART}/sweep_a3m/{RUN}"
OUT = f"{ART}/sweep_out/{RUN}"; LOGS = f"{ART}/{RUN}_logs"
for d in (TMPL, A3M, OUT, LOGS): os.makedirs(d, exist_ok=True)
CACHE_F = "/home/ebrahim/common/release_date_cache.json"
PY = "/home/ebrahim/Protenix/.venv/bin/python"

base = json.load(open(f"{JDIR}/{A.pdb}-final-updated.json"))
base = base[0] if isinstance(base, list) else base
# the natural hmmsearch a3m per chain, which we keep and append to
nat = [e["proteinChain"].get("templatesPath", "") for e in base["sequences"] if "proteinChain" in e]

hits = [[] for _ in nat]        # accumulated self-template hit records, per chain
log = []
for it in range(A.iters):
    TAKEN = {os.path.basename(p)[:-4] for p in glob.glob("/home/ebrahim/mmcif/*.cif")}
    obj = json.loads(json.dumps(base))
    qseqs = [e["proteinChain"]["sequence"] for e in obj["sequences"] if "proteinChain" in e]

    # write this iteration's a3m per chain: natural hits first, then every self-pose so far
    pi = 0
    for e in obj["sequences"]:
        if "proteinChain" not in e: continue
        # SELF-HITS FIRST. Templates are selected in FILE ORDER (a3m hits carry sum_probs=None, so
        # the sort key is 0.0 for all and the stable sort preserves order). The natural a3m has ~670
        # hits, so anything appended after them is never reached -- raising the cap would just pull
        # in more natural templates. Query line first, then our poses, then the natural hits.
        lines = []
        nat_txt = ""
        if nat[pi] and os.path.exists(nat[pi]):
            nat_txt = open(nat[pi]).read().rstrip("\n")
        lines += hits[pi]
        if nat_txt:
            lines.append(nat_txt)
        if lines:
            a3mp = f"{A3M}/{A.pdb}_it{it:03d}_q{pi}.a3m"
            open(a3mp, "w").write("\n".join(lines) + "\n")
            e["proteinChain"]["templatesPath"] = a3mp
        pi += 1

    name = f"{A.pdb}_it{it:03d}"
    obj["name"] = name
    jp = f"{TMPL}/{name}.json"
    json.dump([obj], open(jp, "w"))

    env = dict(os.environ,
               CUDA_VISIBLE_DEVICES=str(A.gpu),
               PROTENIX_TMPL_NODEDUP="1",
               # all accumulated self-poses + the 4 natural templates stock would have used
               PROTENIX_MAX_TEMPLATES=str(len(hits[0]) + 4),
               LAYERNORM_TYPE="openfold")
    lf = f"{LOGS}/{name}.log"
    with open(lf, "w") as fh:
        # Mirror sweep_sched.py's command exactly. Omitting --use_template true skips template
        # featurisation entirely (no "Found N templates" line, run silently untemplated), and the
        # model name/checkpoint must match or a different checkpoint is loaded.
        subprocess.run([PY, "runner/inference.py",
                        "--model_name", "protenix-v2",
                        "--load_checkpoint_dir", "/home/ebrahim/checkpoint",
                        "--input_json_path", jp, "--dump_dir", OUT,
                        "--seeds", "101",
                        "--sample_diffusion.N_sample", "1",
                        "--sample_diffusion.N_step", "200",
                        "--diffusion_chunk_size", "1",
                        "--use_template", "true"],
                       cwd="/home/ebrahim/Protenix",
                       env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=7200)

    found = sorted(set(l.split("Found ")[1].split()[0]
                       for l in open(lf, errors="ignore") if "templates for sequence" in l))
    cif = sorted(glob.glob(f"{OUT}/{name}/**/*_sample_0.cif", recursive=True))
    if not cif:
        # A transient failure (a co-tenant spiking memory -> CUDA OOM) previously aborted the whole
        # complex and then every later complex on that card within seconds, leaving the GPU empty
        # for someone else to claim. Retry before giving up on the complex.
        _fails = sum(1 for e in log[-3:] if "consec_fail" in e) + 1
        print(f"[{A.pdb}] iter {it}: no pose (failure {_fails}/3)", flush=True)
        log.append({"iter": it, "consec_fail": _fails, "templates_found": found})
        json.dump(log, open(f"{ART}/{RUN}_log.json", "w"), indent=1)
        if _fails >= 3:
            print(f"[{A.pdb}] 3 consecutive failures -- moving on", flush=True)
            break
        time.sleep(60)
        continue
    cif = cif[-1]

    # register the new pose as a template for the next iteration
    tid = f"{A.tid_prefix}{A.tid_start + it:03d}"
    # Template ids are a scarce global resource (~555 left under prefix 8, no free first-char), so
    # each GPU owns a 50-id block and RECYCLES it for its next complex. Overwriting is safe because
    # the previous complex has finished and its templates are no longer referenced.
    assert len(tid) == 4, f"template id {tid!r} is not 4 characters"
    # The release-date cache is PRE-REGISTERED for ids 8500-8699 before launch and only READ here.
    # Four workers doing read-modify-write on this one 9 MB file raced and left it transiently
    # invalid, which killed four complexes on gpu0 at the json.load with no log to show for it.
    open(f"/home/ebrahim/mmcif/{tid}.cif", "w").write(convert(open(cif).read(), tid))

    st = gemmi.read_structure(cif); st.setup_entities()
    for pj, q in enumerate(qseqs):
        mc = best_chain(st, q)
        if mc is None: continue
        L = len(q); hs = list(q)
        # Each hit must differ from the query AND from every earlier hit. There are TWO dedups:
        # one on the a3m hit_sequence (before structures load) and one on the mmcif-derived
        # template_sequence (after). PROTENIX_TMPL_NODEDUP only disables the second, so the a3m
        # sequences must genuinely differ -- offset the mutated positions by the iteration index.
        for pos in (L // 7 + it, 3 * L // 7 + it, 5 * L // 7 + it):
            if pos < L:
                hs[pos] = mut(hs[pos])
        hits[pj].append(f">{tid}_{mc.name}/1-{L} [subseq from] mol:protein length:{L}  {RUN}\n"
                        f"{''.join(hs)}")
    log.append({"iter": it, "tid": tid, "templates_found": found, "cif": cif})
    print(f"[{A.pdb}] iter {it:3d}  templates_found={found}  tid={tid}", flush=True)
    json.dump(log, open(f"{ART}/{RUN}_log.json", "w"), indent=1)

print(f"[{A.pdb}] done: {len(log)} iterations", flush=True)
