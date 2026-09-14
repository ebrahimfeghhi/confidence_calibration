#!/usr/bin/env python
"""Dump per-template v for every free-run pose of a complex, at z = z_init.

v = single_template_forward(pose, z_init): the per-template contribution TemplateEmbedder averages,
before the mean, the relu and the 64->128 projection. Shape [N, N, 64].

WHY z_init: v reads the trunk's current z, so the same pose gives a different v at every recycle and
in every run. Pinning z to the prior (recycle 0) makes v a pure function of the pose, which is what
makes poses comparable to each other. The model-side hook (DUMP_TEMPLATE_V) therefore fires on the
FIRST TemplateEmbedder call only.

WHY BATCHES: all 250 templates at once is a [250,N,N,108] feature tensor -- 48 GB at N=942, which
does not fit alongside the model. v for a template depends only on (that template, z_init), and
z_init does not depend on which templates were passed, so batching is exact rather than an
approximation.

CAVEAT: the relu comes AFTER the mean, so v is the right object for comparing or clustering poses,
but NOT for attributing a share of the final z to a pose.

Usage: .../python scripts/dump_template_v.py --pdb 8che --gpu 3 --batch 25 [--limit 5]
"""
import argparse, glob, json, os, shutil, subprocess, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemmi
from cifconv import convert
from decoy_set import EXP, PRED, SEEDS, SAMPLES
from build_dockq_templates import best_chain, mut

ap = argparse.ArgumentParser()
ap.add_argument("--pdb", required=True)
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--batch", type=int, default=25)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--xchain", action="store_true",
                help="PROTENIX_MULTICHAIN_MASK=0: let templates carry CROSS-CHAIN geometry, i.e. the "
                     "dock. With the mask on (default) the pose's epitope is zeroed out of the input "
                     "features and any cross-chain signal in v arrived only indirectly, via the "
                     "linear_z(LN(z)) term and triangle propagation.")
ap.add_argument("--tid-prefix", default="8")
ap.add_argument("--tid-start", type=int, default=700)
A = ap.parse_args()

ART = f"{EXP}/artifacts"; JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
RUN = f"vdump_{A.pdb}" + ("_xchain" if A.xchain else "")
TMPL = f"{ART}/sweep_tmpl/{RUN}"; A3M = f"{ART}/sweep_a3m/{RUN}"
VDIR = f"{ART}/{A.pdb}_template_v" + ("_xchain" if A.xchain else ""); SCRATCH = f"{ART}/{RUN}_scratch"
for d in (TMPL, A3M, VDIR, SCRATCH): os.makedirs(d, exist_ok=True)
PY = "/home/ebrahim/Protenix/.venv/bin/python"

poses = [f"{PRED}/{A.pdb}/seed_{s}/predictions/{A.pdb}_sample_{k}.cif"
         for s in SEEDS for k in SAMPLES]
poses = [p for p in poses if os.path.exists(p)]
if A.limit: poses = poses[:A.limit]
print(f"{A.pdb}: {len(poses)} poses, batches of {A.batch}", flush=True)

base = json.load(open(f"{JDIR}/{A.pdb}-final-updated.json"))
base = base[0] if isinstance(base, list) else base

manifest = []
for b0 in range(0, len(poses), A.batch):
    chunk = poses[b0 : b0 + A.batch]
    obj = json.loads(json.dumps(base))
    qseqs = [e["proteinChain"]["sequence"] for e in obj["sequences"] if "proteinChain" in e]

    tids = []
    for j, cif in enumerate(chunk):
        tid = f"{A.tid_prefix}{A.tid_start + j:03d}"      # reused per batch; previous batch is done
        assert len(tid) == 4
        open(f"/home/ebrahim/mmcif/{tid}.cif", "w").write(convert(open(cif).read(), tid))
        tids.append(tid)

    pi = 0
    for e in obj["sequences"]:
        if "proteinChain" not in e: continue
        q = qseqs[pi]; L = len(q); lines = [f">query\n{q}"]
        for j, (tid, cif) in enumerate(zip(tids, chunk)):
            st = gemmi.read_structure(cif); st.setup_entities()
            mc = best_chain(st, q)
            if mc is None: continue
            hs = list(q)
            for pos in (L // 7 + j, 3 * L // 7 + j, 5 * L // 7 + j):   # unique per hit: dedup
                if pos < L: hs[pos] = mut(hs[pos])
            lines.append(f">{tid}_{mc.name}/1-{L} [subseq from] mol:protein length:{L}  {RUN}\n"
                         f"{''.join(hs)}")
        a3mp = f"{A3M}/{A.pdb}_b{b0:04d}_q{pi}.a3m"
        open(a3mp, "w").write("\n".join(lines) + "\n")
        e["proteinChain"]["templatesPath"] = a3mp
        pi += 1

    obj["name"] = f"{A.pdb}_b{b0:04d}"
    jp = f"{TMPL}/{obj['name']}.json"
    json.dump([obj], open(jp, "w"))

    tmp = f"{SCRATCH}/b{b0:04d}"
    if os.path.isdir(tmp): shutil.rmtree(tmp)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(A.gpu), LAYERNORM_TYPE="openfold",
               DUMP_TEMPLATE_V=tmp, DUMP_TEMPLATE_V_TAG=f"b{b0:04d}",
               PROTENIX_TMPL_NODEDUP="1",
               **({"PROTENIX_MULTICHAIN_MASK": "0"} if A.xchain else {}),
               PROTENIX_MAX_TEMPLATES=str(len(chunk)),
               PROTENIX_MAX_TEMPLATE_CANDIDATES=str(len(chunk) + 8))
    lf = f"{ART}/{RUN}_logs_b{b0:04d}.log"; os.makedirs(os.path.dirname(lf), exist_ok=True)
    with open(lf, "w") as fh:
        subprocess.run([PY, "runner/inference.py", "--model_name", "protenix-v2",
                        "--load_checkpoint_dir", "/home/ebrahim/checkpoint",
                        "--input_json_path", jp, "--dump_dir", f"{SCRATCH}/out",
                        "--seeds", "101", "--sample_diffusion.N_sample", "1",
                        "--sample_diffusion.N_step", "20",   # coords unused; keep decode cheap
                        "--use_template", "true"],
                       cwd="/home/ebrahim/Protenix", env=env,
                       stdout=fh, stderr=subprocess.STDOUT, timeout=10800)

    got = sorted(glob.glob(f"{tmp}/*.npy"))
    found = sorted(set(l.split("Found ")[1].split()[0]
                       for l in open(lf, errors="ignore") if "templates for sequence" in l))
    print(f"  batch {b0:>4}: templates_found={found}  v files={len(got)} (want {len(chunk)})",
          flush=True)
    if len(got) < len(chunk):
        print(f"  *** expected {len(chunk)} v tensors, got {len(got)} -- stopping ***", flush=True)
        break
    for j, f in enumerate(got[: len(chunk)]):
        dst = f"{VDIR}/pose_{b0 + j:04d}.npy"
        shutil.move(f, dst)
        manifest.append({"idx": b0 + j, "pose": os.path.relpath(chunk[j], PRED), "v": dst})
    shutil.rmtree(tmp, ignore_errors=True)
    json.dump(manifest, open(f"{ART}/{A.pdb}_template_v{'_xchain' if A.xchain else ''}_manifest.json",
                             "w"), indent=1)

tot = sum(os.path.getsize(m["v"]) for m in manifest) / 1e9
print(f"\n{len(manifest)} v tensors -> {VDIR}  ({tot:.1f} GB)", flush=True)
