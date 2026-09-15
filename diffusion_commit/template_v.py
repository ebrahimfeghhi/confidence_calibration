#!/usr/bin/env python
"""Push each pose of one seed through the TEMPLATE EMBEDDER alone and save its contribution v.

WHY. ipTM cannot rank the poses inside a seed -- median within-seed ipTM SD is 0.0078 while DockQ can
span the whole scale -- because all five poses share one trunk state and confidence is a property of
that state. The template embedder offers a different route: it turns a POSE into a pair-rep
contribution, so each pose gets a representation derived from itself alone. The question is whether
that representation separates a good pose from a decoy.

HOW. DUMP_TEMPLATE_V saves v per template, before the mean/ReLU/64->128 projection, and only on the
FIRST forward -- recycle 0, where z is z_init. v depends on z, so pinning z to the prior makes v a pure
function of the template geometry and comparable across poses. That is exactly what is wanted here.

ONE POSE PER RUN, deliberately. Feeding five templates at once would be cheaper, but template_id after
dedup/selection is not guaranteed to match the order poses were passed in, and a silent mis-mapping
would attribute each v to the wrong pose. One template per run makes the correspondence unambiguous.
N_cycle=1 and N_sample=1 because nothing past the first forward is used.

Usage: template_v.py --iface 9n6e_GF --seed 300046 --gpu 6
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
DC = "/home/ebrahim/tts_abag/diffusion_commit"
PY = "/home/ebrahim/Protenix/.venv/bin/python"


def load_v(path):
    """bfloat16 stored as uint16 -> float32."""
    raw = np.load(path)
    return raw.astype(np.uint32).__lshift__(16).view(np.float32).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    pdb, chains = A.iface.split("_")
    out = A.out or f"{EXP}/artifacts/template_v/{A.iface}_s{A.seed}"
    os.makedirs(out, exist_ok=True)

    # the poses, and their true quality (used only to label the result, never as input)
    meta = {}
    for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv")):
        if r["pdb"] == pdb and {r["chain1"], r["chain2"]} == set(chains) and r["seed"] == A.seed:
            meta[int(r["sample"])] = (float(r["dockq"]), float(r["pair_iptm"]))
    if len(meta) < 5:
        raise SystemExit(f"expected 5 poses, found {len(meta)} for {A.iface} seed {A.seed}")

    for k in sorted(meta):
        cif = f"{P}/out/{pdb}/{pdb}/seed_{A.seed}/predictions/{pdb}_sample_{k}.cif"
        if not os.path.exists(cif):
            print(f"  pose {k}: cif missing, skipped", flush=True); continue
        vdir = f"{out}/v{k}"
        if os.path.exists(f"{vdir}/t_0000.npy"):
            print(f"  pose {k}: cached", flush=True); continue
        os.makedirs(vdir, exist_ok=True)
        T = f"{out}/tmpl{k}"
        if not os.path.exists(f"{T}/job.json"):
            r = subprocess.run([PY, f"{DC}/build_selftmpl_general.py", "--pdb", pdb, "--out", T,
                                "--poses", cif, "--tids", f"v{k}{int(A.seed) % 100:02d}"],
                               cwd=DC, capture_output=True, text=True,
                               env={**os.environ, "SELFTMPL_JOBS": f"{P}/jsons"})
            if not os.path.exists(f"{T}/job.json"):
                print(f"  pose {k}: template build FAILED", flush=True)
                print((r.stderr or "").strip().splitlines()[-2:], flush=True); continue
        env = {**os.environ,
               "CUDA_VISIBLE_DEVICES": A.gpu, "LAYERNORM_TYPE": "openfold",
               "PROTENIX_MULTICHAIN_MASK": "0", "PROTENIX_TMPL_KEEP_FRAME": "1",
               "PROTENIX_TMPL_NODEDUP": "1", "PROTENIX_MAX_TEMPLATES": "1",
               "DUMP_TEMPLATE_V": vdir, "DUMP_TEMPLATE_V_TAG": "t"}
        log = f"{out}/run{k}.log"
        with open(log, "w") as fh:
            subprocess.run([PY, "runner/inference.py", "--model_name", "protenix-v2",
                            "--load_checkpoint_dir", "/home/ebrahim/checkpoint",
                            "--input_json_path", f"{T}/job.json", "--dump_dir", f"{out}/out{k}",
                            "--seeds", A.seed, "--model.N_cycle", "1",
                            "--sample_diffusion.N_sample", "1", "--sample_diffusion.N_step", "20",
                            "--use_template", "true"],
                           cwd="/home/ebrahim/Protenix", env=env, stdout=fh, stderr=subprocess.STDOUT)
        got = sorted(f for f in os.listdir(vdir) if f.endswith(".npy"))
        ntmpl = subprocess.run(["grep", "-oE", "Found [0-9]+ templates", log],
                               capture_output=True, text=True).stdout.strip().splitlines()
        print(f"  pose {k}: dumped {len(got)} v tensor(s)   {ntmpl[:1]}   "
              f"(DockQ {meta[k][0]:.3f}, ipTM {meta[k][1]:.3f})", flush=True)

    # ---- compare the v's
    V, ks = [], []
    for k in sorted(meta):
        f = f"{out}/v{k}/t_0000.npy"
        if os.path.exists(f):
            V.append(load_v(f).astype(np.float32)); ks.append(k)
    if len(V) < 2:
        print("  not enough v tensors to compare"); return
    print(f"\n  v shape {V[0].shape}   poses {ks}")
    F = [v.ravel() for v in V]
    print(f"\n  {'':<6}" + "".join(f"{'p'+str(k):>8}" for k in ks) + "     DockQ    ipTM")
    for i, k in enumerate(ks):
        cos = ["" for _ in ks]
        for j, _ in enumerate(ks):
            a, b = F[i], F[j]
            cos[j] = f"{float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)):>8.3f}"
        print(f"  {'p'+str(k):<6}" + "".join(cos) + f"   {meta[k][0]:>7.3f} {meta[k][1]:>7.3f}")
    np.savez_compressed(f"{out}/v_all.npz", V=np.stack(V), samples=np.array(ks),
                        dockq=np.array([meta[k][0] for k in ks]),
                        iptm=np.array([meta[k][1] for k in ks]))
    print(f"\n  wrote {out}/v_all.npz")


if __name__ == "__main__":
    main()
