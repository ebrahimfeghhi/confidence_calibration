"""Decode steered s_lm leaves -> 3D structure (CIF) by injecting the steered lm_z
into model.forward (replacing model.language_model), then write CIFs for DockQ.

Usage:
  python decode_dockq.py sanity            # decode baseline s0, print ipTM, write baseline CIF
  python decode_dockq.py <leaves_tag.npy>  # decode every leaf -> CIF in decoded_<tag>/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "run" / "esm"))
import torch
import numpy as np
import forward_grad as fg
from complex_io import load_complex  # noqa
from paths import JSON_DIR  # noqa

OUT = Path(__file__).resolve().parent
TARGET = "8im0"
N_SAMPLES = 2          # diffusion samples per leaf (structure head is stochastic; take best dock)

cfg = fg.configure_target(TARGET)
model, builder, feat, chain_infos, cond = fg.load()
hidden = fg.load_hidden(cfg["baseline"])          # passed as lm_hidden_states; IGNORED by the patch (just must be non-None)


def decode_slm(s_lm, n_samples=N_SAMPLES, seed=0):
    """s_lm [1,L,256] -> list of MolecularComplexResult (each has .complex.to_mmcif(), .iptm)."""
    lm_z = fg.lift_to_lm_z(model, s_lm)
    h = model.language_model.register_forward_hook(lambda mod, inp, out: lm_z)  # inject steered conditioning
    try:
        torch.manual_seed(seed)
        with torch.no_grad():
            out = model(**feat, lm_hidden_states=hidden, num_loops=fg.N_LOOPS,
                        num_sampling_steps=200, num_diffusion_samples=n_samples)
    finally:
        h.remove()
    samples = builder.decode(out, feat, chain_infos, num_diffusion_samples=n_samples, complex_id=TARGET)
    return samples if isinstance(samples, list) else [samples]


def write_best(samples, path_stem):
    """Write each sample's CIF; return (paths, iptms)."""
    paths, iptms = [], []
    for k, s in enumerate(samples):
        p = f"{path_stem}_s{k}.cif"
        with open(p, "w") as f:
            f.write(s.complex.to_mmcif())
        paths.append(p)
        iptms.append(float(s.iptm) if s.iptm is not None else (float(s.ptm) if s.ptm is not None else 0.0))
    return paths, iptms


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "sanity"
    if arg == "sanity":
        s0 = fg.single_rep_from_hidden(model, hidden).detach()    # baseline s_lm (no steering)
        samples = decode_slm(s0, n_samples=N_SAMPLES, seed=0)
        d = OUT / f"decoded_{TARGET}_baseline"
        d.mkdir(exist_ok=True)
        paths, iptms = write_best(samples, str(d / "baseline"))
        print(f"SANITY baseline: iptms={[round(x,3) for x in iptms]}  -> {paths}", flush=True)
    else:
        leaves = np.load(OUT / arg)                                # [n_runs,1,L,256]
        tag = arg.replace("leaves_", "").replace(".npy", "")
        d = OUT / f"decoded_{tag}"; d.mkdir(exist_ok=True)
        print(f"decoding {len(leaves)} leaves from {arg} -> {d}", flush=True)
        for i, leaf in enumerate(leaves):
            s_lm = torch.from_numpy(leaf.astype(np.float32)).cuda()
            samples = decode_slm(s_lm, n_samples=N_SAMPLES, seed=0)
            paths, iptms = write_best(samples, str(d / f"run{i+1}"))
            print(f"  run {i+1}: iptm={[round(x,3) for x in iptms]}", flush=True)
        print("done decoding.", flush=True)
