"""Decode the steered loop16 z's (from steer_horizon.py) into structures so we can
score DockQ — the real verdict on whether a low-W1 steered distogram is an actual
good complex or just a numeric look-alike (off-manifold).

For each K dir under 7txw_steer_horizon/7txw/K*/seed_0/reps/pairformer_z_loop16.npy,
run the diffusion head (reusing run_7txw_loop_diffusion.diffuse_from_z) and write
predictions in the standard layout so eval_ost_foldbench.py can score them:
    ESM_PREDICTIONS/7txw_steer_decode/<Kxx>/7txw/seed_0/predictions/

Run (esm venv, GPU): python analysis/grad_steer/decode_steered.py
Then score:  python eval/eval_ost_foldbench.py --pred_dir <.../7txw_steer_decode/Kxx> --out_csv ...
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "run" / "esm"))
from run_7txw_loop_diffusion import (  # noqa: E402
    load_complex, build_conditioning, diffuse_from_z, save_samples,
)
from esm.models.esmfold2 import ESMFold2InputBuilder  # noqa: E402
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model  # noqa: E402

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import ESM_PREDICTIONS, JSON_DIR  # noqa: E402

STEER = ESM_PREDICTIONS / "7txw_steer_horizon" / "7txw"
OUT = ESM_PREDICTIONS / "7txw_steer_decode"
N_SAMPLES = 5
N_STEPS = 16


def main():
    # CLI: --steer_dir=PATH  --glob=PATTERN  --out=PATH  (default = original K* horizon decode)
    args = sys.argv[1:]
    def _flag(name, default):
        for a in args:
            if a.startswith(name + "="):
                return a.split("=", 1)[1]
        return default
    steer = Path(_flag("--steer_dir", str(STEER)))
    pattern = _flag("--glob", "K*")
    out = Path(_flag("--out", str(OUT)))

    print("loading model...", flush=True)
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
    model.set_chunk_size(32)
    builder = ESMFold2InputBuilder()
    spi = load_complex(str(JSON_DIR), "7txw")
    feat, chain_infos = builder.prepare_input(spi, seed=0, device=model.device)
    with torch.no_grad():
        cond = build_conditioning(model, feat)

    kdirs = sorted(steer.glob(pattern))
    if not kdirs:
        raise SystemExit(f"no dirs matching {pattern!r} under {steer}")
    print(f"decoding {len(kdirs)} dir(s): {[d.name for d in kdirs]}", flush=True)

    for kdir in kdirs:
        z_path = kdir / "seed_0" / "reps" / "pairformer_z_loop16.npy"
        if not z_path.exists():
            print(f"  {kdir.name}: no z yet, skip", flush=True)
            continue
        pred_dir = out / kdir.name / "7txw" / "seed_0" / "predictions"
        if any(pred_dir.glob("7txw_sample_*.cif")):
            print(f"  {kdir.name}: already decoded, skip", flush=True)
            continue
        z_np = np.load(z_path).astype(np.float32)
        with torch.no_grad():
            samples = diffuse_from_z(model, builder, feat, chain_infos, cond, z_np,
                                     N_SAMPLES, N_STEPS, "7txw", rng_seed=12345)
        save_samples(pred_dir, "7txw", samples)
        best = max(s.iptm or s.ptm or 0.0 for s in samples)
        print(f"  {kdir.name}: decoded {len(samples)} samples, best ipTM={best:.3f} "
              f"-> {pred_dir}", flush=True)
        torch.cuda.empty_cache()

    print("\nDecode done. Score with eval_ost_foldbench.py per K dir.", flush=True)


if __name__ == "__main__":
    main()
