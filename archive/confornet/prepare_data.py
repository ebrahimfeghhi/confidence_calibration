"""Build the portable ConforNet data bundle (run ONCE on the MAIN server).

The ConforNet experiment touches the 195 GB `7txw_reps` only to derive three small
centroid distograms (G/B1/B2) and one baseline `esmc_hidden_states.npy`. This script
bakes those — plus the 7txw input JSON, ground truth, and targets CSV needed for the
decode->DockQ step — into `CONFORNET_DATA/` (default `confornet/data/`, ~300 MB), so the
new server needs none of the 195 GB.

After running, ship the bundle:
    rsync -a confornet/  newserver:~/tts_abag/confornet/

Run (esm venv, GPU):  python confornet/prepare_data.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forward_grad import load, compute_z_init, fixed_z0, BASELINE  # noqa: E402
from basins import centroid, B1, B2  # noqa: E402
from forward_grad import GOOD  # noqa: E402

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import (  # noqa: E402
    CONFORNET_DATA, JSON_DIR, GROUND_TRUTHS, TARGETS_CSV,
)

NAMES = {"G": GOOD, "B1": B1, "B2": B2}


def _copy(src: Path, dst: Path):
    if not src.exists():
        print(f"  WARN missing, skipped: {src}", flush=True)
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    mb = dst.stat().st_size / 1e6
    print(f"  copied {src}  ->  {dst}  ({mb:.1f} MB)", flush=True)
    return dst.stat().st_size


def main():
    CONFORNET_DATA.mkdir(parents=True, exist_ok=True)
    print(f"building bundle at {CONFORNET_DATA}", flush=True)
    total = 0

    # --- (1) centroids: compute from REPS, save the 3 small distograms ---
    print("\n[1/4] precomputing G/B1/B2 centroids from 7txw_reps...", flush=True)
    print("loading model...", flush=True)
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]
    cdir = CONFORNET_DATA / "centroids"
    cdir.mkdir(parents=True, exist_ok=True)
    for name, seeds in NAMES.items():
        with torch.no_grad():
            P = centroid(model, pair_mask, seeds)            # normalized, detached
        out = cdir / f"{name}.npy"
        np.save(out, P.float().cpu().numpy())
        sz = out.stat().st_size
        total += sz
        print(f"  {name}: {len(seeds)} seeds -> {tuple(P.shape)}  {out}  ({sz/1e6:.1f} MB)",
              flush=True)
    # touch z_init/z0 so a mismatch surfaces here, not on the new server
    _ = fixed_z0(model, compute_z_init(model, cond), seed=0)

    # --- (2) baseline esmc hidden, laid out so ESM_PREDICTIONS=<bundle>/baseline works ---
    print("\n[2/4] copying baseline esmc_hidden_states.npy...", flush=True)
    base_rel = BASELINE.relative_to(BASELINE.parents[2])     # 7txw_reps_baseline/7txw/seed_0
    base_dst = CONFORNET_DATA / "baseline" / base_rel / "reps"
    total += _copy(BASELINE / "reps" / "esmc_hidden_states.npy",
                   base_dst / "esmc_hidden_states.npy")
    _copy(BASELINE / "reps" / "meta.json", base_dst / "meta.json")

    # --- (3) input JSON for build_conditioning (only the 7txw complex) ---
    print("\n[3/4] copying 7txw input JSON...", flush=True)
    total += _copy(JSON_DIR / "7txw.json", CONFORNET_DATA / "jsons" / "7txw.json")

    # --- (4) DockQ inputs: ground truth + targets CSV ---
    print("\n[4/4] copying ground truth + targets for DockQ...", flush=True)
    total += _copy(GROUND_TRUTHS / "7txw-assembly1.cif",
                   CONFORNET_DATA / "ground_truths" / "7txw-assembly1.cif")
    total += _copy(TARGETS_CSV, CONFORNET_DATA / "targets" / "interface_antibody_antigen.csv")

    print(f"\nbundle ready: {CONFORNET_DATA}  (~{total/1e6:.0f} MB)", flush=True)
    print("ship it:  rsync -a confornet/  newserver:~/tts_abag/confornet/", flush=True)


if __name__ == "__main__":
    main()
