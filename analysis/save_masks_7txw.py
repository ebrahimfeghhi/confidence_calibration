"""Generate + persist the 7txw input masks ONCE (the only GPU step).

Mask regeneration (CUDA RNG, must match the real runs) is the sole compute
bottleneck for every mask-pattern analysis; the masks themselves never change.
Save them once; all downstream analyses load the .npz and run CPU-only.

Writes results/csv_files/mask_matrix_7txw.npz:
  M        int8 [n_seeds, n_res]   1 = residue masked for that seed
  residues int  [n_res, 2]         (asym_idx, res0based) — the column order
  seeds    int  [n_seeds]

Usage: CUDA_VISIBLE_DEVICES=0 ~/esm/.venv/bin/python analysis/save_masks_7txw.py
"""
import sys
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
sys.path.insert(0, str(_p / "analysis"))
from paths import CSV_DIR
import analyze_masks as am


def main():
    n_seeds = 2060
    seeds = list(range(n_seeds))
    if am.torch is None:
        sys.exit("torch/esm not importable — run with ~/esm/.venv/bin/python on a GPU.")

    builder = am.ESMFold2InputBuilder()
    spi, _ = am.build_spi("7txw")
    features, _ = builder.prepare_input(spi, seed=0, device=am.DEVICE)
    lm_ids, expand_maps = am.build_lm_input_ids(features)
    seed_to_masked = am.generate_masks(lm_ids, features, expand_maps, seeds=seeds)

    residues = sorted(am.all_residues(features))
    col_of = {r: i for i, r in enumerate(residues)}
    M = np.zeros((n_seeds, len(residues)), dtype=np.int8)
    for i, s in enumerate(seeds):
        for r in seed_to_masked[s]:
            M[i, col_of[r]] = 1

    out = CSV_DIR / "mask_matrix_7txw.npz"
    np.savez_compressed(out, M=M, residues=np.array(residues, dtype=np.int64),
                        seeds=np.array(seeds, dtype=np.int64))
    print(f"saved {out}")
    print(f"  M={M.shape}  global masked frac={M.mean():.4f}  "
          f"(per-seed mean {M.sum(1).mean():.1f} residues)")


if __name__ == "__main__":
    main()
