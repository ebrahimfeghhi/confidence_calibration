"""Basin labels + centroid distograms for 7txw (G / B1 / B2).

Extracted from the earlier exploratory steering scripts (`steer_sequential.py`,
`steer_deviation.py`) so the ConforNet experiment no longer imports them.

Two ways to get the three centroid distograms (mean loop-16 cross-block distogram
over the known good / B1 / B2 seeds):
  * `load_centroids` — the function the experiment calls. If a precomputed bundle
    exists (`STEER_DATA/centroids/{G,B1,B2}.npy`, produced once by
    `prepare_data.py`), it loads those — this is the **portable / new-server path**
    that needs NO access to the 195 GB `7txw_reps`. Otherwise it falls back to
    computing them from `forward_grad.REPS` (the **main-server path**).
  * `centroid` — the from-reps computation, used by `prepare_data.py`.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forward_grad import cross_distogram_from_z, REPS, GOOD  # noqa: E402,F401

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import STEER_DATA  # noqa: E402

# Basin membership for the 60-seed 7txw_reps run (B1 = default decoy, B2 = confident
# wrong-register decoy). GOOD is the native-pose basin (defined in forward_grad).
B1 = [1, 2, 3, 4, 5, 7, 8, 9, 11, 16, 17, 18, 19, 20, 21, 22, 23, 26, 28, 29, 30, 31, 32,
      34, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49, 52, 55, 56, 57, 58, 59]
B2 = [0, 6, 12, 13, 15, 24, 27, 35, 43, 47, 50, 53, 54]

_NAMES = {"G": GOOD, "B1": B1, "B2": B2}
_CENTROID_DIR = STEER_DATA / "centroids"


def centroid(model, pair_mask, seeds):
    """Mean loop-16 cross-block distogram over `seeds`, normalized, detached.

    Reads `REPS/seed_S/reps/pairformer_z_loop16.npy` for each seed — needs the
    195 GB `7txw_reps` (main server / `prepare_data.py` only)."""
    Ps = [cross_distogram_from_z(
              model, torch.from_numpy(np.load(
                  REPS / f"seed_{s}" / "reps" / "pairformer_z_loop16.npy").astype(np.float32)
              ).cuda(), pair_mask)
          for s in seeds]
    P = torch.stack(Ps).mean(0)
    return (P / P.sum(-1, keepdim=True)).detach()


def load_centroids(model, pair_mask):
    """Return (Pg, Pb1, Pb2). Prefer the precomputed bundle; else compute from REPS."""
    files = {k: _CENTROID_DIR / f"{k}.npy" for k in ("G", "B1", "B2")}
    if all(f.exists() for f in files.values()):
        print(f"loading precomputed centroids from {_CENTROID_DIR}", flush=True)
        return tuple(
            torch.from_numpy(np.load(files[k]).astype(np.float32)).cuda().detach()
            for k in ("G", "B1", "B2")
        )
    print(f"no precomputed centroids at {_CENTROID_DIR}; computing from REPS "
          f"({REPS}) — needs the full 7txw_reps", flush=True)
    return tuple(centroid(model, pair_mask, _NAMES[k]) for k in ("G", "B1", "B2"))
