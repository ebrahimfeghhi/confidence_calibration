"""For the K=3 stars, compute the cross-block distogram wg (W1 to the G centroid) at K=3 (z3)
vs K=17 (z17) -- no decode. Tells whether a 'K=3-failed' run failed at the DISTOGRAM level
(steering didn't commit by K=3) or only at the decoded-STRUCTURE level (z3 too immature to decode)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forward_grad import load, cross_distogram_from_z, w1, GOOD  # noqa: E402
from steer_deviation import centroid  # noqa: E402

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import ESM_PREDICTIONS  # noqa: E402

TAG = "slm_nudge_M5_K3_k3"
STARS = [(0, 7), (1, 4)]   # (trial, run); extend as more land


def wg_of(model, pair_mask, Pg, zpath):
    z = torch.from_numpy(np.load(zpath).astype(np.float32)).cuda()
    with torch.no_grad():
        P = cross_distogram_from_z(model, z, pair_mask)
    return w1(P, Pg).item()


def main():
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]
    Pg = centroid(model, pair_mask, GOOD)
    base = ESM_PREDICTIONS / "7txw_repel" / TAG
    print(f"{'star':<10}{'wg@K3':>9}{'wg@K17':>9}   (G band 0.033-0.079)")
    for t, r in STARS:
        rd = base / f"trial{t:02d}" / f"run{r:02d}" / "seed_0" / "reps"
        wk3 = wg_of(model, pair_mask, Pg, rd / "pairformer_z_K3.npy")
        wk17 = wg_of(model, pair_mask, Pg, rd / "pairformer_z_loop16.npy")
        print(f"t{t:02d}r{r:02d}   {wk3:>9.3f}{wk17:>9.3f}", flush=True)


if __name__ == "__main__":
    main()
