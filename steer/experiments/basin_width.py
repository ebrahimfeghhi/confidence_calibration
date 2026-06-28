"""Within-basin width: mean pairwise cross-block W1 among the members of each
basin (G / B1 / B2) at a few loops. Narrow basin = small within-group distance.
Tests whether B2 (confident decoy) is as narrow as G. CPU/GPU light."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from forward_grad import load, cross_distogram_from_z, AB, C, GOOD, REPS  # noqa: E402
from basins import B1, B2  # noqa: E402

NPAIRS = (AB.stop - AB.start) * (C.stop - C.start)
GROUPS = {"G": GOOD, "B1": B1, "B2": B2}


def cdf(model, pair_mask, seed, loop):
    z = torch.from_numpy(np.load(
        REPS / f"seed_{seed}" / "reps" / f"pairformer_z_loop{loop:02d}.npy"
    ).astype(np.float32)).cuda()
    with torch.no_grad():
        P = cross_distogram_from_z(model, z, pair_mask)
    out = P.cumsum(-1).reshape(-1).half().cpu()
    del z, P
    return out


def within(V):
    n = V.shape[0]
    if n < 2:
        return float("nan")
    d = []
    for i in range(n):
        for j in range(i + 1, n):
            d.append((V[i] - V[j]).abs().sum().item() / NPAIRS)
    return float(np.mean(d))


def main():
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]
    print(f"{'loop':>4}  {'width_G':>8} {'width_B1':>9} {'width_B2':>9}", flush=True)
    for loop in (6, 10, 16):
        Vg = {g: torch.stack([cdf(model, pair_mask, s, loop) for s in seeds]).cuda().float()
              for g, seeds in GROUPS.items()}
        wG, wB1, wB2 = within(Vg["G"]), within(Vg["B1"]), within(Vg["B2"])
        print(f"{loop:>4}  {wG:8.4f} {wB1:9.4f} {wB2:9.4f}", flush=True)
        del Vg
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
