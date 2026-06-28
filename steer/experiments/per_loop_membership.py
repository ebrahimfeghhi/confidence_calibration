"""When does "you are in B1" become a well-defined call? At each loop, build the
B1 centroid from n reference B1 seeds (at that loop) and measure how well
distance-to-centroid separates held-out B1 from G+B2 (= "not B1"). Reports
AUC / clean-separation% / margin per loop, at small (n=5) and large (n=20) n —
so we see the onset loop AND whether more references help early.
"""
import sys
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from forward_grad import load, cross_distogram_from_z, AB, C, GOOD, N_LOOPS, REPS  # noqa: E402
from basins import B1, B2  # noqa: E402

NPAIRS = (AB.stop - AB.start) * (C.stop - C.start)
OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(0)
B1A, OUTA = np.array(B1), np.array(GOOD + B2)


def cdf(model, pair_mask, seed, loop):
    z = torch.from_numpy(np.load(
        REPS / f"seed_{seed}" / "reps" / f"pairformer_z_loop{loop:02d}.npy"
    ).astype(np.float32)).cuda()
    with torch.no_grad():
        P = cross_distogram_from_z(model, z, pair_mask)
    out = P.cumsum(-1).reshape(-1)
    del z, P
    return out


def membership(V, n, R=100):
    """AUC / clean-sep / margin for separating held-out B1 from G+B2 using an
    n-ref B1 centroid, averaged over R subsamples."""
    aucs, cleans, margins = [], [], []
    for _ in range(R):
        ref = RNG.choice(B1A, size=n, replace=False)
        held = np.array([s for s in B1A if s not in set(ref)])
        q = np.concatenate([held, OUTA])
        y = np.concatenate([np.zeros(len(held)), np.ones(len(OUTA))])
        c = V[ref].mean(0)
        s = ((V[q] - c).abs().sum(1) / NPAIRS).cpu().numpy()
        aucs.append(roc_auc_score(y, s))
        margins.append(s[y == 1].min() - s[y == 0].max())
        cleans.append(1.0 if margins[-1] > 0 else 0.0)
    return np.mean(aucs), np.mean(cleans), np.mean(margins)


def main():
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]
    rows = []
    print(f"{'loop':>4} | {'AUC@5':>6} {'clean@5':>8} {'marg@5':>7} | "
          f"{'AUC@20':>6} {'clean@20':>9} {'marg@20':>8}", flush=True)
    for loop in range(N_LOOPS + 1):
        V = torch.stack([cdf(model, pair_mask, s, loop) for s in range(60)])   # [60,D] GPU
        a5, c5, m5 = membership(V, 5)
        a20, c20, m20 = membership(V, 20)
        del V
        torch.cuda.empty_cache()
        rows.append(dict(loop=loop, auc5=round(a5, 4), clean5=round(c5, 3), marg5=round(m5, 4),
                         auc20=round(a20, 4), clean20=round(c20, 3), marg20=round(m20, 4)))
        print(f"{loop:>4} | {a5:6.3f} {100*c5:7.1f}% {m5:+7.3f} | "
              f"{a20:6.3f} {100*c20:8.1f}% {m20:+8.3f}", flush=True)

    with open(OUT / "per_loop_membership.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lp = [r["loop"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(lp, [r["auc5"] for r in rows], "-o", label="AUC (n=5)")
    ax[0].plot(lp, [r["auc20"] for r in rows], "-s", label="AUC (n=20)")
    ax[0].axhline(0.5, color="k", lw=.5)
    ax[0].set_xlabel("loop"); ax[0].set_ylabel("AUC (B1 vs not-B1)")
    ax[0].set_title("when is 'in B1' detectable?"); ax[0].legend()
    ax[1].plot(lp, [r["marg5"] for r in rows], "-o", label="margin (n=5)")
    ax[1].plot(lp, [r["marg20"] for r in rows], "-s", label="margin (n=20)")
    ax[1].axhline(0, color="k", lw=.5)
    ax[1].set_xlabel("loop"); ax[1].set_ylabel("min(out) - max(in) distance")
    ax[1].set_title("separation margin by loop"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(OUT / "per_loop_membership.png", dpi=130)
    print("\nsaved per_loop_membership.csv and .png", flush=True)


if __name__ == "__main__":
    main()
