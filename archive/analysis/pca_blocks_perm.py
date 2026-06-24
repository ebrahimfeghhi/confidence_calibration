"""Is each block's good-seed silhouette above chance? Per-block permutation baseline.

For every chain block, compare the TRUE good set {10,14,33,39} silhouette against the
distribution of silhouettes from random 4-seed subsets (same D, shuffled membership).
Tells us how much of a block's separation is real good-seed structure vs. "any 4 seeds
look somewhat clustered." loop16 only.

Output: prints per-block true sil, null mean/p95, excess over null, and p-value.
"""
from pathlib import Path
import csv
import numpy as np

REPS_DIR = Path("/home/ebrahim/esm/foldbench_predictions/7txw_reps/7txw")
CSV = Path("/home/ebrahim/tts_abag/results/csv_files/dockq_ost_7txw_reps_all_samples.csv")
N_SEEDS, N_CH, LOOP = 60, 256, 16
GOOD_T, MARG_T = 0.5, 0.23
N_PERM = 5000
SPANS = {"A": slice(0, 219), "B": slice(219, 438), "C": slice(438, 609)}
BLOCKS = [("AxC", "A", "C"), ("BxC", "B", "C"), ("CxC", "C", "C"),
          ("AxB", "A", "B"), ("AxA", "A", "A"), ("BxB", "B", "B")]
RNG = np.random.default_rng(0)


def seed_dockq_ac():
    d = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if r["chain1"] == "A" and r["chain2"] == "C":
                s, q = int(r["seed"]), float(r["dockq"])
                d[s] = max(d.get(s, 0.0), q)
    return d


def silhouette(D, mask):
    other = ~mask
    s = []
    for i in np.where(mask)[0]:
        same = mask.copy(); same[i] = False
        a = D[i, same].mean(); b = D[i, other].mean()
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


def block_D(mats):
    X = mats - mats.mean(0)
    G = X @ X.T
    diag = np.diag(G)
    return np.sqrt(np.clip(diag[:, None] + diag[None, :] - 2 * G, 0, None))


def main():
    dq = seed_dockq_ac()
    dvals = np.array([dq[s] for s in range(N_SEEDS)])
    good = sorted(s for s in range(N_SEEDS) if dvals[s] >= GOOD_T)
    assert good == [10, 14, 33, 39], good
    good_mask = np.zeros(N_SEEDS, bool); good_mask[good] = True

    mats = {name: np.empty((N_SEEDS, (SPANS[r].stop - SPANS[r].start)
                            * (SPANS[c].stop - SPANS[c].start) * N_CH), dtype=np.float32)
            for name, r, c in BLOCKS}
    for s in range(N_SEEDS):
        z = np.load(REPS_DIR / f"seed_{s}" / "reps" / f"pairformer_z_loop{LOOP:02d}.npy",
                    mmap_mode="r")
        zf = np.asarray(z[0], dtype=np.float32)
        for name, r, c in BLOCKS:
            mats[name][s] = zf[SPANS[r], SPANS[c], :].reshape(-1)
        del z, zf
    print(f"loaded {N_SEEDS} seeds, loop{LOOP}\n")

    # pre-draw the same random 4-subsets for all blocks (fair comparison)
    perms = [RNG.choice(N_SEEDS, 4, replace=False) for _ in range(N_PERM)]

    print(f"{'block':5} {'true':>7} {'null_mean':>9} {'null_p95':>8} "
          f"{'excess':>7} {'p':>8}")
    rows = []
    for name, r, c in BLOCKS:
        D = block_D(mats[name])
        true = silhouette(D, good_mask)
        null = np.empty(N_PERM)
        for k, idx in enumerate(perms):
            m = np.zeros(N_SEEDS, bool); m[idx] = True
            null[k] = silhouette(D, m)
        p = (1 + (null >= true).sum()) / (1 + N_PERM)
        rows.append((name, true, null.mean(), np.percentile(null, 95),
                     true - null.mean(), p))
        print(f"{name:5} {true:+7.3f} {null.mean():+9.3f} "
              f"{np.percentile(null,95):+8.3f} {true-null.mean():+7.3f} {p:8.4f}")

    print("\nranked by excess-over-chance (true - null_mean):")
    for name, true, nm, p95, exc, p in sorted(rows, key=lambda x: -x[4]):
        print(f"  {name}: excess={exc:+.3f}  (true {true:+.3f}, null {nm:+.3f}, p={p:.4f})")


if __name__ == "__main__":
    main()
