"""Re-run the distogram skip-filter AND loop-trajectory with WASSERSTEIN-1 (in Angstrom)
instead of KL, to confirm conclusions hold and get interpretable thresholds.

W1 for 1D distributions over the distogram's ordered distance bins:
  W1(P,Q) = w * sum_bins |cumsum(P) - cumsum(Q)|   (w = bin spacing in A, read from the
  model's distogram boundaries buffer). Mean over the antibody x antigen cross-block pairs.

PART A: W1(seed16 || baseline) for all 60 seeds -> skip-filter savings (good vs bad).
PART B: per-loop W1 to the loop16 G/B1/B2 centroids (LOO) -> escape-from-B1 trajectory.
"""
import sys, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "run" / "esm"))
from run_7txw_loop_diffusion import load_complex, build_conditioning  # noqa: E402
from esm.models.esmfold2 import ESMFold2InputBuilder  # noqa: E402
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model  # noqa: E402
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import ESM_PREDICTIONS, JSON_DIR, CSV_DIR  # noqa: E402

REPS = ESM_PREDICTIONS / "7txw_reps" / "7txw"
BASE = ESM_PREDICTIONS / "7txw_reps_baseline" / "7txw" / "seed_0" / "reps" / "pairformer_z_loop16.npy"
OUT = Path("/home/ebrahim/tts_abag/analysis/figs")
AB, C = slice(0, 438), slice(438, 609)
NS, NLOOP = 5, 17
G = [10, 14, 33, 39]
B1 = [1, 2, 3, 4, 5, 7, 8, 9, 11, 16, 17, 18, 19, 20, 21, 22, 23, 26, 28, 29, 30, 31, 32,
      34, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49, 52, 55, 56, 57, 58, 59]
B2 = [0, 6, 12, 13, 15, 24, 27, 35, 43, 47, 50, 53, 54]
TRACE = [10, 14, 33, 39, 0, 12, 32, 42]
BASIN = {**{s: "G" for s in G}, **{s: "B1" for s in B1}, **{s: "B2" for s in B2}}
COL = {"G": "#d62728", "B1": "#1f77b4", "B2": "#ff8c00"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
    model.set_chunk_size(32)
    builder = ESMFold2InputBuilder()
    feat, _ = builder.prepare_input(load_complex(str(JSON_DIR), "7txw"), seed=0, device=model.device)
    with torch.no_grad():
        cond = build_conditioning(model, feat)
    pm = cond["pair_mask"]
    # distogram bin spacing (A) from the model's boundaries buffer
    bnd = next(b for n, b in model.named_buffers() if n.endswith("boundaries"))
    w = float((bnd[1:] - bnd[:-1]).mean().cpu())
    print(f"distogram: {len(bnd)+1} bins, range {bnd.min():.2f}-{bnd.max():.2f} A, spacing {w:.4f} A", flush=True)

    @torch.no_grad()
    def disto(path):
        z = torch.from_numpy(np.load(path).astype(np.float32)).cuda()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            z = model.parcae_readout(z); z = model.parcae_coda(z, pair_attention_mask=pm)
        logits = model.distogram_head(z.float() + z.float().transpose(-2, -3))[0]
        return F.softmax(logits, -1)[AB, C, :].reshape(-1, logits.shape[-1]).cpu().numpy().astype(np.float32)

    def W1(P, Q):  # mean over pairs, in Angstrom
        return float(w * np.abs(np.cumsum(P, 1) - np.cumsum(Q, 1)).sum(1).mean())

    # all 60 loop16 distograms (shared by skip + centroids) + baseline
    P0 = disto(BASE)
    D16 = {}
    for s in range(60):
        D16[s] = disto(REPS / f"seed_{s}" / "reps" / "pairformer_z_loop16.npy")
        if s % 15 == 0:
            print(f"  loop16 seed {s}", flush=True)

    # ---------- PART A: skip filter ----------
    dq = {}
    for r in csv.DictReader(open(CSV_DIR / "dockq_ost_7txw_reps_all_samples.csv")):
        if r["chain1"] == "A" and r["chain2"] == "C":
            dq[int(r["seed"])] = max(dq.get(int(r["seed"]), 0.0), float(r["dockq"]))
    good = np.array([dq.get(s, 0) >= 0.5 for s in range(60)])
    w1b = np.array([W1(D16[s], P0) for s in range(60)])
    print(f"\n=== PART A: W1-to-baseline (A) ===")
    print(f"  good={w1b[good].min():.3f}-{w1b[good].max():.3f}  "
          f"bad={w1b[~good].min():.3f}-{w1b[~good].max():.3f}")
    print(f"  {'T(A)':>6} {'skipped':>8} {'%diff_saved':>11} {'GOOD_skipped':>13} {'kept':>5}")
    for T in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        sk = w1b < T
        print(f"  {T:>6.2f} {sk.sum():>8d} {sk.sum()*NS}/{60*NS}={sk.sum()/60:>5.0%} "
              f"{(sk & good).sum():>13d} {(~sk).sum():>5d}")

    # ---------- PART B: trajectory ----------
    g16 = {s: D16[s] for s in G}
    b1_sum = sum(D16[s] for s in B1); b2_sum = sum(D16[s] for s in B2); g_sum = sum(g16.values())
    KLs = {}
    for s in TRACE:
        loops = [disto(REPS / f"seed_{s}" / "reps" / f"pairformer_z_loop{L:02d}.npy")
                 for L in range(NLOOP - 1)] + [D16[s]]
        Ps16 = loops[16]
        gref = (g_sum - g16[s]) / (len(G) - 1) if s in g16 else g_sum / len(G)
        b1ref = (b1_sum - Ps16) / (len(B1) - 1) if s in B1 else b1_sum / len(B1)
        b2ref = (b2_sum - Ps16) / (len(B2) - 1) if s in B2 else b2_sum / len(B2)
        KLs[s] = np.array([[W1(P, gref), W1(P, b1ref), W1(P, b2ref)] for P in loops])
        print(f"  traced seed {s} ({BASIN[s]})", flush=True)

    names = ["G", "B1", "B2"]
    print("\n=== PART B: nearest-basin trajectory (argmin W1) ===")
    nearest = {}
    for s in TRACE:
        nn = [names[i] for i in KLs[s].argmin(1)]; nearest[s] = nn
        commit = next((L for L in range(NLOOP) if all(nn[k] == "G" for k in range(L, NLOOP))), None)
        print(f"  seed {s:2d} [{('GOOD' if s in G else BASIN[s]):4s}]  commit→G @loop "
              f"{commit if commit is not None else 'never':>4}  | {' '.join(nn)}")

    # silhouette of good under W1 distance (loop16, all 60)
    M = np.array([[W1(D16[i], D16[j]) for j in range(60)] for i in range(60)])
    gm = good; om = ~good
    sil = np.mean([(M[i, om].mean() - M[i, gm & (np.arange(60) != i)].mean()) /
                   max(M[i, om].mean(), M[i, gm & (np.arange(60) != i)].mean())
                   for i in np.where(gm)[0]])
    print(f"\ngood-seed silhouette under W1 = {sil:+.3f}  (KL gave +0.96)")

    # figure (trajectory + strip), same layout as the KL version
    fig = plt.figure(figsize=(20, 11)); gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.5])
    for i, s in enumerate(TRACE):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        for j, b in enumerate(names):
            ax.plot(range(NLOOP), KLs[s][:, j], "-o", ms=3, color=COL[b], label=f"W1→{b}")
        ax.set_title(f"seed {s}  ({'GOOD' if s in G else 'bad/'+BASIN[s]})",
                     color=("#d62728" if s in G else "#333"))
        ax.set_xlabel("recycling loop"); ax.set_xticks(range(0, NLOOP, 2))
        if i == 0:
            ax.set_ylabel("cross-block distogram-W1 (Å)"); ax.legend(fontsize=8, frameon=False)
    axs = fig.add_subplot(gs[2, :]); code = {"G": 0, "B1": 1, "B2": 2}
    Mn = np.array([[code[b] for b in nearest[s]] for s in TRACE])
    axs.imshow(Mn, cmap=ListedColormap([COL["G"], COL["B1"], COL["B2"]]), aspect="auto", vmin=0, vmax=2)
    axs.set_yticks(range(len(TRACE)))
    axs.set_yticklabels([f"{s} ({'GOOD' if s in G else BASIN[s]})" for s in TRACE])
    axs.set_xticks(range(NLOOP)); axs.set_xlabel("recycling loop")
    axs.set_title("nearest basin per loop under W1 (red=G, blue=B1, orange=B2)")
    fig.suptitle("7txw — distogram trajectory with WASSERSTEIN-1 (Å)", y=1.0, fontsize=14)
    fig.tight_layout(); fig.savefig(OUT / "distogram_w1_trajectory.png", dpi=130, bbox_inches="tight")
    print("\nwrote", OUT / "distogram_w1_trajectory.png")


if __name__ == "__main__":
    main()
