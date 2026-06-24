"""Trajectory of each seed's cross-block distogram across the 17 recycling loops,
relative to the loop16 basin archetypes G / B1 / B2.

Answers: before a good seed snaps to the good basin, where does it sit? In a B1/B2
DECOY basin (commit-then-correct), diffuse/far from all (separate path), or already
nearest G (born-good)?

For each traced seed s and loop L: cross-block (antibody x antigen) distogram, then
KL to the loop16 centroids G16/B1_16/B2_16. Centroids use leave-one-out for s's own
basin (so the seed's converged endpoint can't leak into its own early-loop reference).
The loop where KL-to-G becomes (and stays) the min = the commitment loop; compare it to
the diffuse-from-loop DockQ snap in dockq_iptm_across_loops_7txw.png.

Near-free: one linear-ish distogram head per loop, NO diffusion decode.
Run with ESM_PYTHON on a GPU. Output: analysis/figs/distogram_loop_trajectory.png
"""
import sys
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
from paths import ESM_PREDICTIONS, JSON_DIR  # noqa: E402

REPS = ESM_PREDICTIONS / "7txw_reps" / "7txw"
OUT = Path("/home/ebrahim/tts_abag/analysis/figs")
AB, C = slice(0, 438), slice(438, 609)
NLOOP = 17
G = [10, 14, 33, 39]
B1 = [1, 2, 3, 4, 5, 7, 8, 9, 11, 16, 17, 18, 19, 20, 21, 22, 23, 26, 28, 29, 30, 31, 32,
      34, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49, 52, 55, 56, 57, 58, 59]
B2 = [0, 6, 12, 13, 15, 24, 27, 35, 43, 47, 50, 53, 54]
TRACE = [10, 14, 33, 39, 0, 12, 32, 42]   # good + B2(0,12) + B1(32,42)
BASIN = {**{s: "G" for s in G}, **{s: "B1" for s in B1}, **{s: "B2" for s in B2}}
COL = {"G": "#d62728", "B1": "#1f77b4", "B2": "#ff8c00"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading model...", flush=True)
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
    model.set_chunk_size(32)
    builder = ESMFold2InputBuilder()
    spi = load_complex(str(JSON_DIR), "7txw")
    feat, _ = builder.prepare_input(spi, seed=0, device=model.device)
    with torch.no_grad():
        cond = build_conditioning(model, feat)
    pair_mask = cond["pair_mask"]

    @torch.no_grad()
    def disto(seed, loop):
        z = torch.from_numpy(np.load(
            REPS / f"seed_{seed}" / "reps" / f"pairformer_z_loop{loop:02d}.npy"
        ).astype(np.float32)).cuda()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            z = model.parcae_readout(z)
            z = model.parcae_coda(z, pair_attention_mask=pair_mask)
        z = z.float()
        logits = model.distogram_head(z + z.transpose(-2, -3))[0]
        P = F.softmax(logits, dim=-1)[AB, C, :].reshape(-1, logits.shape[-1])
        return P.cpu().numpy().astype(np.float32)

    def kl(P, Q):
        return float((P * (np.log(P + 1e-9) - np.log(Q + 1e-9))).sum(1).mean())

    # loop16 basin centroids (sum + count, so we can leave-one-out by subtracting)
    print("building loop16 basin centroids...", flush=True)
    g16 = {s: disto(s, 16) for s in G}                       # keep individually for LOO
    def basin_sum(seeds):
        acc = None
        for s in seeds:
            d = disto(s, 16); acc = d if acc is None else acc + d
        return acc, len(seeds)
    b1_sum, b1_n = basin_sum(B1)
    b2_sum, b2_n = basin_sum(B2)
    g_sum = sum(g16.values())

    # trace each seed across loops
    KLs = {}            # seed -> [NLOOP, 3]  (G, B1, B2)
    for s in TRACE:
        loops = [disto(s, L) for L in range(NLOOP)]
        Ps16 = loops[16]
        # leave-one-out references for s's own basin
        gref = (g_sum - g16[s]) / (len(G) - 1) if s in g16 else g_sum / len(G)
        b1ref = (b1_sum - Ps16) / (b1_n - 1) if s in B1 else b1_sum / b1_n
        b2ref = (b2_sum - Ps16) / (b2_n - 1) if s in B2 else b2_sum / b2_n
        arr = np.array([[kl(P, gref), kl(P, b1ref), kl(P, b2ref)] for P in loops])
        KLs[s] = arr
        print(f"  seed {s} ({BASIN[s]}) traced", flush=True)

    # ---- per-seed summary: nearest basin per loop + commitment loop ----
    names = ["G", "B1", "B2"]
    print("\n=== nearest-basin trajectory (argmin KL per loop) ===")
    nearest = {}
    for s in TRACE:
        nn = [names[i] for i in KLs[s].argmin(1)]
        nearest[s] = nn
        # commitment loop = first loop from which G stays the argmin to the end
        commit = next((L for L in range(NLOOP) if all(nn[k] == "G" for k in range(L, NLOOP))),
                      None)
        pre = nn[max(0, (commit or NLOOP) - 1)] if commit else "—"
        early = "/".join(dict.fromkeys(nn[:3]))   # nearest in loops 0-2
        tag = "GOOD" if s in G else BASIN[s]
        print(f"  seed {s:2d} [{tag:4s}]  early(0-2)={early:6s}  commit→G @loop "
              f"{commit if commit is not None else 'never':>4}  | {' '.join(nn)}")

    # ---- figure: per-seed KL curves + nearest-basin strip ----
    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.5])
    for i, s in enumerate(TRACE):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        for j, b in enumerate(names):
            ax.plot(range(NLOOP), KLs[s][:, j], "-o", ms=3, color=COL[b], label=f"KL→{b}")
        ax.set_yscale("log")
        tag = "GOOD" if s in G else f"bad/{BASIN[s]}"
        ax.set_title(f"seed {s}  ({tag})", color=("#d62728" if s in G else "#333"))
        ax.set_xlabel("recycling loop"); ax.set_xticks(range(0, NLOOP, 2))
        if i == 0:
            ax.set_ylabel("cross-block distogram-KL"); ax.legend(fontsize=8, frameon=False)

    # strip
    axs = fig.add_subplot(gs[2, :])
    code = {"G": 0, "B1": 1, "B2": 2}
    M = np.array([[code[b] for b in nearest[s]] for s in TRACE])
    cmap = ListedColormap([COL["G"], COL["B1"], COL["B2"]])
    axs.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=2)
    axs.set_yticks(range(len(TRACE)))
    axs.set_yticklabels([f"{s} ({'GOOD' if s in G else BASIN[s]})" for s in TRACE])
    axs.set_xticks(range(NLOOP)); axs.set_xlabel("recycling loop")
    axs.set_title("nearest basin per loop  (red=G, blue=B1, orange=B2)")
    fig.suptitle("7txw — distogram trajectory across recycling: do good seeds pass through a "
                 "decoy basin before committing to G?", y=1.0, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "distogram_loop_trajectory.png", dpi=130, bbox_inches="tight")
    print("\nwrote", OUT / "distogram_loop_trajectory.png")


if __name__ == "__main__":
    main()
