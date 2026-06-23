"""Validate the distogram-KL idea: does cross-block (antibody x antigen) distogram-KL
separate the 7txw orientation basins, and is KL-to-baseline a basin-change signal?

For each seed's loop16 z we compute the predicted distogram (parcae_readout -> parcae_coda
-> distogram_head, exactly as the diffusion path does), softmax to a per-pair distance
distribution, and keep the A|B x C cross-block (antibody->antigen pairs — the general,
epitope-free 'interface' region). Then:
  (1) pairwise KL between seeds; does basin (G / B1 / B2) separate? esp. G vs B2 (the
      confident decoy that ipTM CANNOT separate).
  (2) KL(seed || baseline-no-mask/dropout) per basin: is it LOW for same-basin-as-baseline
      seeds and HIGH for the good basin? = the 'perturb until KL-to-baseline is large' premise.

numpy + torch + the ESM model. Run with ESM_PYTHON on a GPU.
Output: prints stats + analysis/figs/distogram_kl_basins.png
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
BASE = ESM_PREDICTIONS / "7txw_reps_baseline" / "7txw" / "seed_0"
OUT = Path("/home/ebrahim/tts_abag/analysis/figs")
AB, C = slice(0, 438), slice(438, 609)        # antibody rows, antigen cols
G = [10, 14, 33, 39]
B1 = [1, 2, 3, 4, 5, 7, 8, 9, 11, 16, 17, 18, 19, 20, 21, 22, 23, 26, 28, 29, 30, 31, 32,
      34, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49, 52, 55, 56, 57, 58, 59]
B2 = [0, 6, 12, 13, 15, 24, 27, 35, 43, 47, 50, 53, 54]


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
    def cross_distogram(z_path):
        z = torch.from_numpy(np.load(z_path).astype(np.float32)).cuda()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            z = model.parcae_readout(z)
            z = model.parcae_coda(z, pair_attention_mask=pair_mask)
        z = z.float()
        logits = model.distogram_head(z + z.transpose(-2, -3))[0]   # [609,609,nbins]
        P = F.softmax(logits, dim=-1)[AB, C, :].reshape(-1, logits.shape[-1])
        return P.cpu().numpy().astype(np.float32)                    # [npairs, nbins]

    seeds = list(range(60))
    P0 = cross_distogram(BASE / "reps" / "pairformer_z_loop16.npy")  # baseline
    npairs, nbins = P0.shape
    print(f"cross-block: {npairs} pairs x {nbins} bins", flush=True)
    Ps = np.empty((60, npairs * nbins), np.float32)
    for s in seeds:
        Ps[s] = cross_distogram(REPS / f"seed_{s}" / "reps" / "pairformer_z_loop16.npy").reshape(-1)
        if s % 15 == 0:
            print(f"  seed {s}", flush=True)
    logPs = np.log(Ps + 1e-9)
    p0 = P0.reshape(-1)[None]; logp0 = np.log(P0.reshape(-1) + 1e-9)[None]

    # pairwise KL(i||j) = mean_pair sum_bin P_i log(P_i/P_j)
    self_ = (Ps * logPs).sum(1) / npairs
    cross = (Ps @ logPs.T) / npairs
    KL = self_[:, None] - cross                     # [60,60]
    KL = 0.5 * (KL + KL.T)                           # symmetrize for reporting
    # KL to baseline
    kl_base = self_ - (Ps @ logp0.T).ravel() / npairs

    lab = np.array(["mid"] * 60, dtype=object)
    for s in G: lab[s] = "G"
    for s in B1: lab[s] = "B1"
    for s in B2: lab[s] = "B2"

    def mean_block(la, lb):
        i = np.where(lab == la)[0]; j = np.where(lab == lb)[0]
        sub = KL[np.ix_(i, j)]
        if la == lb:
            m = ~np.eye(len(i), dtype=bool); return sub[m].mean()
        return sub.mean()

    print("\n=== cross-block distogram-KL between basins (mean) ===")
    for a in ("G", "B1", "B2"):
        print("  " + "  ".join(f"{a}-{b}={mean_block(a,b):.3f}" for b in ("G", "B1", "B2")))
    print("\nHEADLINE — does distogram-KL separate G from the confident decoy B2?")
    print(f"  within-G={mean_block('G','G'):.3f}  within-B2={mean_block('B2','B2'):.3f}  "
          f"G-B2={mean_block('G','B2'):.3f}  (G-B2 >> withins => YES)")

    print("\n=== KL(seed || baseline) by basin  (baseline is no-mask/no-dropout) ===")
    for a in ("G", "B1", "B2", "mid"):
        v = kl_base[lab == a]
        print(f"  {a:3s} n={len(v):2d}  KL-to-baseline mean={v.mean():.3f}  "
              f"[{v.min():.3f},{v.max():.3f}]")

    # silhouette of basin labels under KL distance (G vs rest), like pca_blocks
    def silhouette():
        good = lab == "G"; other = ~good & (lab != "mid")
        s = []
        for i in np.where(good)[0]:
            same = good.copy(); same[i] = False
            a = KL[i, same].mean(); b = KL[i, other].mean()
            s.append((b - a) / max(a, b))
        return float(np.mean(s))
    print(f"\ngood-seed silhouette under distogram-KL distance = {silhouette():+.3f} "
          f"(cf. pairformer-z PCA block A×C ~+0.71)")

    # ---- figure ----
    order = np.concatenate([np.where(lab == l)[0] for l in ("G", "B2", "B1", "mid")])
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    im = ax[0].imshow(KL[np.ix_(order, order)], cmap="magma")
    bounds = np.cumsum([len(G), len(B2), len(B1)])
    for b in bounds[:-1]:
        ax[0].axhline(b - 0.5, color="cyan", lw=0.8); ax[0].axvline(b - 0.5, color="cyan", lw=0.8)
    ax[0].set_title("cross-block distogram-KL (seeds ordered G | B2 | B1 | mid)")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    cols = {"G": "#d62728", "B1": "#7fa8d0", "B2": "#ff8c00", "mid": "0.6"}
    for a in ("G", "B1", "B2", "mid"):
        v = kl_base[lab == a]
        ax[1].scatter([a] * len(v), v, c=cols[a], s=45, alpha=0.8)
    ax[1].set_ylabel("KL(seed || baseline)  on antibody×antigen block")
    ax[1].set_title("distance from the no-mask/dropout baseline by basin")
    fig.tight_layout()
    fig.savefig(OUT / "distogram_kl_basins.png", dpi=140, bbox_inches="tight")
    print("\nwrote", OUT / "distogram_kl_basins.png")


if __name__ == "__main__":
    main()
