"""Pairwise KL and Wasserstein-1 between the 7txw basin centroids (G/B1/B2), on the
cross-block distogram. Centroid = mean distogram over each basin's seeds (loop16).
W1 in Angstrom using the trunk distogram's ~0.81 A/bin (64 bins over 2-52 A)."""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

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
AB, C = slice(0, 438), slice(438, 609)
G = [10, 14, 33, 39]
B1 = [1, 2, 3, 4, 5, 7, 8, 9, 11, 16, 17, 18, 19, 20, 21, 22, 23, 26, 28, 29, 30, 31, 32,
      34, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49, 52, 55, 56, 57, 58, 59]
B2 = [0, 6, 12, 13, 15, 24, 27, 35, 43, 47, 50, 53, 54]
W = 50.0 / 62.0   # distogram bin spacing in A: linspace(2,52,63) boundaries -> 64 bins


def main():
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
    model.set_chunk_size(32)
    builder = ESMFold2InputBuilder()
    feat, _ = builder.prepare_input(load_complex(str(JSON_DIR), "7txw"), seed=0, device=model.device)
    with torch.no_grad():
        cond = build_conditioning(model, feat)
    pm = cond["pair_mask"]

    @torch.no_grad()
    def disto(s):
        z = torch.from_numpy(np.load(REPS / f"seed_{s}" / "reps" / "pairformer_z_loop16.npy").astype(np.float32)).cuda()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            z = model.parcae_readout(z); z = model.parcae_coda(z, pair_attention_mask=pm)
        logits = model.distogram_head(z.float() + z.float().transpose(-2, -3))[0]
        return F.softmax(logits, -1)[AB, C, :].reshape(-1, logits.shape[-1]).cpu().numpy().astype(np.float32)

    def cen(seeds):
        a = None
        for s in seeds:
            d = disto(s); a = d if a is None else a + d
        return a / len(seeds)
    print("building centroids...", flush=True)
    cents = {"G": cen(G), "B1": cen(B1), "B2": cen(B2)}

    kl = lambda P, Q: float((P * (np.log(P + 1e-9) - np.log(Q + 1e-9))).sum(1).mean())
    w1 = lambda P, Q: float(W * np.abs(np.cumsum(P, 1) - np.cumsum(Q, 1)).sum(1).mean())

    print(f"\n{'pair':>7} {'KL':>8} {'W1 (A)':>8}")
    for a, b in [("B1", "G"), ("B1", "B2"), ("G", "B2")]:
        print(f"{a+'-'+b:>7} {0.5*(kl(cents[a],cents[b])+kl(cents[b],cents[a])):>8.3f} "
              f"{w1(cents[a], cents[b]):>8.3f}")


if __name__ == "__main__":
    main()
