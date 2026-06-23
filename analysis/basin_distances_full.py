"""Within- and between-basin KL and Wasserstein-1, seed-pair-mean (consistent), on the
cross-block distogram. within(basin)=mean over seed-pairs inside it; between=across basins."""
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
Wbin = 50.0 / 62.0
LAB = {**{s: "G" for s in G}, **{s: "B1" for s in B1}, **{s: "B2" for s in B2}}


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

    d0 = disto(0); npairs, nbins = d0.shape
    Pf = np.empty((60, npairs * nbins), np.float32); cumf = np.empty_like(Pf)
    for s in range(60):
        d = disto(s); Pf[s] = d.reshape(-1); cumf[s] = np.cumsum(d, 1).reshape(-1)
        if s % 15 == 0:
            print(f"  seed {s}", flush=True)
    logPf = np.log(Pf + 1e-9)
    self_ = (Pf * logPf).sum(1) / npairs
    KL = self_[:, None] - (Pf @ logPf.T) / npairs
    KL = 0.5 * (KL + KL.T)
    W1 = np.zeros((60, 60))
    for i in range(60):
        W1[i] = np.abs(cumf[i][None] - cumf).sum(1) / npairs * Wbin

    lab = np.array([LAB.get(s, "mid") for s in range(60)])

    def agg(A, B, Mx):
        i = np.where(lab == A)[0]; j = np.where(lab == B)[0]
        if A == B:
            m = Mx[np.ix_(i, i)][np.triu_indices(len(i), 1)]
        else:
            m = Mx[np.ix_(i, j)].ravel()
        return m.mean()

    print(f"\n{'set':>7} {'KL':>8} {'W1 (A)':>8}")
    print("-- within --")
    for b in ("G", "B1", "B2"):
        print(f"{b:>7} {agg(b, b, KL):>8.3f} {agg(b, b, W1):>8.3f}")
    print("-- between --")
    for a, b in [("B1", "G"), ("B1", "B2"), ("G", "B2")]:
        print(f"{a+'-'+b:>7} {agg(a, b, KL):>8.3f} {agg(a, b, W1):>8.3f}")


if __name__ == "__main__":
    main()
