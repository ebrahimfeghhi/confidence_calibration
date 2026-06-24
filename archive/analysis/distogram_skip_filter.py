"""Cheap pre-diffusion skip filter: after the trunk but BEFORE the diffusion head,
compute the seed's cross-block distogram-KL to the no-mask/dropout baseline. If it's
~identical to baseline (KL < T) the seed stayed in the default B1 basin -> it will just
reproduce the baseline's (bad) answer, so SKIP its diffusion decode.

How many diffusion runs does this save, and does it ever skip a GOOD seed? (60-seed
with-dropout reps run; 5 diffusion samples/seed.) numpy + torch + model.
"""
import sys, csv
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
from paths import ESM_PREDICTIONS, JSON_DIR, CSV_DIR  # noqa: E402

REPS = ESM_PREDICTIONS / "7txw_reps" / "7txw"
BASE = ESM_PREDICTIONS / "7txw_reps_baseline" / "7txw" / "seed_0" / "reps" / "pairformer_z_loop16.npy"
AB, C = slice(0, 438), slice(438, 609)
NS = 5  # diffusion samples per seed in this run


def main():
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
    model.set_chunk_size(32)
    builder = ESMFold2InputBuilder()
    feat, _ = builder.prepare_input(load_complex(str(JSON_DIR), "7txw"), seed=0, device=model.device)
    with torch.no_grad():
        cond = build_conditioning(model, feat)
    pm = cond["pair_mask"]

    @torch.no_grad()
    def disto(path):
        z = torch.from_numpy(np.load(path).astype(np.float32)).cuda()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            z = model.parcae_readout(z); z = model.parcae_coda(z, pair_attention_mask=pm)
        logits = model.distogram_head(z.float() + z.float().transpose(-2, -3))[0]
        return F.softmax(logits, -1)[AB, C, :].reshape(-1, logits.shape[-1]).cpu().numpy().astype(np.float32)

    kl = lambda P, Q: float((P * (np.log(P + 1e-9) - np.log(Q + 1e-9))).sum(1).mean())
    P0 = disto(BASE)
    klb = np.array([kl(disto(REPS / f"seed_{s}" / "reps" / "pairformer_z_loop16.npy"), P0)
                    for s in range(60)])

    dq = {}
    for r in csv.DictReader(open(CSV_DIR / "dockq_ost_7txw_reps_all_samples.csv")):
        if r["chain1"] == "A" and r["chain2"] == "C":
            s = int(r["seed"]); dq[s] = max(dq.get(s, 0.0), float(r["dockq"]))
    good = np.array([dq.get(s, 0) >= 0.5 for s in range(60)])
    print(f"good seeds: {[s for s in range(60) if good[s]]}  (n={good.sum()})")
    print(f"KL-to-baseline: good={klb[good].min():.3f}-{klb[good].max():.3f}  "
          f"bad={klb[~good].min():.3f}-{klb[~good].max():.3f}\n")

    print(f"{'T':>5} {'skipped':>8} {'%diff_saved':>12} {'GOOD_skipped':>13} {'kept':>5} "
          f"{'good_in_kept':>12}")
    for T in (0.02, 0.03, 0.04, 0.045, 0.05, 0.06, 0.08, 0.10, 0.12):
        skip = klb < T
        gs = (skip & good).sum()
        print(f"{T:>5.3f} {skip.sum():>8d} {skip.sum()*NS}/{60*NS} = {skip.sum()/60:>5.0%} "
              f"{gs:>13d} {(~skip).sum():>5d} {(good & ~skip).sum()}/{(~skip).sum():>5d}")

    safe = (klb < 0.045)
    print(f"\nSAFE filter (skip KL<0.045 = the default-B1 seeds):")
    print(f"  skip {safe.sum()}/60 seeds -> save {safe.sum()*NS}/{60*NS} diffusion decodes "
          f"({safe.sum()/60:.0%}); GOOD seeds skipped = {(safe & good).sum()}")


if __name__ == "__main__":
    main()
