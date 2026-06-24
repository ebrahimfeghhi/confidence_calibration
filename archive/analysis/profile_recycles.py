"""Separate ESMC (fixed, runs once) from the per-recycle trunk cost, so we know what
'fewer trunk recycles' actually saves.

Time builder.fold at several num_loops (each fold = ESMC + (L+1) trunk iters + diffusion5
+ confidence + decode). The post-trunk decode (diffusion+conf+decode = POST) is L-independent.
  total(L) = (ESMC + POST) + (L+1) * t_loop
=> t_loop = slope; ESMC = intercept - POST  (POST≈3.48s measured by profile_stages.py).
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "run" / "esm"))
from run_7txw_loop_diffusion import load_complex  # noqa: E402
from esm.models.esmfold2 import ESMFold2InputBuilder  # noqa: E402
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model  # noqa: E402
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import JSON_DIR  # noqa: E402

POST = 3.48  # diffusion(5)+confidence+readout, from profile_stages.py


def timed(fn, reps=2, warmup=1):
    for _ in range(warmup):
        fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
    model.set_chunk_size(32)
    builder = ESMFold2InputBuilder()
    spi = load_complex(str(JSON_DIR), "7txw")

    @torch.no_grad()
    def fold(L):
        return builder.fold(model, spi, seed=0, num_loops=L, num_sampling_steps=16,
                            num_diffusion_samples=5, lm_mask_pct=0.1, lm_dropout=0.0)

    Ls = [1, 8, 16]
    t = {L: timed(lambda L=L: fold(L)) for L in Ls}
    for L in Ls:
        print(f"  fold(num_loops={L:2d}) = {t[L]:.2f}s   ({L+1} trunk iters)", flush=True)

    t_loop = (t[16] - t[1]) / ((16 + 1) - (1 + 1))     # per-recycle trunk cost
    trunk17 = 17 * t_loop                              # full 17-iter recycling
    esmc = (t[16] - trunk17) - POST                    # fixed LM forward
    full = t[16]
    print(f"\n{'component':<32}{'sec':>8}{'% of full':>10}")
    for nm, v in [("FULL fold (num_loops=16)", full),
                  ("  ESMC LM forward (once, fixed)", esmc),
                  ("  trunk recycling x17", trunk17),
                  ("    per recycle iteration", t_loop),
                  ("  post-trunk decode (diff5+conf)", POST)]:
        print(f"{nm:<32}{v:>8.2f}{100*v/full:>9.1f}%")
    # linearity check
    pred8 = (esmc + POST) + (8 + 1) * t_loop
    print(f"\nlinearity check: predicted fold(8)={pred8:.2f}s vs measured {t[8]:.2f}s")
    print(f"\n>>> per trunk recycle = {t_loop:.2f}s ; ESMC fixed = {esmc:.1f}s "
          f"({100*esmc/full:.0f}% of full)")
    print(f">>> cutting 17->N recycles saves {t_loop:.2f}s each "
          f"(e.g. stop at loop 9 saves ~{8*t_loop:.1f}s = {100*8*t_loop/full:.0f}% of full)")


if __name__ == "__main__":
    main()
