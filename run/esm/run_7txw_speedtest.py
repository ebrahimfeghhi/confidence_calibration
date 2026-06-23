"""Speed test of the constant-injection optimization.

When lm_dropout=0 and there's no MSA (ESMFold2-Fast), the per-loop injection
  u_t = LN(z_init + lm_encoder(lm_z))   and   B·u_t
is CONSTANT across all 17 loops. So we precompute  C = B·LN(z_init + lm_encoder(lm_z))
ONCE and collapse each recycle to  z = a⊙z + C ; z = folding_trunk(z)  — hoisting the
expensive per-loop lm_encoder + LayerNorm + B matmul. Output is mathematically identical.

Compares ORIGINAL vs FAST (`_run_one_loop` monkeypatch) on the two mask-robust good seeds
{10, 33}, 10 runs each, lm_dropout=0. Records wall-time/run; structures saved for DockQ.

Usage (one GPU):
    CUDA_VISIBLE_DEVICES=3 python run/esm/run_7txw_speedtest.py
"""
import csv
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F

from esm.models.esmfold2 import ESMFold2InputBuilder
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

from run_7txw_loop_diffusion import load_complex, save_samples  # same dir

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import ESM_PREDICTIONS, JSON_DIR, CSV_DIR  # noqa: E402

SEEDS = [10, 33]
N_RUNS = 10


def fast_run_one_loop(self, z, z_init, lm_z, _msa_inputs, pair_mask, a, b_mat,
                      tok_mask, total_steps):
    """Drop-in for _run_one_loop, valid ONLY when lm_dropout=0 and no MSA:
    precompute the constant injection C, then collapse the loop."""
    assert _msa_inputs is None, "fast path assumes no MSA"
    refined = None
    if lm_z is not None and self.lm_encoder is not None:
        refined = self.lm_encoder(lm_z.to(z_init.dtype), pair_attention_mask=pair_mask)
    z_inject = z_init
    if lm_z is not None and self.lm_encoder is None:
        z_inject = z_inject + lm_z.to(z_inject.dtype)
    if refined is not None:
        z_inject = z_inject + refined.to(z_inject.dtype)
    C = F.linear(self.parcae_input_norm(z_inject).to(z.dtype), b_mat)   # constant
    for _ in range(total_steps):
        z = a * z + C
        z = self.folding_trunk(z, pair_attention_mask=pair_mask)
    return z


def main():
    name = "7txw"
    print("Loading ESMFold2-Fast...", flush=True)
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
    model.set_chunk_size(32)
    builder = ESMFold2InputBuilder()
    spi = load_complex(str(JSON_DIR), name)

    def fold(seed):
        return builder.fold(model, spi, seed=seed, num_loops=16, num_sampling_steps=16,
                            num_diffusion_samples=5, lm_mask_pct=0.1, lm_dropout=0.0)

    timings = []
    for method in ("orig", "fast"):
        if method == "fast":
            model._run_one_loop = types.MethodType(fast_run_one_loop, model)
        # warm up this code path (kernel caching) — discarded
        with torch.no_grad():
            fold(10)
        torch.cuda.synchronize()
        print(f"--- method={method} (warmed up) ---", flush=True)
        for seed in SEEDS:
            for r in range(N_RUNS):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.no_grad():
                    result = fold(seed)
                torch.cuda.synchronize(); dt = time.perf_counter() - t0
                samples = result if isinstance(result, list) else [result]
                pid = seed * 100 + r          # pseudo-seed: 1000-1009, 3300-3309
                pred_dir = ESM_PREDICTIONS / f"7txw_speed_{method}" / name / f"seed_{pid}" / "predictions"
                save_samples(pred_dir, name, samples)
                best = max(s.iptm or s.ptm or 0.0 for s in samples)
                timings.append(dict(method=method, seed=seed, run=r, pseudo_seed=pid,
                                    seconds=round(dt, 3)))
                print(f"  {method} seed{seed} run{r}  {dt:6.1f}s  best_ipTM={best:.3f}", flush=True)

    out = CSV_DIR / "speedtest_timings.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "seed", "run", "pseudo_seed", "seconds"])
        w.writeheader(); w.writerows(timings)
    # quick summary
    for method in ("orig", "fast"):
        ts = [t["seconds"] for t in timings if t["method"] == method]
        print(f"{method}: mean {sum(ts)/len(ts):.1f}s/run over {len(ts)} runs")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
