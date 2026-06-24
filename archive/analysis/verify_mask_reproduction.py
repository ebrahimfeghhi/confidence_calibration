"""Empirically verify that generate_masks() reproduces the REAL run's input mask.

The reproduction assumes (a) the reconstructed lm_input_ids has the same shape as
the model's internal one (broken if fp8 -> pad_to_multiple=8), and (b) the mask
torch.rand is the first RNG draw after seeding. The ~10% sanity check does NOT
test either. This script tests them directly:

  - load the real model exactly as run_foldbench_abag.py does;
  - swap its _esmc submodule for a stub that captures the masked lm_input_ids and
    aborts before the (expensive) LM forward;
  - call the REAL model._compute_lm_hidden_states under _seed_context(seed) so the
    real pad_to_multiple / fp8 logic is exercised;
  - map captured masked positions to residues and diff against generate_masks().

Run: CUDA_VISIBLE_DEVICES=3 ~/esm/.venv/bin/python analysis/verify_mask_reproduction.py
"""
import sys
from pathlib import Path

import torch

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
sys.path.insert(0, str(_p / "analysis"))

from esm.models.esmfold2.processor import _seed_context
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
import analyze_masks as am

BASE = "7txw"
LM_MASK_PCT = am.LM_MASK_PCT  # 0.1
MASK_TOKEN_ID = 32
DEVICE = am.DEVICE


class _Stop(Exception):
    pass


class StubEsmc(torch.nn.Module):
    """Captures the lm_input_ids passed to the LM forward, then aborts."""
    captured = None

    def forward(self, *args, **kwargs):
        StubEsmc.captured = kwargs.get("input_ids", args[0] if args else None).detach().clone()
        raise _Stop()

    __call__ = forward


def masked_residues_from_ids(lm_ids_row, features, expand_map):
    """{(asym, res)} for residues whose LM position holds the mask token."""
    asym_id = features["asym_id"][0]
    res_idx = features["residue_index"][0]
    mol_type = features["mol_type"][0]
    tok_mask = features["token_attention_mask"][0]
    protein_mask = (mol_type == 0) & tok_mask.bool()
    masked_lm = set((lm_ids_row == MASK_TOKEN_ID).nonzero(as_tuple=True)[0].cpu().tolist())
    out = set()
    for pos in protein_mask.nonzero(as_tuple=True)[0].cpu().tolist():
        lp = expand_map[pos].item()
        if lp >= 0 and lp in masked_lm:
            out.add((asym_id[pos].item(), res_idx[pos].item()))
    return out


def main():
    print("loading real ESMFold2-Fast (same as run_foldbench_abag.py) ...")
    # weights stay on CPU: we stub _esmc, so only the real _esmc_fp8 flag and the
    # real _compute_lm_hidden_states pad/RNG logic matter (the mask draws on the
    # CUDA feature tensors regardless of where the unused weights live).
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").eval()
    print(f"  model class      = {type(model).__module__}.{type(model).__name__}")
    print(f"  _esmc_fp8        = {getattr(model, '_esmc_fp8', 'MISSING')}  "
          f"(True => pad_to_multiple=8 => reproduction would be WRONG)")
    print(f"  _esmc is None?   = {getattr(model, '_esmc', None) is None}")

    builder = am.ESMFold2InputBuilder()
    spi, _ = am.build_spi(BASE)
    features, _ = builder.prepare_input(spi, seed=0, device=DEVICE)
    lm_input_ids_repro, expand_maps = am.build_lm_input_ids(features)
    expand_map = expand_maps[0]
    repro = am.generate_masks(lm_input_ids_repro, features, expand_maps)

    # swap in the capturing stub (keep the real _esmc_fp8 flag / pad logic intact)
    model._esmc = StubEsmc()

    n_mismatch = 0
    shape_real = shape_repro = None
    for s in am.SEEDS:
        StubEsmc.captured = None
        with _seed_context(s):
            try:
                model._compute_lm_hidden_states(
                    features["input_ids"], features["asym_id"],
                    features["residue_index"], features["mol_type"],
                    features["token_attention_mask"], lm_mask_pct=LM_MASK_PCT,
                )
            except _Stop:
                pass
        real_ids = StubEsmc.captured
        shape_real, shape_repro = tuple(real_ids.shape), tuple(lm_input_ids_repro.shape)
        real_set = masked_residues_from_ids(real_ids[0], features, expand_map)
        repro_set = repro[s]
        sym = real_set ^ repro_set
        if sym:
            n_mismatch += 1
            if n_mismatch <= 5:
                print(f"  seed {s}: MISMATCH  |real|={len(real_set)} |repro|={len(repro_set)} "
                      f"sym_diff={len(sym)}")
        elif s < 3:
            print(f"  seed {s}: match  ({len(real_set)} masked residues)")

    print(f"\nshapes: real lm_input_ids={shape_real}  repro={shape_repro}  "
          f"{'OK' if shape_real == shape_repro else 'DIFFER!!'}")
    print(f"seeds with identical masked-residue set: {len(am.SEEDS) - n_mismatch}/{len(am.SEEDS)}")
    print("VERDICT:", "reproduction is EXACT ✓" if n_mismatch == 0
          else f"reproduction WRONG for {n_mismatch} seeds ✗")


if __name__ == "__main__":
    main()
