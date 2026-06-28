"""Causal force-mask test for 7txw: does masking the enriched antigen residues
C48-C50 raise the good-basin rate?

The powered observational analysis (analysis/mask_pattern_powered.py) found that
good 7txw seeds mask antigen residues C48/C49/C50 more than bad seeds. This script
intervenes on the mask to test causation: fold the SAME seeds under three
conditions and compare good-rates.

  none       : natural random LM mask only (baseline)
  candidate  : random mask + FORCE C48,C49,C50 masked
  control    : random mask + FORCE a size-matched inert set (default C65,C70,C98)

Mechanism: compute_lm_hidden_states() draws the random mask and writes the mask
token (32) into lm_input_ids, then calls model._esmc(input_ids=...). We swap
model._esmc for a thin wrapper that sets the forced LM positions to the mask token
right before the real LM forward -- i.e. force-masking ON TOP of the natural draw,
after it, so the per-seed random mask is byte-identical across conditions and only
the forced residues differ. (Same submodule boundary verify_mask_reproduction.py
captures at.) Forced LM positions are derived from the deterministic
(asym_id, residue_index) -> lm-position map (analyze_masks.build_lm_input_ids).

Outputs predictions per condition under ESM_PREDICTIONS/{run_prefix}_{cond}/7txw/
seed_*/predictions (cif + summary_confidence json), scoreable by the normal eval.

Usage (one GPU):
    CUDA_VISIBLE_DEVICES=0 ~/esm/.venv/bin/python run/esm/run_7txw_forcemask.py \
        --seeds $(seq 3000 3049) --conditions none candidate control
"""
import argparse
import json
import string
import sys
from pathlib import Path

import numpy as np
import torch

from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
sys.path.insert(0, str(_p / "analysis"))
from paths import ESM_PREDICTIONS, JSON_DIR
from runlog import save_run_config
import analyze_masks as am

CHAIN_LETTERS = string.ascii_uppercase
MASK_TOKEN_ID = 32


def load_complex(json_dir: str, name: str) -> StructurePredictionInput:
    json_dir = Path(json_dir)
    for json_path in sorted(json_dir.glob("*.json")):
        if "updated" in json_path.name or "update-msa" in json_path.name:
            continue
        with open(json_path) as f:
            data = json.load(f)
        entry = data[0]
        if entry["name"] != name:
            continue
        chain_idx = 0
        inputs = []
        for seq_entry in entry["sequences"]:
            if "proteinChain" not in seq_entry:
                continue
            chain_data = seq_entry["proteinChain"]
            count = chain_data.get("count", 1)
            ids = [CHAIN_LETTERS[chain_idx + i] for i in range(count)]
            chain_idx += count
            inputs.append(ProteinInput(id=ids if count > 1 else ids[0],
                                        sequence=chain_data["sequence"]))
        return StructurePredictionInput(sequences=inputs)
    raise SystemExit(f"Complex {name!r} not found in {json_dir}")


def parse_residues(specs):
    """['C48','C49'] -> {(asym_idx, res_idx0based)}."""
    out = set()
    for s in specs:
        chain, num = s[0], int(s[1:])
        out.add((ord(chain) - ord("A"), num - 1))
    return out


def lm_positions_for_residues(features, expand_maps, residue_keys):
    """LM-input positions (batch 0) for a set of (asym_idx, res_idx0based) keys."""
    asym_id = features["asym_id"][0]
    res_idx = features["residue_index"][0]
    mol_type = features["mol_type"][0]
    tok_mask = features["token_attention_mask"][0]
    protein_mask = (mol_type == 0) & tok_mask.bool()
    expand_map = expand_maps[0]
    found, positions = set(), set()
    for pos in protein_mask.nonzero(as_tuple=True)[0].cpu().tolist():
        key = (asym_id[pos].item(), res_idx[pos].item())
        if key in residue_keys:
            lp = expand_map[pos].item()
            if lp >= 0:
                positions.add(lp)
                found.add(key)
    missing = residue_keys - found
    if missing:
        raise SystemExit(f"residues not found in input: {sorted(missing)}")
    return sorted(positions)


class ForcedMaskEsmc(torch.nn.Module):
    """Wrap the real ESMC: force given LM positions to the mask token, then delegate."""

    def __init__(self, inner, lm_positions):
        super().__init__()
        self.inner = inner
        self.register_buffer("pos", torch.tensor(lm_positions, dtype=torch.long),
                             persistent=False)
        self._reported = False

    def forward(self, *args, **kwargs):
        if "input_ids" in kwargs:
            ids = kwargs["input_ids"].clone()
        elif args:
            ids = args[0].clone()
            args = args[1:]
        else:
            raise RuntimeError("ForcedMaskEsmc: no input_ids")
        pos = self.pos.to(ids.device)
        if not self._reported:
            already = int((ids[:, pos] == MASK_TOKEN_ID).sum().item())
            print(f"      [forcemask] forcing {pos.numel()} LM positions "
                  f"({already} already masked by the random draw)", flush=True)
            self._reported = True
        ids[:, pos] = MASK_TOKEN_ID
        kwargs["input_ids"] = ids
        return self.inner(*args, **kwargs)

    __call__ = forward


def save_samples(samples, pred_dir, name):
    pred_dir.mkdir(parents=True, exist_ok=True)
    for k, sample in enumerate(samples):
        with open(pred_dir / f"{name}_sample_{k}.cif", "w") as f:
            f.write(sample.complex.to_mmcif())
        conf = {
            "ranking_score": sample.iptm if sample.iptm is not None else (sample.ptm or 0.0),
            "iptm": sample.iptm,
            "ptm": sample.ptm,
            "chain_pair_iptm": sample.pair_chains_iptm.tolist()
            if sample.pair_chains_iptm is not None else None,
        }
        with open(pred_dir / f"{name}_summary_confidence_sample_{k}.json", "w") as f:
            json.dump(conf, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_prefix", default="7txw_force")
    ap.add_argument("--complex", default="7txw")
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--conditions", nargs="+", default=["none", "candidate", "control"],
                    choices=["none", "candidate", "control"])
    ap.add_argument("--candidate", nargs="+", default=["C48", "C49", "C50"])
    ap.add_argument("--control", nargs="+", default=["C65", "C70", "C98"])
    ap.add_argument("--lm_mask_pct", type=float, default=0.1)
    ap.add_argument("--lm_dropout", type=float, default=0.3)
    ap.add_argument("--json_dir", default=str(JSON_DIR))
    ap.add_argument("--num_loops", type=int, default=16)
    ap.add_argument("--num_sampling_steps", type=int, default=16)
    ap.add_argument("--num_diffusion_samples", type=int, default=5)
    ap.add_argument("--chunk_size", type=int, default=32)
    ap.add_argument("--no_config_log", action="store_true")
    args = ap.parse_args()

    name = args.complex
    print("Loading ESMFold2-Fast...", flush=True)
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
    model.set_chunk_size(args.chunk_size if args.chunk_size > 0 else None)
    builder = ESMFold2InputBuilder()
    spi = load_complex(args.json_dir, name)
    orig_esmc = model._esmc

    # deterministic (asym,res) -> lm-position map for this complex
    features, _ = builder.prepare_input(spi, seed=0, device="cuda:0")
    _lm_ids, expand_maps = am.build_lm_input_ids(features)
    cond_positions = {
        "candidate": lm_positions_for_residues(features, expand_maps,
                                               parse_residues(args.candidate)),
        "control": lm_positions_for_residues(features, expand_maps,
                                             parse_residues(args.control)),
    }
    print(f"candidate {args.candidate} -> LM pos {cond_positions['candidate']}")
    print(f"control   {args.control} -> LM pos {cond_positions['control']}")

    for cond in args.conditions:
        if cond == "none":
            model._esmc = orig_esmc
        else:
            model._esmc = ForcedMaskEsmc(orig_esmc, cond_positions[cond])
        out_dir = ESM_PREDICTIONS / f"{args.run_prefix}_{cond}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_config_log:
            save_run_config(out_dir, {"model": "biohub/ESMFold2-Fast", "complex": name,
                                      "condition": cond, **vars(args)})
        print(f"\n=== condition {cond} -> {out_dir.name} ===", flush=True)

        for seed_val in args.seeds:
            seed_int = int(seed_val)
            pred_dir = out_dir / name / f"seed_{seed_int}" / "predictions"
            if any(pred_dir.glob(f"{name}_summary_confidence_sample_*.json")):
                print(f"  seed_{seed_int}  (skip — exists)", flush=True)
                continue
            print(f"  seed_{seed_int} ...", end="", flush=True)
            if isinstance(model._esmc, ForcedMaskEsmc):
                model._esmc._reported = False
            try:
                result = builder.fold(
                    model, spi, seed=seed_int,
                    num_loops=args.num_loops,
                    num_sampling_steps=args.num_sampling_steps,
                    num_diffusion_samples=args.num_diffusion_samples,
                    lm_mask_pct=args.lm_mask_pct,
                    lm_dropout=args.lm_dropout,
                )
                samples = result if isinstance(result, list) else [result]
                save_samples(samples, pred_dir, name)
                best_iptm = max(s.iptm or s.ptm or 0.0 for s in samples)
                print(f"  best ipTM={best_iptm:.3f}", flush=True)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()
            finally:
                torch.cuda.empty_cache()

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
