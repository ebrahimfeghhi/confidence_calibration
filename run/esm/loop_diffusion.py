"""Diffuse from each recycling loop's saved pairformer-z and score the result.

For a set of (seed, loop) cells, load the *already-captured* trunk rep
`pairformer_z_loop{LL}.npy`, apply the final parcae readout+coda, and run the
diffusion structure head — i.e. "what structure would the trunk commit to if it
stopped at loop L?". Writes cif + confidence per sample so the normal OST eval
scores DockQ across loops.

Why this is faithful (see modeling_esmfold2.ESMFold2Model.forward):
  * diffusion conditions only on z_trunk + deterministic input features
    (s_inputs=x_inputs, s_trunk=None) — never on the LM rep — so the saved
    per-loop pair z is sufficient and NO ESMC pass is needed.
  * reusing the SAVED z (not re-running the trunk) keeps the diffused rep
    identical to the one analyzed by PCA; re-running would draw fresh per-loop
    LM-dropout/MSA-subsample noise and give a different trajectory.
  * loop16 == final trunk output, so readout(coda(z_loop16)) equals the original
    run's z_trunk -> loop16 DockQ must reproduce the original CSV (built-in check).

This mirrors forward() lines ~886-1081 verbatim, replacing only the trunk loop
(`_run_one_loop`) with a load of the saved z.

Usage (one GPU):
    python run/esm/loop_diffusion.py --cells 10:16 14:16 --run_name 7txw_loops
"""
import argparse
import json
import string
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
from transformers.models.esmfold2.modeling_esmfold2 import (
    ESMFold2Model, NUM_RES_TYPES, MAX_ATOMIC_NUMBER, CHAR_VOCAB_SIZE,
)

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import ESM_PREDICTIONS, JSON_DIR

CHAIN_LETTERS = string.ascii_uppercase


def load_complex(json_dir, name):
    for json_path in sorted(Path(json_dir).glob("*.json")):
        if "updated" in json_path.name or "update-msa" in json_path.name:
            continue
        data = json.load(open(json_path))
        entry = data[0]
        if entry["name"] != name:
            continue
        chain_idx, inputs = 0, []
        for seq_entry in entry["sequences"]:
            if "proteinChain" not in seq_entry:
                continue
            cd = seq_entry["proteinChain"]
            count = cd.get("count", 1)
            ids = [CHAIN_LETTERS[chain_idx + i] for i in range(count)]
            chain_idx += count
            inputs.append(ProteinInput(id=ids if count > 1 else ids[0], sequence=cd["sequence"]))
        return StructurePredictionInput(sequences=inputs)
    raise SystemExit(f"Complex {name!r} not found in {json_dir}")


def build_conditioning(model, feat):
    """Replicate forward()'s deterministic input prep (no LM, no trunk).

    Returns the kwargs bundle that structure_head.sample / confidence_head need;
    identical across seeds and loops, so computed once."""
    tok_mask = feat["token_attention_mask"]
    atm_mask = feat["atom_attention_mask"]
    res_type = feat["res_type"]

    if res_type.dim() == 2:
        res_type_oh = F.one_hot(res_type.long(), num_classes=NUM_RES_TYPES).float()
        res_type_oh = res_type_oh * tok_mask.unsqueeze(-1).float()
    else:
        res_type_oh = res_type.float()

    msa = feat.get("msa")
    if msa is not None:
        msa_oh = F.one_hot(msa.long(), num_classes=NUM_RES_TYPES).float()
        mam = feat.get("msa_attention_mask")
        if mam is not None:
            msa_oh = msa_oh * mam.float().unsqueeze(-1)
            cnt = mam.float().sum(dim=1).clamp(min=1)
            profile = msa_oh.sum(dim=1) / cnt.unsqueeze(-1)
        else:
            profile = msa_oh.mean(dim=1)
    else:
        profile = res_type_oh

    deletion_mean = feat.get("deletion_mean")
    if deletion_mean is None:
        deletion_mean = torch.zeros(res_type.shape[0], res_type.shape[1], device=res_type.device)

    ref_element_oh = F.one_hot(feat["ref_element"].long(), num_classes=MAX_ATOMIC_NUMBER).float()
    ref_name_oh = F.one_hot(feat["ref_atom_name_chars"].long(), num_classes=CHAR_VOCAB_SIZE).float()
    atm_f = atm_mask.float()
    ref_element_oh = ref_element_oh * atm_f.unsqueeze(-1)
    ref_name_oh = ref_name_oh * atm_f.unsqueeze(-1).unsqueeze(-1)
    atom_to_token = feat["atom_to_token"] * atm_mask.long()

    use_amp = res_type.device.type == "cuda"
    with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
        x_inputs = model.inputs_embedder(
            aatype=res_type_oh, profile=profile.float(), deletion_mean=deletion_mean.float(),
            ref_pos=feat["ref_pos"], atom_attention_mask=atm_mask,
            ref_space_uid=feat["ref_space_uid"], ref_charge=feat["ref_charge"],
            ref_element=ref_element_oh, ref_atom_name_chars=ref_name_oh,
            atom_to_token=atom_to_token,
        )
        rel_pos = model.rel_pos(
            residue_index=feat["residue_index"], asym_id=feat["asym_id"],
            sym_id=feat["sym_id"], entity_id=feat["entity_id"], token_index=feat["token_index"],
        )
        token_bonds_encoding = model.token_bonds(feat["token_bonds"].float())
    pair_mask = tok_mask[:, :, None].float() * tok_mask[:, None, :].float()

    return dict(x_inputs=x_inputs, rel_pos=rel_pos, token_bonds_encoding=token_bonds_encoding,
                ref_element_oh=ref_element_oh, ref_name_oh=ref_name_oh,
                atom_to_token=atom_to_token, pair_mask=pair_mask)


def diffuse_from_z(model, builder, feat, chain_infos, cond, z_np, n_samples, n_steps,
                   name, rng_seed, noise_scale=None, step_scale=None):
    """One cell: saved z -> readout/coda -> diffusion -> decode -> samples.

    noise_scale / step_scale override the sampler's churn (lambda) and ODE-step
    (eta) knobs; None keeps the model defaults (noise_scale=0.0, step_scale=1.0),
    i.e. a deterministic probability-flow ODE whose only diversity is the initial
    coordinate draw. Elevated noise_scale injects fresh Gaussian noise at each
    high-sigma step (Idea 1: pressure-test whether a suppressed basin reappears)."""
    tok_mask = feat["token_attention_mask"]
    atm_mask = feat["atom_attention_mask"]
    disto_idx = feat["distogram_atom_idx"]
    use_amp = atm_mask.device.type == "cuda"

    z = torch.from_numpy(z_np).to(atm_mask.device).float()        # [1,609,609,256]
    with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
        z = model.parcae_readout(z)
        z = model.parcae_coda(z, pair_attention_mask=cond["pair_mask"])
    z = z.float()
    distogram_logits = model.distogram_head(z + z.transpose(-2, -3))

    torch.manual_seed(rng_seed)
    structure_output = model.structure_head.sample(
        z_trunk=z, s_inputs=cond["x_inputs"], s_trunk=None,
        relative_position_encoding=cond["rel_pos"],
        ref_pos=feat["ref_pos"], ref_charge=feat["ref_charge"], ref_mask=atm_mask,
        ref_element=cond["ref_element_oh"], ref_atom_name_chars=cond["ref_name_oh"],
        ref_space_uid=feat["ref_space_uid"], tok_idx=cond["atom_to_token"],
        asym_id=feat["asym_id"], residue_index=feat["residue_index"],
        entity_id=feat["entity_id"], token_index=feat["token_index"], sym_id=feat["sym_id"],
        token_attention_mask=tok_mask, num_diffusion_samples=n_samples,
        num_sampling_steps=n_steps, noise_scale=noise_scale, step_scale=step_scale,
        return_atom_repr=False, denoising_early_exit_rmsd=None,
    )
    sample_coords = structure_output["sample_atom_coords"]

    output = {"distogram_logits": distogram_logits, "sample_atom_coords": sample_coords}
    confidence_output = model.confidence_head(
        s_inputs=cond["x_inputs"].detach(), z=z.detach().float(),
        x_pred=sample_coords.detach(), distogram_atom_idx=disto_idx,
        token_attention_mask=tok_mask, atom_to_token=cond["atom_to_token"],
        atom_attention_mask=atm_mask, asym_id=feat["asym_id"], mol_type=feat["mol_type"],
        num_diffusion_samples=n_samples,
        relative_position_encoding=cond["rel_pos"].detach(),
        token_bonds_encoding=cond["token_bonds_encoding"].detach(),
    )
    output.update(confidence_output)
    output["atom_pad_mask"] = atm_mask.unsqueeze(0) if atm_mask.dim() == 1 else atm_mask
    output["residue_index"] = feat["residue_index"]
    output["entity_id"] = feat["entity_id"]

    results = builder.decode(output, feat, chain_infos,
                             num_diffusion_samples=n_samples, complex_id=name)
    return results if isinstance(results, list) else [results]


def save_samples(pred_dir, name, samples):
    pred_dir.mkdir(parents=True, exist_ok=True)
    for k, s in enumerate(samples):
        with open(pred_dir / f"{name}_sample_{k}.cif", "w") as f:
            f.write(s.complex.to_mmcif())
        iptm = s.iptm if s.iptm is not None else (s.ptm or 0.0)
        conf = {"ranking_score": iptm, "iptm": s.iptm, "ptm": s.ptm,
                "chain_pair_iptm": s.pair_chains_iptm.tolist()
                if s.pair_chains_iptm is not None else None}
        with open(pred_dir / f"{name}_summary_confidence_sample_{k}.json", "w") as f:
            json.dump(conf, f)
        if s.plddt is not None:
            np.save(pred_dir / f"{name}_plddt_sample_{k}.npy", s.plddt.half().cpu().numpy())
        if s.pae is not None:
            np.save(pred_dir / f"{name}_pae_sample_{k}.npy", s.pae.half().cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True, help='"seed:loop" pairs')
    ap.add_argument("--run_name", default="7txw_loops")
    ap.add_argument("--reps_run", default="7txw_reps")
    ap.add_argument("--complex", default="7txw")
    ap.add_argument("--num_diffusion_samples", type=int, default=5)
    ap.add_argument("--num_sampling_steps", type=int, default=16)
    ap.add_argument("--chunk_size", type=int, default=32)
    ap.add_argument("--sample_batch", type=int, default=0,
                    help="diffusion samples per sample() call (0 = all at once); "
                         "lower to cap GPU memory when num_diffusion_samples is large")
    ap.add_argument("--noise_scale", type=float, default=None,
                    help="sampler churn lambda; None = model default (0.0). "
                         ">0 injects fresh noise each high-sigma step (Idea 1)")
    ap.add_argument("--step_scale", type=float, default=None,
                    help="sampler ODE-step eta; None = model default (1.0)")
    ap.add_argument("--json_dir", default=str(JSON_DIR))
    args = ap.parse_args()
    name = args.complex
    cells = [(int(c.split(":")[0]), int(c.split(":")[1])) for c in args.cells]

    print("Loading ESMFold2-Fast...", flush=True)
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
    model.set_chunk_size(args.chunk_size if args.chunk_size > 0 else None)
    builder = ESMFold2InputBuilder()
    spi = load_complex(args.json_dir, name)

    feat, chain_infos = builder.prepare_input(spi, seed=0, device=model.device)
    with torch.no_grad():
        cond = build_conditioning(model, feat)
    ns = "default(0.0)" if args.noise_scale is None else f"{args.noise_scale}"
    ss = "default(1.0)" if args.step_scale is None else f"{args.step_scale}"
    print(f"conditioning ready; {len(cells)} cells; {args.num_diffusion_samples} samples "
          f"(batch {args.sample_batch or args.num_diffusion_samples}); "
          f"noise_scale={ns} step_scale={ss}", flush=True)

    reps_root = ESM_PREDICTIONS / args.reps_run / name
    out_root = ESM_PREDICTIONS / args.run_name
    for seed, loop in cells:
        pred_dir = out_root / f"loop{loop:02d}" / name / f"seed_{seed}" / "predictions"
        if any(pred_dir.glob(f"{name}_summary_confidence_sample_*.json")):
            print(f"  seed{seed} loop{loop:02d}  (skip)", flush=True)
            continue
        z_path = reps_root / f"seed_{seed}" / "reps" / f"pairformer_z_loop{loop:02d}.npy"
        if not z_path.exists():
            print(f"  seed{seed} loop{loop:02d}  MISSING {z_path}", flush=True)
            continue
        print(f"  seed{seed} loop{loop:02d} ...", end="", flush=True)
        try:
            z_np = np.load(z_path).astype(np.float32)
            total = args.num_diffusion_samples
            bs = args.sample_batch if args.sample_batch > 0 else total
            samples, done, b = [], 0, 0
            while done < total:
                this = min(bs, total - done)
                with torch.no_grad():
                    samples += diffuse_from_z(
                        model, builder, feat, chain_infos, cond, z_np,
                        this, args.num_sampling_steps, name,
                        rng_seed=1000 * seed + loop + 100000 * (b + 1),
                        noise_scale=args.noise_scale, step_scale=args.step_scale)
                done += this
                b += 1
                torch.cuda.empty_cache()
            save_samples(pred_dir, name, samples)
            best = max(s.iptm or s.ptm or 0.0 for s in samples)
            print(f"  best ipTM={best:.3f} over {len(samples)} samples", flush=True)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()
        finally:
            torch.cuda.empty_cache()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
