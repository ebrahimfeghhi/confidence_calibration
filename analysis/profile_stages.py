"""Per-seed compute breakdown for 7txw: how much of a full fold is the 5 diffusion
samples (the part the distogram skip-filter avoids)?

Stages timed with cuda.synchronize (warmup + median over reps):
  - full fold            (ESMC LM + trunk recycling + diffusion(5) + confidence + decode)
  - parcae readout/coda + distogram  (tiny, post-trunk)
  - structure_head.sample(N=1) and (N=5)   <- the diffusion samples
  - confidence_head(5)
Derived: ESMC+trunk = full - (post-trunk work); fraction spent on the 5 diffusion samples;
fraction the skip-filter saves (= all post-trunk per-seed decode work).
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "run" / "esm"))
from run_7txw_loop_diffusion import load_complex, build_conditioning  # noqa: E402
from esm.models.esmfold2 import ESMFold2InputBuilder  # noqa: E402
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model  # noqa: E402
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import ESM_PREDICTIONS, JSON_DIR  # noqa: E402

Z = ESM_PREDICTIONS / "7txw_reps" / "7txw" / "seed_10" / "reps" / "pairformer_z_loop16.npy"
STEPS = 16


def timed(fn, reps=4, warmup=2):
    for _ in range(warmup):
        fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
    model.set_chunk_size(32)
    builder = ESMFold2InputBuilder()
    spi = load_complex(str(JSON_DIR), "7txw")
    feat, chain_infos = builder.prepare_input(spi, seed=0, device=model.device)
    with torch.no_grad():
        cond = build_conditioning(model, feat)
    tok_mask, atm_mask = feat["token_attention_mask"], feat["atom_attention_mask"]
    disto_idx = feat["distogram_atom_idx"]
    z_np = np.load(Z).astype(np.float32)

    @torch.no_grad()
    def prep_z():
        z = torch.from_numpy(z_np).cuda().float()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            z = model.parcae_readout(z)
            z = model.parcae_coda(z, pair_attention_mask=cond["pair_mask"])
        return z.float()
    zc = prep_z()

    @torch.no_grad()
    def sample(N):
        torch.manual_seed(0)
        return model.structure_head.sample(
            z_trunk=zc, s_inputs=cond["x_inputs"], s_trunk=None,
            relative_position_encoding=cond["rel_pos"], ref_pos=feat["ref_pos"],
            ref_charge=feat["ref_charge"], ref_mask=atm_mask, ref_element=cond["ref_element_oh"],
            ref_atom_name_chars=cond["ref_name_oh"], ref_space_uid=feat["ref_space_uid"],
            tok_idx=cond["atom_to_token"], asym_id=feat["asym_id"],
            residue_index=feat["residue_index"], entity_id=feat["entity_id"],
            token_index=feat["token_index"], sym_id=feat["sym_id"],
            token_attention_mask=tok_mask, num_diffusion_samples=N, num_sampling_steps=STEPS,
            return_atom_repr=False, denoising_early_exit_rmsd=None)

    coords5 = sample(5)["sample_atom_coords"]

    @torch.no_grad()
    def confidence():
        return model.confidence_head(
            s_inputs=cond["x_inputs"].detach(), z=zc.detach().float(), x_pred=coords5.detach(),
            distogram_atom_idx=disto_idx, token_attention_mask=tok_mask,
            atom_to_token=cond["atom_to_token"], atom_attention_mask=atm_mask,
            asym_id=feat["asym_id"], mol_type=feat["mol_type"], num_diffusion_samples=5,
            relative_position_encoding=cond["rel_pos"].detach(),
            token_bonds_encoding=cond["token_bonds_encoding"].detach())

    @torch.no_grad()
    def full_fold():
        return builder.fold(model, spi, seed=0, num_loops=16, num_sampling_steps=STEPS,
                            num_diffusion_samples=5, lm_mask_pct=0.1, lm_dropout=0.0)

    print("timing stages (609 tokens, 16 sampling steps, 5 diffusion samples)...", flush=True)
    t_prep = timed(prep_z)
    t_s1, t_s5 = timed(lambda: sample(1)), timed(lambda: sample(5))
    t_conf = timed(confidence)
    t_full = timed(full_fold, reps=3, warmup=1)

    postz = t_prep + t_s5 + t_conf            # post-trunk per-seed decode work (skip saves this)
    esmc_trunk = t_full - postz               # always runs (skip can't avoid)
    print(f"\n{'stage':<34}{'sec':>8}{'% of full':>10}")
    for name, v in [("FULL fold (ESMC+trunk+diff5+conf)", t_full),
                    ("  ESMC + trunk recycling (always)", esmc_trunk),
                    ("  readout/coda+distogram", t_prep),
                    ("  diffusion sample x5  <<<", t_s5),
                    ("  diffusion sample x1 (ref)", t_s1),
                    ("  confidence head", t_conf),
                    ("post-trunk decode (skip saves)", postz)]:
        print(f"{name:<34}{v:>8.3f}{100*v/t_full:>9.1f}%")
    print(f"\nper-sample diffusion ≈ {(t_s5 - t_s1)/4:.3f}s   fixed diffusion overhead ≈ {t_s1 - (t_s5-t_s1)/4:.3f}s")
    print(f"\n>>> the 5 diffusion samples = {100*t_s5/t_full:.0f}% of a full fold")
    print(f">>> skip-filter avoids {100*postz/t_full:.0f}% of per-seed compute (trunk still runs)")


if __name__ == "__main__":
    main()
