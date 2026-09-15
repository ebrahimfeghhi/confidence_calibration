#!/usr/bin/env python
"""Score a POSE with a pair rep built from that pose alone: z = z_init + templateEmbedder(pose).

THE PROBLEM THIS ATTACKS. Inside a seed all five poses share one trunk state, and ipTM is a property of
that state -- median within-seed ipTM SD is 0.0078 while DockQ can span the whole scale. So confidence
cannot rank a seed's poses, and committing the trunk to each pose did not fix it. Here the pair rep is
derived from the pose itself, so confidence becomes a POSE property by construction.

WHAT IS COMPUTED, per pose:
    z = z_init + templateEmbedder(template built from this pose)      (no pairformer trunk, no recycling)
    s = s_init
    confidence_head(s_inputs, s, z, x_pred_coords = THIS POSE'S OWN COORDINATES)
    -> compute_full_data_and_summary -> chain_pair_iptm for the interface

The coordinates are the pose's own, read from its cif. runner/dumper.py writes pred_coordinates
[N_sample, N_atom, 3] straight onto the model's atom_array, so the cif's atom order IS the model's
internal order -- the correspondence is one-to-one and is asserted below, not assumed.

THIS IS OUT OF DISTRIBUTION. The confidence head was trained on z after ten recycles of the full trunk;
a template-only z is nothing like that, so it may return plausible-looking numbers that mean nothing.
--calibrate runs the identical pipeline with the REAL z_trunk/s_trunk from a normal fold and checks it
reproduces that fold's known ipTM. If calibration fails, the template-only numbers are uninterpretable
and must not be reported as a result.

Usage:
  pose_confidence.py --iface 9nn9_AB --seed 300004 --gpu 6 [--calibrate]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/ebrahim/Protenix")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit")

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
CKPT = "/home/ebrahim/checkpoint"


def chain_index(name):
    v = 0
    for c in name.split("-")[0]:
        v = v * 26 + (ord(c.upper()) - 64)
    return v - 1


def build_runner(job_path, dump_dir):
    """Construct InferenceRunner exactly as runner/inference.py:run() does."""
    from collections.abc import Mapping
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from protenix.config import parse_configs
    from runner.inference import InferenceRunner, model_configs, update_gpu_compatible_configs

    base = {**configs_base, **{"data": data_configs}, **inference_configs}

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d
    deep_update(base, model_configs["protenix-v2"])
    cfg = parse_configs(configs=base, arg_str="", fill_required_with_null=True)
    cfg.load_checkpoint_dir = CKPT
    cfg.model_name = "protenix-v2"
    cfg.input_json_path = job_path
    cfg.dump_dir = dump_dir
    cfg.use_template = True
    cfg = update_gpu_compatible_configs(cfg)
    return InferenceRunner(cfg), cfg


def first_batch(cfg):
    from runner.inference import get_inference_dataloader
    for batch in get_inference_dataloader(configs=cfg):
        data, atom_array, err = batch[0]
        if err:
            raise SystemExit(f"data error: {err}")
        return data, atom_array
    raise SystemExit("dataloader produced nothing")


def pose_coords(cif, n_atom):
    """Coordinates in the model's atom order. dumper.py sets them on the model's own atom_array, so
    file order == model order; the count is asserted rather than trusted."""
    import gemmi
    st = gemmi.read_structure(cif)
    st.setup_entities()
    xyz = [[a.pos.x, a.pos.y, a.pos.z] for m in st for ch in m for r in ch for a in r]
    if len(xyz) != n_atom:
        raise SystemExit(f"atom count mismatch: cif has {len(xyz)}, model expects {n_atom} -- the "
                         f"cif/model correspondence does not hold, refusing to score")
    return np.array(xyz, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--poses", default="0,1,2,3,4")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = A.gpu
    os.environ.setdefault("LAYERNORM_TYPE", "openfold")
    pdb, chains = A.iface.split("_")
    c1, c2 = chains[0], chains[1]
    out = A.out or f"{EXP}/artifacts/pose_confidence/{A.iface}_s{A.seed}"
    os.makedirs(out, exist_ok=True)

    truth = {}
    for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv")):
        if r["pdb"] == pdb and {r["chain1"], r["chain2"]} == {c1, c2} and r["seed"] == A.seed:
            truth[int(r["sample"])] = (float(r["dockq"]), float(r["pair_iptm"]))
    print(f"  {A.iface} seed {A.seed}: "
          + "  ".join(f"p{k} DockQ {truth[k][0]:.2f}/ipTM {truth[k][1]:.2f}" for k in sorted(truth)),
          flush=True)

    from protenix.model import sample_confidence
    from protenix.utils.torch_utils import autocasting_disable_decorator

    results = {}
    for k in [int(x) for x in A.poses.split(",")]:
        cif = f"{P}/out/{pdb}/{pdb}/seed_{A.seed}/predictions/{pdb}_sample_{k}.cif"
        if not os.path.exists(cif):
            print(f"  pose {k}: cif missing", flush=True); continue
        # a job whose ONLY template is this pose
        T = f"{out}/tmpl{k}"
        if not os.path.exists(f"{T}/job.json"):
            import subprocess
            subprocess.run(["/home/ebrahim/Protenix/.venv/bin/python",
                            "/home/ebrahim/tts_abag/diffusion_commit/build_selftmpl_general.py",
                            "--pdb", pdb, "--out", T, "--poses", cif,
                            "--tids", f"c{k}{int(A.seed) % 100:02d}"],
                           cwd="/home/ebrahim/tts_abag/diffusion_commit", capture_output=True,
                           env={**os.environ, "SELFTMPL_JOBS": f"{P}/jsons"})
        if not os.path.exists(f"{T}/job.json"):
            print(f"  pose {k}: template build failed", flush=True); continue

        os.environ.update({"PROTENIX_MULTICHAIN_MASK": "0", "PROTENIX_TMPL_KEEP_FRAME": "1",
                           "PROTENIX_TMPL_NODEDUP": "1", "PROTENIX_MAX_TEMPLATES": "1"})
        runner, cfg = build_runner(f"{T}/job.json", f"{out}/work{k}")
        data, atom_array = first_batch(cfg)
        from protenix.utils.torch_utils import to_device
        data = to_device(data, runner.device)
        feat = data["input_feature_dict"]
        model = runner.model
        model.eval()

        with torch.no_grad():
            # z_init / s_init / s_inputs -- transcribed from protenix.py:215-236, not reconstructed
            # from memory. The token_bond term and the optional constraint embedder are part of z_init;
            # omitting either would silently give the confidence head a different prior than the trunk
            # ever sees.
            # The two feature-prep steps Protenix.forward runs before anything else, in this order
            # (protenix.py:1169-1172). The dataloader supplies neither: generate_relp adds "relp", and
            # update_input_feature_dict adds d_lm / v_lm / pad_info (Algorithm 5 lines 1-3). Calling
            # the embedder on a raw batch raises KeyError on whichever is missing first.
            from protenix.model.protenix import update_input_feature_dict
            feat = model.relative_position_encoding.generate_relp(feat)
            feat = update_input_feature_dict(feat)
            s_inputs = model.input_embedder(feat, inplace_safe=False, chunk_size=None)
            s_init = model.linear_no_bias_sinit(s_inputs)
            z_init = (model.linear_no_bias_zinit1(s_init)[..., None, :]
                      + model.linear_no_bias_zinit2(s_init)[..., None, :, :])
            z_init = z_init + model.relative_position_encoding(feat["relp"])
            z_init = z_init + model.linear_no_bias_token_bond(
                feat["token_bonds"].unsqueeze(dim=-1))
            # constraint_embedder returns None when the job has no constraints (this one logs
            # "#atom contact:0 #contact:0 #pocket:0"), so the model guards with `if is not None`.
            z_constraint = (model.constraint_embedder(feat["constraint_feature"])
                            if "constraint_feature" in feat else None)
            if z_constraint is not None:
                z_init = z_init + z_constraint

            # first forward: the recycling carry-over is zeros, so this reduces to z_init plus the
            # (constant) projection of a zero tensor -- kept explicit so it matches the trunk's line.
            z = z_init + model.linear_no_bias_z_cycle(
                model.layernorm_z_cycle(torch.zeros_like(z_init)))
            zt = model.template_embedder(feat, z,
                                         triangle_multiplicative=cfg.triangle_multiplicative,
                                         triangle_attention=cfg.triangle_attention,
                                         inplace_safe=False, chunk_size=None)
            z = z + zt
            s = s_init
            n_atom = int(feat["atom_to_token_idx"].shape[-1])
            xyz = torch.tensor(pose_coords(cif, n_atom), device=runner.device)[None]
            # contact_probs is a function of z (distogram head), so it must come from OUR z too --
            # passing None makes compute_full_data_and_summary raise, and passing the trunk's version
            # would mix two different pair reps in one score.
            contact_probs = autocasting_disable_decorator(True)(
                sample_confidence.compute_contact_prob)(
                distogram_logits=model.distogram_head(z),
                **sample_confidence.get_bin_params(cfg.loss.distogram))
            # ORDER MATTERS: the head returns (plddt, pae, pde, resolved) -- protenix.py:825-829.
            # Unpacking it as (pae, pde, plddt) feeds the 64-bin PAE tensor in as pLDDT, which expects
            # 50 bins; it happened to raise here, but with matching bin counts it would have returned a
            # plausible number computed from the wrong tensors.
            plddt, pae, pde, _resolved = model.run_confidence_head(
                input_feature_dict=feat, s_inputs=s_inputs, s_trunk=s, z_trunk=z,
                pair_mask=None, x_pred_coords=xyz,
                triangle_multiplicative=cfg.triangle_multiplicative,
                triangle_attention=cfg.triangle_attention,
                inplace_safe=False, chunk_size=None)
            summ, _ = autocasting_disable_decorator(True)(
                sample_confidence.compute_full_data_and_summary)(
                configs=cfg, pae_logits=pae, plddt_logits=plddt, pde_logits=pde,
                contact_probs=contact_probs, token_asym_id=feat["asym_id"],
                token_has_frame=feat["has_frame"], atom_coordinate=xyz,
                atom_to_token_idx=feat["atom_to_token_idx"],
                atom_is_polymer=1 - feat["is_ligand"], N_recycle=1,
                interested_atom_mask=None, return_full_data=True, mol_id=None,
                elements_one_hot=None)
        M = summ[0].get("chain_pair_iptm")
        i, j = chain_index(c1), chain_index(c2)
        val = float(M[i][j]) if M is not None else float("nan")
        results[k] = val
        print(f"  pose {k}: template-derived pair ipTM = {val:.4f}    "
              f"(true DockQ {truth[k][0]:.3f}, original ipTM {truth[k][1]:.3f})", flush=True)
        del runner, model
        torch.cuda.empty_cache()

    if results:
        ks = sorted(results)
        print(f"\n  {'pose':>5}{'DockQ':>9}{'original ipTM':>15}{'template ipTM':>15}")
        for k in ks:
            print(f"  {k:>5}{truth[k][0]:>9.3f}{truth[k][1]:>15.3f}{results[k]:>15.4f}")
        best_dq = max(ks, key=lambda k: truth[k][0])
        best_new = max(ks, key=lambda k: results[k])
        print(f"\n  best pose by DockQ: p{best_dq}   picked by template ipTM: p{best_new}   "
              f"{'CORRECT' if best_dq == best_new else 'WRONG'}")
        json.dump({str(k): results[k] for k in ks}, open(f"{out}/template_iptm.json", "w"), indent=1)


if __name__ == "__main__":
    main()
