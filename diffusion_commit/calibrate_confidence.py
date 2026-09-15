#!/usr/bin/env python
"""Does the offline confidence path in pose_confidence.py reproduce Protenix's OWN ipTM?

WHY THIS EXISTS. pose_confidence.py feeds the confidence head a pair rep built only from a template,
which is nothing like the ten-recycle z_trunk the head was trained on. Any number it returns is
worthless unless the PLUMBING is known to be correct -- and three real wiring bugs turned up while
writing it, two of which would have returned a plausible number instead of raising:

  * the head returns (plddt, pae, pde, resolved); unpacking (pae, pde, plddt) feeds PAE in as pLDDT
  * contact_probs must be derived from the SAME z, or the score mixes two different pair reps
  * Protenix.forward runs generate_relp then update_input_feature_dict BEFORE the embedder

THE TEST. Take a normal fold dumped with DUMP_FULL -- z_trunk, s_trunk, s_inputs, coordinate and the
feature dict, saved for exactly this purpose -- push those REAL tensors through the same offline path,
and compare the chain_pair_iptm produced against what Protenix itself wrote in that run's
summary_confidence jsons.

Samples are compared as a SET. runner/dumper.py ranks the written files by ranking_score, while
`coordinate` is in generation order, so matching by index would compare different poses.

A PASS ONLY VALIDATES THE PLUMBING. It says nothing about whether the confidence head behaves sensibly
on a template-only z, which is out of distribution by construction.

Usage: calibrate_confidence.py --dump <dir> --pred <predictions dir> --job <job.json> --gpu 6 --chains AB
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/ebrahim/Protenix")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit")


def chain_index(name):
    v = 0
    for c in name.split("-")[0]:
        v = v * 26 + (ord(c.upper()) - 64)
    return v - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--chains", default="AB")
    A = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = A.gpu
    os.environ.setdefault("LAYERNORM_TYPE", "openfold")

    reps_f = sorted(glob.glob(f"{A.dump}/*_reps.npz"))
    feat_f = sorted(glob.glob(f"{A.dump}/*_feat.npz"))
    if not reps_f or not feat_f:
        raise SystemExit(f"no DUMP_FULL output under {A.dump}")
    reps = np.load(reps_f[0])
    feats = np.load(feat_f[0])
    print(f"  reps {os.path.basename(reps_f[0])}: z_trunk {reps['z_trunk'].shape}  "
          f"s_trunk {reps['s_trunk'].shape}  coordinate {reps['coordinate'].shape}", flush=True)

    # what Protenix itself recorded, as a set
    c1, c2 = A.chains[0], A.chains[1]
    i, j = chain_index(c1), chain_index(c2)
    ref = []
    for f in sorted(glob.glob(f"{A.pred}/*summary_confidence*.json")):
        M = json.load(open(f)).get("chain_pair_iptm")
        if M:
            ref.append(float(M[i][j]))
    if not ref:
        raise SystemExit(f"no chain_pair_iptm in {A.pred}")
    print(f"  Protenix's own pair ipTM ({c1}-{c2}), {len(ref)} samples: "
          + " ".join(f"{x:.4f}" for x in sorted(ref)), flush=True)

    from pose_confidence import build_runner
    from protenix.model import sample_confidence
    from protenix.utils.torch_utils import autocasting_disable_decorator

    runner, cfg = build_runner(A.job, f"{A.dump}/_calibwork")
    model = runner.model
    model.eval()
    dev = runner.device

    feat = {}
    for k in feats.files:
        v = feats[k]
        feat[k] = torch.tensor(v).to(dev) if v.dtype != np.dtype("O") else v
    # the dumped dict is post-prep, but re-running these is idempotent and guards against a dump that
    # predates either step
    if "relp" not in feat:
        feat = model.relative_position_encoding.generate_relp(feat)
    if "d_lm" not in feat:
        from protenix.model.protenix import update_input_feature_dict
        feat = update_input_feature_dict(feat)

    z = torch.tensor(np.asarray(reps["z_trunk"], dtype=np.float32)).to(dev)
    s = torch.tensor(np.asarray(reps["s_trunk"], dtype=np.float32)).to(dev)
    s_inputs = torch.tensor(np.asarray(reps["s_inputs"], dtype=np.float32)).to(dev)
    xyz = torch.tensor(np.asarray(reps["coordinate"], dtype=np.float32)).to(dev)
    print(f"  loaded z {tuple(z.shape)}  s {tuple(s.shape)}  s_inputs {tuple(s_inputs.shape)}  "
          f"coords {tuple(xyz.shape)}", flush=True)

    with torch.no_grad():
        contact_probs = autocasting_disable_decorator(True)(sample_confidence.compute_contact_prob)(
            distogram_logits=model.distogram_head(z),
            **sample_confidence.get_bin_params(cfg.loss.distogram))
        plddt, pae, pde, _ = model.run_confidence_head(
            input_feature_dict=feat, s_inputs=s_inputs, s_trunk=s, z_trunk=z,
            pair_mask=None, x_pred_coords=xyz,
            triangle_multiplicative=cfg.triangle_multiplicative,
            triangle_attention=cfg.triangle_attention, inplace_safe=False, chunk_size=None)
        summ, _ = autocasting_disable_decorator(True)(
            sample_confidence.compute_full_data_and_summary)(
            configs=cfg, pae_logits=pae, plddt_logits=plddt, pde_logits=pde,
            contact_probs=contact_probs, token_asym_id=feat["asym_id"],
            token_has_frame=feat["has_frame"], atom_coordinate=xyz,
            atom_to_token_idx=feat["atom_to_token_idx"],
            atom_is_polymer=1 - feat["is_ligand"], N_recycle=10,
            interested_atom_mask=None, return_full_data=True, mol_id=None, elements_one_hot=None)

    mine = []
    for d in summ:
        M = d.get("chain_pair_iptm")
        if M is not None:
            mine.append(float(M[i][j]))
    print(f"  recomputed offline, {len(mine)} samples: " + " ".join(f"{x:.4f}" for x in sorted(mine)))

    a, b = sorted(ref), sorted(mine)
    if len(a) != len(b):
        print(f"\n  MISMATCH in sample count: {len(a)} vs {len(b)} -- cannot compare")
        return
    diff = [abs(x - y) for x, y in zip(a, b)]
    print(f"\n  max |difference| = {max(diff):.5f}   mean = {sum(diff)/len(diff):.5f}")
    if max(diff) < 0.005:
        print("  CALIBRATION PASSES -- the offline path reproduces Protenix's own ipTM.")
        print("  (This validates the plumbing ONLY. It says nothing about whether the head is")
        print("   meaningful on a template-only z, which is out of distribution by construction.)")
    else:
        print("  CALIBRATION FAILS -- the offline path does NOT reproduce Protenix's own ipTM, so")
        print("  the template-derived numbers from pose_confidence.py are uninterpretable and must")
        print("  not be reported as a result until this is resolved.")


if __name__ == "__main__":
    main()
