"""Re-decode the SAVED steered s_lm (good_slm.pt, bad_slm.pt) at 16 & 64 loops, writing proper
summary_confidence JSONs with chain_pair_iptm so OST gives antibody-antigen PAIR-ipTM + DockQ.
Also captures trunk-free chain_pair_iptm. This is the decisive readout: once each geometry is FORCED,
does the good-steered structure earn higher A-C confidence than the bad-steered one, or stay tied?"""
import sys, json, os
sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/steer/core")
sys.path.insert(0, "/home/ebrahim/tts_abag/steer/experiments")
import torch, numpy as np
import transformers.models.esmfold2.modeling_esmfold2 as M
_orig = M.ConfidenceHead.forward; _ST = {}
def _patched(self, s_inputs, z, *a, **kw):
    out = _orig(self, s_inputs, z, *a, **kw)
    zln = self.z_norm; w = zln.weight.data.clone(); b = zln.bias.data.clone()
    zln.weight.data.zero_(); zln.bias.data.zero_()
    try: nt = _orig(self, s_inputs, z, *a, **kw)
    finally: zln.weight.data.copy_(w); zln.bias.data.copy_(b)
    _ST['pci_tf'] = nt['pair_chains_iptm'].detach().float().cpu().numpy(); del nt
    return out
M.ConfidenceHead.forward = _patched

import forward_grad as fg
from paths import ESM_PREDICTIONS
FORCE = ESM_PREDICTIONS / "steer_8ts0_force"
OUT = ESM_PREDICTIONS / "steer_8ts0_redecode"

cfg = fg.configure_target("8ts0")
print("loading model...", flush=True)
model, builder, feat, chain_infos, cond = fg.load(chunk_size=128)
pair_mask = cond["pair_mask"]; z_init = fg.compute_z_init(model, cond); z0 = fg.fixed_z0(model, z_init, seed=0)
summary = []
for tag in ["good", "bad"]:
    s_lm = torch.load(FORCE / f"{tag}_slm.pt").cuda()
    lm_z = fg.lift_to_lm_z(model, s_lm)
    for loops in [16, 64]:
        z = fg.final_trunk_z(model, z_init, lm_z, pair_mask, z0, loops + 1)
        _ST.clear()
        res = fg.decode_from_ztrunk(model, builder, feat, chain_infos, cond, z, 2, 16, "8ts0", loops)
        pci_tf = _ST.get('pci_tf')            # [B, nc, nc] trunk-free
        d = OUT / f"{tag}_loops{loops}" / "8ts0" / f"seed_{loops}" / "predictions"
        os.makedirs(d, exist_ok=True)
        for k, sm in enumerate(res):
            open(d / f"8ts0_sample_{k}.cif", "w").write(sm.complex.to_mmcif())
            pci = sm.pair_chains_iptm.detach().float().cpu().numpy()   # trunk-on [nc,nc]
            json.dump({"iptm": float(sm.iptm), "ptm": 0.0, "has_clash": 0,
                       "chain_pair_iptm": pci.tolist()},
                      open(d / f"8ts0_summary_confidence_sample_{k}.json", "w"))
            tf = pci_tf[k] if pci_tf is not None and pci_tf.ndim == 3 else pci_tf
            summary.append(dict(tag=tag, loops=loops, sample=k, global_iptm=float(sm.iptm),
                                pci_AC=float(pci[0][2]), pci_BC=float(pci[1][2]),
                                pci_tf_AC=(float(tf[0][2]) if tf is not None else None),
                                pci_tf_BC=(float(tf[1][2]) if tf is not None else None)))
        print(f"[{tag}] loops={loops} done  A-C on={summary[-1]['pci_AC']:.3f} tf={summary[-1]['pci_tf_AC']}", flush=True)
os.makedirs(OUT, exist_ok=True)
json.dump(summary, open(OUT / "pair_iptm_summary.json", "w"), indent=2)
print("REDECODE_DONE", flush=True)
