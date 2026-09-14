"""Setting 1: 8ts0 baseline (full, no-mask) s_lm + RANDOM z0 -> run trunk 16 & 64 loops -> decode.
Q: does full s_lm + random z0 converge to the GOOD basin (high DockQ)? And what ipTM (trunk-on/off)?"""
import sys, json, os
sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/steer/core")
sys.path.insert(0, "/home/ebrahim/tts_abag/steer/experiments")
import torch, numpy as np
import transformers.models.esmfold2.modeling_esmfold2 as M
# ---- trunk-free ipTM monkeypatch (zero z_norm affine -> faithful trunk drop) ----
_orig = M.ConfidenceHead.forward; _ST = {}
def _patched(self, s_inputs, z, *a, **kw):
    out = _orig(self, s_inputs, z, *a, **kw)
    zln = self.z_norm; w = zln.weight.data.clone(); b = zln.bias.data.clone()
    zln.weight.data.zero_(); zln.bias.data.zero_()
    try: nt = _orig(self, s_inputs, z, *a, **kw)
    finally: zln.weight.data.copy_(w); zln.bias.data.copy_(b)
    _ST['pci_tf'] = nt['pair_chains_iptm'].detach().float().cpu()
    _ST['iptm_tf'] = nt['iptm'].detach().float().cpu()
    del nt
    return out
M.ConfidenceHead.forward = _patched

import forward_grad as fg
cfg = fg.configure_target("8ts0")
print("loading model...", flush=True)
model, builder, feat, chain_infos, cond = fg.load(chunk_size=16)
z_init = fg.compute_z_init(model, cond)
pair_mask = cond["pair_mask"]
hidden = fg.load_hidden(cfg["baseline"])
s_lm = fg.single_rep_from_hidden(model, hidden)         # FULL baseline s_lm
lm_z = fg.lift_to_lm_z(model, s_lm)
print("s_lm", tuple(s_lm.shape), "lm_z", tuple(lm_z.shape), flush=True)

OUT = "/home/ebrahim/esm/foldbench_predictions/steer_8ts0_s1"
recs = []
for loops in [16, 64]:
    steps = loops + 1
    for zs in [0, 1, 2, 3, 4]:
        z0 = fg.fixed_z0(model, z_init, seed=zs)
        z = fg.final_trunk_z(model, z_init, lm_z, pair_mask, z0, steps)
        _ST.clear()
        res = fg.decode_from_ztrunk(model, builder, feat, chain_infos, cond, z, 2, 16, "8ts0", zs)
        for k, sm in enumerate(res):
            d = f"{OUT}/loops{loops}_z{zs}/predictions"; os.makedirs(d, exist_ok=True)
            open(f"{d}/8ts0_sample_{k}.cif", "w").write(sm.complex.to_mmcif())
        pci = res[0].pair_chains_iptm
        rec = {"loops": loops, "z0": zs,
               "iptm": float(res[0].iptm) if res[0].iptm is not None else None,
               "pci": pci.tolist() if pci is not None else None,
               "pci_tf": _ST.get('pci_tf').tolist() if _ST.get('pci_tf') is not None else None}
        recs.append(rec)
        print(f"loops={loops} z0={zs} iptm={rec['iptm']}", flush=True)
        torch.cuda.empty_cache()
json.dump(recs, open(f"{OUT}/recs.json", "w"))
print("STEER_S1_DONE", flush=True)
