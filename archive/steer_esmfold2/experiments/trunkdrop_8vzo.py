"""trunkdrop_8vzo.py — does dropping the trunk pair rep from the confidence head (beta=0, a regime it was
trained with 20% of the time via Bernoulli(0.8) on the trunk term) make ipTM recognize 8vzo's Type-2
good-fluke docks?

The confidence head anchors its confidence to trunk-space (z_base = z_norm(z)); the 8vzo good pose is a
1/5 diffusion fluke the trunk never committed to, so the head scores it through the decoy-committed trunk.
Force beta=0 -> the head must judge from Embed(d_pred) (the decoded structure's own distogram) + inputs.

Folds the 7 known good-fluke 8vzo seeds (+ decoy controls) at N diffusion samples; for EACH decoded
structure computes pair_chains_iptm BOTH ways in one shot: trunk-on (normal) and trunk-off (z_norm zeroed).
Saves the structure + both matrices per sample. Offline: for good-DockQ samples, does trunk-off ipTM
exceed trunk-on and separate good from decoy?
"""
import json, sys
from pathlib import Path
import torch
from esm.models.esmfold2 import ESMFold2InputBuilder
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
import transformers.models.esmfold2.modeling_esmfold2 as M

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists(): _p = _p.parent
sys.path.insert(0, str(_p)); sys.path.insert(0, str(_p / "run" / "esm"))
from paths import ESM_PREDICTIONS, JSON_DIR          # noqa
from run_foldbench_abag import load_complexes         # noqa

# ---- monkey-patch: compute pair_chains_iptm trunk-on AND trunk-off in one call ----
# trunk-off (beta=0) = zero z_base's trunk contribution. z_base = z_norm(z) is the SOLE place the trunk
# enters the head (it only receives s_inputs [pre-trunk] and z [trunk pair]), so zeroing z_norm's output
# is a complete, faithful trunk drop. We do it by temporarily zeroing z_norm's affine (weight,bias) ->
# LayerNorm output is exactly 0 -> no submodule reassignment, no library edit.
_orig = M.ConfidenceHead.forward
_STASH = {}
def _patched(self, s_inputs, z, *a, **kw):
    out = _orig(self, s_inputs, z, *a, **kw)                 # trunk-on (normal)
    zln = self.z_norm
    w = zln.weight.data.clone(); b = zln.bias.data.clone()
    zln.weight.data.zero_(); zln.bias.data.zero_()
    try:
        nt = _orig(self, s_inputs, z, *a, **kw)             # trunk-off (beta=0)
    finally:
        zln.weight.data.copy_(w); zln.bias.data.copy_(b)
    _STASH['pci'] = nt['pair_chains_iptm'].detach().float().cpu()
    _STASH['iptm'] = nt['iptm'].detach().float().cpu()
    del nt
    return out
M.ConfidenceHead.forward = _patched

CX = "8vzo"; N = 8
GOOD = [166, 809, 201, 246, 981, 95, 946]
DECOY = [0, 1, 2]

def main():
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
    model.set_chunk_size(16); model.set_kernel_backend("fused")
    builder = ESMFold2InputBuilder()
    spi = dict(load_complexes(str(JSON_DIR)))[CX]
    outdir = ESM_PREDICTIONS / "trunkdrop_8vzo"
    for tag, seeds in [("good", GOOD), ("decoy", DECOY)]:
        for s in seeds:
            pred = outdir / CX / f"seed_{s}" / "predictions"
            pred.mkdir(parents=True, exist_ok=True)
            _STASH.clear()
            try:
                res = builder.fold(model, spi, seed=s, num_loops=16, num_sampling_steps=16,
                                   num_diffusion_samples=N, lm_mask_pct=0.1, lm_dropout=0.3)
            except Exception as e:
                print(f"{tag} seed {s} ERROR: {e}", flush=True); torch.cuda.empty_cache(); continue
            samples = res if isinstance(res, list) else [res]
            ntp = _STASH.get('pci')
            if ntp is not None:
                print(f"  [check] notrunk pci shape {tuple(ntp.shape)} vs nsamples {len(samples)}", flush=True)
            for k, sm in enumerate(samples):
                (pred / f"{CX}_sample_{k}.cif").write_text(sm.complex.to_mmcif())
                rec = {"iptm": sm.iptm,
                       "pair_chains_iptm": sm.pair_chains_iptm.tolist() if sm.pair_chains_iptm is not None else None,
                       "pair_chains_iptm_notrunk": ntp[k].tolist() if (ntp is not None and k < len(ntp)) else None}
                (pred / f"{CX}_summary_confidence_sample_{k}.json").write_text(json.dumps(rec))
            print(f"{tag} seed {s} done ({len(samples)} samples)", flush=True)
            torch.cuda.empty_cache()
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
