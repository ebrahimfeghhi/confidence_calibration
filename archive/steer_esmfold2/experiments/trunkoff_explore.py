"""trunkoff_explore.py — probe the model's atom layout vs a saved .cif, to design the x_pred
reconstruction for scoring saved structures with the confidence head (trunk on/off).

We need to feed the confidence head an x_pred atom tensor in the MODEL's layout. This prints the
feat shapes (n_atoms, distogram_atom_idx = rep atom slot per token, atom_to_token) and the saved cif's
atom count/order, so we can decide: does the cif parse directly into x_pred, or do we map per-token rep atoms?
"""
import sys
from pathlib import Path
import torch
sys.path.insert(0, "run/esm")
from complex_io import load_complex               # noqa
from esm.models.esmfold2 import ESMFold2InputBuilder   # noqa
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model  # noqa
sys.path.insert(0, ".")
from paths import JSON_DIR                          # noqa
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

CX = "8vzo"
print("loading model...", flush=True)
model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
builder = ESMFold2InputBuilder()
spi = load_complex(str(JSON_DIR), CX)
feat, chain_infos = builder.prepare_input(spi, seed=0, device=model.device)
print("\n=== feat keys ===", flush=True)
print(sorted(feat.keys()))
for k in ['atom_attention_mask','distogram_atom_idx','atom_to_token','ref_pos','ref_atom_name_chars',
          'token_attention_mask','res_type','asym_id','mol_type']:
    if k in feat and hasattr(feat[k],'shape'):
        print(f"  {k}: {tuple(feat[k].shape)} {feat[k].dtype}")
n_atoms = feat['atom_attention_mask'].shape[-1]
n_tok = feat['token_attention_mask'].shape[-1]
print(f"\nn_atoms={n_atoms}  n_tokens={n_tok}")
di = feat['distogram_atom_idx'].flatten()
print(f"distogram_atom_idx: first 8 = {di[:8].tolist()}, min={di.min().item()}, max={di.max().item()}")
a2t = feat['atom_to_token'].flatten()
print(f"atom_to_token: first 24 = {a2t[:24].tolist()}")
# chain_infos: what does it hold (chain -> token range)?
print(f"\nchain_infos type={type(chain_infos)}")
try: print("chain_infos:", chain_infos)
except Exception: pass

# saved cif
cif = Path.home()/f"esm/foldbench_predictions/trunkdrop_8vzo/{CX}/seed_166/predictions/{CX}_sample_0.cif"
d = MMCIF2Dict(str(cif))
grp = d['_atom_site.group_PDB']
n_cif = sum(1 for g in grp if g == 'ATOM')
print(f"\n=== saved cif ({cif.name}) ===")
print(f"cif ATOM records: {n_cif}   (model n_atoms={n_atoms})")
print(f"first 12 atoms (chain,res_id,atom_name): "
      + ", ".join(f"{d['_atom_site.auth_asym_id'][i]}{d['_atom_site.label_seq_id'][i]}:{d['_atom_site.label_atom_id'][i]}"
                  for i in range(12)))
# is the rep atom CA? check atom name at the first token's rep slot
rep0 = di[0].item()
print(f"\nfirst token rep slot={rep0}; atom_to_token[rep0]={a2t[rep0].item()} (should be token 0)")
