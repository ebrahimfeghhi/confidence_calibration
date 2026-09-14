"""trunkoff_validate.py — validate reconstructing x_pred from a saved .cif and scoring trunk-off ipTM,
on 8vzo poses we ALREADY have trunk-off numbers for (the earlier trunkdrop fold-pipeline run).

Two gates:
  (A) distance check: rep-atom distances from the reconstructed x_pred (as the head gathers them) ==
      rep-atom distances read straight from the .cif.  -> reconstruction geometry is right.
  (B) end-to-end: my trunk-off pair_chains_iptm == the trunkdrop run's saved pair_chains_iptm_notrunk
      for the same pose.  -> the whole harness reproduces the model's own trunk-off (no z, no trunk).
If both pass, the reconstruction is trusted and we can score all saved poses.
"""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "run/esm"); sys.path.insert(0, ".")
from complex_io import load_complex, build_conditioning     # noqa
from esm.models.esmfold2 import ESMFold2InputBuilder        # noqa
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model  # noqa
from paths import JSON_DIR                                   # noqa
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

def parse_cif_residues(cif):
    """cif -> list of residues in FILE order: [{atom_name: xyz}], and their (chain,seq) keys."""
    d = MMCIF2Dict(str(cif)); grp = d['_atom_site.group_PDB']
    ch = d['_atom_site.auth_asym_id']; seq = d['_atom_site.label_seq_id']; nm = d['_atom_site.label_atom_id']
    X = d['_atom_site.Cartn_x']; Y = d['_atom_site.Cartn_y']; Z = d['_atom_site.Cartn_z']
    residues, keys, curkey = [], [], None
    for i in range(len(grp)):
        if grp[i] != 'ATOM': continue
        key = (ch[i], seq[i])
        if key != curkey:
            residues.append({}); keys.append(key); curkey = key
        residues[-1][nm[i]] = np.array([float(X[i]), float(Y[i]), float(Z[i])], dtype=np.float32)
    return residues, keys

def decode_atom_names(anc):
    arr = anc.cpu().numpy()
    off = next(o for o in range(20, 80) if ''.join(chr(c + o) for c in arr[0]).strip() == 'N')
    return [''.join(chr(c + off) for c in row).strip() for row in arr], off

def acval(m): return max(m[0][2], m[2][0])   # A-C interface from 3x3 chain_pair_iptm

CX = "8vzo"
print("loading model...", flush=True)
model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
model.set_chunk_size(16)
builder = ESMFold2InputBuilder()
spi = load_complex(str(JSON_DIR), CX)
feat, chain_infos = builder.prepare_input(spi, seed=0, device=model.device)
with torch.no_grad():
    cond = build_conditioning(model, feat)

n_atoms = feat['atom_attention_mask'].shape[-1]
a2t = feat['atom_to_token'][0].cpu().numpy()
di = feat['distogram_atom_idx'][0].cpu().numpy()
names, off = decode_atom_names(feat['ref_atom_name_chars'][0])
L = len(di)
print(f"decode offset={off}; names[:8]={names[:8]}; rep atom (token0) = {names[di[0]]!r}", flush=True)
d_pair = model.confidence_head.z_norm.weight.shape[0]

def reconstruct_xpred(cif):
    residues, keys = parse_cif_residues(cif)
    xp = np.zeros((n_atoms, 3), dtype=np.float32); miss_rep = 0
    for a in range(n_atoms):
        t = a2t[a]
        if t < len(residues) and names[a] in residues[t]:
            xp[a] = residues[t][names[a]]
    for t in range(L):
        if not (t < len(residues) and names[di[t]] in residues[t]): miss_rep += 1
    return xp, residues, len(residues), miss_rep

def trunkoff_pci(xp):
    zln = model.confidence_head.z_norm
    w = zln.weight.data.clone(); b = zln.bias.data.clone()
    z = torch.zeros(1, L, L, d_pair, device=model.device)
    xpt = torch.tensor(xp, device=model.device).unsqueeze(0)      # [1, n_atoms, 3]
    zln.weight.data.zero_(); zln.bias.data.zero_()
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.confidence_head(
                s_inputs=cond["x_inputs"], z=z, x_pred=xpt,
                distogram_atom_idx=feat['distogram_atom_idx'], token_attention_mask=feat['token_attention_mask'],
                atom_to_token=cond["atom_to_token"], atom_attention_mask=feat['atom_attention_mask'],
                asym_id=feat['asym_id'], mol_type=feat['mol_type'], num_diffusion_samples=1,
                relative_position_encoding=cond["rel_pos"], token_bonds_encoding=cond["token_bonds_encoding"])
    finally:
        zln.weight.data.copy_(w); zln.bias.data.copy_(b)
    return out['pair_chains_iptm'][0].float().cpu().numpy(), xpt

for seed, samp in [(166, 0), (166, 1), (809, 0)]:
    base = Path.home()/f"esm/foldbench_predictions/trunkdrop_8vzo/{CX}/seed_{seed}/predictions"
    cif = base/f"{CX}_sample_{samp}.cif"
    if not cif.exists(): print(f"seed {seed} samp {samp}: no cif"); continue
    xp, residues, nres, miss_rep = reconstruct_xpred(cif)
    pci, xpt = trunkoff_pci(xp)
    # GATE A: distance check
    repA = xpt[0, feat['distogram_atom_idx'][0]]                 # gather like the head does
    repB = torch.tensor(np.stack([residues[t].get(names[di[t]], np.zeros(3)) for t in range(L)]), device=model.device)
    dA = torch.cdist(repA[None].float(), repA[None].float())[0]
    dB = torch.cdist(repB[None].float(), repB[None].float())[0]
    max_ab = (dA - dB).abs().max().item()
    # GATE B: match trunkdrop saved trunk-off
    tj = json.load(open(base/f"{CX}_summary_confidence_sample_{samp}.json"))
    saved = tj.get('pair_chains_iptm_notrunk')
    my_ac = acval(pci); saved_ac = acval(saved) if saved else float('nan')
    print(f"seed {seed} samp {samp}: cif_res={nres}(tok={L}) rep_miss={miss_rep} | "
          f"GATE_A max|dA-dB|={max_ab:.4f} | GATE_B my_toff_AC={my_ac:.4f} saved_toff_AC={saved_ac:.4f} "
          f"diff={abs(my_ac-saved_ac):.4f}", flush=True)
print("DONE", flush=True)
