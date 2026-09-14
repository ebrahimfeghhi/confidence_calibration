"""score_saved_trunkoff.py — for the 15 Type-2 confident-decoy interfaces, compute trunk-OFF ipTM on the
saved deep-pool structures (reconstruct x_pred from the .cif, run the confidence head with z_base=0), and
pair with the saved trunk-ON ipTM + OST DockQ. Output per-pose CSV for H1 (lift good vs decoy) and H2
(selection by trunk-on vs trunk-off).

Validated by trunkoff_validate.py (rep-distances match the cif exactly; trunk-off reproduces the trunkdrop
fold-pipeline values to ~1e-3). No trunk, no LM, no decode, no re-fold.
"""
import sys, json, csv
from pathlib import Path
from collections import defaultdict
import numpy as np, torch
sys.path.insert(0, "run/esm"); sys.path.insert(0, ".")
from complex_io import load_complex, build_conditioning     # noqa
from esm.models.esmfold2 import ESMFold2InputBuilder        # noqa
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model  # noqa
from paths import JSON_DIR                                   # noqa
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

DEEP = Path.home()/"esm/foldbench_predictions/esmfold2fast_tts"
OUT = sys.argv[1] if len(sys.argv) > 1 else "results/csv_files/trunkoff_type2.csv"
CXS = set(sys.argv[2:]) if len(sys.argv) > 2 else None      # subset of complexes for this shard
# 15 Type-2 interfaces (9g2g dropped: ipTM selects an acceptable 0.32 there)
T2 = [('8ux6','A','F'),('7ua2','A','C'),('8vzo','A','C'),('8slb','A','C'),('8av2','A','C'),
      ('8w84','B','D'),('8oni','A','F'),('8oni','B','F'),('8oni','A','E'),('8uky','A','C'),
      ('8dcm','B','C'),('7zoz','A','B'),('8ivx','A','C'),('8q3j','A','B'),('8q3j','A','C')]
CIDX = lambda c: ord(c) - ord('A')      # chain letter -> asym_id index (model orders A,B,C,...)
def acval(m, i, j): return max(m[i][j], m[j][i])

def parse_cif_residues(cif):
    d = MMCIF2Dict(str(cif)); grp = d['_atom_site.group_PDB']
    ch = d['_atom_site.auth_asym_id']; seq = d['_atom_site.label_seq_id']; nm = d['_atom_site.label_atom_id']
    X = d['_atom_site.Cartn_x']; Y = d['_atom_site.Cartn_y']; Z = d['_atom_site.Cartn_z']
    residues, curkey = [], None
    for i in range(len(grp)):
        if grp[i] != 'ATOM': continue
        key = (ch[i], seq[i])
        if key != curkey:
            residues.append({}); curkey = key
        residues[-1][nm[i]] = (float(X[i]), float(Y[i]), float(Z[i]))
    return residues

# ---- deep CSV: per interface -> per (seed,sample): (dockq, saved trunk-on pair_iptm) ----
T2SET = {tuple(t) for t in T2}
POSES = defaultdict(dict)     # (cx,c1,c2) -> {(seed,sample): (dockq, pair_iptm)}
for r in csv.DictReader(open("results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv")):
    cx = r['assembly_id'][:4]; key = (cx, r['chain1'], r['chain2'])
    if key not in T2SET: continue
    try: POSES[key][(int(r['seed']), int(r['sample']))] = (float(r['dockq']), float(r['pair_iptm']))
    except: pass

def pick_poses(key):
    """all good (dq>0.4) + top-200 by saved trunk-on ipTM + ~100 spread decoys."""
    items = list(POSES[key].items())          # ((seed,sample), (dockq, pv))
    good = [sk for sk, (dq, pv) in items if dq > 0.4]
    topon = [sk for sk, _ in sorted(items, key=lambda x: -x[1][1])[:200]]
    decoys = [sk for sk, (dq, pv) in items if dq < 0.1]
    rng = decoys[::max(1, len(decoys)//100)][:100]
    return sorted(set(good) | set(topon) | set(rng))

# group poses by complex (so we featurize each complex once)
by_cx = defaultdict(set)
for t in T2:
    for sk in pick_poses(tuple(t)): by_cx[t[0]].add(sk)

print("loading model...", flush=True)
model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
model.set_chunk_size(16)
builder = ESMFold2InputBuilder()

def decode_names(anc):
    arr = anc.cpu().numpy()
    off = next(o for o in range(20, 80) if ''.join(chr(c + o) for c in arr[0]).strip() == 'N')
    return [''.join(chr(c + o) for c in row).strip() for o in [off] for row in arr]

fout = open(OUT, "w", newline=""); W = csv.writer(fout)
W.writerow(["cx", "c1", "c2", "seed", "sample", "dockq", "iptm_on", "iptm_off"]); fout.flush()

for cx in sorted(by_cx):
    if CXS and cx not in CXS: continue
    spi = load_complex(str(JSON_DIR), cx)
    feat, _ = builder.prepare_input(spi, seed=0, device=model.device)
    with torch.no_grad(): cond = build_conditioning(model, feat)
    n_atoms = feat['atom_attention_mask'].shape[-1]
    a2t = feat['atom_to_token'][0].cpu().numpy(); L = feat['token_attention_mask'].shape[-1]
    names = decode_names(feat['ref_atom_name_chars'][0])
    d_pair = model.confidence_head.z_norm.weight.shape[0]
    zln = model.confidence_head.z_norm
    ifaces = [(t[1], t[2]) for t in T2 if t[0] == cx]
    poses = sorted(by_cx[cx]); B = 8 if L < 1000 else (6 if L < 1150 else 4)
    print(f"{cx}: L={L}, {len(poses)} poses, batch={B}", flush=True)
    off_pci = {}                              # (seed,sample) -> trunk-off pci matrix
    for b0 in range(0, len(poses), B):
        batch = poses[b0:b0 + B]; xps = []; ok = []
        for s, k in batch:
            cif = DEEP/cx/f"seed_{s}"/"predictions"/f"{cx}_sample_{k}.cif"
            if not cif.exists(): continue
            res = parse_cif_residues(cif)
            xp = np.zeros((n_atoms, 3), dtype=np.float32)
            for a in range(n_atoms):
                t = a2t[a]
                if t < len(res) and names[a] in res[t]: xp[a] = res[t][names[a]]
            xps.append(xp); ok.append((s, k))
        if not xps: continue
        xpt = torch.tensor(np.stack(xps), device=model.device).unsqueeze(0)   # [1,B,n_atoms,3]
        z = torch.zeros(1, L, L, d_pair, device=model.device)
        w = zln.weight.data.clone(); bb = zln.bias.data.clone()
        zln.weight.data.zero_(); zln.bias.data.zero_()
        try:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.confidence_head(
                    s_inputs=cond["x_inputs"], z=z, x_pred=xpt,
                    distogram_atom_idx=feat['distogram_atom_idx'], token_attention_mask=feat['token_attention_mask'],
                    atom_to_token=cond["atom_to_token"], atom_attention_mask=feat['atom_attention_mask'],
                    asym_id=feat['asym_id'], mol_type=feat['mol_type'], num_diffusion_samples=len(ok),
                    relative_position_encoding=cond["rel_pos"], token_bonds_encoding=cond["token_bonds_encoding"])
        finally:
            zln.weight.data.copy_(w); zln.bias.data.copy_(bb)
        pci = out['pair_chains_iptm'].float().cpu().numpy()
        for n, sk in enumerate(ok): off_pci[sk] = pci[n]
        torch.cuda.empty_cache()
    # write rows per interface (trunk-on from saved json, trunk-off from off_pci)
    for c1, c2 in ifaces:
        i, j = CIDX(c1), CIDX(c2); key = (cx, c1, c2)
        for (s, k), (dq, pv) in POSES[key].items():
            if (s, k) not in off_pci: continue
            jp = DEEP/cx/f"seed_{s}"/"predictions"/f"{cx}_summary_confidence_sample_{k}.json"
            ion = None
            if jp.exists():
                try: ion = acval(json.load(open(jp))['chain_pair_iptm'], i, j)
                except: pass
            ioff = acval(off_pci[(s, k)].tolist(), i, j)
            W.writerow([cx, c1, c2, s, k, f"{dq:.4f}", "" if ion is None else f"{ion:.4f}", f"{ioff:.4f}"])
    fout.flush()
    print(f"  {cx} done, {len(off_pci)} poses scored", flush=True)
fout.close()
print("DONE", flush=True)
