"""score_trunkoff_fast.py — optimized full-pool trunk-off ipTM scorer for the 15 Type-2 interfaces.

Speedups vs score_saved_trunkoff.py:
  (1) fast custom cif parser (no MMCIF2Dict) — ~15x faster
  (2) rep-atoms-only x_pred: fill ONLY the distogram rep slots (CB/CA per token) and leave the rest zero.
      pair_chains_iptm depends on x_pred ONLY through the rep-atom distogram, so this is exact for ipTM.
  (3) parallel pre-parse (Pool) so the GPU never stalls on serial parsing.

Usage:
  validate:  python score_trunkoff_fast.py --validate
  score:     python score_trunkoff_fast.py OUT.csv 8vzo 8slb ...
"""
import sys, json, csv
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np, torch
sys.path.insert(0, "run/esm"); sys.path.insert(0, ".")
from complex_io import load_complex, build_conditioning     # noqa
from esm.models.esmfold2 import ESMFold2InputBuilder        # noqa
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model  # noqa
from paths import JSON_DIR                                   # noqa

DEEP = Path.home()/"esm/foldbench_predictions/esmfold2fast_tts"
T2 = [('8ux6','A','F'),('7ua2','A','C'),('8vzo','A','C'),('8slb','A','C'),('8av2','A','C'),
      ('8w84','B','D'),('8oni','A','F'),('8oni','B','F'),('8oni','A','E'),('8uky','A','C'),
      ('8dcm','B','C'),('7zoz','A','B'),('8ivx','A','C'),('8q3j','A','B'),('8q3j','A','C')]
CIDX = lambda c: ord(c) - ord('A')
def acval(m, i, j): return max(m[i][j], m[j][i])

def flat_reps(cif):
    """token-ordered rep atoms (CB, CA fallback) as [n_tok,3] — fast, no MMCIF2Dict."""
    with open(cif) as f: lines = f.readlines()
    cols = {}; i = 0
    for i, ln in enumerate(lines):
        if ln.startswith('_atom_site.'): cols[ln.strip()] = len(cols)
        elif ln.startswith('ATOM'): break
    C = lambda n: cols['_atom_site.'+n]
    ich, iseq, inm = C('auth_asym_id'), C('label_seq_id'), C('label_atom_id')
    ix, iy, iz = C('Cartn_x'), C('Cartn_y'), C('Cartn_z')
    reps = []; cur = None; atoms = None
    for ln in lines[i:]:
        if not ln.startswith('ATOM'): continue
        p = ln.split(); key = (p[ich], p[iseq])
        if key != cur:
            if atoms is not None: reps.append(atoms.get('CB', atoms.get('CA')))
            cur = key; atoms = {}
        atoms[p[inm]] = (float(p[ix]), float(p[iy]), float(p[iz]))
    if atoms is not None: reps.append(atoms.get('CB', atoms.get('CA')))
    return np.array([r for r in reps if r is not None], dtype=np.float32)

_CX = None
def parse_worker(sk):
    s, k = sk
    cif = DEEP/_CX/f"seed_{s}"/"predictions"/f"{_CX}_sample_{k}.cif"
    if not cif.exists(): return (sk, None)
    try: return (sk, flat_reps(cif))
    except Exception: return (sk, None)
def _init(cx):
    global _CX; _CX = cx

# deep CSV: per interface -> {(seed,sample): (dockq, trunk-on pair_iptm)}
POSES = defaultdict(dict); T2SET = {tuple(t) for t in T2}
for r in csv.DictReader(open("results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv")):
    key = (r['assembly_id'][:4], r['chain1'], r['chain2'])
    if key not in T2SET: continue
    try: POSES[key][(int(r['seed']), int(r['sample']))] = (float(r['dockq']), float(r['pair_iptm']))
    except: pass

def load_model():
    m = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
    m.set_kernel_backend("fused")   # ~3x faster trimul (fused Triton kernel) — the big miss
    m.set_chunk_size(32)
    return m, ESMFold2InputBuilder()

def trunkoff_batch(model, feat, cond, di, n_atoms, L, d_pair, reps_list):
    """reps_list: list of [n_tok,3] arrays -> pair_chains_iptm [B,nchains,nchains] (trunk-off)."""
    dev = model.device; B = len(reps_list)
    xps = torch.zeros(B, n_atoms, 3, device=dev)
    idx = feat['distogram_atom_idx'][0]
    for bi, reps in enumerate(reps_list):
        xps[bi, idx] = torch.tensor(reps, device=dev)
    z = torch.zeros(1, L, L, d_pair, device=dev)
    zln = model.confidence_head.z_norm
    w = zln.weight.data.clone(); b = zln.bias.data.clone()
    zln.weight.data.zero_(); zln.bias.data.zero_()
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.confidence_head(
                s_inputs=cond["x_inputs"], z=z, x_pred=xps.unsqueeze(0),
                distogram_atom_idx=feat['distogram_atom_idx'], token_attention_mask=feat['token_attention_mask'],
                atom_to_token=cond["atom_to_token"], atom_attention_mask=feat['atom_attention_mask'],
                asym_id=feat['asym_id'], mol_type=feat['mol_type'], num_diffusion_samples=B,
                relative_position_encoding=cond["rel_pos"], token_bonds_encoding=cond["token_bonds_encoding"])
    finally:
        zln.weight.data.copy_(w); zln.bias.data.copy_(b)
    return out['pair_chains_iptm'].float().cpu().numpy()

def prep_complex(model, builder, cx):
    spi = load_complex(str(JSON_DIR), cx)
    feat, _ = builder.prepare_input(spi, seed=0, device=model.device)
    with torch.no_grad(): cond = build_conditioning(model, feat)
    n_atoms = feat['atom_attention_mask'].shape[-1]; L = feat['token_attention_mask'].shape[-1]
    d_pair = model.confidence_head.z_norm.weight.shape[0]
    return feat, cond, feat['distogram_atom_idx'], n_atoms, L, d_pair

def validate(model, builder):
    print("=== validation: rep-only fast path vs trunkdrop 8vzo saved values ===")
    feat, cond, di, n_atoms, L, d_pair = prep_complex(model, builder, "8vzo")
    base = Path.home()/"esm/foldbench_predictions/trunkdrop_8vzo/8vzo"
    for s, k in [(166, 0), (166, 1), (809, 0)]:
        cif = base/f"seed_{s}"/"predictions"/f"8vzo_sample_{k}.cif"
        reps = flat_reps(cif)
        if len(reps) != L: print(f"  seed {s}/{k}: rep count {len(reps)} != L {L} !!"); continue
        pci = trunkoff_batch(model, feat, cond, di, n_atoms, L, d_pair, [reps])[0]
        mine = acval(pci.tolist(), 0, 2)
        saved = json.load(open(base/f"seed_{s}"/"predictions"/f"8vzo_summary_confidence_sample_{k}.json")).get('pair_chains_iptm_notrunk')
        sv = acval(saved, 0, 2) if saved else float('nan')
        print(f"  seed {s}/{k}: fast={mine:.4f}  saved={sv:.4f}  diff={abs(mine-sv):.4f}")

def main():
    if sys.argv[1] == "--validate":
        m, b = load_model(); validate(m, b); return
    OUT = sys.argv[1]; CXS = set(sys.argv[2:]) if len(sys.argv) > 2 else None
    by_cx = defaultdict(set)
    for t in T2:
        if CXS and t[0] not in CXS: continue
        for sk in POSES[tuple(t)].keys(): by_cx[t[0]].add(sk)
    model, builder = load_model()
    fout = open(OUT, "w", newline=""); W = csv.writer(fout)
    W.writerow(["cx", "c1", "c2", "seed", "sample", "dockq", "iptm_on", "iptm_off"]); fout.flush()
    for cx in sorted(by_cx):
        feat, cond, di, n_atoms, L, d_pair = prep_complex(model, builder, cx)
        poses = sorted(by_cx[cx]); B = 8 if L < 1000 else (6 if L < 1150 else 4)
        print(f"{cx}: L={L}, {len(poses)} poses, batch={B} — parsing...", flush=True)
        with Pool(40, initializer=_init, initargs=(cx,)) as p:
            parsed = dict(p.map(parse_worker, poses))
        ok = [(sk, parsed[sk]) for sk in poses if parsed.get(sk) is not None and len(parsed[sk]) == L]
        print(f"  parsed {len(ok)}/{len(poses)} (rep-count==L); scoring...", flush=True)
        off = {}
        for b0 in range(0, len(ok), B):
            chunk = ok[b0:b0+B]
            pci = trunkoff_batch(model, feat, cond, di, n_atoms, L, d_pair, [r for _, r in chunk])
            for n, (sk, _) in enumerate(chunk): off[sk] = pci[n]
            torch.cuda.empty_cache()
        for c1, c2 in [(t[1], t[2]) for t in T2 if t[0] == cx]:
            i, j = CIDX(c1), CIDX(c2); key = (cx, c1, c2)
            for (s, k), (dq, pv) in POSES[key].items():
                if (s, k) not in off: continue
                W.writerow([cx, c1, c2, s, k, f"{dq:.4f}", f"{pv:.4f}", f"{acval(off[(s,k)].tolist(), i, j):.4f}"])
        fout.flush(); print(f"  {cx} done ({len(off)} scored)", flush=True)
    fout.close(); print("DONE", flush=True)

if __name__ == "__main__":
    main()
