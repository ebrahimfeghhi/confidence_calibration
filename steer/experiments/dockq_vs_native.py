"""dockq_vs_native.py — recompute a model's DockQ against the CRYSTAL STRUCTURE, correctly.

WHY THIS EXISTS. Every DockQ number in this project's ground-truth column came from OST
(`eval/eval_ost_foldbench.py`). We have never independently verified it. And we have already been
burned once by exactly the thing that could be wrong here: the scoring CSV names chains by their
NATIVE letters, the model files use DIFFERENT letters, and the `chain_mapping` column (ref -> model)
translates -- a mapping that on multi-copy assemblies CHANGES FROM POSE TO POSE. Ignoring it made us
score the wrong pair of molecules on 17 of 48 interfaces.

WHAT THIS DOES. For each pose: take the native assembly, use that pose's OWN chain_mapping to pick
the model chains, align native<->model sequences per chain (the native is a gapped subsequence of
the model -- unresolved residues), subset BOTH to the aligned residues so correspondence is exact,
then run the same validated DockQ math as dockq_torch.py.

    DockQ = ( fnat + 1/(1+(iRMS/1.5)^2) + 1/(1+(LRMS/8.5)^2) ) / 3
      fnat  = |contacts(native) & contacts(model)| / |contacts(native)|   (5.0 A, all-atom min dist)
      iRMSD = RMSD after superposing the model's INTERFACE backbone onto the native's
              (interface = native residues within 10.0 A across the chains -- taken from the NATIVE)
      LRMSD = superpose on the RECEPTOR (the chain with MORE native residues) backbone, then RMSD of
              the LIGAND backbone. No second superposition.

This is one-vs-many (native vs every pose), so it is O(N), not O(N^2), and runs fine on CPU.

Usage:
    python steer/experiments/dockq_vs_native.py --model protenix          # all 47 interfaces
    python steer/experiments/dockq_vs_native.py --model protenix --tags 8q3j_AB
"""
import argparse, csv, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import gemmi

sys.path.insert(0, str(Path(__file__).parent))
from dockq_torch import kabsch_rms, kabsch_rt, BACKBONE, FNAT_T2, IFACE_T2

GT_DIR = Path.home() / "FoldBench/examples/ground_truths"
PROT_DIR = Path.home() / "Protenix/foldbench_abag/predictions_template"
ESM_DIR = Path.home() / "esm/foldbench_predictions/esmfold2fast_tts"
OUT = Path("results/dockq_vs_native"); OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- chain selection
# OST (and therefore chain1/chain2 in every scoring CSV) names NATIVE chains by their
# label_asym_id. gemmi and the DockQ package default to auth_asym_id. On 8ts0 the antibody heavy
# chain is auth "H" but label "A", and one auth chain can even hold TWO polymer segments (auth H =
# label A + label M). Selecting by auth name therefore grabs the wrong molecule, or a chain that
# does not exist. Native chains MUST be selected by label_asym_id.
def native_residues(st, label_asym):
    """Residues of the native polymer segment with this label_asym_id (what OST scored)."""
    out = []
    for ch in st[0]:
        for res in ch:
            if res.subchain == label_asym:
                out.append(res)
    return out


def model_residues(st, auth_chain):
    """Predicted structures use simple auth chain names (A, B, C...) with one polymer each."""
    return list(st[0][auth_chain])


def _seq_of(residues):
    return gemmi.one_letter_code([r.name for r in residues])


def align_idx(nat_res, mod_res):
    """Residue-index correspondence between a native polymer segment and a model chain.

    The native is the crystal: it has UNRESOLVED residues (gaps) and often lacks expression tags the
    model predicts. So the native is a gapped subsequence of the model. Align, keep only the matched
    columns, and return the two index lists -- after this, correspondence is EXACT and the DockQ math
    (which assumes it) is valid.
    """
    sn, sm = _seq_of(nat_res), _seq_of(mod_res)
    aln = gemmi.align_string_sequences(list(sn), list(sm), [])
    ni = mi = 0
    nidx, midx = [], []
    for item in aln.cigar_str_to_ops() if hasattr(aln, "cigar_str_to_ops") else _cigar(aln.cigar_str()):
        op, n = item
        if op == "M":
            nidx += list(range(ni, ni + n)); midx += list(range(mi, mi + n)); ni += n; mi += n
        elif op == "I":          # in native only (model missing it) -- drop
            ni += n
        elif op == "D":          # in model only (native unresolved / tag) -- drop
            mi += n
    return nidx, midx, aln.calculate_identity()


def _cigar(s):
    out, num = [], ""
    for c in s:
        if c.isdigit(): num += c
        else: out.append((c, int(num or 1))); num = ""
    return out


# ---------------------------------------------------------------- coordinate extraction
def coords(residues, idx):
    """all-atom coords + per-atom residue index + backbone coords, restricted to residues `idx`."""
    allx, seg, bb = [], [], []
    for k, ri in enumerate(idx):
        res = residues[ri]
        d = {}
        for a in res:
            allx.append([a.pos.x, a.pos.y, a.pos.z]); seg.append(k)
            if a.name in BACKBONE:
                d[a.name] = [a.pos.x, a.pos.y, a.pos.z]
        ca = d.get("CA", [0.0, 0.0, 0.0])
        bb.append([d.get(a, ca) for a in BACKBONE])
    return (np.asarray(allx, np.float32), np.asarray(seg, np.int64),
            np.asarray(bb, np.float32))


def res_sqdist(a1, s1, n1, a2, s2, n2):
    """[n1, n2] squared MIN distance between every residue pair (min over all atom pairs)."""
    d = torch.cdist(torch.from_numpy(a1).unsqueeze(0), torch.from_numpy(a2).unsqueeze(0))[0].pow_(2)
    t = torch.full((n1, d.shape[1]), float("inf"))
    t.scatter_reduce_(0, torch.from_numpy(s1).unsqueeze(1).expand(-1, d.shape[1]), d, reduce="amin")
    u = torch.full((n1, n2), float("inf"))
    u.scatter_reduce_(1, torch.from_numpy(s2).unsqueeze(0).expand(n1, -1), t, reduce="amin")
    return u


# ---------------------------------------------------------------- DockQ, native vs one pose
def dockq_native_vs_pose(nat_parts, mod_parts):
    """nat_parts / mod_parts = ((all,seg,bb) chain1, (all,seg,bb) chain2). Correspondence exact."""
    (na1, ns1, nb1), (na2, ns2, nb2) = nat_parts
    (ma1, ms1, mb1), (ma2, ms2, mb2) = mod_parts
    n1, n2 = nb1.shape[0], nb2.shape[0]

    Dn = res_sqdist(na1, ns1, n1, na2, ns2, n2)      # native residue-residue distances
    Cn = Dn < FNAT_T2
    nat_total = int(Cn.sum())
    if nat_total == 0:
        return None                                   # native has no interface -> DockQ undefined

    Dm = res_sqdist(ma1, ms1, n1, ma2, ms2, n2)
    Cm = Dm < FNAT_T2
    fnat = float((Cm & Cn).sum()) / nat_total

    # interface residues come from the NATIVE
    m = Dn < IFACE_T2
    i1 = m.any(1).numpy(); i2 = m.any(0).numpy()
    ref = np.concatenate([nb1[i1].reshape(-1, 3), nb2[i2].reshape(-1, 3)], 0)
    mod = np.concatenate([mb1[i1].reshape(-1, 3), mb2[i2].reshape(-1, 3)], 0)
    irms = float(kabsch_rms(torch.from_numpy(mod).unsqueeze(0),
                            torch.from_numpy(ref).unsqueeze(0))[0])

    # receptor = the chain with MORE native residues
    if n1 > n2:
        rN, rM, lN, lM = nb1, mb1, nb2, mb2
    else:
        rN, rM, lN, lM = nb2, mb2, nb1, mb1
    R, t = kabsch_rt(torch.from_numpy(rM.reshape(1, -1, 3)),
                     torch.from_numpy(rN.reshape(1, -1, 3)))
    lr = torch.from_numpy(lM.reshape(1, -1, 3)) @ R.transpose(1, 2) + t
    lrms = float((lr - torch.from_numpy(lN.reshape(1, -1, 3))).pow(2).sum(-1).mean(-1).sqrt()[0])

    dq = (fnat + 1 / (1 + (irms / 1.5) ** 2) + 1 / (1 + (lrms / 8.5) ** 2)) / 3
    return dict(dockq=dq, fnat=fnat, irms=irms, lrms=lrms, nat_contacts=nat_total)


# ---------------------------------------------------------------- driver
def rows_for(model, tag):
    """[(pose_path, native_chain1, native_chain2, model_chain1, model_chain2, seed, sample, ost)]"""
    cx, ch = tag.split("_"); c1, c2 = ch[0], ch[1]
    src = ("results/csv_files/Protenix_dockq.csv" if model == "protenix"
           else "results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv")
    out = []
    for r in csv.DictReader(open(src)):
        if (r['assembly_id'][:4], r['chain1'], r['chain2']) != (cx, c1, c2):
            continue
        try:
            cm = json.loads(r['chain_mapping'])
            m1, m2 = cm[c1], cm[c2]                    # ref -> model, PER POSE
            ost = float(r['dockq'])
        except Exception:
            continue                                   # unusable mapping / unscored
        s, k = r['seed'], r['sample']
        if model == "protenix":
            p = PROT_DIR / cx / f"seed_{s}" / "predictions" / f"{cx}_sample_{k}.cif"
        else:
            sub = "baseline_seed" if s == "baseline_seed" else f"seed_{s}"
            p = ESM_DIR / cx / sub / "predictions" / f"{cx}_sample_{k}.cif"
        if p.exists():
            out.append((p, c1, c2, m1, m2, s, k, ost))
    return out


def run_tag(model, tag, verbose=True):
    cx = tag.split("_")[0]
    npath = GT_DIR / f"{cx}-assembly1.cif"
    if not npath.exists():
        print(f"[{tag}] NO NATIVE at {npath}", flush=True); return []
    nat = gemmi.read_structure(str(npath)); nat.setup_entities()

    rows = rows_for(model, tag)
    if not rows:
        print(f"[{tag}] no {model} poses", flush=True); return []

    # cache the alignment per (model_chain1, model_chain2) -- it depends only on the chain pair
    cache = {}
    out = []
    t0 = time.time()
    for p, c1, c2, m1, m2, s, k, ost in rows:
        try:
            mod = gemmi.read_structure(str(p)); mod.setup_entities()
            # c1/c2 are LABEL asym ids on the native; m1/m2 are auth chains on the model
            nr1, nr2 = native_residues(nat, c1), native_residues(nat, c2)
            mr1, mr2 = model_residues(mod, m1), model_residues(mod, m2)
            if not (nr1 and nr2 and mr1 and mr2):
                raise ValueError(f"empty chain: native {c1}/{c2}, model {m1}/{m2}")
            key = (m1, m2)
            if key not in cache:
                n1i, m1i, id1 = align_idx(nr1, mr1)
                n2i, m2i, id2 = align_idx(nr2, mr2)
                cache[key] = (n1i, m1i, n2i, m2i, id1, id2)
            n1i, m1i, n2i, m2i, id1, id2 = cache[key]
            natp = (coords(nr1, n1i), coords(nr2, n2i))
            modp = (coords(mr1, m1i), coords(mr2, m2i))
            r = dockq_native_vs_pose(natp, modp)
            if r is None:
                continue
            r.update(interface=tag, seed=s, sample=k, model_chain1=m1, model_chain2=m2,
                     ost_dockq=ost, seq_identity=min(id1, id2))
            out.append(r)
        except Exception as e:
            if verbose:
                print(f"  [{tag}] seed {s} sample {k}: {type(e).__name__}: {e}", flush=True)
    if out:
        a = np.array([r['dockq'] for r in out]); b = np.array([r['ost_dockq'] for r in out])
        print(f"[{tag}] {len(out)}/{len(rows)} poses | ours max {a.max():.3f} vs OST max {b.max():.3f}"
              f" | mean |diff| {np.abs(a-b).mean():.3f} | {time.time()-t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="protenix", choices=("protenix", "esm"))
    ap.add_argument("--tags", nargs="+")
    a = ap.parse_args()
    torch.set_num_threads(8)
    split = json.load(open("results/csv_files/train_test_split.json"))
    tags = a.tags or sorted(set(split['train']) | set(split['test']))
    allr = []
    for t in tags:
        allr += run_tag(a.model, t)
    if allr:
        f = OUT / f"{a.model}_vs_native.csv"
        with open(f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(allr[0].keys())); w.writeheader(); w.writerows(allr)
        print(f"\n-> {f} ({len(allr)} poses)")
