#!/usr/bin/env python
"""Fold quality of the TOP-ipTM pose, for decoys vs goods, restricted to domain antigens.

HYPOTHESIS (user, 2026-08-01): where the model folds the paratope/epitope accurately it has a good
structural representation of the binding site and does not invent a confident-but-wrong docking mode.
Where those regions are inaccurate, it mistakes a wrong site for a real one -> high ipTM, low DockQ.
PREDICTION: goods fold better than decoys, and the gap should be LARGER at the paratope/epitope than
whole-chain. If whole-chain separates them just as well, the result is "goods are easier targets".

WHY A NEW COMPUTATION: artifacts/binding_dependence/binding_dependence_results.json already has
global + interface lDDT for the decoys, but as BEST-OF-250. This asks about the pose the model
actually selected (top interface pair_iptm), which is a different question. Goods have no lDDT at all.

DESIGN NOTES
  - Peptide antigens (<30 native residues) are EXCLUDED: they have no independently stable fold, so
    "fold accuracy" is ill-posed and confounded with the docking we are evaluating. Decoys have 4,
    goods 1 -- leaving them in would fake a goods-fold-better result.
  - Antibody/antigen role from the validated kind() classifier (scripts/derive_antigen_idx.py).
  - Epitope/paratope = native residues within 5 A of the partner chain across the interface.
  - lDDT is per-chain vs the native chain, so it is INDEPENDENT of where the pose docked -- a decoy's
    wrong docking does not contaminate its paratope score.
  - The full model is given to OST against a single-chain native, so OST picks the best-matching
    predicted chain/copy itself.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/fold_quality_toppose.py
Writes: artifacts/fold_quality_toppose.json
"""
import json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemmi
from decoy_set import native_chains, GT, OST, ENV, PRED, EXP, SEEDS, SAMPLES

OUTJ = f"{EXP}/artifacts/fold_quality_toppose.json"
WORK = f"{EXP}/artifacts/fold_quality_toppose"
MIN_ANTIGEN_LEN = 30          # below this = peptide antigen, excluded (see DESIGN NOTES)
IFACE_CUTOFF = 5.0
os.makedirs(WORK, exist_ok=True)
TLC = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
       'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
       'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def kind(s):
    """ANTIGEN / LIGHT / HEAVY from sequence. Port of the validated classifier in
    scripts/derive_antigen_idx.py (do not re-derive: it was validated against the known decoy labels)."""
    hasW = re.search(r'WG.G[TS]', s); hasF = re.search(r'FG.G[TS]', s); w = s[:24]
    ntL = re.search(r'([DE]I[QV][ML]TQ|[DE]VVMTQ|EIVLTQ|QSVLTQ|DIQLTQ|SYELTQ|DIVMTQ)', w)
    ntH = re.search(r'([QE][VL][QK]L[VLEQ][EQ]S|QVKLEES|VESGG[GA]|VQSGAE|VETGGG|QESGGG)', w)
    if not (95 <= len(s) <= 275): return 'ANTIGEN'
    light = bool(ntL) or (bool(hasF) and not hasW); heavy = bool(ntH) or bool(hasW)
    if not (light or heavy): return 'ANTIGEN'
    if light and not ntH: return 'LIGHT'
    return 'HEAVY'


def native_residues(pdb, label_chain):
    """Ordered standard-aa residues of one native chain, selected by label_asym_id (gemmi exposes it
    as residue.subchain; residue.name-based chain lookup would use auth ids and silently mismatch)."""
    st = gemmi.read_structure(f"{GT}/{pdb}-assembly1.cif"); st.setup_entities()
    rs = [r for c in st[0] for r in c if r.subchain == label_chain and r.name in TLC]
    if not rs:
        raise KeyError(f"{pdb}: no residues for label chain {label_chain}")
    return rs


def interface_residue_idx(pdb, c1, c2, cutoff=IFACE_CUTOFF):
    """0-based indices (into native_residues(pdb,c1)) of c1 residues contacting c2."""
    import numpy as np
    D = native_chains(pdb)
    XA, IA = D[c1]; XB, _ = D[c2]
    m = np.linalg.norm(XA[:, None, :] - XB[None, :, :], axis=-1) < cutoff
    hit = sorted({IA[i] for i in np.where(m.any(1))[0]})
    order = {v: k for k, v in enumerate(sorted(set(IA), key=lambda x: int(x)))}
    return {order[h] for h in hit if h in order}


def native_chain_pdb(pdb, label_chain):
    p = f"{WORK}/nat_{pdb}_{label_chain}.pdb"
    if not os.path.exists(p):
        rs = native_residues(pdb, label_chain)
        st = gemmi.Structure(); st.add_model(gemmi.Model("1")); ch = gemmi.Chain("A")
        for r in rs: ch.add_residue(r)
        st[0].add_chain(ch); st.setup_entities(); st.write_pdb(p)
    return p


def top_iptm_pose(cache_dir, label, pdb):
    """(seed, sample) of the pose with the highest interface pair_iptm -- what the model selects."""
    best = (-2.0, None)
    for s in SEEDS:
        for k in SAMPLES:
            o = f"{cache_dir}/{label}_{s}_{k}.json"
            if not os.path.exists(o): continue
            try: ifs = json.load(open(o)).get("dockq_interfaces") or []
            except Exception: continue
            if not ifs: continue
            mc = (ifs[0][2], ifs[0][3])
            try:
                M = json.load(open(f"{PRED}/{pdb}/seed_{s}/predictions/"
                                  f"{pdb}_summary_confidence_sample_{k}.json"))["chain_pair_iptm"]
                a, b = ord(mc[0]) - 65, ord(mc[1]) - 65
                pi = M[a][b] if (0 <= a < len(M) and 0 <= b < len(M)) else -1
            except Exception: continue
            if pi > best[0]: best = (pi, (s, k))
    if best[1] is None:
        raise RuntimeError(f"{label}: no scorable pose found in {cache_dir}")
    return best[1], best[0]


def lddt_for_chain(pdb, label_chain, seed, sample, tag):
    """(global lDDT, {native_residue_index: local lDDT}) for one native chain vs the top pose."""
    ref = native_chain_pdb(pdb, label_chain)
    mdl = f"{PRED}/{pdb}/seed_{seed}/predictions/{pdb}_sample_{sample}.cif"
    o = f"{WORK}/ld_{tag}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset", "-c", "0-31", OST, "compare-structures", "-m", mdl, "-r", ref,
                        "-o", o, "--fault-tolerant", "--min-pep-length", "4", "--lddt", "--local-lddt"],
                       env=ENV, capture_output=True, timeout=900)
    d = json.load(open(o))
    g = d.get("lddt")
    ll = d.get("local_lddt") or {}
    aln = (d.get("aln") or [None])[0]
    cmap = d.get("chain_mapping") or {}
    mdl_ch = cmap.get("A")
    if not aln or mdl_ch is None:
        return g, {}
    parts = aln.split("\n")
    ref_seq, mdl_seq = parts[1], parts[3]
    per, ri, mi = {}, 0, 0
    for a, b in zip(ref_seq, mdl_seq):          # walk the alignment: native index -> model resnum
        if a != "-" and b != "-":
            per[ri] = ll.get(f"{mdl_ch}.{mi + 1}.")
        if a != "-": ri += 1
        if b != "-": mi += 1
    return g, {k: v for k, v in per.items() if v is not None}


# ---- build the target list: both sets, domain antigens only -------------------------------------
targets = []
dec = json.load(open(f"{EXP}/artifacts/decoy_set_101_150.json"))["per_interface"]
for label, v in dec.items():
    if v["confident_decoy"]:
        targets.append(("decoy", label, v["pdb"], *v["chains"].split("-"), f"{EXP}/artifacts/decoy_set_ost"))
for e in json.load(open(f"{EXP}/artifacts/safety15.json")):
    targets.append(("good", e["iface"], e["pdb"], e["c1"], e["c2"], f"{EXP}/artifacts/goods_set_ost"))

rows, skipped = [], []
for grp, label, pdb, c1, c2, cache in targets:
    try:
        seqs = {c: "".join(TLC[r.name] for r in native_residues(pdb, c)) for c in (c1, c2)}
        roles = {c: kind(s) for c, s in seqs.items()}
        ag = [c for c in (c1, c2) if roles[c] == "ANTIGEN"]
        if len(ag) != 1:
            skipped.append(f"{label}: role assignment ambiguous {roles}"); continue
        ag = ag[0]; ab = c2 if ag == c1 else c1
        if len(seqs[ag]) < MIN_ANTIGEN_LEN:
            skipped.append(f"{label}: peptide antigen ({len(seqs[ag])} res) -- excluded by design"); continue
        (seed, sample), iptm = top_iptm_pose(cache, label, pdb)
        rec = {"group": grp, "interface": label, "pdb": pdb, "seed": seed, "sample": sample,
               "top_pair_iptm": round(iptm, 3), "antibody_chain": ab, "antigen_chain": ag,
               "antigen_len": len(seqs[ag])}
        for role, ch, key in (("paratope", ab, "ab"), ("epitope", ag, "ag")):
            g, per = lddt_for_chain(pdb, ch, seed, sample, f"{label}_{ch}_{seed}_{sample}")
            idx = interface_residue_idx(pdb, ch, ag if ch == ab else ab)
            vals = [per[i] for i in idx if i in per]
            rec[f"{key}_global_lddt"] = round(g, 3) if g is not None else None
            rec[f"{key}_{role}_lddt"] = round(sum(vals) / len(vals), 3) if vals else None
            rec[f"{key}_{role}_n_res"] = len(vals)
        rows.append(rec)
        print(f"{grp:5} {label:10} ipTM={rec['top_pair_iptm']:.3f} | Ab global={rec['ab_global_lddt']} "
              f"paratope={rec['ab_paratope_lddt']} | Ag global={rec['ag_global_lddt']} "
              f"epitope={rec['ag_epitope_lddt']}", flush=True)
    except Exception as ex:
        skipped.append(f"{label}: {type(ex).__name__} {str(ex)[:90]}")
        print(f"SKIP {label}: {type(ex).__name__} {str(ex)[:90]}", flush=True)

json.dump({"min_antigen_len": MIN_ANTIGEN_LEN, "iface_cutoff": IFACE_CUTOFF,
           "rows": rows, "skipped": skipped}, open(OUTJ, "w"), indent=1)


def summarize(key, name):
    import statistics
    out = []
    for g in ("decoy", "good"):
        v = [r[key] for r in rows if r["group"] == g and r.get(key) is not None]
        out.append((g, len(v), statistics.median(v) if v else None, statistics.mean(v) if v else None))
    d, go = out
    if d[2] is None or go[2] is None: return
    print(f"{name:22} decoy n={d[1]:2d} median={d[2]:.3f} mean={d[3]:.3f} | "
          f"good n={go[1]:2d} median={go[2]:.3f} mean={go[3]:.3f} | "
          f"gap(good-decoy) median={go[2]-d[2]:+.3f}")


print("\n=== SUMMARY (domain antigens only, top-ipTM pose) ===")
for k, n in (("ab_global_lddt", "antibody whole-chain"), ("ab_paratope_lddt", "PARATOPE"),
             ("ag_global_lddt", "antigen whole-chain"), ("ag_epitope_lddt", "EPITOPE")):
    summarize(k, n)
print(f"\nskipped {len(skipped)}"); [print("  " + s) for s in skipped]
print(f"WROTE {OUTJ}")
