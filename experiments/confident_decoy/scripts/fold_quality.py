#!/usr/bin/env python
"""Fold quality (lDDT vs native + pLDDT) at the interface, decoys vs goods, domain antigens only.

HYPOTHESIS (user, 2026-08-01): where the model folds the paratope/epitope accurately it has a good
structural representation of the binding site and does not invent a confident-but-wrong docking mode.
Inaccurate binding-site geometry -> the model mistakes a wrong site for a real one -> high ipTM, low DockQ.
PREDICTION: goods fold better than decoys, and the gap should be LARGER at paratope/epitope than
whole-chain. If whole-chain separates them equally well, the story is just "goods are easier targets".

POSES
  decoys: BOTH the top-ipTM pose (what the model selects; wrong) and the best-DockQ pose (the correct
          pose that exists in the pool). Comparing them separates "can't fold it" from "folded it fine
          but ranked the wrong pose".
  goods:  top-ipTM pose only -- there it is already the correct pose, so best-DockQ adds nothing.

METHOD
  - Domain antigens only (native antigen >= 30 res). Peptides have no independently stable fold, so
    "fold accuracy" is ill-posed and confounded with the docking being evaluated. Decoys have 4 such
    interfaces, goods 1; leaving them in would manufacture a goods-fold-better result.
  - Antibody/antigen role from the validated kind() classifier (scripts/derive_antigen_idx.py).
  - SINGLE-CHAIN vs SINGLE-CHAIN lDDT: the model chain is matched to the native chain by sequence and
    both are written out alone. (An earlier version handed OST the full model against a one-chain
    native; OST then mapped the wrong predicted chain and returned lDDT ~0.004. Absurd values = suspect
    the code first.)
  - lDDT is intra-chain, so it does NOT depend on where the pose docked: a decoy's wrong docking cannot
    contaminate its paratope score.
  - epitope/paratope = native residues within 5 A of the partner chain; mapped onto model residues via
    OST's own alignment.
  - pLDDT read from the model CIF B-factor column (Protenix writes pLDDT there, 0-100; rescaled to 0-1).

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/fold_quality.py
Writes: artifacts/fold_quality.json
"""
import json, os, re, subprocess, sys, statistics, difflib
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gemmi
from decoy_set import native_chains, GT, OST, ENV, PRED, EXP, SEEDS, SAMPLES

OUTJ = f"{EXP}/artifacts/fold_quality.json"
WORK = f"{EXP}/artifacts/fold_quality"
MIN_ANTIGEN_LEN, IFACE_CUTOFF, NWORK = 30, 5.0, 30
os.makedirs(WORK, exist_ok=True)
TLC = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
       'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
       'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def kind(s):
    """ANTIGEN / LIGHT / HEAVY. Port of the validated classifier in scripts/derive_antigen_idx.py."""
    hasW = re.search(r'WG.G[TS]', s); hasF = re.search(r'FG.G[TS]', s); w = s[:24]
    ntL = re.search(r'([DE]I[QV][ML]TQ|[DE]VVMTQ|EIVLTQ|QSVLTQ|DIQLTQ|SYELTQ|DIVMTQ)', w)
    ntH = re.search(r'([QE][VL][QK]L[VLEQ][EQ]S|QVKLEES|VESGG[GA]|VQSGAE|VETGGG|QESGGG)', w)
    if not (95 <= len(s) <= 275): return 'ANTIGEN'
    light = bool(ntL) or (bool(hasF) and not hasW); heavy = bool(ntH) or bool(hasW)
    if not (light or heavy): return 'ANTIGEN'
    if light and not ntH: return 'LIGHT'
    return 'HEAVY'


def nat_res(pdb, ch):
    """Ordered standard residues of one native chain, selected by label_asym_id (gemmi's residue
    .subchain); chain.name would be auth_asym_id and silently mismatch for 8bf4/8che/8fbw/8slb/8the."""
    st = gemmi.read_structure(f"{GT}/{pdb}-assembly1.cif"); st.setup_entities()
    rs = [r for c in st[0] for r in c if r.subchain == ch and r.name in TLC]
    if not rs: raise KeyError(f"{pdb}: no residues for label chain {ch}")
    return rs


def nat_pdb(pdb, ch):
    p = f"{WORK}/nat_{pdb}_{ch}.pdb"
    if not os.path.exists(p):
        st = gemmi.Structure(); st.add_model(gemmi.Model("1")); c = gemmi.Chain("A")
        for r in nat_res(pdb, ch): c.add_residue(r)
        st[0].add_chain(c); st.setup_entities(); st.write_pdb(p)
    return p


def iface_idx(pdb, ch, other):
    """0-based indices into nat_res(pdb,ch) of residues contacting `other`."""
    D = native_chains(pdb)
    XA, IA = D[ch]; XB, _ = D[other]
    m = np.linalg.norm(XA[:, None, :] - XB[None, :, :], axis=-1) < IFACE_CUTOFF
    order = {v: i for i, v in enumerate(sorted(set(IA), key=lambda x: int(x)))}
    return {order[IA[i]] for i in np.where(m.any(1))[0] if IA[i] in order}


def model_chain(pdb, seed, sample, target_seq):
    """Model chain best matching target_seq, written alone; returns (path, per-residue pLDDT list)."""
    cif = f"{PRED}/{pdb}/seed_{seed}/predictions/{pdb}_sample_{sample}.cif"
    st = gemmi.read_structure(cif); st.setup_entities()
    best, br = None, -1
    for c in st[0]:
        rs = [r for r in c if r.name in TLC]
        if not rs: continue
        r_ = difflib.SequenceMatcher(None, "".join(TLC[x.name] for x in rs), target_seq,
                                     autojunk=False).ratio()
        if r_ > br: br, best = r_, rs
    if best is None: raise RuntimeError(f"{pdb}: no model chain matched")
    p = f"{WORK}/mdl_{pdb}_{seed}_{sample}_{len(target_seq)}.pdb"
    if not os.path.exists(p):
        ns = gemmi.Structure(); ns.add_model(gemmi.Model("1")); c = gemmi.Chain("A")
        for r in best: c.add_residue(r)
        ns[0].add_chain(c); ns.setup_entities(); ns.write_pdb(p)
    plddt = [statistics.mean(a.b_iso for a in r) / 100.0 for r in best]
    return p, plddt


def lddt(pdb, ch, seed, sample, tag):
    """(global lDDT, {native_res_idx: local lDDT}, {native_res_idx: model_res_idx})."""
    ref = nat_pdb(pdb, ch)
    seq = "".join(TLC[r.name] for r in nat_res(pdb, ch))
    mdl, plddt = model_chain(pdb, seed, sample, seq)
    o = f"{WORK}/ld_{tag}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset", "-c", "0-99", OST, "compare-structures", "-m", mdl, "-r", ref,
                        "-o", o, "--fault-tolerant", "--min-pep-length", "4", "--lddt", "--local-lddt"],
                       env=ENV, capture_output=True, timeout=900)
    d = json.load(open(o))
    g, ll = d.get("lddt"), d.get("local_lddt") or {}
    aln = (d.get("aln") or [None])[0]
    per, mp = {}, {}
    if aln:
        parts = aln.split("\n"); rs_, ms_ = parts[1], parts[3]
        ri = mi = 0
        for a, b in zip(rs_, ms_):
            if a != "-" and b != "-":
                v = ll.get(f"A.{mi + 1}.")
                if v is not None: per[ri] = v
                mp[ri] = mi
            if a != "-": ri += 1
            if b != "-": mi += 1
    return g, per, mp, plddt


def measure(pdb, ab, ag, seed, sample, tag):
    out = {}
    for role, ch, other, pre in (("paratope", ab, ag, "ab"), ("epitope", ag, ab, "ag")):
        g, per, mp, plddt = lddt(pdb, ch, seed, sample, f"{tag}_{ch}")
        idx = iface_idx(pdb, ch, other)
        lv = [per[i] for i in idx if i in per]
        pv = [plddt[mp[i]] for i in idx if i in mp and mp[i] < len(plddt)]
        out[f"{pre}_global_lddt"] = round(g, 3) if g is not None else None
        out[f"{pre}_{role}_lddt"] = round(statistics.mean(lv), 3) if lv else None
        out[f"{pre}_global_plddt"] = round(statistics.mean(plddt), 3) if plddt else None
        out[f"{pre}_{role}_plddt"] = round(statistics.mean(pv), 3) if pv else None
        out[f"{pre}_{role}_n"] = len(lv)
    return out


def poses(cache, label, pdb, c1, c2):
    """(top-ipTM pose, best-DockQ pose) as (seed, sample) from the cached OST outputs.

    MUST match the requested (c1,c2) interface: a PDB contributing two interfaces (7z2m/8slb/8yxi/8che)
    has several entries in dockq_interfaces, and taking dockq[0] silently reads the WRONG one."""
    bi, bd = (-2.0, None), (-2.0, None)
    for s in SEEDS:
        for k in SAMPLES:
            o = f"{cache}/{label}_{s}_{k}.json"
            if not os.path.exists(o): continue
            try: d = json.load(open(o))
            except Exception: continue
            qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
            q = mc = None
            for i, t in enumerate(ifs):
                if {t[0], t[1]} == {c1, c2} and i < len(qs):
                    q, mc = qs[i], (t[2], t[3]); break
            if q is None and len(qs) == 1:        # native subset -> single interface present
                q, mc = qs[0], (ifs[0][2], ifs[0][3])
            if q is None: continue
            try:
                M = json.load(open(f"{PRED}/{pdb}/seed_{s}/predictions/"
                                  f"{pdb}_summary_confidence_sample_{k}.json"))["chain_pair_iptm"]
                a, b = ord(mc[0]) - 65, ord(mc[1]) - 65
                pi = M[a][b] if (0 <= a < len(M) and 0 <= b < len(M)) else -1
            except Exception: continue
            if pi > bi[0]: bi = (pi, (s, k))
            if q > bd[0]: bd = (q, (s, k))
    return bi, bd


def main():
    targets = []
    for label, v in json.load(open(f"{EXP}/artifacts/decoy_set_101_150.json"))["per_interface"].items():
        if v["confident_decoy"]:
            targets.append(("decoy", label, v["pdb"], *v["chains"].split("-"), f"{EXP}/artifacts/decoy_set_ost"))
    for e in json.load(open(f"{EXP}/artifacts/safety15.json")):
        targets.append(("good", e["iface"], e["pdb"], e["c1"], e["c2"], f"{EXP}/artifacts/goods_set_ost"))


    def run(t):
        grp, label, pdb, c1, c2, cache = t
        seqs = {c: "".join(TLC[r.name] for r in nat_res(pdb, c)) for c in (c1, c2)}
        roles = {c: kind(s) for c, s in seqs.items()}
        ag = [c for c in (c1, c2) if roles[c] == "ANTIGEN"]
        if len(ag) != 1: return None, f"{label}: ambiguous roles {roles}"
        ag = ag[0]; ab = c2 if ag == c1 else c1
        if len(seqs[ag]) < MIN_ANTIGEN_LEN:
            return None, f"{label}: peptide antigen ({len(seqs[ag])} res) - excluded by design"
        (pi, tp), (dq, bp) = poses(cache, label, pdb, c1, c2)
        rec = {"group": grp, "interface": label, "pdb": pdb, "antibody_chain": ab, "antigen_chain": ag,
               "antigen_len": len(seqs[ag]), "top_iptm": round(pi, 3), "best_dockq": round(dq, 3)}
        rec["top_iptm_pose"] = {"seed": tp[0], "sample": tp[1], **measure(pdb, ab, ag, *tp, f"{label}_top")}
        if grp == "decoy":
            rec["best_dockq_pose"] = {"seed": bp[0], "sample": bp[1], **measure(pdb, ab, ag, *bp, f"{label}_bdq")}
        return rec, None


    rows, skipped = [], []
    with ThreadPoolExecutor(NWORK) as ex:
        for rec, err in ex.map(run, targets):
            if err: skipped.append(err); print("SKIP " + err, flush=True)
            else:
                rows.append(rec)
                t = rec["top_iptm_pose"]
                print(f"{rec['group']:5} {rec['interface']:10} Ab[g={t['ab_global_lddt']} "
                      f"para={t['ab_paratope_lddt']}] Ag[g={t['ag_global_lddt']} epi={t['ag_epitope_lddt']}]",
                      flush=True)

    json.dump({"min_antigen_len": MIN_ANTIGEN_LEN, "iface_cutoff": IFACE_CUTOFF,
               "rows": rows, "skipped": skipped}, open(OUTJ, "w"), indent=1)


    def med(grp, pose, key):
        v = [r[pose][key] for r in rows if r["group"] == grp and r.get(pose, {}).get(key) is not None]
        return statistics.median(v) if v else None, len(v)


    print("\n=== MEDIANS (domain antigens only) ===")
    print(f"{'metric':26} {'decoy@topIpTM':>14} {'decoy@bestDockQ':>16} {'good@topIpTM':>13} {'good-decoy':>11}")
    for key, name in (("ab_global_lddt", "antibody whole lDDT"), ("ab_paratope_lddt", "PARATOPE lDDT"),
                      ("ag_global_lddt", "antigen whole lDDT"), ("ag_epitope_lddt", "EPITOPE lDDT"),
                      ("ab_global_plddt", "antibody whole pLDDT"), ("ab_paratope_plddt", "PARATOPE pLDDT"),
                      ("ag_global_plddt", "antigen whole pLDDT"), ("ag_epitope_plddt", "EPITOPE pLDDT")):
        d, nd = med("decoy", "top_iptm_pose", key)
        b, nb = med("decoy", "best_dockq_pose", key)
        g, ng = med("good", "top_iptm_pose", key)
        gap = f"{g - d:+.3f}" if (g is not None and d is not None) else "n/a"
        print(f"{name:26} {d if d is None else f'{d:.3f}':>14} {b if b is None else f'{b:.3f}':>16} "
              f"{g if g is None else f'{g:.3f}':>13} {gap:>11}")
    print(f"\nn: decoy={sum(1 for r in rows if r['group']=='decoy')} good={sum(1 for r in rows if r['group']=='good')}")
    print(f"skipped {len(skipped)}"); [print("  " + s) for s in skipped]
    print(f"WROTE {OUTJ}")


if __name__ == "__main__":
    main()
