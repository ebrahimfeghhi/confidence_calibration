#!/usr/bin/env python
"""Does good paratope folding predict a SMALLER confidence gap? Plus a per-interface deep dive.

Q1 (across decoys): the hypothesis in report S2 says bad paratope folding causes false confidence. If
so, decoys whose paratope IS well folded should show a smaller ipTM gap -- the model should be closer
to ranking a correct pose first.
    gap = pair_iptm(top pose) - pair_iptm(most confident ACCEPTABLE pose, DockQ >= 0.23)
    gap = 0 would mean the correct pose already ranks first (i.e. not a decoy).

Q2 (deep dive, --interface): for one interface, characterise the failure:
    - are the top-ipTM pose and the most-confident-acceptable pose nearly tied in confidence?
    - are they the SAME BINDING MODE slightly off, or genuinely different? Measured as pose-vs-pose
      DockQ (>=0.23 = same basin). Never ligand-RMSD -- see memory 'leaders-always-via-dockq'.
    - is the wrong pose on a DIFFERENT EPITOPE? Compare the antigen residues contacted in the predicted
      pose against the native epitope (Jaccard overlap), using OST's own native<->model alignment.

Usage:
  .../python scripts/paratope_vs_gap.py                       # Q1 table
  .../python scripts/paratope_vs_gap.py --interface 8slb_AC   # Q1 + deep dive
Writes: artifacts/paratope_vs_gap.json  (+ artifacts/deepdive_<interface>.json when --interface given)
"""
import argparse, json, os, subprocess, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gemmi
from decoy_set import native_chains, GT, OST, ENV, PRED, EXP, SEEDS, SAMPLES
from fold_quality import TLC, nat_res, iface_idx, kind, WORK as FQWORK

ap = argparse.ArgumentParser(); ap.add_argument("--interface", default=None)
ARGS = ap.parse_args()
CACHE = f"{EXP}/artifacts/decoy_set_ost"
WORK = f"{EXP}/artifacts/paratope_vs_gap"; os.makedirs(WORK, exist_ok=True)
DEC = json.load(open(f"{EXP}/artifacts/decoy_set_101_150.json"))["per_interface"]
FQ = {r["interface"]: r for r in json.load(open(f"{EXP}/artifacts/fold_quality.json"))["rows"]}


def all_poses(label, pdb, c1, c2):
    """[(seed, sample, pair_iptm, dockq)] over the pool, for the REQUESTED interface.

    Matching (c1,c2) is required: PDBs with two interfaces (7z2m/8slb/8yxi/8che) have several entries
    in dockq_interfaces and dockq[0] would silently be the wrong interface."""
    out = []
    for s in SEEDS:
        for k in SAMPLES:
            o = f"{CACHE}/{label}_{s}_{k}.json"
            if not os.path.exists(o): continue
            try: d = json.load(open(o))
            except Exception: continue
            qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
            q = mc = None
            for i, t in enumerate(ifs):
                if {t[0], t[1]} == {c1, c2} and i < len(qs):
                    q, mc = qs[i], (t[2], t[3]); break
            if q is None and len(qs) == 1:
                q, mc = qs[0], (ifs[0][2], ifs[0][3])
            if q is None: continue
            try:
                M = json.load(open(f"{PRED}/{pdb}/seed_{s}/predictions/"
                                  f"{pdb}_summary_confidence_sample_{k}.json"))["chain_pair_iptm"]
                a, b = ord(mc[0]) - 65, ord(mc[1]) - 65
                pi = M[a][b] if (0 <= a < len(M) and 0 <= b < len(M)) else -1
            except Exception: continue
            out.append((s, k, pi, q))
    return out


# ---------------- Q1: paratope quality vs confidence gap ----------------
rows = []
for label, v in DEC.items():
    if not v["confident_decoy"] or label not in FQ: continue
    ps = all_poses(label, v["pdb"], *v["chains"].split("-"))
    if not ps: continue
    top = max(ps, key=lambda x: x[2])
    ok = [p for p in ps if p[3] >= 0.23]
    if not ok: continue
    bestok = max(ok, key=lambda x: x[2])
    rows.append({"interface": label, "pdb": v["pdb"],
                 "paratope_lddt": FQ[label]["top_iptm_pose"]["ab_paratope_lddt"],
                 "epitope_lddt": FQ[label]["top_iptm_pose"]["ag_epitope_lddt"],
                 "top_iptm": round(top[2], 3), "top_dockq": round(top[3], 3),
                 "best_ok_iptm": round(bestok[2], 3), "best_ok_dockq": round(bestok[3], 3),
                 "iptm_gap": round(top[2] - bestok[2], 3), "n_acceptable": len(ok)})

rows.sort(key=lambda r: r["paratope_lddt"])
print(f"{'interface':10} {'paratope':>8} {'epitope':>8} {'top ipTM':>9} {'top DockQ':>10} "
      f"{'bestOK ipTM':>12} {'bestOK DockQ':>13} {'ipTM gap':>9}")
for r in rows:
    print(f"{r['interface']:10} {r['paratope_lddt']:8.3f} {r['epitope_lddt']:8.3f} {r['top_iptm']:9.3f} "
          f"{r['top_dockq']:10.3f} {r['best_ok_iptm']:12.3f} {r['best_ok_dockq']:13.3f} {r['iptm_gap']:+9.3f}")

x = [r["paratope_lddt"] for r in rows]; y = [r["iptm_gap"] for r in rows]
corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else None
lo = [r["iptm_gap"] for r in rows if r["paratope_lddt"] < 0.75]
hi = [r["iptm_gap"] for r in rows if r["paratope_lddt"] >= 0.75]
print(f"\nPearson r(paratope lDDT, ipTM gap) = {corr:+.3f}  (n={len(rows)})")
print(f"  poor paratope (<0.75, n={len(lo)}): median gap {statistics.median(lo):+.3f}")
print(f"  good paratope (>=0.75, n={len(hi)}): median gap {statistics.median(hi):+.3f}")
json.dump({"rows": rows, "pearson_r": corr}, open(f"{EXP}/artifacts/paratope_vs_gap.json", "w"), indent=1)

# ---------------- Q2: deep dive ----------------
if not ARGS.interface:
    sys.exit(0)
L = ARGS.interface; v = DEC[L]; pdb = v["pdb"]; c1, c2 = v["chains"].split("-")
seqs = {c: "".join(TLC[r.name] for r in nat_res(pdb, c)) for c in (c1, c2)}
ag = [c for c in (c1, c2) if kind(seqs[c]) == "ANTIGEN"][0]; ab = c2 if ag == c1 else c1
ps = all_poses(L, pdb, c1, c2)
top = max(ps, key=lambda x: x[2])
ok = [p for p in ps if p[3] >= 0.23]
bestok = max(ok, key=lambda x: x[2]); bestdq = max(ps, key=lambda x: x[3])
print(f"\n=== DEEP DIVE {L}  (antibody {ab}, antigen {ag}) ===")
for name, p in (("top-ipTM (selected)", top), ("most confident ACCEPTABLE", bestok), ("best-DockQ", bestdq)):
    print(f"  {name:26} seed {p[0]:3d} sample {p[1]}  ipTM {p[2]:.3f}  DockQ {p[3]:.3f}")
print(f"  ipTM gap (top - most confident acceptable): {top[2]-bestok[2]:+.3f}")


def cif(p): return f"{PRED}/{pdb}/seed_{p[0]}/predictions/{pdb}_sample_{p[1]}.cif"


def model_chain_letter(p, seq):
    """Which MODEL chain letter carries `seq` in this pose (model chain naming, not native)."""
    import difflib
    st = gemmi.read_structure(cif(p)); st.setup_entities()
    best, br = None, -1
    for c in st[0]:
        rs = [r for r in c if r.name in TLC]
        if not rs: continue
        r_ = difflib.SequenceMatcher(None, "".join(TLC[x.name] for x in rs), seq, autojunk=False).ratio()
        if r_ > br: br, best = r_, c.name
    return best


def pose_pose_dockq(pa, pb, tag):
    """DockQ of pose A treating pose B as reference, AT THE ab-ag INTERFACE: >=0.23 => same binding mode.

    Must select the ab-ag pair explicitly -- taking max() over dockq would, for a PDB with two
    interfaces (8slb has A-B and A-C), silently report similarity at the other interface."""
    o = f"{WORK}/pp_{L}_{tag}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset", "-c", "0-63", OST, "compare-structures", "-m", cif(pa), "-r", cif(pb),
                        "-o", o, "--fault-tolerant", "--min-pep-length", "4", "--dockq"],
                       env=ENV, capture_output=True, timeout=900)
    try: d = json.load(open(o))
    except Exception: return None
    q, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
    want = {model_chain_letter(pb, seqs[ab]), model_chain_letter(pb, seqs[ag])}
    for i, t in enumerate(ifs):
        if {t[0], t[1]} == want and i < len(q):
            return q[i]
    return None


same_ok = pose_pose_dockq(top, bestok, "top_vs_bestok")
same_dq = pose_pose_dockq(top, bestdq, "top_vs_bestdq")
print(f"  pose-vs-pose DockQ, top vs most-confident-acceptable: {same_ok}"
      f"   ({'SAME binding mode' if (same_ok or 0) >= 0.23 else 'DIFFERENT binding mode'})")
print(f"  pose-vs-pose DockQ, top vs best-DockQ:                {same_dq}"
      f"   ({'SAME binding mode' if (same_dq or 0) >= 0.23 else 'DIFFERENT binding mode'})")


def predicted_epitope(p):
    """Native-indexed antigen residues contacted by the antibody in this predicted pose.
    Model->native index mapping comes from the cached OST alignment written by fold_quality.py."""
    st = gemmi.read_structure(cif(p)); st.setup_entities()
    chains = {}
    for c in st[0]:
        rs = [r for r in c if r.name in TLC]
        if rs: chains[c.name] = rs
    import difflib
    def best(seq):
        return max(chains, key=lambda k: difflib.SequenceMatcher(
            None, "".join(TLC[r.name] for r in chains[k]), seq, autojunk=False).ratio())
    mag, mab = chains[best(seqs[ag])], chains[best(seqs[ab])]
    XA = np.array([[a.pos.x, a.pos.y, a.pos.z] for r in mag for a in r])
    IA = np.array([i for i, r in enumerate(mag) for _ in r])
    XB = np.array([[a.pos.x, a.pos.y, a.pos.z] for r in mab for a in r])
    m = np.linalg.norm(XA[:, None, :] - XB[None, :, :], axis=-1) < 5.0
    hit = {int(IA[i]) for i in np.where(m.any(1))[0]}
    # model residue index -> native residue index, via the alignment fold_quality.py cached
    o = f"{FQWORK}/ld_{L}_top_{ag}.json"
    m2n = {}
    if os.path.exists(o):
        aln = (json.load(open(o)).get("aln") or [None])[0]
        if aln:
            pr = aln.split("\n"); ri = mi = 0
            for a, b in zip(pr[1], pr[3]):
                if a != "-" and b != "-": m2n[mi] = ri
                if a != "-": ri += 1
                if b != "-": mi += 1
    return {m2n[h] for h in hit if h in m2n}


nat_epi = iface_idx(pdb, ag, ab)
pred_epi = predicted_epitope(top)
inter = nat_epi & pred_epi
jac = len(inter) / len(nat_epi | pred_epi) if (nat_epi | pred_epi) else 0
print(f"\n  NATIVE epitope: {len(nat_epi)} residues | PREDICTED (top-ipTM pose): {len(pred_epi)}")
print(f"  overlap: {len(inter)} residues | Jaccard {jac:.3f}"
      f"   -> {'SAME epitope' if jac > 0.3 else 'DIFFERENT epitope (docks elsewhere on the antigen)'}")
fq = FQ.get(L, {}).get("top_iptm_pose", {})
print(f"  fold quality at this pose: paratope lDDT {fq.get('ab_paratope_lddt')}, "
      f"epitope lDDT {fq.get('ag_epitope_lddt')}, antigen whole {fq.get('ag_global_lddt')}")
json.dump({"interface": L, "top": top, "best_acceptable": bestok, "best_dockq_pose": bestdq,
           "iptm_gap": top[2] - bestok[2], "posepose_dockq_top_vs_bestok": same_ok,
           "posepose_dockq_top_vs_bestdq": same_dq, "native_epitope_n": len(nat_epi),
           "predicted_epitope_n": len(pred_epi), "epitope_overlap_n": len(inter),
           "epitope_jaccard": jac}, open(f"{EXP}/artifacts/deepdive_{L}.json", "w"), indent=1)
print(f"\nWROTE artifacts/deepdive_{L}.json")
