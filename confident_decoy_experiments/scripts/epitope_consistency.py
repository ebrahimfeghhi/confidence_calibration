#!/usr/bin/env python
"""Where does the model put the antibody when it fails, and does it put it in the SAME wrong place?

Motivation: 11 generation failures have near-native folds (0.93-0.97 lDDT) and still produce zero
acceptable poses in 250, even with both chains templated on the native structure. Two explanations
make opposite predictions:

  SEARCH FAILURE      the sampler never explores the right basin -> wrong poses scatter over the
                      antigen surface, low pose-to-pose epitope agreement.
  WRONG LEARNED PRIOR the model confidently believes some other patch is the interface -> wrong poses
                      agree with EACH OTHER while disagreeing with the native epitope.

Two quantities per interface, over the free-run poses:
  self-consistency  mean pairwise Jaccard of the predicted epitope between poses
  native overlap    mean Jaccard of the predicted epitope against the true epitope

CRITICAL: model and native residue numbering do NOT correspond (native chains have unresolved
residues). An earlier version compared the two index spaces directly and reported ~0 native overlap
even for SOLVED interfaces, which is nonsense. Model indices are mapped to native indices through
OST's own sequence alignment, exactly as in the 8slb_AC analysis.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/epitope_consistency.py
Writes: artifacts/epitope_consistency.json
"""
import json, os, sys, subprocess, difflib, itertools, statistics, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemmi, numpy as np
from decoy_set import OST, ENV, GT, PRED, EXP, SEEDS
from fold_quality import TLC, nat_res, nat_pdb, iface_idx, kind

W = f"{EXP}/artifacts/epitope_consistency"; os.makedirs(W, exist_ok=True)
NSEED = 16


def model_to_native_map(pdb, ch, seed, sample=0):
    """{model residue index -> native residue index} from OST's alignment of the two chain sequences."""
    ref = nat_pdb(pdb, ch)
    seq = "".join(TLC[r.name] for r in nat_res(pdb, ch))
    cif = f"{PRED}/{pdb}/seed_{seed}/predictions/{pdb}_sample_{sample}.cif"
    if not os.path.exists(cif):
        return None, None
    st = gemmi.read_structure(cif); st.setup_entities()
    best, br = None, -1
    for c in st[0]:
        rs = [r for r in c if r.name in TLC]
        if not rs: continue
        r_ = difflib.SequenceMatcher(None, "".join(TLC[x.name] for x in rs), seq, autojunk=False).ratio()
        if r_ > br: br, best = r_, rs
    if best is None:
        return None, None
    mp = f"{W}/m_{pdb}_{ch}_{seed}.pdb"
    ns = gemmi.Structure(); ns.add_model(gemmi.Model("1")); c = gemmi.Chain("A")
    for r in best: c.add_residue(r)
    ns[0].add_chain(c); ns.setup_entities(); ns.write_pdb(mp)
    o = f"{W}/ld_{pdb}_{ch}_{seed}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset", "-c", "0-47", OST, "compare-structures", "-m", mp, "-r", ref,
                        "-o", o, "--fault-tolerant", "--min-pep-length", "4", "--lddt"],
                       env=ENV, capture_output=True, timeout=600)
    try:
        aln = (json.load(open(o)).get("aln") or [None])[0]
    except Exception:
        return None, None
    if not aln:
        return None, None
    p = aln.split("\n"); ri = mi = 0; m2n = {}
    for a, b in zip(p[1], p[3]):
        if a != "-" and b != "-":
            m2n[mi] = ri
        if a != "-": ri += 1
        if b != "-": mi += 1
    return m2n, best


def predicted_epitope_native_indexed(pdb, ag, abs_, seed):
    m2n, agres = model_to_native_map(pdb, ag, seed)
    if not m2n:
        return None
    cif = f"{PRED}/{pdb}/seed_{seed}/predictions/{pdb}_sample_0.cif"
    st = gemmi.read_structure(cif); st.setup_entities()
    ch = {c.name: [r for r in c if r.name in TLC] for c in st[0]}
    ch = {k: v for k, v in ch.items() if v}
    seqs = {c: "".join(TLC[r.name] for r in nat_res(pdb, c)) for c in [ag] + abs_}
    pick = lambda q: max(ch, key=lambda k: difflib.SequenceMatcher(
        None, "".join(TLC[r.name] for r in ch[k]), q, autojunk=False).ratio())
    A = ch[pick(seqs[ag])]
    XA = np.array([[a.pos.x, a.pos.y, a.pos.z] for r in A for a in r])
    IA = np.array([i for i, r in enumerate(A) for _ in r])
    hit = set()
    for x in abs_:
        B = ch[pick(seqs[x])]
        XB = np.array([[a.pos.x, a.pos.y, a.pos.z] for r in B for a in r])
        m = np.linalg.norm(XA[:, None, :] - XB[None, :, :], axis=-1) < 5.0
        hit |= {int(IA[i]) for i in np.where(m.any(1))[0]}
    return {m2n[h] for h in hit if h in m2n}          # NATIVE indexing


cls = json.load(open(f"{EXP}/artifacts/abag_all_101_150.json"))
scores = json.load(open(f"{EXP}/artifacts/native_arm_scores.json"))
gen_rescued = {k for k, v in scores.items() if v["free_status"] == "no_acceptable_pose" and v["rescued"]}
groups = collections.defaultdict(list)
for k, v in cls.items():
    if v["status"] == "solved":
        groups["solved (control)"].append(k)
    elif v["status"] == "no_acceptable_pose":
        groups["gen-failure, rescued by folds" if k in gen_rescued
               else "gen-failure, unmoved by folds"].append(k)
    elif v["status"] == "confident_decoy":
        groups["confident decoy"].append(k)

out = {}
print(f"{'group':30} {'interface':12} {'self-consist':>13} {'native overlap':>15}")
for g in ("solved (control)", "confident decoy", "gen-failure, rescued by folds",
          "gen-failure, unmoved by folds"):
    picks = sorted(groups[g])[:6]
    for lab in picks:
        v = cls[lab]; pdb = v["pdb"]
        parts = v["chains"].split("-")
        c1, c2 = (parts[0], "-".join(parts[1:])) if len(parts) > 2 else (parts[0], parts[1])
        try:
            seqs = {c: "".join(TLC[r.name] for r in nat_res(pdb, c)) for c in (c1, c2)}
        except Exception:
            continue
        ag = [c for c in (c1, c2) if kind(seqs[c]) == "ANTIGEN"]
        if len(ag) != 1:
            continue
        ag = ag[0]; abs_ = [c for c in (c1, c2) if c != ag]
        eps = []
        for s in list(SEEDS)[:NSEED]:
            e = predicted_epitope_native_indexed(pdb, ag, abs_, s)
            if e: eps.append(e)
        if len(eps) < 4:
            continue
        nat = iface_idx(pdb, ag, abs_[0])
        sc = statistics.mean(len(a & b) / len(a | b) for a, b in itertools.combinations(eps, 2) if (a | b))
        ov = statistics.mean(len(e & nat) / len(e | nat) for e in eps if (e | nat))
        out[lab] = {"group": g, "self_consistency": round(sc, 3), "native_overlap": round(ov, 3),
                    "n_poses": len(eps)}
        print(f"{g:30} {lab:12} {sc:13.3f} {ov:15.3f}", flush=True)

json.dump(out, open(f"{EXP}/artifacts/epitope_consistency.json", "w"), indent=1)
print()
for g in dict.fromkeys(v["group"] for v in out.values()):
    v = [x for x in out.values() if x["group"] == g]
    print(f"  {g:30} n={len(v)}  self-consistency {statistics.median(x['self_consistency'] for x in v):.3f}"
          f"   native overlap {statistics.median(x['native_overlap'] for x in v):.3f}")
print(f"\nWROTE {EXP}/artifacts/epitope_consistency.json")
