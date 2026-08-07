#!/usr/bin/env python
"""`duo2` arm: feed the model BOTH competing folds at once -- leader 1 (the decoy) AND leader 2.

Every arm so far gave the model ONE fold and asked what it does with it. dockqdonor (good fold) rescues
9/25; topiptm26 (the decoy's own fold) collapses coverage to zero on 9/16. So a single fold largely
determines the outcome. This asks the natural next question:

    given BOTH folds simultaneously, does the model prefer the correct one?

Leader 1 is ALWAYS the top-ipTM pose -- that is what makes an interface a confident decoy, and why
leaders top-1 is 0/26. So "leader 1 + leader 2" is precisely "the decoy's fold + a competing fold",
selected by greedy NMS over the free run WITHOUT knowing the answer. That matters: unlike dockqdonor
(donor picked by DockQ = oracle), a top-2 leader shortlist is deployable.

Restricted to interfaces where leader 2 IS acceptable (DockQ >= 0.23), so the question is well-posed:
the correct fold is present in the pair. Those are the cases in leaders_topk.json with
leader_rank_first_ok == 2.

Protenix takes max_templates=4 per chain (template_featurizer.py:764), so 2 hits is within budget --
but note every previous arm used exactly 1, so multi-template behaviour here is untested.

Each hit carries its OWN label string (two changed letters, at different positions per hit). The label
only marks the row as a TEMPLATE rather than an MSA sequence -- the coordinates come from the
registered mmCIF -- but Protenix de-duplicates hits by that label, so identical labels silently collapse
two templates into one. Both hits carry multichain_mask, so neither leaks cross-chain/docking info.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/build_duo_templates.py
Writes: artifacts/sweep_tmpl/duo2/<interface>.json, artifacts/sweep_a3m/duo2_*.a3m,
        artifacts/duo2_jobs.json, artifacts/duo2_donors.json
"""
import csv, json, os, sys, glob, difflib, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gemmi
from cifconv import convert
from decoy_set import EXP, PRED, THRESH

JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
TMPL = f"{EXP}/artifacts/sweep_tmpl/duo2"; A3M = f"{EXP}/artifacts/sweep_a3m"
MATS = f"{EXP}/artifacts/pose_pose_dockq"
CACHE_F = "/home/ebrahim/common/release_date_cache.json"
os.makedirs(TMPL, exist_ok=True); os.makedirs(A3M, exist_ok=True)

TLC = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
       'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
       'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

sc = json.load(open(f"{EXP}/artifacts/native_arm_scores.json"))
cache = json.load(open(CACHE_F))
TAKEN = {os.path.basename(p)[:-4] for p in glob.glob("/home/ebrahim/mmcif/*.cif")}
_n = [0]


def new_tid():
    while True:
        _n[0] += 1
        t = f"u{_n[0]:03d}"
        if t not in TAKEN:
            TAKEN.add(t); return t


def mut(a): return "A" if a != "A" else "G"


def native_seqs(pdb):
    path = f"/home/ebrahim/FoldBench/examples/ground_truths/{pdb}-assembly1.cif"
    block = gemmi.cif.read(path).sole_block()
    cols = [list(block.find_values("_atom_site." + c)) for c in
            ("label_asym_id", "label_comp_id", "label_seq_id", "group_PDB")]
    seen, out = collections.defaultdict(set), collections.defaultdict(list)
    for a, comp, sid, g in zip(*cols):
        if g != "ATOM" or comp not in TLC or sid in seen[a]:
            continue
        seen[a].add(sid); out[a].append(TLC[comp])
    return {k: "".join(v) for k, v in out.items() if len(v) >= 4}


def best_chain(st, q):
    best, br = None, -1
    for c in st[0]:
        s = "".join(TLC[r.name] for r in c if r.name in TLC)
        if not s: continue
        r_ = difflib.SequenceMatcher(None, s, q, autojunk=False).ratio()
        if r_ > br: br, best = r_, c
    return best


def leaders(M, iptm, thresh=THRESH):
    order = np.argsort(-iptm); alive = np.ones(len(iptm), bool); out = []
    for i in order:
        if not alive[i]: continue
        out.append(int(i)); alive &= ~(M[i] >= thresh); alive[i] = False
    return out


jobs, donors, skipped = [], {}, []
for lab in sorted(k for k, v in sc.items() if v["free_status"] == "confident_decoy"
                  and k not in {"7z2m_AJ", "8udz_AC"}):
    npy = f"{MATS}/dockq_{lab}_seeds101-150.npy"; idx = f"{MATS}/dockq_{lab}_seeds101-150_index.csv"
    if not (os.path.exists(npy) and os.path.exists(idx)):
        continue
    r = list(csv.DictReader(open(idx)))
    dq = np.array([float(x["dockq_gt"]) for x in r])
    ip = np.array([float(x["iptm"]) for x in r])
    L = leaders(np.load(npy), ip)
    if len(L) < 2 or dq[L[1]] < THRESH:
        continue                       # leader 2 must be the acceptable one for the test to be posed
    p1, p2 = L[0], L[1]
    if dq[p1] >= THRESH:
        skipped.append(f"{lab}: leader 1 already acceptable -- not a decoy shortlist"); continue

    v = sc[lab]; pdb = v["pdb"]
    poses = [(int(r[p]["seed"]), int(r[p]["sample"])) for p in (p1, p2)]
    cifs = [f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif" for s, k in poses]
    if not all(os.path.exists(c) for c in cifs):
        skipped.append(f"{lab}: donor cif missing"); continue

    base = json.load(open(f"{JDIR}/{pdb}-final-updated.json"))
    obj = json.loads(json.dumps(base[0] if isinstance(base, list) else base))
    qseqs = [e["proteinChain"]["sequence"] for e in obj["sequences"] if "proteinChain" in e]
    natseq = native_seqs(pdb)
    tids, structs = [], []
    for c in cifs:
        t = new_tid(); cache.setdefault(t, {"release_date": "2000-01-01"})
        open(f"/home/ebrahim/mmcif/{t}.cif", "w").write(convert(open(c).read(), t))
        tids.append(t)
        s_ = gemmi.read_structure(c); s_.setup_entities(); structs.append(s_)

    pi, info, ok = 0, [], True
    for e in obj["sequences"]:
        if "proteinChain" not in e:
            continue
        q = qseqs[pi]
        nc = max(natseq, key=lambda c: difflib.SequenceMatcher(None, natseq[c], q,
                                                               autojunk=False).ratio())
        L_ = len(q)
        lines = [">query", q]
        # Protenix de-duplicates template hits BY THEIR LABEL STRING (template_utils.py:1002-1008,
        # `seq = hit.hit_sequence...; if seq not in seen_seq`). Two hits whose label text matches are
        # collapsed to one, however different their STRUCTURES are. The label is only a marker saying
        # "treat this row as a template, not an MSA sequence" -- the coordinates come from the tid --
        # so giving each hit its own label positions is free, and without it the second template is
        # silently dropped (observed: "Found 1 templates" for a 2-template a3m).
        LABEL_POS = [(1 / 3, 2 / 3), (1 / 4, 3 / 4), (1 / 5, 4 / 5), (2 / 5, 3 / 5)]
        for ti, (t, s_) in enumerate(zip(tids, structs)):
            mc = best_chain(s_, q)
            if mc is None:
                ok = False; break
            hs = list(q)
            for frac in LABEL_POS[ti % len(LABEL_POS)]:
                pos = max(0, min(L_ - 1, int(L_ * frac)))
                hs[pos] = mut(hs[pos])
            lines += [f">{t}_{mc.name}/1-{L_} [subseq from] mol:protein length:{L_}  duo2",
                      "".join(hs)]
        if not ok:
            skipped.append(f"{lab} q{pi}: no model chain matched"); break
        a3mp = f"{A3M}/duo2_{lab}_q{pi}.a3m"
        open(a3mp, "w").write("\n".join(lines) + "\n")
        e["proteinChain"]["templatesPath"] = a3mp
        info.append(f"q{pi}({nc})")
        pi += 1
    if not ok:
        continue

    donors[lab] = {"pdb": pdb, "leader1": {"seed": poses[0][0], "sample": poses[0][1],
                                           "dockq": round(float(dq[p1]), 3),
                                           "iptm": round(float(ip[p1]), 3)},
                   "leader2": {"seed": poses[1][0], "sample": poses[1][1],
                               "dockq": round(float(dq[p2]), 3),
                               "iptm": round(float(ip[p2]), 3)},
                   "tids": tids, "n_chains": len(info)}
    c1, c2 = (v["chains"].split("-") + [""])[:2]
    obj["name"] = f"{pdb}_{c1}{c2}_duo2".replace("-", "")
    jp = f"{TMPL}/{lab}.json"
    json.dump([obj], open(jp, "w"))
    jobs.append({"run": "duo2", "pdb": pdb, "interface": lab, "json": jp,
                 "name": obj["name"], "seeds": list(range(101, 151)), "set": "decoy"})
    print(f"{lab:11} leader1 seed{poses[0][0]}/s{poses[0][1]} DockQ={dq[p1]:.3f} ipTM={ip[p1]:.3f}  +  "
          f"leader2 seed{poses[1][0]}/s{poses[1][1]} DockQ={dq[p2]:.3f} ipTM={ip[p2]:.3f}  "
          f"[{len(info)} chains x 2 templates]", flush=True)

json.dump(cache, open(CACHE_F, "w"))
json.dump(jobs, open(f"{EXP}/artifacts/duo2_jobs.json", "w"), indent=1)
json.dump(donors, open(f"{EXP}/artifacts/duo2_donors.json", "w"), indent=1)
print(f"\nBUILT {len(jobs)} interface jobs -> artifacts/duo2_jobs.json")
for s_ in skipped:
    print("  SKIP " + s_)
