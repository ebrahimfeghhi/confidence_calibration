#!/usr/bin/env python
"""Build the `dockqdonor` sweep arm: template from the folds of the pose that DOCKED correctly.

§4's four predicted-fold arms all pick the donor pose by FOLD quality (whole/interface lDDT or pLDDT).
This picks it by DockQ instead: among poses that got the docking right (DockQ >= 0.23) but which ipTM
ranked low, take the best one and template from its folds.

Templates carry only intra-chain geometry (multichain_mask), so NONE of the donor's docking arrangement
is transferred -- only the backbone conformations it happened to adopt. The question is whether a pose
that docked correctly holds conformations (CDR-H3 above all, which is induced-fit and therefore coupled
to the bound pose) that fold-accuracy scoring does not specifically reward.

WHY THIS IS NOT REDUNDANT with `ifacelddt` (scripts/dockq_vs_lddt_donor.py, artifacts/
dockq_vs_lddt_donor.json): on decoy interfaces the two criteria choose a DIFFERENT donor for 79% of
chains, and the trade is lopsided -- the DockQ donor gives up a median 0.012 interface lDDT, while the
lDDT donor gives up a median 0.291 DockQ. So the two donors have near-identical binding-site fold
accuracy and differ mainly in whether their pose docked. Any rescue difference is therefore hard to
attribute to fold accuracy as interface lDDT measures it.

DONOR GRANULARITY: ONE donor pose per INTERFACE -- the pose with the best DockQ on that interface --
result is directly comparable to `ifacelddt` (9/15). A single-donor-pose-per-complex variant is a
different experiment and is not built here.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/build_dockq_templates.py
Writes: artifacts/sweep_tmpl/dockqdonor/<pdb>.json, artifacts/sweep_a3m/dockqdonor_*.a3m,
        artifacts/dockq_arm_jobs.json, artifacts/dockq_arm_donors.json
"""
import csv, json, os, sys, glob, difflib, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemmi
from cifconv import convert
from decoy_set import EXP, PRED, SEEDS, SAMPLES, THRESH

JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
TMPL = f"{EXP}/artifacts/sweep_tmpl/dockqdonor"; A3M = f"{EXP}/artifacts/sweep_a3m"
CACHE_F = "/home/ebrahim/common/release_date_cache.json"
CACHES = [f"{EXP}/artifacts/abag_all_ost", f"{EXP}/artifacts/decoy_set_ost"]
EXCLUDE = {"7z2m_AJ", "8udz_AC"}
os.makedirs(TMPL, exist_ok=True); os.makedirs(A3M, exist_ok=True)

TLC = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
       'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
       'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

sc = json.load(open(f"{EXP}/artifacts/native_arm_scores.json"))
DECOYS = {k: v for k, v in sc.items()
          if v["free_status"] == "confident_decoy" and k not in EXCLUDE}
PDBS = sorted({v["pdb"] for v in DECOYS.values()})

pairs = collections.defaultdict(list)
for r in csv.DictReader(open("/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv")):
    pairs[r["pdb_id"].split("-")[0]].append((r["interface_chain_id_1"], r["interface_chain_id_2"]))

cache = json.load(open(CACHE_F))
TAKEN = {os.path.basename(p)[:-4] for p in glob.glob("/home/ebrahim/mmcif/*.cif")}
_n = [0]


def new_tid():
    while True:
        _n[0] += 1
        t = f"d{_n[0]:03d}"
        if t not in TAKEN:
            TAKEN.add(t); return t


def mut(a): return "A" if a != "A" else "G"


def native_seqs(pdb):
    """{label_asym_id: sequence} for the ground truth.

    MUST be label_asym_id: the targets CSV, meta.json and OST's dockq_interfaces all speak it, while
    gemmi's read_structure() names chains by auth_asym_id. They differ for 8bf4/8che/8fbw/8slb/8the --
    an auth-keyed lookup matched NO labelled interface for those pdbs and silently left every chain
    un-templated. Same failure mode decoy_set.native_chains() exists to prevent; this is its sequence
    counterpart, built the same way (gemmi.cif, not read_structure).
    """
    path = f"/home/ebrahim/FoldBench/examples/ground_truths/{pdb}-assembly1.cif"
    block = gemmi.cif.read(path).sole_block()
    cols = [list(block.find_values("_atom_site." + c)) for c in
            ("label_asym_id", "label_comp_id", "label_seq_id", "group_PDB")]
    seen, out = collections.defaultdict(set), collections.defaultdict(list)
    for a, comp, sid, g in zip(*cols):
        if g != "ATOM" or comp not in TLC or sid in seen[a]:
            continue
        seen[a].add(sid)
        out[a].append(TLC[comp])
    got = {k: "".join(v) for k, v in out.items() if len(v) >= 4}
    if not got:
        raise ValueError(f"{pdb}: no polymer chains parsed from {path}")
    return got


def best_chain(st, q):
    """Model chain whose sequence best matches the query (poses have no stable chain naming)."""
    best, br = None, -1
    for c in st[0]:
        s = "".join(TLC[r.name] for r in c if r.name in TLC)
        if not s: continue
        r_ = difflib.SequenceMatcher(None, s, q, autojunk=False).ratio()
        if r_ > br: br, best = r_, c
    return best


def dockq_by_pose(pdb, c1, c2):
    """{(seed, sample): dockq} for one labelled interface, from the cached OST runs."""
    out = {}
    for s in SEEDS:
        for k in SAMPLES:
            for cdir in CACHES:
                o = f"{cdir}/{pdb}_{s}_{k}.json"
                if not (os.path.exists(o) and os.path.getsize(o)):
                    continue
                try:
                    d = json.load(open(o))
                except Exception:
                    continue
                qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
                for i, t in enumerate(ifs):
                    if {t[0], t[1]} == {c1, c2} and i < len(qs):     # by NAME, never dockq[0]
                        out[(s, k)] = qs[i]; break
                if (s, k) in out:
                    break
    return out


jobs, donors, skipped = [], {}, []
for lab in sorted(DECOYS):
    v = DECOYS[lab]; pdb = v["pdb"]
    parts = v["chains"].split("-")
    c1, c2 = (parts[0], "-".join(parts[1:])) if len(parts) > 2 else (parts[0], parts[1])

    # ONE donor pose: the pose that docked THIS interface best. Not a per-chain argmax -- DockQ is a
    # property of an interface, not of a chain, and taking max() over the interfaces a chain belongs to
    # let an unrelated interface choose that chain's folds (8yxi C was picked by B-C while A-C was
    # under test). Every chain of the complex is templated from this single pose, so the only thing
    # that differs from the free run is whose folds are used.
    dq = dockq_by_pose(pdb, c1, c2)
    acc = {k: q for k, q in dq.items() if q >= THRESH}
    if not acc:
        skipped.append(f"{lab}: no acceptable pose on this interface -- cannot build"); continue
    (s, k) = max(acc, key=lambda x: acc[x])
    cif = f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif"
    if not os.path.exists(cif):
        skipped.append(f"{lab}: donor cif missing {cif}"); continue

    base = json.load(open(f"{JDIR}/{pdb}-final-updated.json"))
    obj = json.loads(json.dumps(base[0] if isinstance(base, list) else base))
    qseqs = [e["proteinChain"]["sequence"] for e in obj["sequences"] if "proteinChain" in e]
    natseq = native_seqs(pdb)
    st = gemmi.read_structure(cif); st.setup_entities()

    tid = new_tid(); cache.setdefault(tid, {"release_date": "2000-01-01"})
    open(f"/home/ebrahim/mmcif/{tid}.cif", "w").write(convert(open(cif).read(), tid))

    pi, info, ok, chains = 0, [], True, {}
    for e in obj["sequences"]:
        if "proteinChain" not in e:
            continue
        q = qseqs[pi]
        nc = max(natseq, key=lambda c: difflib.SequenceMatcher(None, natseq[c], q,
                                                               autojunk=False).ratio())
        mc = best_chain(st, q)
        if mc is None:
            skipped.append(f"{lab} q{pi}: no model chain matched in the donor pose"); ok = False; break
        L = len(q); hs = list(q)
        for pos in (L // 3, 2 * L // 3):
            hs[pos] = mut(hs[pos])
        a3mp = f"{A3M}/dockqdonor_{lab}_q{pi}.a3m"
        open(a3mp, "w").write(
            f">query\n{q}\n>{tid}_{mc.name}/1-{L} [subseq from] mol:protein length:{L}  dockqdonor\n"
            f"{''.join(hs)}\n")
        e["proteinChain"]["templatesPath"] = a3mp
        chains[f"q{pi}"] = nc
        info.append(f"q{pi}({nc})<-{mc.name}")
        pi += 1
    if not ok:
        continue

    donors[lab] = {"pdb": pdb, "interface": [c1, c2], "seed": s, "sample": k,
                   "donor_dockq": round(acc[(s, k)], 3), "n_acceptable": len(acc),
                   "n_chains_templated": len(chains), "chains": chains, "tid": tid}
    obj["name"] = f"{pdb}_{c1}{c2}_dockqdonor".replace("-", "")     # 8im0-style hyphens break the a3m
    jp = f"{TMPL}/{lab}.json"
    json.dump([obj], open(jp, "w"))
    jobs.append({"run": "dockqdonor", "pdb": pdb, "interface": lab, "json": jp,
                 "name": obj["name"], "seeds": list(SEEDS), "set": "decoy"})
    print(f"{lab:12} donor seed{s}/s{k} dq={acc[(s,k)]:.3f} "
          f"({len(acc)} acceptable)  chains: {' '.join(info)}", flush=True)

json.dump(cache, open(CACHE_F, "w"))
json.dump(jobs, open(f"{EXP}/artifacts/dockq_arm_jobs.json", "w"), indent=1)
json.dump(donors, open(f"{EXP}/artifacts/dockq_arm_donors.json", "w"), indent=1)
assert len(jobs) == len(set(j["interface"] for j in jobs)), "duplicate job per interface"
print(f"\nBUILT {len(jobs)} interface jobs (of {len(DECOYS)} decoys) -> artifacts/dockq_arm_jobs.json")
if skipped:
    print(f"SKIPPED {len(skipped)}:")
    for s in skipped:
        print("  " + s)
