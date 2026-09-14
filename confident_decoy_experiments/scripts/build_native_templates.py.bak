#!/usr/bin/env python
"""Build NATIVE-fold templates (both chains) for arbitrary pdbs -- the `native50` arm.

WHY THIS EXISTS: the native templates used so far were copied out of an ephemeral /tmp scratch
directory; no script in this repo could regenerate them, and none existed for pdbs we had not already
run. This reconstructs the recipe from the a3m format so any pdb can be templated.

RECIPE (same as README 6.3 / build_pddag.py, only the fold source differs):
  - register the ground-truth assembly as a fake-old mmCIF under a fresh tid (release_date 2000-01-01,
    so Protenix's date filter accepts it as a legitimate template)
  - for every protein chain in the Protenix input JSON, emit a 2-line a3m whose single hit points at
    the matching NATIVE chain of that tid
  - two residues in the hit are mutated so Protenix treats the row as a template, not an MSA sequence
  - multichain_mask means the template carries ZERO cross-chain information: it fixes each chain's
    FOLD and says nothing about how the chains are arranged relative to each other

Usage:
  .../python scripts/build_native_templates.py 8pn0 7zau ...      # explicit pdbs
  .../python scripts/build_native_templates.py --from-status confident_decoy,no_acceptable_pose
Writes: artifacts/native_tmpl/<pdb>-native50.json, artifacts/native_a3m_built/<pdb>_q<N>.a3m,
        artifacts/native_tmpl_jobs.json
"""
import argparse, json, os, sys, glob, difflib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemmi
from cifconv import convert
from decoy_set import EXP, GT

JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
TMPL = f"{EXP}/artifacts/native_tmpl"; A3M = f"{EXP}/artifacts/native_a3m_built"
CACHE_F = "/home/ebrahim/common/release_date_cache.json"
SEEDS = list(range(101, 151))
os.makedirs(TMPL, exist_ok=True); os.makedirs(A3M, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("pdbs", nargs="*")
ap.add_argument("--from-status", default=None,
                help="comma-separated statuses to pull from artifacts/abag_all_101_150.json")
A = ap.parse_args()

pdbs = list(A.pdbs)
if A.from_status:
    want = set(A.from_status.split(","))
    cls = json.load(open(f"{EXP}/artifacts/abag_all_101_150.json"))
    pdbs += sorted({v["pdb"] for v in cls.values() if v.get("status") in want})
pdbs = sorted(set(pdbs))
if not pdbs:
    sys.exit("no pdbs given")

TLC = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
       'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
       'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
cache = json.load(open(CACHE_F))
TAKEN = {os.path.basename(p)[:-4] for p in glob.glob("/home/ebrahim/mmcif/*.cif")}
_n = [0]


def new_tid():
    while True:
        _n[0] += 1
        t = f"v{_n[0]:03d}"
        if t not in TAKEN:
            TAKEN.add(t); return t


def seqof(ch): return "".join(TLC[r.name] for r in ch if r.name in TLC)
def mut(a): return "A" if a != "A" else "G"


jobs, skipped = [], []
for pdb in pdbs:
    base = f"{JDIR}/{pdb}-final-updated.json"
    nat = f"{GT}/{pdb}-assembly1.cif"
    if not (os.path.exists(base) and os.path.exists(nat)):
        skipped.append(f"{pdb}: missing input json or ground truth"); continue
    st = gemmi.read_structure(nat); st.setup_entities()
    nat_chains = [(c.name, seqof(c)) for c in st[0]]
    nat_chains = [(n, s) for n, s in nat_chains if len(s) >= 4]
    if not nat_chains:
        skipped.append(f"{pdb}: no polymer chains in ground truth"); continue
    tid = new_tid()
    cache.setdefault(tid, {"release_date": "2000-01-01"})
    open(f"/home/ebrahim/mmcif/{tid}.cif", "w").write(convert(open(nat).read(), tid))

    o = json.load(open(base)); obj = json.loads(json.dumps(o[0] if isinstance(o, list) else o))
    pi, info, ok = 0, [], True
    for e in obj["sequences"]:
        if "proteinChain" not in e:
            continue
        q = e["proteinChain"]["sequence"]

        def score(nseq):
            """Plain ratio penalises a native chain that is heavily truncated (8vzo D: 70 of 215
            residues resolved -> ratio 0.49 despite being the right chain). Take the better of the
            ratio and the containment of the longest common block, as done elsewhere in this project
            for partially-resolved peptides."""
            sm = difflib.SequenceMatcher(None, nseq, q, autojunk=False)
            m = sm.find_longest_match(0, len(nseq), 0, len(q))
            contain = m.size / min(len(nseq), len(q)) if min(len(nseq), len(q)) else 0.0
            return max(sm.ratio(), contain)

        nm, r = max(((n, score(s)) for n, s in nat_chains), key=lambda x: x[1])
        if r < 0.5:
            # This query chain has no counterpart in the deposited structure (e.g. 7yv1 q3, an
            # 11-residue chain that is simply unresolved). Leave it FREE rather than failing the whole
            # pdb -- the other chains can still be templated.
            e["proteinChain"].pop("templatesPath", None)
            info.append(f"q{pi}:free(no native match {r:.2f})")
            skipped.append(f"{pdb} q{pi}: no native chain (best {r:.2f}) -> left un-templated")
            pi += 1
            continue
        L = len(q); hs = list(q)
        for p in (L // 3, 2 * L // 3):
            hs[p] = mut(hs[p])
        a3mp = f"{A3M}/{pdb}_q{pi}.a3m"
        open(a3mp, "w").write(
            f">query\n{q}\n>{tid}_{nm}/1-{L} [subseq from] mol:protein length:{L}  native\n"
            f"{''.join(hs)}\n")
        e["proteinChain"]["templatesPath"] = a3mp
        info.append(f"q{pi}->{nm}({r:.2f})")
        pi += 1
    if not ok:
        continue
    obj["name"] = f"{pdb}_native50"
    jp = f"{TMPL}/{pdb}-native50.json"
    json.dump([obj], open(jp, "w"))
    jobs.append({"run": "native50", "pdb": pdb, "json": jp, "name": obj["name"], "seeds": SEEDS})
    print(f"{pdb}: tid {tid}  " + " ".join(info), flush=True)

json.dump(cache, open(CACHE_F, "w"))
json.dump(jobs, open(f"{EXP}/artifacts/native_tmpl_jobs.json", "w"), indent=1)
print(f"\nBUILT {len(jobs)} native50 template sets -> artifacts/native_tmpl_jobs.json")
if skipped:
    print(f"SKIPPED {len(skipped)}:")
    for s in skipped:
        print("  " + s)
