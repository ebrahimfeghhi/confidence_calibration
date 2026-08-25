#!/usr/bin/env python
"""Template a complex from ARBITRARY predicted CIFs -- one job per donor structure.

WHY. build_allpose_generic.py templates from the cached FREE POOL (seed/sample indices into
predictions_template), and build_native_templates.py templates from the ground-truth assembly.
Neither can template from "these 5 CIF files I just produced", which is what the diffusion-feedback
loop needs: run K cycles, decode 5 poses, then continue from each pose as a template.

RECIPE is copied from build_native_templates.py -- register the donor CIF as a fake-old mmCIF under a
fresh 4-char tid (release_date 2000-01-01 so Protenix's date filter accepts it), then emit a 2-line
a3m per protein chain pointing at the matching donor chain, with two residues mutated so Protenix
reads the row as a TEMPLATE and not as an MSA sequence.

TRAPS GUARDED (each has burned an arm before):
  - tids must be exactly 4 characters or Protenix logs 'Found 0 templates' and silently runs
    UNTEMPLATED while still emitting plausible poses
  - obj["name"] must be unique per job or every donor writes into one output directory
  - the a3m hit must differ from the query or it is parsed as an MSA row

NOTE ON THE CROSS-CHAIN MASK: by default multichain_mask zeroes all cross-chain template features, so
a donor transmits only intra-chain folds. Run with PROTENIX_MULTICHAIN_MASK=0 to let the donor's
inter-chain geometry (the dock) through -- which is the point of the feedback loop.

Usage:
  .../python scripts/build_selftmpl.py --pdb 8che --run selftmpl8che --tid-prefix w \
      --cifs <a.cif> <b.cif> ... [--seeds 48]
Writes: artifacts/selftmpl_tmpl/<run>/*.json, artifacts/selftmpl_a3m/<run>/*.a3m,
        artifacts/<run>_jobs.json
"""
import argparse, difflib, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemmi
from cifconv import convert
from decoy_set import EXP

JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
CACHE_F = "/home/ebrahim/common/release_date_cache.json"

ap = argparse.ArgumentParser()
ap.add_argument("--pdb", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--tid-prefix", required=True, help="1 char; tids become <p>000.. (4 chars total)")
ap.add_argument("--cifs", nargs="+", required=True, help="donor structures to template from")
ap.add_argument("--seeds", type=int, nargs="+", default=[101])
A = ap.parse_args()

assert len(A.tid_prefix) == 1, "--tid-prefix must be exactly 1 char (tids must be 4 chars)"
TMPL = f"{EXP}/artifacts/selftmpl_tmpl/{A.run}"
A3M = f"{EXP}/artifacts/selftmpl_a3m/{A.run}"
os.makedirs(TMPL, exist_ok=True); os.makedirs(A3M, exist_ok=True)

TLC = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
       'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
       'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
cache = json.load(open(CACHE_F))
TAKEN = {os.path.basename(p)[:-4] for p in glob.glob("/home/ebrahim/mmcif/*.cif")}


def seqof(ch): return "".join(TLC[r.name] for r in ch if r.name in TLC)
def mut(a): return "A" if a != "A" else "G"


# Scan for FREE ids rather than assuming <prefix>000.. is an unused block. /home/ebrahim/mmcif is
# shared across every arm this project has ever built, so most prefixes are partly consumed -- the
# first attempt died on w001 because an earlier run had taken it while leaving w000 free.
_alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
_pool = (f"{A.tid_prefix}{a}{b}{c}" for a in _alphabet for b in _alphabet for c in _alphabet)


def new_tid():
    for t in _pool:
        if t not in TAKEN:
            TAKEN.add(t)
            return t
    sys.exit(f"no free 4-char tids left under prefix {A.tid_prefix!r}")


base = f"{JDIR}/{A.pdb}-final-updated.json"
jobs = []
for di, cifp in enumerate(sorted(A.cifs)):
    if not os.path.exists(cifp):
        sys.exit(f"missing donor cif: {cifp}")
    tid = new_tid()
    assert len(tid) == 4, f"tid {tid!r} is not 4 chars"

    st = gemmi.read_structure(cifp); st.setup_entities()
    donor_chains = [(c.name, seqof(c)) for c in st[0]]
    donor_chains = [(n, s) for n, s in donor_chains if len(s) >= 4]
    if not donor_chains:
        sys.exit(f"{cifp}: no polymer chains")
    cache.setdefault(tid, {"release_date": "2000-01-01"})
    open(f"/home/ebrahim/mmcif/{tid}.cif", "w").write(convert(open(cifp).read(), tid))

    o = json.load(open(base)); obj = json.loads(json.dumps(o[0] if isinstance(o, list) else o))
    pi, info = 0, []
    for e in obj["sequences"]:
        if "proteinChain" not in e:
            continue
        q = e["proteinChain"]["sequence"]

        def score(nseq):
            sm = difflib.SequenceMatcher(None, nseq, q, autojunk=False)
            m = sm.find_longest_match(0, len(nseq), 0, len(q))
            contain = m.size / min(len(nseq), len(q)) if min(len(nseq), len(q)) else 0.0
            return max(sm.ratio(), contain)

        nm, r = max(((n, score(s)) for n, s in donor_chains), key=lambda x: x[1])
        if r < 0.5:
            e["proteinChain"].pop("templatesPath", None)
            info.append(f"q{pi}:free({r:.2f})")
            pi += 1
            continue
        L = len(q); hs = list(q)
        for p in (L // 3, 2 * L // 3):
            hs[p] = mut(hs[p])
        a3mp = f"{A3M}/{A.pdb}_d{di}_q{pi}.a3m"
        open(a3mp, "w").write(
            f">query\n{q}\n>{tid}_{nm}/1-{L} [subseq from] mol:protein length:{L}  selftmpl\n"
            f"{''.join(hs)}\n")
        e["proteinChain"]["templatesPath"] = a3mp
        info.append(f"q{pi}->{nm}({r:.2f})")
        pi += 1

    name = f"{A.pdb}_{A.run}_d{di}"
    obj["name"] = name
    jp = f"{TMPL}/{name}.json"
    json.dump([obj], open(jp, "w"))
    jobs.append({"run": A.run, "pdb": A.pdb, "json": jp, "name": name,
                 "donor_cif": cifp, "donor_idx": di, "seeds": list(A.seeds)})
    print(f"d{di} <- {os.path.basename(cifp)}  tid {tid}  " + " ".join(info), flush=True)

json.dump(cache, open(CACHE_F, "w"))
json.dump(jobs, open(f"{EXP}/artifacts/{A.run}_jobs.json", "w"), indent=1)
assert len({j["name"] for j in jobs}) == len(jobs), "duplicate job name -- outputs would collide"
print(f"\nBUILT {len(jobs)} donor jobs x {len(A.seeds)} seeds -> artifacts/{A.run}_jobs.json")
