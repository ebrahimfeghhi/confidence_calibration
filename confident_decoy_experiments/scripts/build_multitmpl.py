#!/usr/bin/env python
"""Template a complex from SEVERAL predicted CIFs AT ONCE -- all donors in a single job.

DIFFERENCE FROM build_selftmpl.py, which this is otherwise copied from: that script emits ONE JOB PER
DONOR, each with a single-hit a3m, so the donors are alternative continuations. Here every donor
becomes another hit row in ONE a3m per query chain, so the model sees all of them simultaneously and
its template embedder decides what to do with them. One job, N templates.

THE HYPOTHESIS THIS SERVES. Within an interface, the monomer folds and the interface patch geometry
are essentially identical across all 250 poses -- per-chain lDDT moves with SD 0.007 (antibody) and
0.014 (antigen), epitope/paratope patch RMSD with SD 0.10-0.18 A, while epitope Jaccard moves with SD
0.13. So poses differ in PLACEMENT, not in folding. If that is true, spending 10 recycles to
re-derive a fold we already have is waste: hand the fold over as templates and run ONE recycle, and
the seed should still be free to vary the epitope. That is the thing being tested.

THE CROSS-CHAIN MASK MUST STAY ON (i.e. do NOT set PROTENIX_MULTICHAIN_MASK=0). multichain_mask zeroes
every cross-chain template feature, so a donor transmits only its INTRA-CHAIN backbone conformation
and cannot tell the model where the antigen sat. That is exactly what this experiment wants: give away
the fold, keep the docking decision free. Every previous self-templating arm here deliberately set the
mask to 0 to pass the dock through -- that is the opposite configuration and a different experiment
(and it lost to the free run: selected DockQ 0.587 -> 0.511, n=35).

TRAPS GUARDED, each of which has silently corrupted an arm before:
  - TEMPLATE IDS MUST BE EXACTLY 4 CHARACTERS. Otherwise Protenix logs 'Found 0 templates' and runs
    UNTEMPLATED while still emitting perfectly plausible poses. Three arms were scored and believed
    for a week in that state. Ids are scanned for availability against /home/ebrahim/mmcif rather than
    assumed free, because that directory is shared across every arm this project has built.
  - THE A3M HIT MUST DIFFER FROM THE QUERY or it is parsed as an MSA row and not a template; two
    residues are mutated per hit.
  - obj["name"] must be unique per job or every job writes into one output directory.
  - donor chain -> query chain is matched by sequence similarity, and a chain scoring < 0.5 is left
    UNTEMPLATED rather than force-matched.

Usage:
  .../python scripts/build_multitmpl.py --pdb 7txw --run mt7txw --tid-prefix m \
      --cifs pose0.cif ... pose4.cif --seeds 102 103 ...
Writes artifacts/selftmpl_tmpl/<run>/*.json, artifacts/selftmpl_a3m/<run>/*.a3m,
       artifacts/<run>_jobs.json
"""
import argparse
import difflib
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemmi  # noqa: E402
from cifconv import convert  # noqa: E402
from decoy_set import EXP  # noqa: E402

# --json-dir so this works for PXM-2025 complexes too, whose jsons live under artifacts/pxm2025.
JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
CACHE_F = "/home/ebrahim/common/release_date_cache.json"

ap = argparse.ArgumentParser()
ap.add_argument("--pdb", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--tid-prefix", required=True, help="1 char; tids become <p>NNN (4 chars total)")
ap.add_argument("--cifs", nargs="+", required=True, help="donor structures, ALL used together")
ap.add_argument("--seeds", type=int, nargs="+", default=[101])
# --keep-stock APPENDS the donors to the chain's existing hmmsearch templates instead of replacing
# them. The replace-only arms conflated two changes: templates were ADDED and the stock ones were
# REMOVED. 8vyl's stock pool is 26,271 real nanobody hits and ~3,000 real haemoglobin structures, so
# removing it is not a small edit. This mode isolates "does adding a wrong pose poison a good prior".
#
# ORDER MATTERS: get_templates() truncates the candidate pool to the first
# PROTENIX_MAX_TEMPLATE_CANDIDATES (default 20) AFTER sorting by sum_probs. HmmsearchA3MParser sets
# sum_probs=None for every a3m hit (template_parser.py:689), so all keys tie at 0.0 and Python's
# stable sort preserves file order. The donors are therefore written FIRST, immediately after the
# query, or they would fall outside the window behind thousands of stock hits.
# --donor-chains forces the donor chain used for each query entity, in order, e.g. "A,C,F".
# NEEDED FOR MULTI-COPY ASSEMBLIES with the cross-chain mask OFF: chains are matched by sequence,
# and when every copy has the same sequence the pick among ties is arbitrary. On 9x1h the real
# protomers are (A,C,F) and (B,D,E) -- A-C 209 contacts, A-F 53, C-F 55, but A-E only 17 -- so the
# default pick of A,C,E pairs a Fab with the antigen copy it is NOT bound to, transmitting a
# nonsense arrangement. Harmless with the mask ON (only intra-chain folds are read), fatal with it
# off. Trimming the donor cif instead does NOT work: a gemmi round-trip makes Protenix report
# "No protein chains found", so the raw prediction cif must be used as-is.
ap.add_argument("--donor-chains", default="",
                help="comma list, one donor chain per protein entity in order (e.g. A,C,F)")
ap.add_argument("--json-dir", default=JDIR, help="where <pdb>-final-updated.json lives")
ap.add_argument("--keep-stock", action="store_true",
                help="append donors to the chain's existing templates instead of replacing them")
A = ap.parse_args()

assert len(A.tid_prefix) == 1, "--tid-prefix must be exactly 1 char (tids must be 4 chars)"
TMPL = f"{EXP}/artifacts/selftmpl_tmpl/{A.run}"
A3M = f"{EXP}/artifacts/selftmpl_a3m/{A.run}"
os.makedirs(TMPL, exist_ok=True)
os.makedirs(A3M, exist_ok=True)

TLC = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
       'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
       'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
cache = json.load(open(CACHE_F))
TAKEN = {os.path.basename(p)[:-4] for p in glob.glob("/home/ebrahim/mmcif/*.cif")}


def seqof(ch):
    return "".join(TLC[r.name] for r in ch if r.name in TLC)


def mut(a):
    return "A" if a != "A" else "G"


_alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
_pool = (f"{A.tid_prefix}{a}{b}{c}" for a in _alphabet for b in _alphabet for c in _alphabet)


def new_tid():
    for t in _pool:
        if t not in TAKEN:
            TAKEN.add(t)
            return t
    sys.exit(f"no free 4-char tids left under prefix {A.tid_prefix!r}")


# ---- register every donor as a fake-old mmCIF, once, and keep its chain sequences
donors = []
for cifp in sorted(A.cifs):
    if not os.path.exists(cifp):
        sys.exit(f"missing donor cif: {cifp}")
    tid = new_tid()
    assert len(tid) == 4, f"tid {tid!r} is not 4 chars"
    st = gemmi.read_structure(cifp)
    st.setup_entities()
    chains = [(c.name, seqof(c)) for c in st[0]]
    chains = [(n, s) for n, s in chains if len(s) >= 4]
    if not chains:
        sys.exit(f"{cifp}: no polymer chains")
    cache.setdefault(tid, {"release_date": "2000-01-01"})
    open(f"/home/ebrahim/mmcif/{tid}.cif", "w").write(convert(open(cifp).read(), tid))
    donors.append({"tid": tid, "cif": cifp, "chains": chains})

# ---- ONE job, whose a3m for each query chain lists EVERY donor as a separate hit
base = f"{A.json_dir}/{A.pdb}-final-updated.json"
o = json.load(open(base))
obj = json.loads(json.dumps(o[0] if isinstance(o, list) else o))

pi, info, n_hits_total = 0, [], 0
for e in obj["sequences"]:
    if "proteinChain" not in e:
        continue
    q = e["proteinChain"]["sequence"]
    L = len(q)

    def score(nseq):
        sm = difflib.SequenceMatcher(None, nseq, q, autojunk=False)
        m = sm.find_longest_match(0, len(nseq), 0, len(q))
        contain = m.size / min(len(nseq), len(q)) if min(len(nseq), len(q)) else 0.0
        return max(sm.ratio(), contain)

    forced = [c.strip() for c in A.donor_chains.split(",") if c.strip()]
    lines, used, seen = [f">query\n{q}"], [], set()
    for di, d in enumerate(donors):
        if forced:
            if pi >= len(forced):
                sys.exit(f"--donor-chains has {len(forced)} entries, need one per protein entity")
            nm = forced[pi]
            have = dict(d["chains"])
            if nm not in have:
                sys.exit(f"donor {d['cif']} has no chain {nm!r} (has {sorted(have)})")
            r = score(have[nm])
            if r < 0.5:
                sys.exit(f"forced chain {nm} matches query q{pi} at only {r:.2f} -- wrong chain?")
        else:
            nm, r = max(((n, score(s)) for n, s in d["chains"]), key=lambda x: x[1])
            if r < 0.5:
                continue
        # EACH DONOR NEEDS A DISTINCT HIT SEQUENCE. get_templates() de-duplicates by hit sequence
        # (template_utils.py:1002), so mutating the same two positions in every hit collapses all N
        # donors to ONE -- observed as 'Found 1 templates' on the first attempt, which is a silent
        # wrong-configuration rather than an error. The mutated positions are therefore offset by the
        # donor index. Two residues are still mutated per hit: fewer than two and Protenix parses the
        # row as an MSA sequence instead of a template.
        hs = list(q)
        p1 = L // 3
        p2 = (L // 2 + 1 + di) % L
        if p2 == p1:
            p2 = (p2 + 1) % L
        for p in (p1, p2):
            hs[p] = mut(hs[p])
        hit = "".join(hs)
        assert hit != q, "hit identical to query -- would be parsed as an MSA row"
        assert hit not in seen, (f"duplicate hit sequence for donor {di} -- Protenix would "
                                 f"de-duplicate and silently drop it")
        seen.add(hit)
        lines.append(f">{d['tid']}_{nm}/1-{L} [subseq from] mol:protein length:{L}  multitmpl\n"
                     f"{hit}")
        used.append(f"{d['tid']}:{nm}({r:.2f})")

    if not used:
        # no donor chain resembles this query chain -- leave it untemplated rather than force a match
        e["proteinChain"].pop("templatesPath", None)
        info.append(f"q{pi}:FREE")
    else:
        n_stock = 0
        if A.keep_stock:
            stock = e["proteinChain"].get("templatesPath", "")
            if stock and os.path.exists(stock):
                body = [ln for ln in open(stock).read().splitlines() if ln.strip()]
                # drop the stock file's own ">query" header+sequence; ours is already line 0
                if body and body[0].startswith(">") and "mol:protein" not in body[0]:
                    body = body[2:]
                lines += body
                n_stock = sum(1 for ln in body if ln.startswith(">"))
            info.append(f"q{pi}:+{n_stock} stock hits")
        a3mp = f"{A3M}/{A.pdb}_q{pi}.a3m"
        open(a3mp, "w").write("\n".join(lines) + "\n")
        e["proteinChain"]["templatesPath"] = a3mp
        n_hits_total += len(used)
        info.append(f"q{pi}<-{len(used)} hits [{' '.join(used)}]")
    pi += 1

name = f"{A.pdb}_{A.run}"
obj["name"] = name
jp = f"{TMPL}/{name}.json"
json.dump([obj], open(jp, "w"))
json.dump(cache, open(CACHE_F, "w"))
jobs = [{"run": A.run, "pdb": A.pdb, "json": jp, "name": name,
         "donor_cifs": [d["cif"] for d in donors], "tids": [d["tid"] for d in donors],
         "seeds": list(A.seeds)}]
json.dump(jobs, open(f"{EXP}/artifacts/{A.run}_jobs.json", "w"), indent=1)

for line in info:
    print("  " + line)
assert n_hits_total > 0, "no templates attached -- would silently run UNTEMPLATED"
print(f"\nBUILT 1 job with {len(donors)} donors, {n_hits_total} template hits across {pi} chains")
print(f"  json  {jp}")
print(f"  seeds {A.seeds[0]}..{A.seeds[-1]} ({len(A.seeds)})")
