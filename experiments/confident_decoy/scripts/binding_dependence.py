#!/usr/bin/env python
"""BINDING-DEPENDENCE ASSAY — how much is each native fold shaped by its binding partner?

For every chain (antigen AND antibody heavy/light), compares the best fold the model produces
WITHOUT the partner (isolation run) vs WITH the partner (free complex run), scored vs native:
  - GLOBAL   chain lDDT
  - INTERFACE lDDT: epitope (antigen residues <5A of any antibody chain in the NATIVE complex)
                    paratope (antibody-chain residues <5A of the antigen in the NATIVE complex)
binding_dependence = complex_best - isolation_best.  ~0 => native fold is partner-independent
(rigid; native template wins on ACCURACY). Large + => the native conformation only forms against
the partner (holo/induced-fit; native template leaks bound-state geometry no apo predictor can match).
Reporting interface lDDT separately tests whether binding-dependence is localized to the interface.

Chain roles from corrected artifacts/agonly_antigen_idx.json + oracle_chainmap.json.
Run with Protenix venv python (gemmi). OST for lDDT (--local-lddt gives per-residue).
"""
import gemmi, subprocess, os, json, glob, statistics, difflib, sys
from concurrent.futures import ThreadPoolExecutor

EXP  = "/home/ebrahim/confident_decoy_experiments"
PRED = "/home/ebrahim/Protenix/foldbench_abag/predictions_template"
JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
GT   = "/home/ebrahim/FoldBench/examples/ground_truths"
OLD  = "/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl"
ISO  = f"{OLD}/mono50_out"; ISOJ = f"{OLD}/mono50_json"; NATD = f"{OLD}/native_chains"
OST  = "/home/ebrahim/miniconda3/envs/foldbench/bin/ost"
OUT  = f"{EXP}/artifacts/binding_dependence"
os.makedirs(f"{OUT}/tmp", exist_ok=True); os.makedirs(f"{OUT}/ost", exist_ok=True)
ENV  = dict(os.environ); ENV["LD_LIBRARY_PATH"] = "/home/ebrahim/miniconda3/envs/foldbench/lib"

agidx    = json.load(open(f"{EXP}/artifacts/agonly_antigen_idx.json"))     # corrected
chainmap = json.load(open(f"{EXP}/artifacts/oracle_chainmap.json"))        # qidx -> native letter
TLC = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
       'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def std(ch): return [r for r in ch if r.name in TLC]
def seqof(rl): return "".join(TLC[r.name] for r in rl)

def best_match_chain(st, target):
    best, br = None, -1
    for ch in st[0]:
        s = seqof(std(ch))
        if len(s) < 4: continue
        r = difflib.SequenceMatcher(None, s, target, autojunk=False).ratio()
        if r > br: br, best = r, ch
    return (best, br)

_unit_cache = {}
def resolve_unit(pdb, qseqs, aidx, cutoff=5.0):
    """Copy-aware chain assignment. Native assemblies often contain MULTIPLE copies of the Fab and
    antigen (e.g. 7z2m: Fabs A+B and E+F, antigens J and K), and the stored chain_map can mix copies
    or point at a crystal contact (7z2m's labelled A-J interface is only 3 residues; the real ones are
    A/B-K and E/F-J). Here we pick, from the sequence-matching candidates, the antigen copy with the
    most antibody contacts and then each antibody chain's copy that contacts THAT antigen most.
    Returns (qidx -> native_letter, qidx -> set(interface resnums))."""
    if pdb in _unit_cache: return _unit_cache[pdb]
    st = gemmi.read_structure(f"{GT}/{pdb}-assembly1.cif"); st.setup_entities()
    model = st[0]
    chains = {c.name: std(c) for c in model if len(std(c)) >= 8}
    ns = gemmi.NeighborSearch(model, st.cell, cutoff).populate()
    def contact_res(a, b):
        out = set()
        for res in chains.get(a, []):
            hit = False
            for at in res:
                if at.element == gemmi.Element('H'): continue
                for m in ns.find_atoms(at.pos, '\0', radius=cutoff):
                    cra = m.to_cra(model)
                    if cra.chain.name == b and cra.atom.element != gemmi.Element('H'): hit = True; break
                if hit: break
            if hit: out.add(res.seqid.num)
        return out
    # candidate native letters per model chain (by sequence identity)
    def match_score(nat, qs):
        """ratio, or containment for partially-resolved chains (peptide antigens are often only
        partly ordered in the crystal, e.g. 8che: 37-mer construct but 15 residues modelled)."""
        r = difflib.SequenceMatcher(None, nat, qs, autojunk=False).ratio()
        sm = difflib.SequenceMatcher(None, nat, qs, autojunk=False)
        m = sm.find_longest_match(0, len(nat), 0, len(qs))
        contain = m.size / min(len(nat), len(qs)) if min(len(nat), len(qs)) else 0.0
        return max(r, contain)
    cand = {}
    for qi, qs in enumerate(qseqs):
        c = []
        for nm, rl in chains.items():
            r = match_score(seqof(rl), qs)
            if r >= 0.90: c.append((r, nm))
        c.sort(reverse=True)
        cand[qi] = [nm for _, nm in c]
    ab_q = [q for q in range(len(qseqs)) if q not in aidx]
    ab_all = set(l for q in ab_q for l in cand.get(q, []))
    # 1) antigen copy with most antibody contacts
    letters = {}; ifaces = {}
    for gq in aidx:
        best_g, best_n = None, -1
        for g in cand.get(gq, []):
            n = sum(len(contact_res(g, a)) for a in ab_all)
            if n > best_n: best_n, best_g = n, g
        if best_g is None: continue
        letters[gq] = best_g
        ifaces[gq] = set().union(*[contact_res(best_g, a) for a in ab_all]) if ab_all else set()
    # 2) each antibody chain's copy that contacts the chosen antigen(s) most
    ag_letters = set(letters.values())
    for aq in ab_q:
        best_a, best_n = None, -1
        for a in cand.get(aq, []):
            n = sum(len(contact_res(a, g)) for g in ag_letters)
            if n > best_n: best_n, best_a = n, a
        if best_a is None: continue
        letters[aq] = best_a
        ifaces[aq] = set().union(*[contact_res(best_a, g) for g in ag_letters]) if ag_letters else set()
    _unit_cache[pdb] = (letters, ifaces)
    return letters, ifaces

def interface_resnums(pdb, this_letter, partner_letters, cutoff=5.0):
    """Native residues of `this_letter` with a heavy atom within cutoff of any partner chain."""
    st = gemmi.read_structure(f"{GT}/{pdb}-assembly1.cif"); st.setup_entities()
    model = st[0]
    me = next((c for c in model if c.name == this_letter), None)
    if me is None: return set()
    ns = gemmi.NeighborSearch(model, st.cell, cutoff).populate()
    hit = set()
    for res in me:
        done = False
        for at in res:
            if at.element == gemmi.Element('H'): continue
            for m in ns.find_atoms(at.pos, '\0', radius=cutoff):
                cra = m.to_cra(model)
                if cra.chain.name in partner_letters and cra.atom.element != gemmi.Element('H'):
                    hit.add(res.seqid.num); done = True; break
            if done: break
    return hit

def native_ref(pdb, letter):
    p = f"{NATD}/{pdb}_{letter}.pdb"
    if os.path.exists(p): return p
    st = gemmi.read_structure(f"{GT}/{pdb}-assembly1.cif"); st.setup_entities()
    ch = next(c for c in st[0] if c.name == letter)
    p = f"{OUT}/tmp/{pdb}_{letter}_native.pdb"
    ns = gemmi.Structure(); ns.add_model(gemmi.Model("1")); ns[0].add_chain(ch); ns.setup_entities(); ns.write_pdb(p)
    return p

def native_seq(pdb, letter):
    st = gemmi.read_structure(f"{GT}/{pdb}-assembly1.cif"); st.setup_entities()
    ch = next((c for c in st[0] if c.name == letter), None)
    return seqof(std(ch)) if ch else None

def score_pose(a):
    cif, qseq, natref, natseq, iface_nat, tag = a
    try: st = gemmi.read_structure(cif); st.setup_entities()
    except Exception: return None
    ch, ratio = best_match_chain(st, qseq)
    if ch is None or ratio < 0.5: return None
    mres = std(ch); mseq = seqof(mres)
    mp = f"{OUT}/tmp/{tag}.pdb"
    ns = gemmi.Structure(); ns.add_model(gemmi.Model("1")); ns[0].add_chain(ch); ns.setup_entities(); ns.write_pdb(mp)
    o = f"{OUT}/ost/{tag}.json"
    if not (os.path.exists(o) and os.path.getsize(o) > 0):
        subprocess.run(["taskset","-c","0-31",OST,"compare-structures","-m",mp,"-r",natref,"-o",o,
                        "--lddt","--local-lddt","--fault-tolerant","--min-pep-length","4"],
                       env=ENV, capture_output=True, timeout=180)
    try: os.remove(mp)
    except Exception: pass
    try:
        d = json.load(open(o)); g = d.get("lddt"); loc = d.get("local_lddt")
    except Exception: return None
    if g is None: return None
    ifl = None
    if loc and iface_nat:
        sm = difflib.SequenceMatcher(None, mseq, natseq, autojunk=False)
        mresnums = [r.seqid.num for r in mres]
        natst = gemmi.read_structure(natref); natst.setup_entities()
        natres = std(natst[0][0]); natnums = [r.seqid.num for r in natres]
        m2n = {}
        for i, j, n in sm.get_matching_blocks():
            for k in range(n):
                if i+k < len(mresnums) and j+k < len(natnums): m2n[mresnums[i+k]] = natnums[j+k]
        vals = []
        for key, v in loc.items():
            if v is None: continue
            try: rn = int(key.split('.')[1])
            except Exception: continue
            if m2n.get(rn) in iface_nat: vals.append(v)
        if vals: ifl = statistics.mean(vals)
    return {"global": g, "iface": ifl}

# map isolation target -> its sequences
isotargets = {}
for f in glob.glob(f"{ISOJ}/*.json"):
    name = os.path.basename(f)[:-5]
    try:
        d = json.load(open(f)); o = d[0] if isinstance(d, list) else d
        isotargets[name] = [e["proteinChain"]["sequence"] for e in o["sequences"] if "proteinChain" in e]
    except Exception: pass

def iso_poses_for(pdb, qseq):
    """isolation target containing this chain (mono_q* for single, abpair for Fab H+L)."""
    best, br = None, -1
    for name, seqs in isotargets.items():
        if not name.startswith(pdb + "_"): continue
        for s in seqs:
            r = difflib.SequenceMatcher(None, s, qseq, autojunk=False).ratio()
            if r > br: br, best = r, name
    if best is None or br < 0.9: return [], None
    # seed-match the complex pool: cap at 50 seeds (isolation runs have 25-50 seeds/chain)
    seeds = sorted(int(os.path.basename(d).split('_')[1]) for d in glob.glob(f"{ISO}/{best}/seed_*"))[:50]
    out = []
    for s in seeds:
        out += sorted(glob.glob(f"{ISO}/{best}/seed_{s}/predictions/{best}_sample_*.cif"))
    return out, f"{best}(seeds={len(seeds)})"

def cplx_poses(pdb):
    out = []
    for s in range(101, 151): out += sorted(glob.glob(f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_*.cif"))
    return out

rows = []
for pdb in sorted(agidx):
    aidx = [int(x) for x in agidx[pdb]]
    d = json.load(open(f"{JDIR}/{pdb}-final-updated.json")); o = d[0] if isinstance(d, list) else d
    qseqs = [e["proteinChain"]["sequence"] for e in o["sequences"] if "proteinChain" in e]
    unit_letters, unit_ifaces = resolve_unit(pdb, qseqs, aidx)
    for qi, qseq in enumerate(qseqs):
        letter = unit_letters.get(qi)
        if letter is None: continue
        is_ag = qi in aidx
        role = "ANTIGEN" if is_ag else "ANTIBODY"
        iface = unit_ifaces.get(qi, set())
        natref = native_ref(pdb, letter); natseq = native_seq(pdb, letter)
        if not natseq: continue
        ipose, itgt = iso_poses_for(pdb, qseq)
        cpose = cplx_poses(pdb)
        res = {}
        for label, plist in (("iso", ipose), ("cplx", cpose)):
            if not plist: res[label] = None; continue
            args = [(c, qseq, natref, natseq, iface, f"{pdb}_q{qi}_{letter}_{label}_{i}") for i, c in enumerate(plist)]
            sc = [x for x in ThreadPoolExecutor(6).map(score_pose, args) if x]
            if not sc: res[label] = None; continue
            res[label] = {"n": len(sc), "best_global": max(x["global"] for x in sc),
                          "best_iface": max([x["iface"] for x in sc if x["iface"] is not None], default=None)}
        if not res.get("iso") or not res.get("cplx"):
            print(f"  {pdb} q{qi} {role}: missing pool (iso={bool(res.get('iso'))} cplx={bool(res.get('cplx'))})", flush=True)
            continue
        dg = res["cplx"]["best_global"] - res["iso"]["best_global"]
        di = (res["cplx"]["best_iface"] - res["iso"]["best_iface"]) if (res["cplx"]["best_iface"] is not None and res["iso"]["best_iface"] is not None) else None
        row = {"pdb": pdb, "qidx": qi, "role": role, "native_chain": letter, "chain_len": len(qseq),
               "n_iface_res": len(iface), "iso_target": itgt,
               "iso_n": res["iso"]["n"], "cplx_n": res["cplx"]["n"],
               "iso_best_global": round(res["iso"]["best_global"], 3), "cplx_best_global": round(res["cplx"]["best_global"], 3),
               "binddep_global": round(dg, 3),
               "iso_best_iface": (round(res["iso"]["best_iface"], 3) if res["iso"]["best_iface"] is not None else None),
               "cplx_best_iface": (round(res["cplx"]["best_iface"], 3) if res["cplx"]["best_iface"] is not None else None),
               "binddep_iface": (round(di, 3) if di is not None else None)}
        rows.append(row)
        print(f"{pdb} q{qi} {role:8} ch{letter} len={len(qseq):4d} iface={len(iface):3d} | "
              f"GLOBAL iso={row['iso_best_global']:.3f} cplx={row['cplx_best_global']:.3f} d={row['binddep_global']:+.3f} | "
              f"IFACE iso={row['iso_best_iface']} cplx={row['cplx_best_iface']} d={row['binddep_iface']}", flush=True)
        json.dump(rows, open(f"{OUT}/binding_dependence_results.json", "w"), indent=1)

json.dump(rows, open(f"{OUT}/binding_dependence_results.json", "w"), indent=1)
print(f"\nWROTE {OUT}/binding_dependence_results.json  ({len(rows)} chains)")
