#!/usr/bin/env python
"""Localization check: is the near-native antigen fold a GLOBAL structural outlier
(separable by structural NMS / leader-compression) or a LOCAL epitope-loop variant
of the wrong folds (not separable)?

For each antigen chain, compares three folds over the canonical pool (seeds 101-150, 250 poses):
  - ORACLE   = best lDDT-to-native antigen fold (cached in oracle_folds.json)
  - PLDDT    = highest mean-pLDDT antigen fold (what the naive selector picks)
  - NATIVE   = ground-truth antigen chain
Metrics (OST --local-lddt, official per-residue lDDT):
  - global lddt(oracle,native), lddt(plddt,native)  -> the selection gap
  - global lddt(oracle, plddt)                        -> NMS separability
       HIGH  (~>0.85) => folds globally near-identical => NOT an outlier => leaders CANNOT separate
       LOW           => globally distinct            => outlier          => leaders CAN separate
  - per-residue lddt(oracle vs plddt) split epitope / non-epitope
       difference concentrated at epitope => LOCAL variant; spread => GLOBAL difference
Run with the Protenix venv python (has gemmi). OST for lddt.
"""
import gemmi, subprocess, os, json, glob, statistics, difflib, sys
from concurrent.futures import ThreadPoolExecutor

OST  = "/home/ebrahim/miniconda3/envs/foldbench/bin/ost"
PRED = "/home/ebrahim/Protenix/foldbench_abag/predictions_template"
GT   = "/home/ebrahim/FoldBench/examples/ground_truths"
EXP  = "/home/ebrahim/confident_decoy_experiments"
OUT  = f"{EXP}/artifacts/antigen_localization"
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/chains", exist_ok=True); os.makedirs(f"{OUT}/ost", exist_ok=True)
ENV = dict(os.environ); ENV["LD_LIBRARY_PATH"] = "/home/ebrahim/miniconda3/envs/foldbench/lib"

meta      = json.load(open(f"{EXP}/artifacts/meta.json"))
agidx     = json.load(open(f"{EXP}/artifacts/agonly_antigen_idx.json"))
oracle    = json.load(open(f"{EXP}/artifacts/oracle_folds.json"))
chainmap  = json.load(open(f"{EXP}/artifacts/oracle_chainmap.json"))   # pdb -> {qidx: native_letter}

TLC = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
       'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def seqof(ch): return "".join(TLC[r.name] for r in ch if r.name in TLC)

def write_chain(ch, path):
    st = gemmi.Structure(); st.add_model(gemmi.Model("1")); st[0].add_chain(ch); st.setup_entities()
    st.write_pdb(path)

def ost_lddt(m, r, tag, local=False):
    o = f"{OUT}/ost/{tag}.json"
    cmd = ["taskset","-c","0-31",OST,"compare-structures","-m",m,"-r",r,"-o",o,
           "--lddt","--fault-tolerant","--min-pep-length","4"]
    if local: cmd.append("--local-lddt")
    if not (os.path.exists(o) and os.path.getsize(o) > 0):
        subprocess.run(cmd, env=ENV, capture_output=True, timeout=180)
    try:
        d = json.load(open(o)); return d.get("lddt"), d.get("local_lddt")
    except Exception:
        return None, None

def antigen_seq_native(pdb, letter):
    st = gemmi.read_structure(f"{GT}/{pdb}-assembly1.cif"); st.setup_entities()
    for ch in st[0]:
        if ch.name == letter:
            return seqof(ch)
    return None

def epitope_native_resnums(pdb, ag_letter, ab_letters, cutoff=5.0):
    """Native antigen residue numbers with any heavy atom within cutoff of any antibody heavy atom."""
    st = gemmi.read_structure(f"{GT}/{pdb}-assembly1.cif"); st.setup_entities()
    model = st[0]
    ag = next((c for c in model if c.name == ag_letter), None)
    if ag is None: return set(), 0
    ns = gemmi.NeighborSearch(model, st.cell, cutoff).populate()
    epi = set()
    for res in ag:
        for at in res:
            if at.element == gemmi.Element('H'): continue
            marks = ns.find_atoms(at.pos, '\0', radius=cutoff)
            for m in marks:
                cra = m.to_cra(model)
                if cra.chain.name in ab_letters and cra.atom.element != gemmi.Element('H'):
                    epi.add(res.seqid.num); break
            if res.seqid.num in epi: break
    nag = sum(1 for _ in ag)
    return epi, nag

def poses(pdb):
    out = []
    for s in range(101, 151):
        out += sorted(glob.glob(f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_*.cif"))
    return out

def best_match_chain(st, target):
    best, br = None, -1
    for ch in st[0]:
        s = seqof(ch)
        if len(s) < 4: continue
        r = difflib.SequenceMatcher(None, s, target, autojunk=False).ratio()
        if r > br: br, best = r, ch
    return best

def plddt_of_pose(args):
    cif, target = args
    try:
        st = gemmi.read_structure(cif); st.setup_entities()
    except Exception:
        return None
    ch = best_match_chain(st, target)
    if ch is None: return None
    pl = statistics.mean([at.b_iso for r in ch for at in r]) / 100.0
    return (pl, cif, ch.name)

results = []
for ifc, d in meta.items():
    pdb = d['pdb']
    for qi in agidx.get(pdb, []):
        qi = str(qi)
        tag = f"{pdb}_q{qi}"
        if any(r['tag'] == tag for r in results):   # dedup per pdb-antigen
            continue
        ag_letter = chainmap[pdb][qi]
        ab_letters = set(v for k, v in chainmap[pdb].items() if k != qi)
        nat_seq = antigen_seq_native(pdb, ag_letter)
        nat_ref = f"/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl/native_chains/{pdb}_{ag_letter}.pdb"
        if not os.path.exists(nat_ref):
            st = gemmi.read_structure(f"{GT}/{pdb}-assembly1.cif"); st.setup_entities()
            ch = next(c for c in st[0] if c.name == ag_letter)
            nat_ref = f"{OUT}/chains/{pdb}_{ag_letter}_native.pdb"; write_chain(ch, nat_ref)
        nag = len(nat_seq)
        is_peptide = nag < 30
        epi, _ = epitope_native_resnums(pdb, ag_letter, ab_letters)

        # ORACLE fold
        orc = oracle[pdb][qi]
        st = gemmi.read_structure(orc['cif']); st.setup_entities()
        orc_ch = next(c for c in st[0] if c.name == orc['model_chain'])
        orc_pdb = f"{OUT}/chains/{tag}_oracle.pdb"; write_chain(orc_ch, orc_pdb)

        # PLDDT-pick fold over the pool
        pl_list = [x for x in ThreadPoolExecutor(8).map(plddt_of_pose, [(c, nat_seq) for c in poses(pdb)]) if x]
        pl_list.sort(reverse=True)
        best_pl, best_cif, best_chname = pl_list[0]
        st = gemmi.read_structure(best_cif); st.setup_entities()
        pl_ch = next(c for c in st[0] if c.name == best_chname)
        pl_pdb = f"{OUT}/chains/{tag}_plddt.pdb"; write_chain(pl_ch, pl_pdb)

        # OST comparisons
        l_orc_nat, _ = ost_lddt(orc_pdb, nat_ref, f"{tag}_orc_nat")
        l_pl_nat,  _ = ost_lddt(pl_pdb,  nat_ref, f"{tag}_pl_nat")
        l_orc_pl, loc = ost_lddt(orc_pdb, pl_pdb, f"{tag}_orc_pl", local=True)  # NMS separability

        # localize oracle-vs-plddt difference at epitope (map model resnum -> native resnum -> epitope)
        epi_frac = non_frac = epi_n = non_n = None
        if loc and not is_peptide:
            # align oracle-model seq to native seq to map model resnum -> native resnum
            orc_seq = seqof(orc_ch)
            orc_resnums = [r.seqid.num for r in orc_ch if r.name in TLC]
            nat_resnums_full = None
            stn = gemmi.read_structure(f"{GT}/{pdb}-assembly1.cif"); stn.setup_entities()
            natch = next(c for c in stn[0] if c.name == ag_letter)
            nat_resnums_full = [r.seqid.num for r in natch if r.name in TLC]
            sm = difflib.SequenceMatcher(None, orc_seq, nat_seq, autojunk=False)
            model2nat = {}
            for a, b, n in sm.get_matching_blocks():
                for k in range(n):
                    model2nat[orc_resnums[a+k]] = nat_resnums_full[b+k]
            epi_scores, non_scores = [], []
            for key, val in loc.items():
                if val is None: continue
                parts = key.split('.')
                try: rn = int(parts[1])
                except Exception: continue
                natrn = model2nat.get(rn)
                if natrn is None: continue
                (epi_scores if natrn in epi else non_scores).append(val)
            if epi_scores: epi_frac = round(statistics.mean(epi_scores), 3); epi_n = len(epi_scores)
            if non_scores: non_frac = round(statistics.mean(non_scores), 3); non_n = len(non_scores)

        row = {
            "tag": tag, "pdb": pdb, "antigen_qidx": qi, "native_chain": ag_letter,
            "n_residues_native": nag, "is_peptide": is_peptide, "n_epitope": len(epi),
            "lddt_oracle_native": l_orc_nat, "lddt_plddt_native": l_pl_nat,
            "selection_gap": (round(l_orc_nat - l_pl_nat, 3) if (l_orc_nat and l_pl_nat) else None),
            "lddt_oracle_vs_plddt": l_orc_pl,          # NMS separability: high => not separable
            "plddt_pick_pose": os.path.relpath(best_cif, PRED), "plddt_pick_meanpLDDT": round(best_pl, 3),
            "orcVSpl_lddt_epitope": epi_frac, "orcVSpl_lddt_nonepitope": non_frac,
            "orcVSpl_epi_n": epi_n, "orcVSpl_non_n": non_n,
        }
        results.append(row)
        print(f"{tag:12s} nres={nag:4d} pep={int(is_peptide)} | orc/nat={l_orc_nat} pl/nat={l_pl_nat} gap={row['selection_gap']} | orcVSpl={l_orc_pl} | epi={epi_frac} non={non_frac}", flush=True)

json.dump(results, open(f"{OUT}/localization_results.json", "w"), indent=1)
print("\nWROTE", f"{OUT}/localization_results.json")
