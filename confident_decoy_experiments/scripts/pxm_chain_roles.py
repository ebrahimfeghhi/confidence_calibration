#!/usr/bin/env python
"""Which chain of each PXM-2025 interface is the ANTIBODY and which is the ANTIGEN?

WHY THIS SCRIPT EXISTS. The epitope is defined on the antigen side, so every downstream epitope
measurement needs the role. PXMeter's artifacts/pxm2025/interfaces.csv does NOT carry it: the chain
pair is stored ALPHABETICALLY SORTED (every id is A_B, A_C, A_E...), which destroys any ordering the
`[antibody_H-protein]` subset tag might have had. Verified empirically -- taking chain_id_1 as the
antibody agrees with the shared-chain rule 52 times and disagrees 93 times, i.e. worse than chance.
This is not an oversight by PXMeter: DockQ is symmetric over the interface, so a scoring benchmark
never needs the role. We are asking their file a question it was not built to answer.

Getting it backwards is SILENT: the epitope script still runs and still emits a Jaccard, it just
measures the PARATOPE instead. Since epitope change explains r^2 = 0.82 of DockQ gain and paratope
change explains ~0.006 uniquely, a flip yields a plausible-looking but meaningless replication. Same
bug class as the FoldBench chain-label error that hit 45% of interfaces (see scripts/iface_chains.py).

THREE INDEPENDENT SIGNALS.

  ig      hmmscan of the chain's full entity sequence against ANARCI's Ig V-domain profile library
          (dat/HMMs/ALL.hmm: per-species VH/VK/VL/VA profiles). A chain that hits is an
          immunoglobulin variable domain. This is the strongest and most objective signal -- it reads
          the sequence, not a human's free-text annotation -- and it is the primary authority here.

          NOTE we call hmmscan DIRECTLY rather than going through the anarci python package. ANARCI is
          installed but its result parsing is broken against the HMMER 3.4 in PATH: run_anarci returns
          None for a textbook VH that hmmscan itself matches at bitscore 190 / E-value 2e-59. The HMM
          library is fine; only the wrapper is broken. So we use the library and skip the wrapper.

          FALSE-POSITIVE RISK is the immunoglobulin SUPERFAMILY -- Nectin-4, CD antigens, MHC and T-cell
          receptors share the V-set fold and are genuine ANTIGENS here. Real V domains score ~150-200;
          the cutoff is IG_BITS and every score is written out so borderline calls surface rather than
          hide.

  desc    _entity.pdbx_description keyword match ('NB10 Nanobody', 'mAb 2E9 Fab heavy chain'). Strong
          but it is depositor free text, so incomplete and unsafe alone: 'light chain' also occurs in
          Myosin/Ferritin/Clathrin light chain, which are ANTIGENS. ANTIGEN_TRAPS vetoes those.

  shared  within one complex, a chain in MORE THAN ONE ab-ag interface row is the antigen (a Fab's H
          and L both contact one antigen). DEMOTED TO A DIAGNOSTIC, NEVER A VOTE, on the evidence
          below. It inverts whenever the ANTIGEN is a multi-chain assembly, because then it is the
          antibody that contacts several antigen chains and so becomes the shared one. That is not
          rare here: it accounts for ALL 17 rows where `shared` contradicted `ig`+`desc` --
          8w5d/e/f/i/m/o/p/q/u/v (one Fab chain touching 2-4 copies of 'Minor capsid protein A1'),
          9g4e/9g4f (Fab across both SbmA protomers), and 8yw9 (a nanobody across the pyruvate-carrier
          1+2 HETEROdimer, so it is multi-chain rather than strictly multi-copy). Verified by checking
          each shared chain's partners; the column is kept for inspection only.

Chain LENGTH is deliberately NOT a vote. Fab chains (~215) are longer than many real antigens (45-140
here), so 'shorter chain is the antibody' is wrong roughly as often as it is right. It is reported as
a column for eyeballing only.

RESOLUTION. `ig` decides whenever exactly one side is an Ig V-domain; `desc` must not contradict it.
`shared` is reported but does not vote (see above).

ON THE CUTOFF. The two populations do not merely separate, they are DISJOINT: over the 296 interfaces
resolved on a strict bitscore >= 80, the antibody side scores 144-203 (median 178) and the antigen
side scores exactly 0.0 -- every one. No immunoglobulin-superfamily antigen registered any hit at all,
Nectin-4 and the MHC/CD-type antigens included. So on this dataset any non-zero hit is diagnostic, and
the 80 cutoff was costing a real antibody: 9kqc chain B is a VNAR, the shark IgNAR variable domain,
which scores 54.8 against ALL.hmm's mammalian+camelid profiles -- low because it is evolutionarily
distant, not because it is doubtful. IG_BITS is therefore set at 25, with anything landing in
[25, 100) written as status 'ok_low_ig' so a weak hit is reviewed rather than silently accepted.

Usage: /home/ebrahim/Protenix/.venv/bin/python scripts/pxm_chain_roles.py [--jobs 8]
Outputs artifacts/pxm2025/chain_roles.csv; prints a summary and every unresolved case.
"""
import argparse
import collections
import csv
import os
import re
import subprocess
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
from Bio.PDB.MMCIF2Dict import MMCIF2Dict  # noqa: E402

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
OUT = f"{P}/chain_roles.csv"
HMM = ("/home/ebrahim/Protenix/.venv/lib/python3.12/site-packages/anarci/dat/HMMs/ALL.hmm")
IG_BITS = 25.0        # see "ON THE CUTOFF" above: antigen side scores 0.0 across the board
IG_LOW = 100.0        # a hit below this is real but weak (VNAR); flagged for review, not rejected

ANTIBODY_PAT = re.compile(
    r"\b(nanobod\w*|vhh|vnar|sdab|single[- ]domain antibod\w*|megabod\w*|sybod\w*|"
    r"fab\w*|scfv|\bfv\b|diabod\w*|minibod\w*|hcab\w*|"
    r"antibod\w*|immunoglobulin|\bigg\w*|\bige\b|\biga\b|\bigm\b|\bnb[- ]?\w*\d|"
    r"heavy chain|light chain|kappa chain|lambda chain|\bmab\b)\b")
ANTIGEN_TRAPS = re.compile(
    r"\b(myosin|ferritin|clathrin|neurofilament|dynein|kinesin|spectrin|"
    r"tubulin|titin|troponin|myelin|hemoglobin)\b")


def chain_info(pdb):
    """label_asym_id -> (entity description, canonical one-letter sequence)."""
    d = MMCIF2Dict(f"{P}/cif/{pdb}.cif")
    desc = dict(zip(d["_entity.id"], d["_entity.pdbx_description"]))
    seq = {e: s.replace("\n", "").replace(" ", "")
           for e, s in zip(d["_entity_poly.entity_id"],
                           d["_entity_poly.pdbx_seq_one_letter_code_can"])}
    return {a: (desc.get(e, ""), seq.get(e, ""))
            for a, e in zip(d["_struct_asym.id"], d["_struct_asym.entity_id"])}


def ig_scores(seqs, jobs):
    """{key: best Ig V-domain bitscore}. One hmmscan over every unique sequence."""
    if not seqs:
        return {}
    with tempfile.TemporaryDirectory() as td:
        fa, dt = f"{td}/q.fa", f"{td}/d.txt"
        with open(fa, "w") as fh:
            for k, s in seqs.items():
                fh.write(f">{k}\n{s}\n")
        subprocess.run(["hmmscan", "--cpu", str(jobs), "--domtblout", dt, HMM, fa],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        best = collections.defaultdict(float)
        for line in open(dt):
            if line.startswith("#"):
                continue
            f = line.split()
            # domtblout: 0 target, 3 query, 13 domain bitscore
            best[f[3]] = max(best[f[3]], float(f[13]))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    A = ap.parse_args()

    rows = list(csv.DictReader(open(f"{P}/interfaces.csv")))

    info, missing = {}, set()
    for r in rows:
        pdb = r["entry_id"]
        if pdb in info or pdb in missing:
            continue
        if os.path.exists(f"{P}/cif/{pdb}.cif"):
            info[pdb] = chain_info(pdb)
        else:
            missing.add(pdb)
    if missing:
        print(f"WARNING no reference cif for {len(missing)} complexes: "
              f"{','.join(sorted(missing))}", file=sys.stderr)

    # one hmmscan for every (pdb, chain) that appears in an interface row
    need = {}
    for r in rows:
        for c in (r["chain_id_1"], r["chain_id_2"]):
            k = f'{r["entry_id"]}|{c}'
            s = info.get(r["entry_id"], {}).get(c, ("", ""))[1]
            if s:
                need[k] = s
    print(f"hmmscan: {len(need)} chains vs ANARCI Ig V-domain profiles ...", file=sys.stderr)
    bits = ig_scores(need, A.jobs)

    seen = collections.Counter()
    for r in rows:
        seen[(r["entry_id"], r["chain_id_1"])] += 1
        seen[(r["entry_id"], r["chain_id_2"])] += 1

    out = []
    for r in rows:
        pdb, c1, c2 = r["entry_id"], r["chain_id_1"], r["chain_id_2"]
        d1, _ = info.get(pdb, {}).get(c1, ("", ""))
        d2, _ = info.get(pdb, {}).get(c2, ("", ""))
        b1, b2 = bits.get(f"{pdb}|{c1}", 0.0), bits.get(f"{pdb}|{c2}", 0.0)

        ig1, ig2 = b1 >= IG_BITS, b2 >= IG_BITS
        v_ig = "c1" if (ig1 and not ig2) else ("c2" if (ig2 and not ig1) else None)

        def is_ab(t):
            t = t.lower()
            return bool(ANTIBODY_PAT.search(t)) and not ANTIGEN_TRAPS.search(t)
        a1, a2 = is_ab(d1), is_ab(d2)
        v_desc = "c1" if (a1 and not a2) else ("c2" if (a2 and not a1) else None)

        n1, n2 = seen[(pdb, c1)], seen[(pdb, c2)]
        v_shared = "c2" if (n1 > 1 and n2 == 1) else ("c1" if (n2 > 1 and n1 == 1) else None)

        # `shared` is a diagnostic only; `desc` is the one cross-check that can block.
        if v_ig is None:
            status = ("AMB_both_ig" if (ig1 and ig2) else "AMB_no_ig")
            role = None
        elif v_desc and v_desc != v_ig:
            status, role = "AMB_conflict", None
        else:
            role = v_ig
            status = ("ok_low_ig" if max(b1, b2) < IG_LOW else "ok")

        out.append(dict(
            interface=r["id"], pdb=pdb, chain_1=c1, chain_2=c2,
            ab_chain=("" if role is None else (c1 if role == "c1" else c2)),
            ag_chain=("" if role is None else (c2 if role == "c1" else c1)),
            status=status,
            v_ig=v_ig or "", v_desc=v_desc or "", v_shared=v_shared or "",
            ig_bits_1=round(b1, 1), ig_bits_2=round(b2, 1),
            len_1=r["resolved_seq_length_1"], len_2=r["resolved_seq_length_2"],
            subset=r["subset"], desc_1=d1, desc_2=d2))

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    ok = [o for o in out if o["status"].startswith("ok")]
    bad = [o for o in out if not o["status"].startswith("ok")]
    print(f"\n{len(out)} interfaces: {len(ok)} resolved, {len(bad)} unresolved")
    for k, v in collections.Counter(o["status"] for o in ok).most_common():
        print(f"  {v:>4}  {k}")
    print(f"\ncross-checks on the {len(ok)} resolved rows:")
    d = [o for o in ok if o["v_desc"]]
    dn = sum(o["v_desc"] != o["v_ig"] for o in d)
    s = [o for o in ok if o["v_shared"]]
    sn = sum(o["v_shared"] != o["v_ig"] for o in s)
    print(f"  entity description   fires on {len(d):>3}/{len(ok)}, disagrees with Ig on {dn}")
    print(f"  shared-chain rule    fires on {len(s):>3}/{len(ok)}, disagrees with Ig on {sn}"
          f"   (diagnostic only -- multi-chain antigens invert it)")
    b = [o["ig_bits_1"] if o["ab_chain"] == o["chain_1"] else o["ig_bits_2"] for o in ok]
    ag = [o["ig_bits_2"] if o["ab_chain"] == o["chain_1"] else o["ig_bits_1"] for o in ok]
    print(f"\n  Ig bitscore, antibody side: min {min(b):.0f}  median {sorted(b)[len(b)//2]:.0f}"
          f"  max {max(b):.0f}")
    print(f"  Ig bitscore, antigen  side: min {min(ag):.0f}  median {sorted(ag)[len(ag)//2]:.0f}"
          f"  max {max(ag):.0f}   (cutoff {IG_BITS:.0f})")
    for k, v in collections.Counter(o["status"] for o in bad).most_common():
        print(f"\n--- {v} rows: {k} ---")
        for o in bad:
            if o["status"] != k:
                continue
            print(f"  {o['interface']:<12} ig[{o['ig_bits_1']:>6.1f}/{o['ig_bits_2']:>6.1f}] "
                  f"desc={o['v_desc'] or '-':<3} shared={o['v_shared'] or '-':<3}  "
                  f"{o['chain_1']}={o['len_1']:>4} {o['desc_1'][:38]!r} | "
                  f"{o['chain_2']}={o['len_2']:>4} {o['desc_2'][:38]!r}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
