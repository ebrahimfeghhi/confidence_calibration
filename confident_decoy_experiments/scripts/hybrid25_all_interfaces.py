#!/usr/bin/env python
"""Re-extract hybrid25 results for EVERY antibody-antigen interface, not just one per complex.

The workers recorded only the interface named in units.csv, which was one representative per complex.
That discards real data: PXM-2025 has 89 confident-decoy interfaces across 45 complexes, and a run
predicts the whole assembly, so every interface of that assembly is already scored in the PXMeter
JSONs sitting on disk. This re-reads those JSONs -- no re-scoring, no GPU -- and emits one row per
(interface, sample).

WHICH PAIRS COUNT AS ab-ag comes from artifacts/pxm2025/chain_roles.csv, the resolved table built by
hmmscan against the Ig HMMs. It is keyed on REFERENCE chain names, which is what PXMeter's interface
keys use -- model chain letters differ and must not be substituted (45% of interfaces in this project
were once measured on the wrong chains by making exactly that mistake).

ipTM needs the opposite direction: chain_pair_iptm is indexed by MODEL asym order, so each pose's own
ref_to_model_chain_mapping is required to line the two up.

Usage: hybrid25_all_interfaces.py [--out artifacts/hybrid25/results_all.csv]
"""
import argparse
import collections
import csv
import glob
import json
import os
import re

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
M = f"{EXP}/artifacts/hybrid25"


def chain_idx(name):
    v = 0
    for c in name.split("-")[0]:
        v = v * 26 + (ord(c.upper()) - 64)
    return v - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="",
                    help="batch root: artifacts/hybrid25 (A) or artifacts/hybrid25b (B). Batch B "
                         "templates from donor seeds 300025-300049 under run seeds 300050-300074.")
    ap.add_argument("--out", default="")
    A = ap.parse_args()
    root = A.root or M
    A.out = A.out or f"{root}/results_all.csv"

    # ab-ag reference chain pairs per complex
    abag = collections.defaultdict(set)
    for r in csv.DictReader(open(f"{P}/chain_roles.csv")):
        if r["status"] == "ok":
            abag[r["pdb"]].add(frozenset((r["chain_1"], r["chain_2"])))

    # class per interface, from the regenerated protocol
    cls = {}
    for r in csv.DictReader(open(f"{P}/pxm2025_selection_protocol.csv")):
        try:
            s, o = float(r["sel50_dockq"]), float(r["oracle_dockq"])
        except (TypeError, ValueError):
            continue
        cls[r["interface"]] = "decoy" if s < 0.23 <= o else ("nocov" if o < 0.23 else "solved")

    rows, seen_units, skipped = [], set(), collections.Counter()
    for jf in glob.glob(f"{root}/*/s*/pxm_json/*.json"):
        m = re.search(rf"{root}/([^/]+)/s(\d+)/pxm_json/(\d+)\.json$", jf)
        if not m:
            continue
        pdb, dseed, samp = m.group(1), int(m.group(2)), m.group(3)
        try:
            J = json.load(open(jf))
        except Exception:
            skipped["unreadable"] += 1
            continue
        r2m = J.get("ref_to_model_chain_mapping", {})
        conf = None
        cf = glob.glob(f"{root}/{pdb}/s{dseed}/out/**/*summary_confidence_sample_{samp}.json",
                       recursive=True)
        if cf:
            try:
                conf = json.load(open(cf[0])).get("chain_pair_iptm")
            except Exception:
                conf = None
        for k, v in (J.get("interface") or {}).items():
            if v.get("dockq") is None:
                continue
            c1, c2 = k.split(",")
            if frozenset((c1, c2)) not in abag.get(pdb, ()):
                continue                       # not an antibody-antigen interface
            iface = f"{pdb}_{''.join(sorted((c1, c2)))}"
            ip = ""
            if conf and c1 in r2m and c2 in r2m:
                i, j = chain_idx(r2m[c1]), chain_idx(r2m[c2])
                if 0 <= i < len(conf) and 0 <= j < len(conf[i]):
                    ip = f"{float(conf[i][j]):.4f}"
            rows.append(dict(pdb=pdb, interface=iface, cls=cls.get(iface, "?"),
                             donor_seed=dseed, new_seed=dseed + 25, sample=samp,
                             dockq=f"{float(v['dockq']):.4f}", iptm=ip))
        seen_units.add((pdb, dseed))

    with open(A.out, "w", newline="\n") as fh:
        w = csv.DictWriter(fh, fieldnames=["pdb", "interface", "cls", "donor_seed", "new_seed",
                                           "sample", "dockq", "iptm"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    ifaces = {r["interface"] for r in rows}
    byc = collections.Counter(r["cls"] for r in rows)
    print(f"units read: {len(seen_units)}   rows: {len(rows)}   interfaces: {len(ifaces)}")
    print(f"   complexes: {len({r['pdb'] for r in rows})}")
    print(f"   rows by class: {dict(byc)}")
    print(f"   interfaces by class: "
          f"{dict(collections.Counter(cls.get(i,'?') for i in ifaces))}")
    if skipped:
        print(f"   skipped: {dict(skipped)}")
    print(f"wrote {A.out}")


if __name__ == "__main__":
    main()
