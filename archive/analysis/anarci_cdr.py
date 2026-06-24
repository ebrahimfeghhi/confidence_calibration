"""Annotate CDR residues (IMGT) for sequences, using a WORKING ANARCI install.

Run with an interpreter whose anarci works (e.g. the bioconda `anarci` env), with
that env's bin on PATH so hmmscan is found:
    {anarci_py} analysis/anarci_cdr.py SEQ1 [SEQ2 ...]

Prints one JSON line: {"0": [seq_idx,...], "1": [...]}  (key = position in argv list,
value = 0-based sequence indices that fall in a CDR). IMGT CDRs: 27-38, 56-65, 105-117.
"""
import json
import sys

from anarci import anarci

CDR = {1: range(27, 39), 2: range(56, 66), 3: range(105, 118)}


def main():
    seqs = sys.argv[1:]
    res = anarci([(f"s{i}", s) for i, s in enumerate(seqs)], scheme="imgt", output=False)
    numbering = res[0]
    out = {}
    for i, dom_list in enumerate(numbering):
        idxs = []
        if dom_list:
            for domnum, start, _end in dom_list:  # V (and maybe C) domains
                idx = start
                for (num, _ins), aa in domnum:
                    if aa == "-":
                        continue
                    if any(num in rg for rg in CDR.values()):
                        idxs.append(idx)
                    idx += 1
        out[str(i)] = sorted(set(idxs))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
