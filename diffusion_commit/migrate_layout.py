#!/usr/bin/env python
"""One-off: consolidate experimental_data into four top-level entries.

BEFORE -- five run trees, two of them free folds under different names, plus the pool's own folds
buried under the pool inputs:

    pxm2025/out/<pdb>/<pdb>/seed_<N>/predictions/    the original 50-75 seeds
    seed100/free/<pdb>/<seed>/<rid>/                 the 100-seed deepening
    freespread/free/<pdb>/<seed>/<rid>/              the fp32-vs-bf16 spread test
    st/self_template/<pdb>/<seed>/<rid>/             self-templating, fp32
    st_bf16/self_template/<pdb>/<seed>/<rid>/        self-templating, bf16

AFTER:

    pxm2025/            the pool: targets, ground truth, job jsons, MSAs, score tables
    free/               EVERY free fold, one results.csv
    self_template/      EVERY self-templated fold, one results.csv, plus templates/ (the namespace)
    archive/            everything else

WHY the precision folders disappear: st vs st_bf16 encoded precision in a directory name, but
run.json already records `precision` and the arm names carry it too (b1/b2/b3, freebf16r1,
freefp32r1). A fact stored in two places is a fact that can disagree with itself.

THE LEGACY MARKER. The pool's folds predate runmeta: no run id, no run.json, and their exact
configuration is not recoverable from disk (precision and recycle count are known only because
bf16 and N_cycle=10 were the defaults at the time). Rather than mint a plausible-looking run id for
them, they land under

    free/<pdb>/<seed>/legacy-no-provenance/

so the absence is visible in the path instead of disguised as a normal run.

    migrate_layout.py --root <experimental_data> [--apply] [--stage runs|archive|all]
"""
import argparse
import csv
import json
import os
import shutil

# Everything that is NOT archived. Anything else at the top level is a previous era.
KEEP = {"pxm2025", "free", "self_template", "archive"}

# tree -> (subdirectory holding the run dirs, destination)
RUN_TREES = [
    ("seed100", "free", "free"),
    ("freespread", "free", "free"),
    ("st", "self_template", "self_template"),
    ("st_bf16", "self_template", "self_template"),
]
CSV_MERGE = [
    (["seed100/results.csv", "freespread/results.csv"], "free/results.csv"),
    (["st/results.csv", "st_bf16/results.csv"], "self_template/results.csv"),
]
LEGACY = "legacy-no-provenance"


def plan_runs(root):
    moves = []
    # 1. the pool's own folds -> free/<pdb>/<seed>/legacy-no-provenance/
    out = os.path.join(root, "pxm2025", "out")
    if os.path.isdir(out):
        for pdb in sorted(os.listdir(out)):
            inner = os.path.join(out, pdb, pdb)
            if not os.path.isdir(inner):
                continue                      # .log files and other strays stay put
            for sd in sorted(os.listdir(inner)):
                if not sd.startswith("seed_"):
                    continue
                seed = sd[len("seed_"):]
                moves.append((os.path.join(inner, sd),
                              os.path.join(root, "free", pdb, seed, LEGACY)))
    # 2. the four modern trees -> free/ or self_template/
    for tree, sub, dest in RUN_TREES:
        base = os.path.join(root, tree, sub)
        if not os.path.isdir(base):
            continue
        for pdb in sorted(os.listdir(base)):
            pdir = os.path.join(base, pdb)
            if not os.path.isdir(pdir):
                continue
            for seed in sorted(os.listdir(pdir)):
                sdir = os.path.join(pdir, seed)
                if not os.path.isdir(sdir):
                    continue
                for rid in sorted(os.listdir(sdir)):
                    src = os.path.join(sdir, rid)
                    if os.path.isdir(src):
                        moves.append((src, os.path.join(root, dest, pdb, seed, rid)))
    # 3. the template namespace is an input to self-templated folds
    ns = os.path.join(root, "st", "ns")
    if os.path.isdir(ns):
        moves.append((ns, os.path.join(root, "self_template", "templates")))
    return moves


def merge_csvs(root, apply):
    """Concatenate same-schema csvs, dropping duplicate (iface, seed, arm, sample) rows."""
    for srcs, dst in CSV_MERGE:
        rows, header, seen, dup = [], None, set(), 0
        for s in srcs:
            p = os.path.join(root, s)
            if not os.path.exists(p):
                continue
            with open(p) as fh:
                r = csv.reader(fh)
                h = next(r, None)
                if h is None:
                    continue
                if header is None:
                    header = h
                elif h != header:
                    raise SystemExit(f"{s}: schema {h} != {header}")
                for row in r:
                    k = (row[0], row[1], row[2], row[4])
                    if k in seen:
                        dup += 1
                        continue
                    seen.add(k)
                    rows.append(row)
        if header is None:
            continue
        print(f"  {dst}: {len(rows)} rows from {len(srcs)} files ({dup} duplicates dropped)")
        if apply:
            os.makedirs(os.path.join(root, os.path.dirname(dst)), exist_ok=True)
            with open(os.path.join(root, dst), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(header)
                w.writerows(rows)


def plan_archive(root):
    return [e for e in sorted(os.listdir(root)) if e not in KEEP]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--stage", choices=("runs", "archive", "all"), default="runs")
    A = ap.parse_args()

    if A.stage in ("runs", "all"):
        moves = plan_runs(A.root)
        clash = [(s, d) for s, d in moves if os.path.exists(d)]
        legacy = sum(1 for _, d in moves if d.endswith(LEGACY))
        print(f"run dirs to move : {len(moves)}  (of which {legacy} legacy-no-provenance)")
        print(f"destination clash: {len(clash)}")
        if clash:
            print("REFUSING -- a destination already exists:")
            for _, d in clash[:5]:
                print(f"  {d}")
            return 1
        for s, d in moves[:2]:
            print(f"  {s}\n    -> {d}")
        print("csv merge:")
        merge_csvs(A.root, A.apply)
        if A.apply:
            for src, dst in moves:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)
            print(f"  moved {len(moves)} run dirs")

    if A.stage in ("archive", "all"):
        entries = plan_archive(A.root)
        print(f"\ntop-level entries to archive: {len(entries)}")
        print("  " + ", ".join(entries[:6]) + (" ..." if len(entries) > 6 else ""))
        if A.apply:
            adir = os.path.join(A.root, "archive")
            os.makedirs(adir, exist_ok=True)
            for e in entries:
                dst = os.path.join(adir, e)
                if os.path.exists(dst):
                    print(f"  skip (exists in archive): {e}")
                    continue
                shutil.move(os.path.join(A.root, e), dst)
            print(f"  archived {len(entries)} entries")

    if not A.apply:
        print("\ndry run -- pass --apply to perform the moves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
