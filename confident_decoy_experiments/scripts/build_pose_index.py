#!/usr/bin/env python
"""One index of every antibody-antigen pose on disk, wherever it was written.

WHY: poses for the same complex are scattered across prediction trees written by different arms --
Protenix/foldbench_abag/predictions_template (the base 0-150 sweep), and per-arm directories under
artifacts/sweep_out/<arm>/ for the deep runs, which are additionally CHUNKED (8aok_freedeep_a..d
cover seeds 151-738, 739-1326, 1327-1914, 1915-2499). Looking only at predictions_template made six
complexes appear to have 151 seeds when they actually have 2500, and nearly cost a week of GPU time
regenerating data we already had.

Writes artifacts/pose_index.csv.gz: pdb, seed, sample, arm, path -- one row per pose file.
Use scripts/pose_index.py to query it rather than globbing a directory.
"""
import csv, glob, gzip, os, re, sys, time

ROOTS = ["/home/ebrahim/Protenix/foldbench_abag/predictions_template",
         "/home/ebrahim/Protenix/foldbench_abag/predictions_seed_test",
         "/home/ebrahim/tts_abag/confident_decoy_experiments/artifacts/sweep_out",
         "/home/ebrahim/tts_abag/confident_decoy_experiments/predictions",
         "/home/ebrahim/Protenix/artifacts/sweep_out"]
PAT = re.compile(r"seed_(\d+)/predictions/(.+?)_sample_(\d+)\.cif$")

def main():
    out, t0, n = [], time.time(), 0
    for root in ROOTS:
        if not os.path.isdir(root):
            continue
        for f in glob.iglob(f"{root}/**/seed_*/predictions/*_sample_*.cif", recursive=True):
            m = PAT.search(f)
            if not m:
                continue
            seed, name, smp = int(m.group(1)), m.group(2), int(m.group(3))
            pdb = name[:4].lower()
            # arm = the path component under the root, so deep chunks stay distinguishable
            rel = os.path.relpath(f, root).split("/seed_")[0]
            arm = f"{os.path.basename(root)}:{rel}" if rel != name else os.path.basename(root)
            out.append((pdb, seed, smp, arm, f))
            n += 1
            if n % 50000 == 0:
                print(f"  {n} poses, {time.time()-t0:.0f}s", flush=True)
    with gzip.open("artifacts/pose_index.csv.gz", "wt", newline="") as fh:
        w = csv.writer(fh); w.writerow(["pdb", "seed", "sample", "arm", "path"]); w.writerows(out)
    print(f"indexed {len(out)} poses in {time.time()-t0:.0f}s -> artifacts/pose_index.csv.gz")

if __name__ == "__main__":
    main()
