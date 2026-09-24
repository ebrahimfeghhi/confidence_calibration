#!/usr/bin/env python
"""Write a queue of untemplated folds -- the pool of candidate poses everything else selects from.

    build_free_queue.py --out <dir>
    call_folder.py --queue <dir>/queue_free.tsv --out <dir> --gpus 1,5,6 --run

SUMMARY
    Decides WHAT to fold and writes it down. Nothing is launched here; call_folder.py runs the queue,
    and runs the templated queues too, so both kinds of run take the identical launch path.

INPUTS
    the dataset's job jsons   one per complex, matched as `*-final-updated.json` (the MSA-attached
                             input). A complex whose json is named otherwise is silently absent from
                             the queue, so the caller reports what it asked for and did not get.
    --seeds, --start-seed    how many folds per complex, and the first seed

    DEFAULTS
        --dataset      pxm2025          the other choice is foldbench
        --complexes    every complex with a job json (131 for pxm2025)
        --seeds        50               folds per complex
        --start-seed   300000           so seeds are 300000..300049
        --out          paths.FREE_DIR   experimental_data/free

OUTPUTS
    <out>/queue_free.tsv   one line per fold, tab separated:

        kind   pdb    target   tag    seed
        free   8ror   *        free   300000

    `target` is `*` because a free fold is not aimed at one interface -- it is scored against every
    interface of its complex. `tag` names the arm in the results csv.

FUNCTIONS
    complexes()           which complexes the dataset has a job json for
    queue_lines()         one queue line per (complex, seed)
    existing_seeds()      which of those already have a fold on disk
    unreadable_warning()  whether later steps will be able to FIND what this writes
    main()                write the queue, or refuse if the folds already exist
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, "/home/ebrahim/tts_abag")

import paths  # noqa: E402


def complexes(dataset, want=None):
    """in: dataset name, optional set of complexes.  out: (found, not found)

    Matches `*-final-updated.json`, the MSA-attached input. An input named anything else is absent
    from the queue rather than an error, so the caller reports what it asked for and did not get.
    """
    jobs, _, _ = paths.dataset(dataset)
    found = {os.path.basename(p).split("-")[0]
             for p in glob.glob(f"{jobs}/*-final-updated.json")}
    return sorted(found if not want else (found & want)), sorted((want or set()) - found)


def queue_lines(pdbs, seeds, start_seed):
    """in: complexes, how many seeds, first seed.  out: one queue line per (complex, seed)."""
    return [f"free\t{pdb}\t*\tfree\t{start_seed + i}"
            for pdb in pdbs for i in range(seeds)]


def existing_seeds(out, pdbs, seeds, start_seed):
    """in: output dir and the folds about to be queued.  out: those that already exist on disk.

    Reported because the driver's skip is keyed on (complex, seed, template) from the csv, so folds
    written under a different tag or into a different csv do not count as done and would be redone.
    """
    return [(pdb, start_seed + i) for pdb in pdbs for i in range(seeds)
            if glob.glob(f"{out}/{pdb}/{start_seed + i}-*")]


def unreadable_warning(out):
    """in: output dir.  out: a warning if later steps will not be able to find what lands there.

    paths.pool_pose() builds '<seed>-legacy-no-provenance', while new folds get a random run id. Poses
    outside the legacy naming are on disk and invisible to selection -- already true of 25 seeds per
    complex in the existing pool.
    """
    return (f"NOTE: poses here get a random run id, but paths.pool_pose() only resolves "
            f"'<seed>-{paths.LEGACY_RUN}'. Until it resolves by glob, poses under {out} will not be "
            f"selected by later steps.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(paths.DATASET_NAMES), default="pxm2025")
    ap.add_argument("--complexes", default=None, help="comma-separated; default is every complex")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--start-seed", type=int, default=300000)
    ap.add_argument("--out", default=None, help=f"default {paths.FREE_DIR}")
    ap.add_argument("--force", action="store_true", help="queue folds that already exist on disk")
    A = ap.parse_args()

    want = set(A.complexes.split(",")) if A.complexes else None
    pdbs, missing = complexes(A.dataset, want)
    if missing:
        print(f"  NOT FOUND in {A.dataset}: {','.join(missing)}")
    if not pdbs:
        raise SystemExit("no complexes matched")

    out = A.out or str(paths.FREE_DIR)
    lines = queue_lines(pdbs, A.seeds, A.start_seed)
    os.makedirs(out, exist_ok=True)
    qpath = f"{out}/queue_free.tsv"
    open(qpath, "w").write("\n".join(lines) + "\n")

    print(f"  dataset {A.dataset}   complexes {len(pdbs)}   seeds {A.seeds} from {A.start_seed}")
    print(f"  folds {len(lines)}   queue -> {qpath}")
    print(f"\n  {unreadable_warning(out)}")

    seen = existing_seeds(out, pdbs, A.seeds, A.start_seed)
    if seen and not A.force:
        hit = sorted({p for p, _ in seen})
        print(f"  {len(seen)} of {len(lines)} folds already exist under {out} "
              f"({len(hit)} complexes, e.g. {', '.join(hit[:4])}).")
        raise SystemExit("  refusing. --out <new dir> for a fresh pool, or --force to queue anyway.")

    print(f"\n  next: call_folder.py --queue {qpath} --out {out} --gpus <ids> --run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
