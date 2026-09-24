#!/usr/bin/env python
"""STEP 1 -- generate the untemplated pose pool that every later step selects from.

The method does not invent poses. It chooses which of these to feed back in as templates, so this is
the input, and its two numbers set the ceiling on everything downstream:

    seeds    independent folds per complex. ipTM is largely a property of the SEED -- a seed commits to
             a docking basin early and stays there -- so seeds are what buys pose diversity.
    samples  poses per fold, from diffusion sampling. Poses within a seed are far more alike than poses
             across seeds, which is why 50x5 is not interchangeable with 5x50.

The published pool is 50 seeds x 5 samples = 250 poses per complex; every result assumes it. Both are
arguments rather than constants so changing them is a visible choice.

    free_poses.py                                    # what would run, nothing launched
    free_poses.py --run --gpus 1,5,6                 # all complexes of pxm2025, 50 seeds
    free_poses.py --dataset foldbench --seeds 20 --complexes 8ror,9k94 --run --gpus 0

WHY IT DOES NOT PICK GPUS. This runs on a shared machine. A script that finds "free" cards will
eventually take one mid-job from someone else, so devices are always explicit and more than four is
refused unless asked for. Workers go in tmux, one session per GPU, because a detached background job
dies with the shell that made it and leaves its claim behind -- which silently shrinks the sweep.

NOT IDEMPOTENT ON THE FILESYSTEM, only on the csv. fold.py skips a (complex, seed) already present in
results.csv, but run directories are named with a RANDOM id (folds do not reproduce, so two folds of one
seed are different data and must not overwrite). Point at a fresh csv over a populated output directory
and you get a second copy of every seed, silently double-weighting it downstream. The csv is the
authority on what is done; the directory only says what exists.
"""
import argparse
import glob
import os
import shlex
import subprocess
import sys

sys.path.insert(0, "/home/ebrahim/tts_abag")

import paths  # noqa: E402

FOLD = f"{paths.CODE_DIR}/fold.py"
MAX_GPUS = 4


def complexes(dataset, want=None):
    """Complexes the dataset has a Protenix input json for, optionally restricted to `want`.

    The glob is on `*-final-updated.json`, the MSA-attached input. A complex whose json is named
    anything else is silently absent rather than an error, so `want` is checked against what was found
    and the difference is reported by the caller.
    """
    jobs, _, _ = paths.dataset(dataset)
    found = {os.path.basename(p).split("-")[0]
             for p in glob.glob(f"{jobs}/*-final-updated.json")}
    return sorted(found if not want else (found & want)), sorted((want or set()) - found)


def queue_lines(pdbs, seeds, start_seed):
    """One line per (complex, seed): kind, pdb, target, tag, seed.

    `target` is "*" -- a free fold is not aimed at an interface, and one fold produces the whole
    complex, so it is scored against every interface the complex has. The tag is constant because
    fold.py identifies a unit as (seed, tag); a per-seed tag would duplicate the seed column.
    """
    return [f"free\t{pdb}\t*\tfree\t{start_seed + i}"
            for pdb in pdbs for i in range(seeds)]


def existing_folds(out, pdbs, seeds, start_seed):
    """(complex, seed) pairs this run would produce that ALREADY have a run directory.

    The skip logic in fold.py is keyed on (interface, seed, TAG) from the csv, so it protects against
    repeating work only when the tag matches. The existing pool was written under tags free100,
    freebf16r1, freefp32r1 -- a fresh run tagged `free` matches none of them, would refold every seed,
    and because run directories carry a random id the new folds would sit BESIDE the old ones rather
    than replacing them. Two folds of one seed, both valid, both picked up by anything globbing
    <seed>-*, silently double-weighting that seed in every average. The csv cannot see this; only the
    filesystem can, so the filesystem is what gets checked.
    """
    hits = []
    for pdb in pdbs:
        for i in range(seeds):
            seed = start_seed + i
            if glob.glob(f"{out}/{pdb}/{seed}-*"):
                hits.append((pdb, seed))
    return hits


def launch(gpus, queue, out, dataset, nsample, session_prefix):
    """One tmux session per GPU, all on the same queue; claims stop them colliding."""
    cmds = []
    for g in gpus:
        cmd = (f"cd {paths.CODE_DIR} && CUDA_VISIBLE_DEVICES={g} LAYERNORM_TYPE=openfold "
               f"{paths.PROTENIX_PYTHON} fold.py --queue {shlex.quote(queue)} "
               f"--out {shlex.quote(out)} --csv {shlex.quote(out)}/results.csv "
               f"--claim-dir {shlex.quote(out)}/claims --dataset {dataset} "
               f"--nsample {nsample} --drop-cifs 2>&1 | tee -a {shlex.quote(out)}/w{g}.log")
        s = f"{session_prefix}{g}"
        subprocess.run(["tmux", "new-session", "-d", "-s", s, cmd], check=True)
        cmds.append(s)
    return cmds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(paths.DATASET_NAMES), default="pxm2025")
    ap.add_argument("--complexes", default=None, help="comma-separated; default is every complex")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--start-seed", type=int, default=300000)
    ap.add_argument("--nsample", type=int, default=5)
    # one choice today. Present so adding a model is a new entry rather than a redesign, and so the
    # flag cannot silently mean "whatever driver happens to be wired up".
    ap.add_argument("--model", choices=("protenix",), default="protenix")
    ap.add_argument("--out", default=None, help=f"default {paths.FREE_DIR}")
    ap.add_argument("--gpus", default=None, help="explicit device ids, e.g. 1,5,6")
    ap.add_argument("--allow-many", action="store_true",
                    help=f"permit more than {MAX_GPUS} GPUs on this shared machine")
    ap.add_argument("--run", action="store_true", help="launch; otherwise only report the plan")
    ap.add_argument("--force", action="store_true",
                    help="proceed even though --out already holds folds for these seeds")
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

    print(f"  dataset {A.dataset}   model {A.model}")
    print(f"  complexes {len(pdbs)}   seeds {A.seeds} from {A.start_seed}   samples {A.nsample}")
    print(f"  folds {len(lines)}   poses {len(lines) * A.nsample}")
    print(f"  queue -> {qpath}")
    print(f"  out   -> {out}")

    clash = existing_folds(out, pdbs, A.seeds, A.start_seed)
    if clash:
        pdbs_hit = sorted({p for p, _ in clash})
        print(f"\n  {len(clash)} of {len(lines)} (complex, seed) pairs ALREADY have folds under {out}")
        print(f"  affecting {len(pdbs_hit)} complexes, e.g. {', '.join(pdbs_hit[:4])}")
        print("  these would be folded AGAIN into new directories beside the existing ones, because")
        print("  the run id is random and the csv is keyed on a tag this run does not share.")
        if not A.force:
            raise SystemExit("  refusing. use --out <new dir> for a fresh pool, or --force if you "
                             "really intend duplicate folds.")

    if not A.run:
        print("\n  nothing launched. re-run with --run --gpus <ids>")
        return 0
    if not A.gpus:
        raise SystemExit("--run needs --gpus; this machine is shared and devices are never guessed")
    gpus = [g.strip() for g in A.gpus.split(",") if g.strip()]
    if len(gpus) > MAX_GPUS and not A.allow_many:
        raise SystemExit(f"{len(gpus)} GPUs requested; cap is {MAX_GPUS}. Pass --allow-many to override.")
    sessions = launch(gpus, qpath, out, A.dataset, A.nsample, f"free{A.dataset}w")
    print(f"\n  launched {len(sessions)} workers: {', '.join(sessions)}")
    print(f"  watch: tmux attach -t {sessions[0]}    |    tail -f {out}/w{gpus[0]}.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
