#!/usr/bin/env python
"""STEP 1 -- generate the free pose pool, then hand it to folder.py.

Decides WHAT to fold (which complexes, which seeds) and launches the driver. The method does not
invent poses: it chooses which of these to feed back in as templates later, so this is the input.

    call_folder.py                                   plan only, nothing launched
    call_folder.py --run --gpus 1,5,6                every complex of pxm2025, 50 seeds
    call_folder.py --dataset foldbench --seeds 20 --complexes 8ror,9k94 --run --gpus 0

Functions
    complexes()         which complexes the dataset has an input json for
    queue_lines()       one queue line per (complex, seed)
    existing_seeds()    which of those already have a fold on disk
    unreadable_warning() whether later steps will be able to FIND what this writes
    launch()            one tmux session per GPU, all sharing the queue

seeds vs samples: ipTM is largely a property of the SEED -- a seed commits to a docking basin early
and stays there -- so seeds buy pose diversity and samples within a seed do not. 50 seeds x 5 samples
is not interchangeable with 5 x 50. The published pool is 50 x 5 = 250 poses per complex and every
result assumes it.

It will not guess GPUs. This is a shared machine, so devices are explicit and more than four is
refused. Workers go in tmux, not background jobs, which die with the shell and leave claims behind.
"""
import argparse
import glob
import os
import shlex
import subprocess
import sys

sys.path.insert(0, "/home/ebrahim/tts_abag")

import paths  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# the driver lives beside this script now, not in paths.CODE_DIR
FOLD = "folder.py"
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
    folder.py identifies a unit as (seed, tag); a per-seed tag would duplicate the seed column.
    """
    return [f"free\t{pdb}\t*\tfree\t{start_seed + i}"
            for pdb in pdbs for i in range(seeds)]


def unreadable_warning(out):
    """Will the poses this run writes be findable by the rest of the pipeline?

    paths.pool_pose() hardcodes the LEGACY run id:

        f"{FREE_DIR}/{pdb}/{seed}-{LEGACY_RUN}/predictions/{pdb}_sample_{sample}.cif"

    while the driver names every new fold <seed>-<random id>. So a fresh pool is written correctly and
    then never selected from -- not duplicated, not overwritten, simply invisible. This is already true
    of data on disk: experimental_data/free/8ror holds 100 seeds and pool_pose can address 50.

    Checked and reported rather than fixed here, because making pool_pose glob changes which poses every
    existing result was computed from, and that is not a change to make as a side effect of step 1.
    """
    return (f"NOTE: poses written here get a random run id, but paths.pool_pose() only resolves "
            f"'<seed>-{paths.LEGACY_RUN}'. Until pool_pose resolves by glob, poses under {out} will "
            f"not be selected by later steps.")


def existing_seeds(out, pdbs, seeds, start_seed):
    """(complex, seed) pairs that already have a fold directory under `out`.

    Not a duplication hazard -- one fold per seed is the normal layout and re-running a seed is
    legitimate, since folds do not reproduce. It is reported so the size of the run is not a surprise:
    the driver skips a (complex, seed, TAG) already in the csv, and the existing pool was written under
    tags free100/freebf16r1/freefp32r1, so a run tagged `free` will refold rather than skip.
    """
    return [(pdb, start_seed + i) for pdb in pdbs for i in range(seeds)
            if glob.glob(f"{out}/{pdb}/{start_seed + i}-*")]


def launch(gpus, queue, out, dataset, nsample, session_prefix):
    """One tmux session per GPU, all on the same queue; claims stop them colliding."""
    cmds = []
    for g in gpus:
        cmd = (f"cd {HERE} && CUDA_VISIBLE_DEVICES={g} LAYERNORM_TYPE=openfold "
               f"{paths.PROTENIX_PYTHON} {FOLD} --queue {shlex.quote(queue)} "
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

    print(f"\n  {unreadable_warning(out)}")
    seen = existing_seeds(out, pdbs, A.seeds, A.start_seed)
    if seen:
        hit = sorted({p for p, _ in seen})
        print(f"  {len(seen)} of {len(lines)} (complex, seed) pairs already have a fold under {out} "
              f"({len(hit)} complexes, e.g. {', '.join(hit[:4])}).")
        print("  Those will be folded again: the driver skips on (interface, seed, TAG) from the csv, "
              "and the existing pool carries other tags.")
        if not A.force:
            raise SystemExit("  refusing. --out <new dir> for a fresh pool, or --force to refold.")

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
