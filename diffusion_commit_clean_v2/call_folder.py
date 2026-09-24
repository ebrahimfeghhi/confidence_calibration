#!/usr/bin/env python
"""Launch folder.py workers on a queue -- any queue, free or self-templated.

    call_folder.py --queue <dir>/queue_free.tsv --out <dir> --gpus 1,5,6 --run
    call_folder.py --queue <dir>/queue.tsv --out <dir> --templates <dir>/ns --gpus 1,5,6 --run

SUMMARY
    One launcher for both kinds of run: same driver, same claim directory, same GPU rules. Assembling
    the folder.py command by hand instead is how workers end up pointed at the wrong paths.

    Several workers read one queue at once. Each claims a line by creating a directory under the claim
    directory before folding it, so two workers never take the same fold and there is nothing to split
    up by hand. A worker killed mid-fold leaves a claim behind, which folder.py releases at startup.

    Devices are never guessed -- this is a shared machine -- and more than four is refused. Workers go
    in tmux, not background jobs, which die with the shell that started them and leave claims behind.

INPUTS
    --queue        the work list, free or templated
    --templates    template namespace, required if the queue has any tmpl lines
    --gpus         explicit device ids; required by --run
    --out          where folds land
    --csv          default <out>/results.csv
    --claim-dir    default <out>/claims
    --log-dir      default <out>; worker logs are w<gpu>.log

    DEFAULTS
        --out       paths.FREE_DIR
        --dataset   pxm2025
        --nsample   5
        --run       off; without it the plan is printed and nothing starts

OUTPUTS
    one tmux session per GPU, named <basename of log dir>w<gpu>, each running folder.py on the queue

FUNCTIONS
    check()   refuse a queue that cannot run: missing, templated with no namespace, or a templated
              queue aimed at the shared free pool
    launch()  one tmux session per GPU, all reading the same queue
    main()    validate, then launch
"""
import argparse
import os
import shlex
import subprocess
import sys

sys.path.insert(0, "/home/ebrahim/tts_abag")

import paths  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_GPUS = 4


def check(queue, out, templates):
    """in: queue path, output dir, namespace or None.  out: (n folds, n templated); raises on refusal."""
    if not os.path.exists(queue):
        raise SystemExit(f"no queue at {queue}")
    kinds = [l.split("\t")[0] for l in open(queue) if l.strip()]
    n_tmpl = sum(1 for k in kinds if k != "free")
    if n_tmpl and not templates:
        raise SystemExit(f"{n_tmpl} templated folds need --templates (run the template builder first)")
    if n_tmpl and out == str(paths.FREE_DIR):
        raise SystemExit("refusing to write a templated queue into the free pool; pass --out")
    return len(kinds), n_tmpl


def launch(gpus, queue, out, templates, dataset, nsample, prefix, csv_path, claim_dir, log_dir):
    """in: devices and the run's paths.  out: the tmux session names started, one per GPU."""
    names = []
    for g in gpus:
        cmd = (f"cd {HERE} && CUDA_VISIBLE_DEVICES={g} LAYERNORM_TYPE=openfold "
               f"{paths.PROTENIX_PYTHON} folder.py --queue {shlex.quote(queue)} "
               f"--out {shlex.quote(out)} --csv {shlex.quote(csv_path)} "
               f"--claim-dir {shlex.quote(claim_dir)} --dataset {dataset} "
               f"--nsample {nsample} "
               + (f"--templates {shlex.quote(templates)} " if templates else "")
               + f"--drop-cifs 2>&1 | tee -a {shlex.quote(log_dir)}/w{g}.log")
        s = f"{prefix}{g}"
        subprocess.run(["tmux", "new-session", "-d", "-s", s, cmd], check=True)
        names.append(s)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--out", default=None, help=f"default {paths.FREE_DIR}")
    ap.add_argument("--templates", default=None, help="namespace, required for templated queues")
    ap.add_argument("--csv", default=None, help="default <out>/results.csv")
    ap.add_argument("--claim-dir", default=None, help="default <out>/claims")
    ap.add_argument("--log-dir", default=None, help="default <out>; worker logs are w<gpu>.log")
    ap.add_argument("--dataset", choices=sorted(paths.DATASET_NAMES), default="pxm2025")
    ap.add_argument("--nsample", type=int, default=5)
    ap.add_argument("--gpus", default=None, help="explicit device ids, e.g. 1,5,6")
    ap.add_argument("--allow-many", action="store_true",
                    help=f"permit more than {MAX_GPUS} GPUs on this shared machine")
    ap.add_argument("--run", action="store_true", help="launch; otherwise only report the plan")
    A = ap.parse_args()

    out = A.out or str(paths.FREE_DIR)
    n, n_tmpl = check(A.queue, out, A.templates)
    print(f"  queue {A.queue}\n  folds {n} ({n_tmpl} templated)\n  out   {out}")
    print(f"  csv   {A.csv or out + '/results.csv'}\n  claims {A.claim_dir or out + '/claims'}")

    if not A.run:
        print("\n  nothing launched. re-run with --run --gpus <ids>")
        return 0
    if not A.gpus:
        raise SystemExit("--run needs --gpus; this machine is shared and devices are never guessed")
    gpus = [g.strip() for g in A.gpus.split(",") if g.strip()]
    if len(gpus) > MAX_GPUS and not A.allow_many:
        raise SystemExit(f"{len(gpus)} GPUs requested; cap is {MAX_GPUS}. Pass --allow-many.")
    csv_path = A.csv or f"{out}/results.csv"
    claim_dir = A.claim_dir or f"{out}/claims"
    log_dir = A.log_dir or out
    prefix = f"{os.path.basename(os.path.normpath(log_dir))}w"
    names = launch(gpus, A.queue, out, A.templates, A.dataset, A.nsample,
                   prefix, csv_path, claim_dir, log_dir)
    print(f"\n  launched {len(names)} workers: {', '.join(names)}")
    print(f"  watch: tmux attach -t {names[0]}   |   tail -f {out}/w{gpus[0]}.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
