#!/usr/bin/env python
"""Queue the top-5-pose self-templating experiment: oracle templates versus confident templates.

THE COMPARISON. For each interface take its 5 best poses two ways -- by ground-truth DockQ (oracle)
and by ipTM (what a user could actually pick) -- and self-template on each set. The oracle arm is an
UPPER BOUND, not a method: it answers "if we could identify the good poses, would self-templating
exploit them?" If it does not beat the confidence arm, no amount of better clustering helps, because
clustering can at best find what the oracle already had.

The two sets are nearly disjoint and that is the point: on 87 of 91 decoy interfaces they share ZERO
poses, and the oracle set averages DockQ 0.369 against the confidence set's 0.061.

Emits fold.py's queue format, which takes explicit pose paths:

    all5  <pdb>  <interface>  <tag>  <seed>  <pose1> ... <pose5>

`all5` means "these N poses are one template set, averaged by TemplateEmbedder" -- it does NOT
require them to come from one seed, despite the name.

FRESH SEEDS (400000+) so nothing collides with the pool's 300000-300049 run directories.

    make_queue.py --out queue.tsv [--classes decoy,solved,nocov] [--nseeds 5] [--ntmpl 5]
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paths  # noqa: E402
from separation import pool_classes  # noqa: E402

SEED0 = 400000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--matrices", default="matrices")
    ap.add_argument("--classes", default="decoy,solved,nocov")
    ap.add_argument("--nseeds", type=int, default=5)
    ap.add_argument("--ntmpl", type=int, default=5)
    A = ap.parse_args()

    keep = set(A.classes.split(","))
    cls = pool_classes()
    lines, stats = [], {"oracle": 0, "conf": 0}
    for f in sorted(glob.glob(f"{A.matrices}/*.npz")):
        iface = os.path.basename(f)[:-4]
        if cls.get(iface) not in keep:
            continue
        pdb = iface.split("_")[0]
        z = np.load(f)
        if len(z["seed"]) < A.ntmpl:
            continue
        paths_by_i = [paths.pool_pose(pdb, s, int(x)) for s, x in zip(z["seed"], z["sample"])]
        for arm, order in (("oracle", np.argsort(-z["dockq_gt"].astype(float))),
                           ("conf", np.argsort(-z["pair_iptm"].astype(float)))):
            sel = [paths_by_i[i] for i in order[:A.ntmpl]]
            missing = [p for p in sel if not os.path.exists(p)]
            if missing:
                print(f"  skip {iface} {arm}: {len(missing)} pose files absent", flush=True)
                continue
            for k in range(A.nseeds):
                lines.append("\t".join(["all5", pdb, iface, arm, str(SEED0 + k), " ".join(sel)]))
            stats[arm] += 1

    with open(A.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote {A.out}: {len(lines)} folds "
          f"({stats['oracle']} oracle + {stats['conf']} conf interfaces x {A.nseeds} seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
