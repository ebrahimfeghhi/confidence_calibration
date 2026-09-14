#!/usr/bin/env python
"""Does deeper sampling find BETTER EPITOPES, even though it fails to improve selection?

Selection churns with depth (scripts/cd_depth_1024.py): the ipTM argmax is displaced by each new
pose, so selected DockQ bounces. That says nothing about COVERAGE. This measures the oracle epitope
jaccard -- the best epitope any pose reaches -- as a function of how many seeds you draw, on the
confident decoys that already have >=1024 seeds on disk.

If the oracle keeps climbing, deep sampling IS finding new/better epitopes and the failure is purely
selection. If it plateaus early, the model exhausts its epitope repertoire quickly and depth adds
nothing at all.

Poses come from scripts/pose_index.py so the deep arms are actually included; globbing
predictions_template would silently cap this at 151 seeds.
"""
import collections, csv, os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "scripts")
from pose_index import poses
from epitope_overlap import score_pose
from epitope_rmsd import native_epitope
from iface_chains import first_pose_with
from cdr_h3_rmsd import GT_DIR, _ost

FREE = ("predictions_template", "freedeep", "allpose", "free1k")
TARGETS = ["8aok_BA", "8av2_CA", "8che_AE", "8che_BE", "8fas_BC", "8slb_BA", "8slb_CA", "9jbq_AC"]


def main():
    _, io, _ = _ost()
    out = {}
    for iface in TARGETS:
        pdb, ch = iface.rsplit("_", 1)
        ab, ag = ch[0], ch[1]
        try:
            c0, _, _ = first_pose_with(pdb, ab, ag)
        except KeyError as e:
            print(f"{iface}: {e}"); continue
        gt = f"{GT_DIR}/{pdb}-assembly1.cif"
        native = native_epitope(io.LoadMMCIF(gt, fault_tolerant=True), c0["ref_ab"], c0["ref_ag"])
        if len(native) < 4:
            print(f"{iface}: native epitope too small"); continue
        seen, work = set(), []
        for s, k, arm, f in poses(pdb):
            if not any(t in arm for t in FREE) or "nomsa" in arm or "fullmsa" in arm:
                continue
            if (s, k) in seen:
                continue
            seen.add((s, k))
            work.append((pdb, ab, ag, f, gt, str(s), str(k), native))
        with Pool(28) as p:
            res = [r for r in p.imap_unordered(score_pose, work, chunksize=8)
                   if r and r[0] != "__ERR__"]
        # r = [pdb, ab, ag, seed, sample, n_native, n_pred, n_inter, recall, jaccard]
        rows = sorted((int(r[3]), float(r[9])) for r in res)
        out[iface] = rows
        print(f"{iface}: {len(rows)} poses scored, {len({s for s,_ in rows})} seeds", flush=True)
    np.save("artifacts/cd_epitope_depth.npy", out, allow_pickle=True)
    print("\nORACLE epitope jaccard (best of the first n seeds)\n")
    D = [10, 25, 50, 100, 250, 512, 1024, 2048]
    print(f'{"interface":<12}' + "".join(f"{d:>8}" for d in D))
    for iface, rows in out.items():
        seeds = sorted({s for s, _ in rows})
        line = f"{iface:<12}"
        for d in D:
            keep = set(seeds[:d])
            v = [j for s, j in rows if s in keep]
            line += f"{max(v):>8.2f}" if v else f'{"-":>8}'
        print(line)


if __name__ == "__main__":
    main()
