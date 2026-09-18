#!/usr/bin/env python
"""One-off: drop the interface level from run directories.

    <kind>/<pdb>/<iface>/<seed>/<run_id>/   ->   <kind>/<pdb>/<seed>/<run_id>/

WHY. Every fold produces the whole complex, so an interface in the path implied the output was
specific to one interface when the coordinates cover all of them. Worse, the level held two different
things: the old driver passed the pdb into that slot (giving <pdb>/<pdb>/) while the newer one wrote
"all", so one kind of fold sat at two shapes in the same dataset.

Nothing reads these directory names -- the csv carries `iface` per row and run.json carries the rest --
so this is presentation, not data. Run ids are unique (timestamp-pid-hash), so a collision at the
destination would mean two runs claimed one id; that is reported rather than merged.

    migrate_run_dirs.py --root <out> [--apply]      # dry run unless --apply
"""
import argparse
import os
import shutil

# "self_template" is the older mode name, still on disk in the st/ and st_bf16/ trees.
KINDS = ("free", "all5", "perpose", "self_template")


def plan(root):
    """Every (src, dst) run-directory move, plus the interface levels left empty behind them."""
    moves, levels = [], []
    for kind in KINDS:
        kdir = os.path.join(root, kind)
        if not os.path.isdir(kdir):
            continue
        for pdb in sorted(os.listdir(kdir)):
            pdir = os.path.join(kdir, pdb)
            if not os.path.isdir(pdir):
                continue
            for mid in sorted(os.listdir(pdir)):
                mdir = os.path.join(pdir, mid)
                # a seed directory is all digits; anything else at this depth is the interface level
                if not os.path.isdir(mdir) or mid.isdigit():
                    continue
                levels.append(mdir)
                for seed in sorted(os.listdir(mdir)):
                    sdir = os.path.join(mdir, seed)
                    if not os.path.isdir(sdir):
                        continue
                    for rid in sorted(os.listdir(sdir)):
                        src = os.path.join(sdir, rid)
                        if os.path.isdir(src):
                            moves.append((src, os.path.join(pdir, seed, rid)))
    return moves, levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--apply", action="store_true", help="without this, only report")
    A = ap.parse_args()

    moves, levels = plan(A.root)
    clash = [(s, d) for s, d in moves if os.path.exists(d)]
    print(f"run dirs to move : {len(moves)}")
    print(f"interface levels : {len(levels)}")
    print(f"destination clash: {len(clash)}")
    for s, d in moves[:3]:
        print(f"  {s}\n    -> {d}")
    if clash:
        print("REFUSING: a destination already exists, which means two runs share a run id")
        for s, d in clash[:5]:
            print(f"  {d}")
        return 1
    if not A.apply:
        print("\ndry run -- pass --apply to perform the moves")
        return 0

    for src, dst in moves:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
    pruned = 0
    for mdir in levels:                     # remove the interface level, only if truly empty
        # os.walk(topdown=False) captures each directory's `dirs` list BEFORE its children are
        # removed, so the outermost directory still looks non-empty and gets skipped. Hence the
        # explicit rmdir afterwards rather than trusting the walk to reach the top.
        for cur, dirs, files in os.walk(mdir, topdown=False):
            if not dirs and not files:
                os.rmdir(cur)
                pruned += 1
        if os.path.isdir(mdir) and not os.listdir(mdir):
            os.rmdir(mdir)
            pruned += 1
    print(f"moved {len(moves)} run dirs, pruned {pruned} empty directories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
