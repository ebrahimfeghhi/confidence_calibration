"""Score every per-loop diffusion dir with the standard OST/DockQ eval.

Runs eval_ost_foldbench.py once per loop dir (ESM_PREDICTIONS/7txw_loops/loopLL),
writing CSV_DIR/dockq_7txw_loopLL.csv. Idempotent (eval resumes per row).

Usage: python eval/score_7txw_loops.py [--loops 0 1 ... 16]
"""
import argparse
import subprocess
import sys
from pathlib import Path

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import ESM_PREDICTIONS, ESM_PYTHON, CSV_DIR

EVAL = Path(__file__).resolve().parent / "eval_ost_foldbench.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loops", type=int, nargs="+", default=list(range(17)))
    ap.add_argument("--run_name", default="7txw_loops")
    args = ap.parse_args()

    for L in args.loops:
        pred_dir = ESM_PREDICTIONS / args.run_name / f"loop{L:02d}"
        if not pred_dir.exists():
            print(f"loop{L:02d}: no pred dir, skip", flush=True)
            continue
        out_csv = CSV_DIR / f"dockq_7txw_loop{L:02d}.csv"
        print(f"=== scoring loop{L:02d} -> {out_csv.name} ===", flush=True)
        subprocess.run([str(ESM_PYTHON), str(EVAL),
                        "--pred_dir", str(pred_dir), "--out_csv", str(out_csv)],
                       check=False)
    print("scoring done.", flush=True)


if __name__ == "__main__":
    main()
