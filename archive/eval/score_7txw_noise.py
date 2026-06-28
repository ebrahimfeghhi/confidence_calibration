"""Score the Idea-1 noise-sweep diffusion dirs with the standard OST/DockQ eval.

For each noise_scale, runs eval_ost_foldbench.py on
    ESM_PREDICTIONS/7txw_noise{ns}/loop{LL}
and writes CSV_DIR/dockq_7txw_noise{ns}.csv (one row per seed x sample x interface).
Idempotent: the eval resumes per row, so re-running after the sweep finishes only
scores the newly-written samples.

Usage:
    python eval/score_7txw_noise.py                      # all 4 noise scales, loop16
    python eval/score_7txw_noise.py --noise 1.0 2.0
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


def _fmt(ns: float) -> str:
    """Match the run_name suffix written by launch.sh (e.g. 0.5, 1.0)."""
    return f"{ns:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--loop", type=int, default=16)
    args = ap.parse_args()

    for ns in args.noise:
        tag = _fmt(ns)
        pred_dir = ESM_PREDICTIONS / f"7txw_noise{tag}" / f"loop{args.loop:02d}"
        if not pred_dir.exists():
            print(f"noise {tag}: no pred dir ({pred_dir}), skip", flush=True)
            continue
        out_csv = CSV_DIR / f"dockq_7txw_noise{tag}.csv"
        print(f"=== scoring noise {tag} -> {out_csv.name} ===", flush=True)
        subprocess.run([str(ESM_PYTHON), str(EVAL),
                        "--pred_dir", str(pred_dir), "--out_csv", str(out_csv)],
                       check=False)
    print("scoring done.", flush=True)


if __name__ == "__main__":
    main()
