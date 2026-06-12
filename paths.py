"""
Centralized paths for the tts_abag benchmark.

Single source of truth for every absolute path the run / eval / plot / analysis
scripts depend on. Each value defaults to a location under $HOME and can be
overridden with an environment variable of the same name, e.g.

    FOLDBENCH_DIR=/data/FoldBench python eval/eval_ost_foldbench.py ...

Scripts import this module via a small repo-root bootstrap (they walk up parent
dirs until they find this file, add the repo root to sys.path, then
`from paths import ...`). That keeps them runnable from any cwd and under any of
the model venvs without needing an editable install.

NOTE: this module points at *existing* data/output locations in the original
repos. The tts_abag repo holds scripts only — predictions and ground-truth data
are NOT moved here.
"""

import os
from pathlib import Path

HOME = Path.home()

# Repo root (directory containing this file) and default plot/CSV output dir.
REPO_ROOT = Path(__file__).resolve().parent


def _path(env_var: str, default: Path) -> Path:
    """Return Path from env override or default."""
    val = os.environ.get(env_var)
    return Path(val).expanduser() if val else default


# --- External input data (FoldBench ground truths + targets) ---------------
FOLDBENCH_DIR = _path("FOLDBENCH_DIR", HOME / "FoldBench")
GROUND_TRUTHS = _path("GROUND_TRUTHS", FOLDBENCH_DIR / "examples" / "ground_truths")
TARGETS_CSV = _path(
    "TARGETS_CSV", FOLDBENCH_DIR / "targets" / "interface_antibody_antigen.csv"
)

# Input JSONs (sequence/complex definitions) — currently live in the Protenix repo.
JSON_DIR = _path("JSON_DIR", HOME / "Protenix" / "foldbench_abag" / "jsons")

# --- Model repos / environments / checkpoints ------------------------------
# run_benchmark.sh must run from the Protenix repo (it calls runner/inference.py).
ESM_REPO = _path("ESM_REPO", HOME / "esm")
PROTENIX_REPO = _path("PROTENIX_REPO", HOME / "Protenix")
ESM_PYTHON = _path("ESM_PYTHON", ESM_REPO / ".venv" / "bin" / "python")
PROTENIX_PYTHON = _path("PROTENIX_PYTHON", PROTENIX_REPO / ".venv" / "bin" / "python")
PROTENIX_CKPT = _path("PROTENIX_CKPT", HOME / "checkpoint")

# --- OST (lives in the `foldbench` conda env) ------------------------------
OST_BIN = _path("OST_BIN", HOME / "miniconda3" / "envs" / "foldbench" / "bin" / "ost")
OST_LIB = _path("OST_LIB", HOME / "miniconda3" / "envs" / "foldbench" / "lib")

# --- Prediction outputs (data stays in the original repos) -----------------
ESM_PREDICTIONS = _path("ESM_PREDICTIONS", HOME / "esm" / "foldbench_predictions")
PROTENIX_PREDICTIONS = _path(
    "PROTENIX_PREDICTIONS",
    HOME / "Protenix" / "foldbench_abag" / "predictions_template",
)

# --- Plot / CSV outputs ----------------------------------------------------
RESULTS_DIR = _path("RESULTS_DIR", REPO_ROOT / "results")


def _sh():
    """Print `export NAME=value` lines so shell scripts can source these paths:

        eval "$(python paths.py --sh)"
    """
    for name in (
        "FOLDBENCH_DIR", "GROUND_TRUTHS", "TARGETS_CSV", "JSON_DIR",
        "ESM_REPO", "PROTENIX_REPO", "ESM_PYTHON", "PROTENIX_PYTHON",
        "PROTENIX_CKPT", "OST_BIN", "OST_LIB",
        "ESM_PREDICTIONS", "PROTENIX_PREDICTIONS", "RESULTS_DIR",
    ):
        print(f'export {name}="{globals()[name]}"')


if __name__ == "__main__":
    import sys

    if "--sh" in sys.argv:
        _sh()
    else:
        for name, val in sorted(globals().items()):
            if name.isupper() and isinstance(val, Path):
                exists = "ok " if val.exists() else "MISSING"
                print(f"[{exists}] {name} = {val}")
