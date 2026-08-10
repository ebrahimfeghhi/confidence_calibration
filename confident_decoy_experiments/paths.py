"""
Centralized paths for the confident_decoy_experiments project.

Mirrors the convention in tts_abag/paths.py: env-var-overridable Path
constants, defaulting to existing on-disk locations.

Phase 2 DONE (2026-08-10): artifacts/ (91G) and predictions/ (12G) now live
INSIDE this repo. Everything the project produces is under tts_abag.
/home/ebrahim/confident_decoy_experiments/{artifacts,predictions} are left as
SYMLINKS to here, so the absolute paths baked into old logs, configs and
running jobs still resolve. Both are gitignored -- the repo tracks code and
docs, not 100G of run output.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Pre-move standalone root. Now only symlinks to this repo; kept so absolute
# paths recorded in old logs/configs keep resolving.
_OLD_EXP_ROOT = Path("/home/ebrahim/confident_decoy_experiments")


def _path(env_var: str, default: Path) -> Path:
    val = os.environ.get(env_var)
    return Path(val).expanduser() if val else default


# --- Code-relative (already moved) ------------------------------------------
SCRIPTS_DIR = _path("CDE_SCRIPTS_DIR", REPO_ROOT / "scripts")
CONFIGS_DIR = _path("CDE_CONFIGS_DIR", REPO_ROOT / "configs")

# --- Data (moved into the repo 2026-08-10) ----------------------------------
ARTIFACTS_DIR = _path("CDE_ARTIFACTS_DIR", REPO_ROOT / "artifacts")
PREDICTIONS_DIR = _path("CDE_PREDICTIONS_DIR", REPO_ROOT / "predictions")

# --- Backwards-compat aliases ------------------------------------------------
# Scripts historically did `EXP = "/home/ebrahim/confident_decoy_experiments"`
# and built every subpath off that one root. With Phase 2 done, EXP and
# CODE_ROOT have collapsed back into a single root: this repo.
EXP = REPO_ROOT
CODE_ROOT = REPO_ROOT


def _sh():
    for name in ("SCRIPTS_DIR", "CONFIGS_DIR", "ARTIFACTS_DIR", "PREDICTIONS_DIR"):
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
