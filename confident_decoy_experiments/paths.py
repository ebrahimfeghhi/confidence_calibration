"""
Centralized paths for the confident_decoy_experiments project.

Mirrors the convention in tts_abag/paths.py: env-var-overridable Path
constants, defaulting to existing on-disk locations.

Phase 2 (2026-08-10): artifacts/ and predictions/ belong INSIDE this repo, and
are gitignored -- the repo tracks code and docs, not 100G of run output.

PER-MACHINE: the move is a local filesystem operation, so a clone elsewhere has
the data wherever that machine put it. These defaults therefore RESOLVE AT
IMPORT rather than assuming: repo-local if it exists, else the pre-move
standalone root, else repo-local (so a fresh machine gets the intended layout).
Override explicitly with CDE_ARTIFACTS_DIR / CDE_PREDICTIONS_DIR.

On the origin box the data is in the repo and
/home/ebrahim/confident_decoy_experiments/{artifacts,predictions} are SYMLINKS
back to it, so absolute paths baked into old logs, configs and running jobs keep
resolving. That symlink layout is an origin-box detail, NOT a guarantee
elsewhere -- alderaan had the real directory at the old path and nothing in the
repo, which the resolver below handles.
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

# --- Data (repo-local by intent, resolved per machine) ----------------------
def _resolve_data(env_var: str, name: str) -> Path:
    """Repo-local if present, else the pre-move root, else repo-local.

    The Phase 2 move is per-machine, so hardcoding either location breaks the
    other. Explicit env var always wins.
    """
    val = os.environ.get(env_var)
    if val:
        return Path(val).expanduser()
    in_repo = REPO_ROOT / name
    if in_repo.exists():
        return in_repo
    legacy = _OLD_EXP_ROOT / name
    if legacy.exists():
        return legacy
    return in_repo


ARTIFACTS_DIR = _resolve_data("CDE_ARTIFACTS_DIR", "artifacts")
PREDICTIONS_DIR = _resolve_data("CDE_PREDICTIONS_DIR", "predictions")

# --- Backwards-compat aliases ------------------------------------------------
# Scripts historically did `EXP = "/home/ebrahim/confident_decoy_experiments"`
# and built every subpath off that one root as f"{EXP}/artifacts/...". EXP must
# therefore point at whichever root actually HOLDS artifacts/ on this machine,
# which is the repo once Phase 2 has been done here and the old root otherwise.
EXP = ARTIFACTS_DIR.parent
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
