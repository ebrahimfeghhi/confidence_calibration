"""
Centralized paths for the confident_decoy_experiments project.

Mirrors the convention in tts_abag/paths.py: env-var-overridable Path
constants, defaulting to existing on-disk locations.

NOTE: artifacts/ and predictions/ (87G) are NOT yet moved into this repo —
they still live at the old standalone location because a run
(allpose8che10) was actively reading/writing them via absolute paths when
the code was moved. Once that run finishes, move the directories here and
flip these two defaults to repo-relative paths (see
docs/confident_decoy_localization_handoff.md Phase 2 in the move plan).
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Old standalone project root (pre-move). artifacts/ and predictions/ still
# live under here until Phase 2 relocates them into this repo.
_OLD_EXP_ROOT = Path("/home/ebrahim/confident_decoy_experiments")


def _path(env_var: str, default: Path) -> Path:
    val = os.environ.get(env_var)
    return Path(val).expanduser() if val else default


# --- Code-relative (already moved) ------------------------------------------
SCRIPTS_DIR = _path("CDE_SCRIPTS_DIR", REPO_ROOT / "scripts")
CONFIGS_DIR = _path("CDE_CONFIGS_DIR", REPO_ROOT / "configs")

# --- Data (not yet moved — Phase 2) -----------------------------------------
ARTIFACTS_DIR = _path("CDE_ARTIFACTS_DIR", _OLD_EXP_ROOT / "artifacts")
PREDICTIONS_DIR = _path("CDE_PREDICTIONS_DIR", _OLD_EXP_ROOT / "predictions")

# --- Backwards-compat aliases ------------------------------------------------
# Scripts historically did `EXP = "/home/ebrahim/confident_decoy_experiments"`
# and built every subpath off that one root via f"{EXP}/artifacts/...",
# f"{EXP}/scripts/...", etc. Now that code has moved here but artifacts/
# predictions have not (Phase 2), that single root has split in two:
#   EXP       -> still the OLD data root (artifacts/, predictions/ live here)
#   CODE_ROOT -> this repo location (scripts/, configs/, RESULTS.md, logs/)
# Scripts were updated to use whichever alias matches what they're building a
# path under; once Phase 2 moves artifacts/predictions here too, EXP and
# CODE_ROOT collapse back into one and callers can go back to a single root.
EXP = _OLD_EXP_ROOT
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
