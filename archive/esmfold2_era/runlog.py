"""
Per-run config logging — one self-contained JSON snapshot per invocation.

Each call writes a UNIQUELY-named file into <out_dir>/configs/, so concurrent GPU
processes never write the same file (this replaces the old shared config.json that
every process read-modify-wrote, which raced). The snapshot records the run params
plus timestamp, host, pid, and the tts_abag git commit for reproducibility.
"""
import json
import os
import socket
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=_REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def save_run_config(out_dir, config: dict) -> Path:
    """Write a uniquely-named config snapshot under <out_dir>/configs/ and return its path."""
    cfg_dir = Path(out_dir) / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {
        "timestamp_utc": ts,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "tts_abag_commit": _git_sha(),
        **config,
    }

    # timestamp + short uuid => unique even for processes started in the same second.
    path = cfg_dir / f"{ts}__{uuid.uuid4().hex[:8]}.json"
    path.write_text(json.dumps(record, indent=2, default=str))
    return path
