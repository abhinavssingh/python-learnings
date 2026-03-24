from __future__ import annotations

import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# --------------------------------------------------------------------
# Locate project root = two levels up from this file (lib/../)
# --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------
# Load settings.json for log cleanup configuration
# --------------------------------------------------------------------
CONFIG_FILE = ROOT / "settings.json"
LOG_CLEANUP_CONFIG = {}

if CONFIG_FILE.exists():
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        LOG_CLEANUP_CONFIG = data.get("log_cleanup", {})
    except Exception:
        LOG_CLEANUP_CONFIG = {}
else:
    LOG_CLEANUP_CONFIG = {}

# --------------------------------------------------------------------
# Log directory initialization
# --------------------------------------------------------------------
LOG_DIR = ROOT / "log"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE: Optional[Path] = None
_CLEANUP_DONE = False  # ensure cleanup runs only once per session


# --------------------------------------------------------------------
# Cleanup logic
# --------------------------------------------------------------------
def _cleanup_logs() -> None:
    """Delete log files based on rules in paths.json."""
    global _CLEANUP_DONE

    if _CLEANUP_DONE:
        return

    if not LOG_CLEANUP_CONFIG.get("enabled", False):
        _CLEANUP_DONE = True
        return

    max_days = LOG_CLEANUP_CONFIG.get("max_days")
    max_files = LOG_CLEANUP_CONFIG.get("max_files")

    log_files = sorted(LOG_DIR.glob("run_*.log"))

    # --- Rule 1: Delete based on age ---
    if max_days is not None:
        cutoff = datetime.now() - timedelta(days=max_days)
        for f in log_files:
            if datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
                try:
                    f.unlink()
                except Exception:
                    pass

    # Refresh list after deletion
    log_files = sorted(LOG_DIR.glob("run_*.log"))

    # --- Rule 2: Keep only latest N files ---
    if max_files is not None and len(log_files) > max_files:
        to_delete = log_files[: len(log_files) - max_files]
        for f in to_delete:
            try:
                f.unlink()
            except Exception:
                pass

    _CLEANUP_DONE = True


# --------------------------------------------------------------------
# Log file creation
# --------------------------------------------------------------------
def _get_log_file() -> Path:
    """Return the log file path, create new timestamped one if needed."""
    global LOG_FILE

    # Perform cleanup first
    _cleanup_logs()

    if LOG_FILE is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        LOG_FILE = LOG_DIR / f"run_{timestamp}.log"
        LOG_FILE.touch()

    return LOG_FILE


# --------------------------------------------------------------------
# Log writing internals
# --------------------------------------------------------------------
def _write(level: str, message: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {message}\n"

    log_file = _get_log_file()
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line)


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------
def log_info(msg: str):
    _write("INFO", msg)


def log_warn(msg: str):
    _write("WARN", msg)


def log_error(msg: str):
    _write("ERROR", msg)


def log_debug(msg: str):
    _write("DEBUG", msg)
