from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


class Logger:
    """
    File-based logger with automatic cleanup based on settings.json.

    Features:
    - Auto-detect project root
    - Timestamped log files
    - Cleanup by age and max file count
    - Simple log level API
    """

    _LOG_FILE: Optional[Path] = None
    _CLEANUP_DONE: bool = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    @classmethod
    def _get_project_root(cls) -> Path:
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "settings.json").exists():
                return parent
        raise RuntimeError("Project root not found (settings.json missing)")

    ROOT = _get_project_root.__func__(None)

    CONFIG_FILE = ROOT / "settings.json"
    LOG_DIR = ROOT / "log"
    LOG_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Load log cleanup config
    # ------------------------------------------------------------------
    if CONFIG_FILE.exists():
        try:
            _CONFIG = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            LOG_CLEANUP_CONFIG = _CONFIG.get("log_cleanup", {})
        except Exception:
            LOG_CLEANUP_CONFIG = {}
    else:
        LOG_CLEANUP_CONFIG = {}

    # ------------------------------------------------------------------
    # Cleanup logic
    # ------------------------------------------------------------------
    @classmethod
    def _cleanup_logs(cls) -> None:
        if cls._CLEANUP_DONE:
            return

        if not cls.LOG_CLEANUP_CONFIG.get("enabled", False):
            cls._CLEANUP_DONE = True
            return

        max_days = cls.LOG_CLEANUP_CONFIG.get("max_days")
        max_files = cls.LOG_CLEANUP_CONFIG.get("max_files")

        log_files = sorted(cls.LOG_DIR.glob("run_*.log"))

        # Delete old logs
        if max_days is not None:
            cutoff = datetime.now() - timedelta(days=max_days)
            for f in log_files:
                if datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
                    try:
                        f.unlink()
                    except Exception:
                        pass

        # Refresh list
        log_files = sorted(cls.LOG_DIR.glob("run_*.log"))

        # Keep only latest N logs
        if max_files is not None and len(log_files) > max_files:
            for f in log_files[: len(log_files) - max_files]:
                try:
                    f.unlink()
                except Exception:
                    pass

        cls._CLEANUP_DONE = True

    # ------------------------------------------------------------------
    # Log file management
    # ------------------------------------------------------------------
    @classmethod
    def _get_log_file(cls) -> Path:
        cls._cleanup_logs()

        if cls._LOG_FILE is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cls._LOG_FILE = cls.LOG_DIR / f"run_{ts}.log"
            cls._LOG_FILE.touch()

        return cls._LOG_FILE

    # ------------------------------------------------------------------
    # Core write method
    # ------------------------------------------------------------------
    @classmethod
    def _write(cls, level: str, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{level}] {message}\n"

        log_file = cls._get_log_file()
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @classmethod
    def info(cls, msg: str) -> None:
        cls._write("INFO", msg)

    @classmethod
    def warn(cls, msg: str) -> None:
        cls._write("WARN", msg)

    @classmethod
    def error(cls, msg: str) -> None:
        cls._write("ERROR", msg)

    @classmethod
    def debug(cls, msg: str) -> None:
        cls._write("DEBUG", msg)
