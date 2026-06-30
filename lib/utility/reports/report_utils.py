from __future__ import annotations

import os
import webbrowser
from functools import wraps
from pathlib import Path
from typing import Optional, Union

from lib.utility.logger import Logger

StrPath = Union[str, os.PathLike]


class ReportUtils:
    """
    Utility class for filesystem and output/report handling.

    Provides helpers for:
    - Script directory resolution
    - Safe directory creation
    - Output path construction
    - Saving and opening HTML reports
    """

    @staticmethod
    def _log_exceptions(method_name: str):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    Logger.error(f"ReportUtils.{method_name} failed: {e}")
                    raise

            return wrapper

        return decorator

    # -------------------------------------------------
    # Script / path helpers
    # -------------------------------------------------
    @staticmethod
    @_log_exceptions.__func__("script_dir")
    def script_dir(file_dunder: str) -> Path:
        """
        Return the absolute directory containing the given script.

        Usage:
            SCRIPT_DIR = FileUtils.script_dir(__file__)
        """
        return Path(file_dunder).resolve().parent

    @staticmethod
    @_log_exceptions.__func__("ensure_dir")
    def ensure_dir(path: StrPath) -> Path:
        """
        Create the directory if it does not exist and return it as Path.
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # -------------------------------------------------
    # Output path construction
    # -------------------------------------------------
    @staticmethod
    @_log_exceptions.__func__("build_output_path")
    def build_output_path(
        script_file: str,
        filename: str,
        subfolder: Optional[str] = None,
    ) -> Path:
        """
        Build an output path relative to the script's directory.

        If `subfolder` is provided, it is created under the script folder.
        """
        base = ReportUtils.script_dir(script_file)

        if subfolder:
            base = ReportUtils.ensure_dir(base / subfolder)

        return base / filename

    # -------------------------------------------------
    # Report utilities
    # -------------------------------------------------
    @staticmethod
    @_log_exceptions.__func__("save_html_report")
    def save_html_report(
        script_file: str,
        filename: str,
        html_text: str,
        subfolder: Optional[str] = None,
        open_in_browser: bool = True,
    ) -> Path:
        """
        Save an HTML report relative to the script's directory
        and optionally open it in the default browser.

        HTML is written exactly as provided (CSS-safe).
        """

        output_path = ReportUtils.build_output_path(
            script_file=script_file,
            filename=filename,
            subfolder=subfolder,
        )

        output_path.write_text(html_text, encoding="utf-8")

        if open_in_browser:
            try:
                webbrowser.open(output_path.as_uri())
            except Exception:
                # Fallback for platforms / browsers that don't support URI
                webbrowser.open(str(output_path))

        return output_path
