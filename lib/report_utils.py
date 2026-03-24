from __future__ import annotations

import os
import webbrowser
from pathlib import Path
from typing import Optional, Union

StrPath = Union[str, os.PathLike]


def script_dir(file_dunder: str) -> Path:
    """
    Return the absolute directory containing the current script.

    Usage in a script:  SCRIPT_DIR = script_dir(__file__)
    Works even when the script is launched with `-m` from elsewhere.
    """
    return Path(file_dunder).resolve().parent


def ensure_dir(path: StrPath) -> Path:
    """
    Create the directory if it does not exist, return it as Path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_output_path(script_file: str,
                      filename: str,
                      subfolder: Optional[str] = None) -> Path:
    """
    Build an output path, relative to the script's folder.
    If `subfolder` is provided, it is created under the script's folder.
    """
    base = script_dir(script_file)
    if subfolder:
        base = ensure_dir(base / subfolder)
    return base / filename


def save_html_report(
    script_file: str,
    filename: str,
    html_text: str,
    subfolder: Optional[str] = None,
    open_in_browser: bool = True
) -> Path:
    """
    Save an HTML report relative to the script's folder and (optionally) open it.
    Fully preserves CSS and HTML as provided.
    """

    # Build final output path
    output_path = build_output_path(script_file, filename, subfolder)

    # Write raw HTML exactly as passed (critical for Tailwind)
    output_path.write_text(html_text, encoding="utf-8")

    # Open in browser if requested
    if open_in_browser:
        try:
            webbrowser.open(output_path.as_uri())
        except Exception:
            # Windows fallback if URI opening fails
            webbrowser.open(str(output_path))

    return output_path
