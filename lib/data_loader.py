from pathlib import Path
import pandas as pd
import json


def _get_project_root() -> Path:
    """Find project root by locating path.json."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "settings.json").exists():
            return parent
    raise RuntimeError("Project root not found (settings.json missing).")


def get_dataset_path(filename: str) -> Path:
    """
    Return the absolute Path of a dataset file stored in /datasets.

    Usage:
        path = get_dataset_path("sales.csv")
    """
    root = _get_project_root()
    dataset_path = root / "datasets" / filename

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    return dataset_path


def read_dataset(filename: str):
    """
    Generic dataset reader.
    Automatically detects file type based on extension.

    Supported:
    - CSV
    - Excel (.xls, .xlsx)
    - JSON
    """

    path = get_dataset_path(filename)
    ext = path.suffix.lower()

    if ext == ".csv":
        return pd.read_csv(path)

    if ext in {".xls", ".xlsx"}:
        return pd.read_excel(path)

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(f"Unsupported dataset format: {ext}")
