from pathlib import Path
import pandas as pd
import json
from typing import Union, Any

from lib.utility.dataframe.df_helper import DataFrameHelper as dfh


class DataLoader:
    """
    Service class for dataset loading and path resolution.

    Responsibilities:
    - Locate project root using settings.json
    - Resolve dataset paths
    - Read datasets (CSV, Excel, JSON)
    - Optionally optimize DataFrame dtypes
    """

    # ===============================
    # Project root resolution
    # ===============================
    @staticmethod
    def _get_project_root() -> Path:
        """Find project root by locating settings.json."""
        current = Path(__file__).resolve()

        for parent in current.parents:
            if (parent / "settings.json").exists():
                return parent

        raise RuntimeError("Project root not found (settings.json missing).")

    # ===============================
    # Dataset path resolution
    # ===============================
    @classmethod
    def get_dataset_path(cls, filename: str) -> Path:
        """
        Return the absolute Path of a dataset file stored in /datasets.
        """
        root = cls._get_project_root()
        dataset_path = root / "datasets" / filename

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        return dataset_path

    # ===============================
    # Dataset reading
    # ===============================
    @classmethod
    def read_dataset(
        cls,
        filename: str,
        *,
        optimize: bool = True,
    ) -> Union[pd.DataFrame, Any]:
        """
        Generic dataset reader.

        Supported:
        - CSV
        - Excel (.xls, .xlsx)
        - JSON (returns Python object)

        Parameters
        ----------
        filename : str
            Dataset name inside /datasets directory.
        optimize : bool, default True
            Whether to optimize numeric dtypes for DataFrames.

        Returns
        -------
        pd.DataFrame or Any
            DataFrame for tabular data, raw object for JSON.
        """

        path = cls.get_dataset_path(filename)
        ext = path.suffix.lower()

        if ext == ".csv":
            df = pd.read_csv(path)

        elif ext in {".xls", ".xlsx"}:
            df = pd.read_excel(path)

        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        else:
            raise ValueError(f"Unsupported dataset format: {ext}")

        # ✅ Optimize only DataFrames
        if optimize:
            df = dfh.optimize_numeric_dtypes(df)

        return df
