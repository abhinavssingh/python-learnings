import json
from pathlib import Path
from typing import Any, Literal, Union, overload

import pandas as pd

from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.logger import Logger


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

    @staticmethod
    def _handle_unnamed_columns(
        df: pd.DataFrame,
        *,
        action: Literal["drop", "rename", "ignore"] = "drop",
        rename_prefix: str = "col",
    ) -> pd.DataFrame:
        """
            Handle unnamed columns in a DataFrame.

            Parameters
            ----------
            df : pd.DataFrame
                Input DataFrame
            action : {"drop", "rename", "ignore"}, default "drop"
                How to handle unnamed columns.
            rename_prefix : str, default "col"
                Prefix to use when renaming unnamed columns.

            Returns
            -------
            pd.DataFrame
                Cleaned DataFrame
            """

        unnamed_cols = [
            col
            for col in df.columns
            if col is None
            or (isinstance(col, str) and col.strip() == "")
            or (isinstance(col, str) and col.startswith("Unnamed"))
        ]

        if not unnamed_cols:
            return df  # ✅ Nothing to do

        Logger.warn(f"Detected unnamed columns: {unnamed_cols}")

        if action == "drop":
            df = df.drop(columns=unnamed_cols)
            Logger.info("Unnamed columns dropped")

        elif action == "rename":
            new_names = {}
            for i, col in enumerate(unnamed_cols, start=1):
                new_names[col] = f"{rename_prefix}_{i}"

            df = df.rename(columns=new_names)
            Logger.info(f"Unnamed columns renamed: {new_names}")

        elif action == "ignore":
            pass

        else:
            raise ValueError(
                "Invalid action. Use 'drop', 'rename', or 'ignore'."
            )

        return df

    @overload
    @classmethod
    def read_dataset(
        cls,
        filename: str,
        *,
        optimize: bool = ...,
        handle_unnamed: Literal["drop", "rename", "ignore"] = ...,
        drop_rows_with_nulls: bool = ...,
        fill_numeric_with: Literal["mean", "median", None] = ...,
        return_report: Literal[True],
    ) -> tuple[pd.DataFrame, str]:
        ...

    @overload
    @classmethod
    def read_dataset(
        cls,
        filename: str,
        *,
        optimize: bool = ...,
        handle_unnamed: Literal["drop", "rename", "ignore"] = ...,
        drop_rows_with_nulls: bool = ...,
        fill_numeric_with: Literal["mean", "median", None] = ...,
        return_report: Literal[False] = ...,
    ) -> pd.DataFrame:
        ...

    # ===============================
    # Dataset reading
    # ===============================

    @classmethod
    def read_dataset(
        cls,
        filename: str,
        *,
        optimize: bool = True,
        handle_unnamed: Literal["drop", "rename", "ignore"] = "drop",
        drop_rows_with_nulls: bool = False,
        fill_numeric_with: Literal["mean", "median", None] = None,
        return_report: bool = False,
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, str], Any]:
        """
        Generic dataset reader.

        Supported:
        - CSV
        - Excel (.xls, .xlsx)
        - JSON (returns Python object)

        Parameters
        ----------
        optimize : bool
            Whether to run null check and numeric dtype optimization
        drop_rows_with_nulls : bool
            Drop rows containing nulls before optimization
        fill_numeric_with : {'mean', 'median', None}
            Strategy to fill numeric nulls
        return_report : bool
            If True, return (DataFrame, optimization_report)
        """

        path = cls.get_dataset_path(filename)
        ext = path.suffix.lower()

        # ----------------------
        # Load dataset
        # ----------------------
        if ext == ".csv":
            df = pd.read_csv(path)

        elif ext in {".xls", ".xlsx"}:
            df = pd.read_excel(path)

        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        else:
            raise ValueError(f"Unsupported dataset format: {ext}")

        # ----------------------
        # Handle unnamed columns FIRST
        # ----------------------
        df = cls._handle_unnamed_columns(
            df,
            action=handle_unnamed,
        )

        # ----------------------
        # Null check & optimization
        # ----------------------
        if optimize:
            optimized_df, pre_text = dfh.null_check_and_optimize(
                df,
                drop_rows_with_nulls=drop_rows_with_nulls,
                fill_numeric_with=fill_numeric_with,
            )

            if return_report:
                return optimized_df, pre_text

            return optimized_df

        return df
