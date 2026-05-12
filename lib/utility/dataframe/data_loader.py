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
    ) -> tuple[pd.DataFrame, str]:
        """
        Handle unnamed columns in a DataFrame and trim whitespace
        from column names. Returns a report string.
        """

        pre_text_lines: list[str] = []

        # ---------------------------------
        # Trim leading/trailing whitespace
        # ---------------------------------

        renamed_columns: dict[str, str] = {}

        for col in df.columns:
            if isinstance(col, str):
                stripped = col.strip()
                if col != stripped:
                    renamed_columns[col] = stripped

        if renamed_columns:
            df = df.rename(columns=renamed_columns)

            msg = (
                "Trimmed leading/trailing whitespace from column names:\n"
                + "\n".join(
                    f"  '{old}' -> '{new}'"
                    for old, new in renamed_columns.items()
                )
            )

            Logger.info(msg)
            pre_text_lines.append(msg)

        # ---------------------------------
        # Detect unnamed columns
        # ---------------------------------
        unnamed_cols = [
            col
            for col in df.columns
            if col is None
            or (isinstance(col, str) and col == "")
            or (isinstance(col, str) and col.startswith("Unnamed"))
        ]

        if not unnamed_cols:
            return df, "\n".join(pre_text_lines)

        detected_msg = f"Detected unnamed columns: {unnamed_cols}"
        Logger.warn(detected_msg)
        pre_text_lines.append(detected_msg)

        # ---------------------------------
        # Handle unnamed columns
        # ---------------------------------
        if action == "drop":
            df = df.drop(columns=unnamed_cols)
            msg = f"Dropped unnamed columns ({len(unnamed_cols)} columns)."
            Logger.info(msg)
            pre_text_lines.append(msg)

        elif action == "rename":
            new_names = {
                col: f"{rename_prefix}_{i}"
                for i, col in enumerate(unnamed_cols, start=1)
            }
            df = df.rename(columns=new_names)
            msg = f"Renamed unnamed columns: {new_names}"
            Logger.info(msg)
            pre_text_lines.append(msg)

        elif action == "ignore":
            msg = "Unnamed columns ignored."
            Logger.info(msg)
            pre_text_lines.append(msg)

        else:
            raise ValueError(
                "Invalid action. Use 'drop', 'rename', or 'ignore'."
            )

        return df, "\n".join(pre_text_lines)

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
        df, unnamed_report = cls._handle_unnamed_columns(
            df,
            action=handle_unnamed,
        )

        reports: list[str] = []

        if unnamed_report:
            reports.append(unnamed_report)

        # ----------------------
        # Null check & optimization
        # ----------------------
        if optimize:
            optimized_df, pre_text = dfh.null_check_and_optimize(
                df,
                drop_rows_with_nulls=drop_rows_with_nulls,
                fill_numeric_with=fill_numeric_with,
            )

            if pre_text:
                reports.append(pre_text)

            final_report = "\n\n".join(reports)

            if return_report:
                return optimized_df, final_report

            return optimized_df

        # No optimization case
        if return_report:
            return df, "\n\n".join(reports)

        return df
