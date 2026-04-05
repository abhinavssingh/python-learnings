from __future__ import annotations

import io
from typing import Any, Callable, Iterable, Literal, Set, Union

import numpy as np
import pandas as pd

from lib.utility.logger import Logger

InsertValue = Union[Any, pd.Series]
ValueFactory = Callable[[pd.DataFrame], InsertValue]


class DataFrameHelper:
    """
    Utility class for DataFrame manipulation, calendar enrichment,
    dtype optimization, and inspection.
    """

    # ===============================
    # Column insertion utilities
    # ===============================
    @staticmethod
    def insert_column_after(
        df: pd.DataFrame,
        after_col: str,
        new_col: str,
        values: Union[InsertValue, ValueFactory],
        inplace: bool = False,
    ) -> pd.DataFrame:

        if after_col not in df.columns:
            raise KeyError(f"Column '{after_col}' not found")

        target_df = df if inplace else df.copy()

        if new_col in target_df.columns:
            raise ValueError(f"Column '{new_col}' already exists")

        # Resolve callable BEFORE insert (type‑safe)
        resolved_values: InsertValue
        if callable(values):
            resolved_values = values(target_df)
        else:
            resolved_values = values

        pos = target_df.columns.get_loc(after_col)
        if not isinstance(pos, int):
            raise TypeError("Ambiguous column location")

        target_df.insert(pos + 1, new_col, resolved_values)
        return target_df

    # ===============================
    # Fiscal calendar utilities
    # ===============================
    FISCAL_YEAR_START_MONTH = {
        "India": 4,
        "USA": 10,
        "UK": 4,
        "Australia": 7,
        "Japan": 4,
    }

    # ✅ Define all supported fields centrally
    ALL_CALENDAR_FIELDS: Set[str] = {
        "year",
        "month",
        "month_name",
        "day_name",
        "weekday",
        "is_weekend",
        "quarter",
    }

    ALL_FISCAL_FIELDS: Set[str] = {
        "fiscal_year",
        "fiscal_quarter",
        "fy_label",
        "fq_label",
    }

    @staticmethod
    def add_fiscal_calendar(
        df: pd.DataFrame,
        date_col: str,
        country: str,
        *,
        calendar_fields: Iterable[str] | None = None,
        fiscal_fields: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """
        Add calendar and fiscal fields derived from a date column.

        Behavior
        --------
        - If calendar_fields and fiscal_fields are None:
            → add ALL supported fields
        - If fields are provided:
            → add ONLY the specified fields
        """

        if country not in DataFrameHelper.FISCAL_YEAR_START_MONTH:
            raise ValueError(f"Fiscal rules not defined for {country}")

        # ✅ Resolve effective field sets
        calendar_fields = (
            DataFrameHelper.ALL_CALENDAR_FIELDS
            if calendar_fields is None
            else set(calendar_fields)
        )

        fiscal_fields = (
            DataFrameHelper.ALL_FISCAL_FIELDS
            if fiscal_fields is None
            else set(fiscal_fields)
        )

        out = df.copy()
        dt = out[date_col]
        fy_start = DataFrameHelper.FISCAL_YEAR_START_MONTH[country]

        # ==========================
        # Calendar fields
        # ==========================
        if "year" in calendar_fields:
            out["Year"] = dt.dt.year

        if "month" in calendar_fields:
            out["Month"] = dt.dt.month

        if "month_name" in calendar_fields:
            out["Month_Name"] = dt.dt.month_name()

        if "day_name" in calendar_fields:
            out["Day_Name"] = dt.dt.day_name()

        if {"weekday", "is_weekend"} & calendar_fields:
            out["_Weekday"] = dt.dt.weekday

        if "weekday" in calendar_fields:
            out["Weekday"] = out["_Weekday"]

        if "is_weekend" in calendar_fields:
            out["IsWeekend"] = out["_Weekday"].isin([5, 6])

        if "quarter" in calendar_fields:
            out["Calendar_Quarter"] = dt.dt.quarter

        # ==========================
        # Fiscal fields
        # ==========================
        if fiscal_fields:
            month = dt.dt.month
            year = dt.dt.year

            out["_FY_Start"] = np.where(month >= fy_start, year, year - 1)
            out["_FY_End"] = out["_FY_Start"] + 1
            out["_Fiscal_Month"] = ((month - fy_start) % 12) + 1
            out["_Fiscal_Quarter"] = ((out["_Fiscal_Month"] - 1) // 3) + 1

        if "fiscal_year" in fiscal_fields:
            out["Fiscal_Year"] = (
                out["_FY_Start"].astype(str)
                + "-"
                + out["_FY_End"].astype(str).str[-2:]
            )

        if "fiscal_quarter" in fiscal_fields:
            out["Fiscal_Quarter"] = out["_Fiscal_Quarter"]

        if "fy_label" in fiscal_fields:
            out["FY_Label"] = "FY " + out["Fiscal_Year"]

        if "fq_label" in fiscal_fields:
            out["FQ_Label"] = "Q" + out["_Fiscal_Quarter"].astype(str)

        # ==========================
        # Cleanup internal columns
        # ==========================
        out.drop(
            columns=[c for c in out.columns if c.startswith("_")],
            errors="ignore",
            inplace=True,
        )

        return out

    # ===============================
    # Numeric dtype optimization
    # ===============================

    @staticmethod
    def optimize_numeric_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """
        Downcast integer and float columns to the smallest safe dtype
        based on value ranges.
        """

        before = df.memory_usage(deep=True).sum()

        for col in df.columns:
            dtype = df[col].dtype

            # -------- INTEGER TYPES --------
            if pd.api.types.is_integer_dtype(dtype):
                c_min, c_max = df[col].min(), df[col].max()

                if c_min >= 0:
                    if c_max <= np.iinfo(np.uint8).max:
                        df[col] = df[col].astype(np.uint8)
                    elif c_max <= np.iinfo(np.uint16).max:
                        df[col] = df[col].astype(np.uint16)
                    elif c_max <= np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)
                else:
                    if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)

            # -------- FLOAT TYPES --------
            elif pd.api.types.is_float_dtype(dtype):
                c_min, c_max = df[col].min(), df[col].max()

                if (
                    c_min >= np.finfo(np.float16).min
                    and c_max <= np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min >= np.finfo(np.float32).min
                    and c_max <= np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)

        after = df.memory_usage(deep=True).sum()

        Logger.info(
            f"Memory usage before optimization: {before / 1024**2:.2f} MB")
        Logger.info(
            f"Memory usage after optimization: {after / 1024**2:.2f} MB")
        Logger.info(
            f"Memory reduction: {(before - after) / before * 100:.2f}%")

        return df

    # ===============================
    # DataFrame inspection
    # ===============================
    @staticmethod
    def get_dataframe_info_str(df: pd.DataFrame) -> str:
        """
        Return the output of DataFrame.info() as a string.
        """
        buffer = io.StringIO()
        df.info(buf=buffer)
        return buffer.getvalue()

    # ===============================================================================
    # DataFrame rendering information as preformatted text blocks
    # using dynamic column detection and safe access patterns.
    # using itertuples by default for better performance, with an option for iterrows.
    # ===============================================================================
    @staticmethod
    def dataframe_rows_as_pre(
        df: pd.DataFrame,
        method: Literal["itertuples", "iterrows"] = "itertuples",
        include_index: bool = True,
        index_label: str = "Index",
    ) -> str:
        """
        Convert each DataFrame row into preformatted text blocks
        using dynamic column detection.

        ✅ Safe for column names with spaces / special characters
        """

        blocks: list[str] = []
        columns = list(df.columns)

        if method == "itertuples":
            for row in df.itertuples():
                row_dict = row._asdict()
                lines = []

                if include_index:
                    lines.append(f"{index_label:<15}: {row_dict['Index']}")

                for col in columns:
                    value = row_dict.get(col.replace(
                        " ", "_"), row_dict.get(col))
                    lines.append(f"{col:<15}: {value}")

                blocks.append("\n".join(lines))

        elif method == "iterrows":
            for idx, row in df.iterrows():
                lines = []

                if include_index:
                    lines.append(f"{index_label:<15}: {idx}")

                for col in columns:
                    lines.append(f"{col:<15}: {row[col]}")

                blocks.append("\n".join(lines))

        else:
            raise ValueError(
                "Invalid method. Use 'itertuples' (default) or 'iterrows'."
            )

        return "\n\n".join(blocks)
