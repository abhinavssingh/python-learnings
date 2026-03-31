from __future__ import annotations
from typing import Literal

from typing import Any, Callable, Union
import pandas as pd
import numpy as np
import io

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

    @staticmethod
    def add_fiscal_calendar(
        df: pd.DataFrame,
        date_col: str,
        country: str,
    ) -> pd.DataFrame:
        """
        Adds calendar and fiscal fields based on country‑specific rules.
        """

        if country not in DataFrameHelper.FISCAL_YEAR_START_MONTH:
            raise ValueError(f"Fiscal rules not defined for {country}")

        fy_start = DataFrameHelper.FISCAL_YEAR_START_MONTH[country]
        df = df.copy()

        df["Year"] = df[date_col].dt.year
        df["Month"] = df[date_col].dt.month
        df["Day"] = df[date_col].dt.day_name()
        df["Month_Name"] = df[date_col].dt.month_name()
        df["Weekday"] = df[date_col].dt.weekday
        df["IsWeekend"] = df["Weekday"].isin([5, 6])
        df["Calendar_Quarter"] = df[date_col].dt.quarter

        df["FY_Start"] = np.where(
            df["Month"] >= fy_start,
            df["Year"],
            df["Year"] - 1,
        )

        df["FY_End"] = df["FY_Start"] + 1

        df["Fiscal_Month"] = ((df["Month"] - fy_start) % 12) + 1
        df["Fiscal_Quarter"] = ((df["Fiscal_Month"] - 1) // 3) + 1

        df["Fiscal_Year"] = (
            df["FY_Start"].astype(str)
            + "-"
            + df["FY_End"].astype(str).str[-2:]
        )

        df["FY_Label"] = "FY " + df["Fiscal_Year"]
        df["FQ_Label"] = "Q" + df["Fiscal_Quarter"].astype(str)

        return df

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
                    value = row_dict.get(col.replace(" ", "_"), row_dict.get(col))
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
