import pandas as pd


class DataCleaner:
    """
    Handles data cleaning for visualization.
    Ensures no NaN crashes and consistent data.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def drop_na(self, cols):
        """Drop rows where key columns are missing"""
        return self.df.dropna(subset=cols)

    def fill_na(self, col, strategy="median"):
        """Fill NaN values safely"""
        if col not in self.df.columns:
            return self.df

        if strategy == "median":
            self.df[col] = self.df[col].fillna(self.df[col].median())
        elif strategy == "mean":
            self.df[col] = self.df[col].fillna(self.df[col].mean())
        elif strategy == "zero":
            self.df[col] = self.df[col].fillna(0)

        return self.df

    def ensure_columns(self, required_cols):
        """Validate required columns"""
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return self.df

    def clean(self, required_cols=None, fill_strategy="median"):
        """
        Clean dataset safely:
        - Drop rows missing required columns
        - Fill NaN in numeric columns only
        """

        df = self.df.copy()

        # ✅ Drop rows where required cols are missing
        if required_cols:
            existing_cols = [col for col in required_cols if col in df.columns]

            # ✅ Only drop if columns exist
            if existing_cols:
                df = df.dropna(subset=existing_cols)

        # ✅ Only operate on numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns

        for col in numeric_cols:
            if fill_strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif fill_strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif fill_strategy == "zero":
                df[col] = df[col].fillna(0)

        return df

    def flatten_cv_results(self, cv_results, model_name, mode):
        """
        Convert sklearn cv_results_ dict into cleaned tabular rows.

        Returns:
            List of dictionaries (each row = one parameter combination)
        """

        df = pd.DataFrame(cv_results)

        # ✅ extract param columns
        param_cols = [col for col in df.columns if col.startswith("param_")]

        if not param_cols:
            return []

        # ✅ keep only useful columns
        df = df[param_cols + ["mean_test_score"]]

        # ✅ rename for consistency
        df = df.rename(columns={"mean_test_score": "score"})

        # ✅ clean NaN safely (reuse existing logic)
        cleaner = DataCleaner(df)
        df = cleaner.clean(required_cols=param_cols + ["score"])

        # ✅ add metadata
        df["model"] = model_name
        df["type"] = "tuned"
        df["mode"] = mode

        return df.to_dict(orient="records")
