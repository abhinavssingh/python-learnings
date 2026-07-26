from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)


@dataclass
class PreprocessingResult:

    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any

    preprocessor: ColumnTransformer

    label_encoders: dict


class DataPreprocessor:

    @staticmethod
    def prepare_classification_data(
        df: pd.DataFrame,
        target_column: str,
        drop_columns: list[str] | None = None,
        label_encode_columns: list[str] | None = None,
        one_hot_columns: list[str] | None = None,
        scale_numeric: bool = True,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> PreprocessingResult:

        df = df.copy()

        # ==================================================
        # DROP COLUMNS
        # ==================================================

        if drop_columns:
            df = df.drop(
                columns=drop_columns,
                errors="ignore",
            )

        # ==================================================
        # TARGET
        # ==================================================

        X = df.drop(
            columns=[target_column]
        )

        y = df[target_column]

        # ==================================================
        # LABEL ENCODING
        # ==================================================

        label_encoders = {}

        if label_encode_columns:

            for column in label_encode_columns:

                encoder = LabelEncoder()

                X[column] = encoder.fit_transform(
                    X[column]
                )

                label_encoders[column] = encoder

        # ==================================================
        # TRAIN TEST SPLIT
        # ==================================================

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        # ==================================================
        # NUMERIC FEATURES
        # ==================================================

        numeric_features = [
            col
            for col in X.columns
            if col not in (one_hot_columns or [])
        ]

        transformers = []

        # ==================================================
        # ONE HOT ENCODING
        # ==================================================

        if one_hot_columns:

            transformers.append(
                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        drop="first",
                    ),
                    one_hot_columns,
                )
            )

        # ==================================================
        # NUMERIC PIPELINE
        # ==================================================

        if scale_numeric:

            numeric_pipeline = Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                ]
            )

            transformers.append(
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_features,
                )
            )

        preprocessor = ColumnTransformer(
            transformers=transformers
        )

        X_train = preprocessor.fit_transform(
            X_train
        )

        X_test = preprocessor.transform(
            X_test
        )

        return PreprocessingResult(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            preprocessor=preprocessor,
            label_encoders=label_encoders,
        )

    @staticmethod
    def transform_new_data(
        df: pd.DataFrame,
        preprocessor,
        label_encoders: dict,
    ):

        df = df.copy()

        for column, encoder in label_encoders.items():

            df[column] = encoder.transform(
                df[column]
            )

        return preprocessor.transform(
            df
        )
