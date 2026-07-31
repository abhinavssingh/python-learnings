from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


@dataclass
class TFImageClassificationDataBundle:
    train_generator: tf.keras.preprocessing.image.DirectoryIterator
    validation_generator: tf.keras.preprocessing.image.DirectoryIterator
    test_generator: tf.keras.preprocessing.image.DirectoryIterator
    class_indices: dict[str, int]
    class_names: list[str]
    train_samples: int
    validation_samples: int
    test_samples: int


class TFImageClassificationDataLoader:
    """
    Utility class for image classification data loading via ImageDataGenerator.

    Supports:
    - Single directory split into train/validation/test
    - Train/test directory layouts with validation split from train
    """

    @staticmethod
    def _build_dataframe_from_directory(directory: str | Path) -> pd.DataFrame:
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        class_dirs = [
            p for p in sorted(directory.iterdir()) if p.is_dir()
        ]

        if not class_dirs:
            raise ValueError(
                f"No class folders found in: {directory}"
            )

        rows: list[dict[str, str]] = []

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

        for class_dir in class_dirs:
            label = class_dir.name
            for image_path in class_dir.rglob("*"):
                if image_path.suffix.lower() in valid_extensions:
                    rows.append(
                        {
                            "filename": str(image_path.resolve()),
                            "class": label,
                        }
                    )

        if not rows:
            raise ValueError(
                f"No supported image files found in: {directory}"
            )

        return pd.DataFrame(rows)

    @staticmethod
    def _build_generators_from_dataframes(
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
        image_size: tuple[int, int] = (128, 128),
        batch_size: int = 32,
        train_augmentation: bool = True,
        seed: int = 42,
    ) -> TFImageClassificationDataBundle:
        if train_augmentation:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1.0 / 255.0,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
            )
        else:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1.0 / 255.0,
            )

        eval_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255.0,
        )

        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            x_col="filename",
            y_col="class",
            target_size=image_size,
            color_mode="rgb",
            class_mode="categorical",
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        )

        validation_generator = eval_datagen.flow_from_dataframe(
            dataframe=validation_df,
            x_col="filename",
            y_col="class",
            target_size=image_size,
            color_mode="rgb",
            class_mode="categorical",
            batch_size=batch_size,
            shuffle=False,
        )

        test_generator = eval_datagen.flow_from_dataframe(
            dataframe=test_df,
            x_col="filename",
            y_col="class",
            target_size=image_size,
            color_mode="rgb",
            class_mode="categorical",
            batch_size=batch_size,
            shuffle=False,
        )

        class_indices = train_generator.class_indices
        class_names = sorted(class_indices, key=class_indices.get)

        return TFImageClassificationDataBundle(
            train_generator=train_generator,
            validation_generator=validation_generator,
            test_generator=test_generator,
            class_indices=class_indices,
            class_names=class_names,
            train_samples=train_generator.samples,
            validation_samples=validation_generator.samples,
            test_samples=test_generator.samples,
        )

    @staticmethod
    def from_single_directory(
        directory: str | Path,
        image_size: tuple[int, int] = (128, 128),
        batch_size: int = 32,
        validation_split: float = 0.2,
        test_split: float = 0.1,
        train_augmentation: bool = True,
        seed: int = 42,
    ) -> TFImageClassificationDataBundle:
        """
        Build train/validation/test generators from one class-folder dataset.
        """
        if validation_split <= 0 or test_split <= 0:
            raise ValueError("validation_split and test_split must be > 0")

        if validation_split + test_split >= 1.0:
            raise ValueError(
                "validation_split + test_split must be < 1"
            )

        df = TFImageClassificationDataLoader._build_dataframe_from_directory(directory)

        train_df, holdout_df = train_test_split(
            df,
            test_size=validation_split + test_split,
            stratify=df["class"],
            random_state=seed,
        )

        relative_test_ratio = test_split / (validation_split + test_split)

        validation_df, test_df = train_test_split(
            holdout_df,
            test_size=relative_test_ratio,
            stratify=holdout_df["class"],
            random_state=seed,
        )

        return TFImageClassificationDataLoader._build_generators_from_dataframes(
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
            image_size=image_size,
            batch_size=batch_size,
            train_augmentation=train_augmentation,
            seed=seed,
        )

    @staticmethod
    def from_train_test_directories(
        train_directory: str | Path,
        test_directory: str | Path,
        image_size: tuple[int, int] = (128, 128),
        batch_size: int = 32,
        validation_split: float = 0.2,
        train_augmentation: bool = True,
        seed: int = 42,
    ) -> TFImageClassificationDataBundle:
        """
        Build generators when train/ and test/ folders are already available.
        """
        if validation_split <= 0 or validation_split >= 1.0:
            raise ValueError("validation_split must be in range (0, 1)")

        train_df_full = TFImageClassificationDataLoader._build_dataframe_from_directory(
            train_directory
        )
        test_df = TFImageClassificationDataLoader._build_dataframe_from_directory(
            test_directory
        )

        train_df, validation_df = train_test_split(
            train_df_full,
            test_size=validation_split,
            stratify=train_df_full["class"],
            random_state=seed,
        )

        return TFImageClassificationDataLoader._build_generators_from_dataframes(
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
            image_size=image_size,
            batch_size=batch_size,
            train_augmentation=train_augmentation,
            seed=seed,
        )
