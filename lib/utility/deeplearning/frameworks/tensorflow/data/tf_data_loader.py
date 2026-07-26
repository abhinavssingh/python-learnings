from typing import Any

import tensorflow as tf

from lib.utility.deeplearning.abstractions.base_data_loader import (
    BaseDataLoader,
)
from lib.utility.deeplearning.frameworks.tensorflow.data.tf_dataset_builder import (
    TFDatasetBuilder,
)


class TFDataLoader(
    BaseDataLoader[tf.data.Dataset]
):
    """
    TensorFlow data loader responsible for
    preparing tf.data.Dataset objects.

    This class does NOT load files.

    File loading is handled by the generic
    DataLoader service.

    Responsibilities:

    - Accept processed X and y
    - Build tf.data.Dataset
    - Support train/test datasets
    """

    def __init__(
        self,
        X: Any,
        y: Any = None,
        batch_size: int = 32,
        shuffle: bool = True,
        prefetch: bool = True,
    ):
        self.X = X
        self.y = y

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.prefetch = prefetch

    # =====================================================
    # RAW DATA
    # =====================================================

    def load_data(
        self,
    ) -> tuple[Any, Any]:

        return self.X, self.y

    # =====================================================
    # PREPROCESS
    # =====================================================

    def preprocess(
        self,
        X: Any,
        y: Any = None,
    ) -> tuple[Any, Any]:

        return X, y

    # =====================================================
    # TF DATASET
    # =====================================================

    def get_dataset(
        self,
    ) -> tf.data.Dataset:

        X, y = self.load_data()

        X, y = self.preprocess(
            X,
            y,
        )

        return TFDatasetBuilder.build(
            X=X,
            y=y,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            prefetch=self.prefetch,
        )

    # =====================================================
    # FACTORY METHODS
    # =====================================================

    @staticmethod
    def build_training_dataset(
        X,
        y,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> tf.data.Dataset:

        return TFDatasetBuilder.build(
            X=X,
            y=y,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    @staticmethod
    def build_inference_dataset(
        X,
        batch_size: int = 32,
    ) -> tf.data.Dataset:

        return TFDatasetBuilder.build(
            X=X,
            y=None,
            batch_size=batch_size,
            shuffle=False,
        )
