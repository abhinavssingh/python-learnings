from typing import Any

import tensorflow as tf

from lib.utility.deeplearning.abstractions.base_data_loader import (
    BaseDataLoader,
)
from lib.utility.deeplearning.frameworks.tensorflow.data.tf_dataset_builder import (
    TFDatasetBuilder,
)


class TFDataLoader(BaseDataLoader[tf.data.Dataset]):
    """
    TensorFlow data loader responsible for
    preparing tf.data.Dataset objects.
    """

    def __init__(
        self,
        X: Any,
        y: Any = None,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def load_data(self) -> tuple[Any, Any]:
        """
        Load raw features and targets.
        """
        return self.X, self.y

    def preprocess(
        self,
        X: Any,
        y: Any = None,
    ) -> tuple[Any, Any]:
        """
        Override in subclasses for custom preprocessing.
        """
        return X, y

    def get_dataset(self) -> tf.data.Dataset:
        """
        Build TensorFlow dataset.
        """
        X, y = self.load_data()

        X, y = self.preprocess(X, y)

        return TFDatasetBuilder.build(
            X=X,
            y=y,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )
