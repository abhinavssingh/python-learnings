import tensorflow as tf


class TFDatasetBuilder:

    @staticmethod
    def build(
        X,
        y=None,
        batch_size: int = 32,
        shuffle: bool = True,
        prefetch: bool = True,
        cache: bool = False,
    ) -> tf.data.Dataset:
        """
        Build TensorFlow dataset.
        """

        if y is not None:

            dataset = tf.data.Dataset.from_tensor_slices(
                (X, y)
            )

        else:

            dataset = tf.data.Dataset.from_tensor_slices(
                X
            )

        if cache:

            dataset = dataset.cache()

        if shuffle:

            dataset = dataset.shuffle(
                buffer_size=len(X)
            )

        dataset = dataset.batch(
            batch_size
        )

        if prefetch:

            dataset = dataset.prefetch(
                tf.data.AUTOTUNE
            )

        return dataset

    # ==================================================
    # TRAIN / TEST DATASETS
    # ==================================================

    @staticmethod
    def build_train_test(
        X_train,
        y_train,
        X_test,
        y_test,
        batch_size: int = 32,
    ):

        train_ds = TFDatasetBuilder.build(
            X_train,
            y_train,
            batch_size=batch_size,
            shuffle=True,
        )

        test_ds = TFDatasetBuilder.build(
            X_test,
            y_test,
            batch_size=batch_size,
            shuffle=False,
        )

        return train_ds, test_ds
