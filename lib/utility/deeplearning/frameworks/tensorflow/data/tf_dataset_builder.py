import tensorflow as tf


class TFDatasetBuilder:
    @staticmethod
    def build(X, y=None, batch_size=32, shuffle=True, prefetch=True):

        if y is not None:
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(X)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))

        dataset = dataset.batch(batch_size)

        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
