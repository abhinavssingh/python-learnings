import tensorflow as tf


class LossFactory:
    @staticmethod
    def create(framework: str, loss_name: str, **kwargs):

        framework = framework.lower()
        loss_name = loss_name.lower()

        if framework == "tensorflow":

            losses = {
                "binary_crossentropy": tf.keras.losses.BinaryCrossentropy(),
                "categorical_crossentropy": tf.keras.losses.CategoricalCrossentropy(),
                "sparse_categorical_crossentropy": tf.keras.losses.SparseCategoricalCrossentropy(),
                "mse": tf.keras.losses.MeanSquaredError(),
                "mean_squared_error": tf.keras.losses.MeanSquaredError(),
                "mae": tf.keras.losses.MeanAbsoluteError(),
                "mean_absolute_error": tf.keras.losses.MeanAbsoluteError(),
                "huber": tf.keras.losses.Huber(),
            }

            if loss_name not in losses:
                raise ValueError(f"Unsupported TensorFlow loss: {loss_name}")

            return losses[loss_name]

        elif framework == "pytorch":
            raise NotImplementedError("PyTorch loss functions not implemented.")

        raise ValueError(f"Unsupported framework: {framework}")
