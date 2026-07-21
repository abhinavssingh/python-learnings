import tensorflow as tf


class OptimizerFactory:
    @staticmethod
    def create(
        framework: str,
        optimizer_name: str,
        learning_rate: float,
    ):
        if framework.lower() == "tensorflow":

            optimizers = {
                "adam": tf.keras.optimizers.Adam(learning_rate=learning_rate),
                "adamw": tf.keras.optimizers.AdamW(learning_rate=learning_rate),
                "sgd": tf.keras.optimizers.SGD(learning_rate=learning_rate),
                "rmsprop": tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                "adagrad": tf.keras.optimizers.Adagrad(learning_rate=learning_rate),
            }

            if optimizer_name not in optimizers:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")

            return optimizers[optimizer_name]

        raise ValueError(f"Unsupported framework: {framework}")
