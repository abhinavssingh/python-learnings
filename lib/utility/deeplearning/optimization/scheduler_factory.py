import tensorflow as tf


class SchedulerFactory:
    @staticmethod
    def create(framework: str, scheduler_name: str | None, **kwargs):

        if scheduler_name is None:
            return None

        framework = framework.lower()
        scheduler_name = scheduler_name.lower()

        if framework == "tensorflow":

            learning_rate = kwargs.get("learning_rate", 0.001)

            decay_steps = kwargs.get("decay_steps", 1000)

            decay_rate = kwargs.get("decay_rate", 0.96)

            if scheduler_name == "exponential_decay":

                return tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=learning_rate,
                    decay_steps=decay_steps,
                    decay_rate=decay_rate,
                    staircase=True,
                )

            elif scheduler_name == "cosine_decay":

                return tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=learning_rate,
                    decay_steps=decay_steps,
                )

            elif scheduler_name == "piecewise_constant":

                return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                    boundaries=kwargs.get("boundaries", [1000, 5000]),
                    values=kwargs.get(
                        "values",
                        [
                            learning_rate,
                            learning_rate * 0.1,
                            learning_rate * 0.01,
                        ],
                    ),
                )

            elif scheduler_name == "polynomial_decay":

                return tf.keras.optimizers.schedules.PolynomialDecay(
                    initial_learning_rate=learning_rate,
                    decay_steps=decay_steps,
                    end_learning_rate=kwargs.get("end_learning_rate", 1e-5),
                    power=kwargs.get("power", 1.0),
                )

            elif scheduler_name == "reduce_lr_on_plateau":

                return tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=kwargs.get("monitor", "val_loss"),
                    factor=kwargs.get("factor", 0.5),
                    patience=kwargs.get("patience", 3),
                    min_lr=kwargs.get("min_lr", 1e-6),
                    verbose=1,
                )

            raise ValueError(f"Unsupported TensorFlow scheduler: " f"{scheduler_name}")

        elif framework == "pytorch":
            raise NotImplementedError("PyTorch schedulers not implemented.")

        raise ValueError(f"Unsupported framework: {framework}")
