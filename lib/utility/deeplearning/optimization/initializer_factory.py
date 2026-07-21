import tensorflow as tf


class InitializerFactory:
    @staticmethod
    def create(framework: str, initializer_name: str, **kwargs):

        framework = framework.lower()
        initializer_name = initializer_name.lower()

        if framework == "tensorflow":

            initializers = {
                "he_normal": tf.keras.initializers.HeNormal(),
                "he_uniform": tf.keras.initializers.HeUniform(),
                "glorot_normal": tf.keras.initializers.GlorotNormal(),
                "glorot_uniform": tf.keras.initializers.GlorotUniform(),
                "lecun_normal": tf.keras.initializers.LecunNormal(),
                "lecun_uniform": tf.keras.initializers.LecunUniform(),
                "random_normal": tf.keras.initializers.RandomNormal(),
                "random_uniform": tf.keras.initializers.RandomUniform(),
                "zeros": tf.keras.initializers.Zeros(),
                "ones": tf.keras.initializers.Ones(),
            }

            if initializer_name not in initializers:
                raise ValueError(
                    f"Unsupported TensorFlow initializer: " f"{initializer_name}"
                )

            return initializers[initializer_name]

        elif framework == "pytorch":
            raise NotImplementedError("PyTorch initializers not implemented.")

        raise ValueError(f"Unsupported framework: {framework}")
