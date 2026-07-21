import tensorflow as tf


class RegularizerFactory:
    @staticmethod
    def create(
        framework: str,
        regularizer_name: str | None,
        **kwargs,
    ):

        if regularizer_name is None:
            return None

        framework = framework.lower()
        regularizer_name = regularizer_name.lower()

        if framework == "tensorflow":

            l1_value = kwargs.get("l1", 0.01)
            l2_value = kwargs.get("l2", 0.01)

            regularizers = {
                "l1": tf.keras.regularizers.L1(l1=l1_value),
                "l2": tf.keras.regularizers.L2(l2=l2_value),
                "l1_l2": tf.keras.regularizers.L1L2(
                    l1=l1_value,
                    l2=l2_value,
                ),
                "none": None,
            }

            if regularizer_name not in regularizers:
                raise ValueError(
                    f"Unsupported TensorFlow regularizer: " f"{regularizer_name}"
                )

            return regularizers[regularizer_name]

        elif framework == "pytorch":
            raise NotImplementedError("PyTorch regularizers not implemented.")

        raise ValueError(f"Unsupported framework: {framework}")
