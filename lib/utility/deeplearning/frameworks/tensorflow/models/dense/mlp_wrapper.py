import tensorflow as tf

from ..tensorflow_model_wrapper import TensorFlowModelWrapper


class MLPWrapper(TensorFlowModelWrapper):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list[int]
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers

        self.model = self.build_model()

    def build_model(self):

        model = tf.keras.Sequential(
            name="MLP"
        )

        model.add(
            tf.keras.layers.Input(
                shape=(self.input_dim,)
            )
        )

        for units in self.hidden_layers:
            model.add(
                tf.keras.layers.Dense(
                    units,
                    activation="relu"
                )
            )

        model.add(
            tf.keras.layers.Dense(
                self.output_dim
            )
        )

        return model
