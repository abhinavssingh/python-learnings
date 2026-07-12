import tensorflow as tf

from .base_model_wrapper import BaseModelWrapper


class MLPWrapper(BaseModelWrapper):

    def __init__(
        self,
        input_dim,
        hidden_layers=[64, 32],
        output_units=1,
        output_activation="sigmoid"
    ):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_units = output_units
        self.output_activation = output_activation

        self.model = self.build_model()

    def build_model(self):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.Input(shape=(self.input_dim,))
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
                self.output_units,
                activation=self.output_activation
            )
        )

        return model

    def get_model(self):
        return self.model
