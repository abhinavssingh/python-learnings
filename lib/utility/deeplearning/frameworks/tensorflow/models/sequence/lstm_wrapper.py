import tensorflow as tf

from ..functional_wrapper import FunctionalWrapper


class LSTMWrapper(FunctionalWrapper):
    def __init__(
        self,
        input_shape: tuple,
        output_units: int = 1,
        output_activation: str = "sigmoid",
        lstm_units: int = 64,
        dropout_rate: float = 0.3,
    ):
        self.input_shape = input_shape
        self.output_units = output_units
        self.output_activation = output_activation
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate

        super().__init__()

    def build_model(self) -> tf.keras.Model:

        inputs = tf.keras.Input(
            shape=self.input_shape,
            name="input_layer",
        )

        x = tf.keras.layers.LSTM(self.lstm_units)(inputs)

        x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        outputs = tf.keras.layers.Dense(
            self.output_units,
            activation=self.output_activation,
        )(x)

        return tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="LSTM",
        )
