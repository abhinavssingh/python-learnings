import tensorflow as tf

from ..functional_wrapper import FunctionalWrapper


class TransformerWrapper(FunctionalWrapper):
    def __init__(
        self,
        input_shape: tuple,
        output_units: int = 1,
        output_activation: str = "sigmoid",
        num_heads: int = 4,
        ff_dim: int = 128,
        dropout_rate: float = 0.1,
    ):
        self.input_shape = input_shape
        self.output_units = output_units
        self.output_activation = output_activation
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        super().__init__()

    def build_model(self) -> tf.keras.Model:

        inputs = tf.keras.Input(
            shape=self.input_shape,
            name="input_layer",
        )

        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.input_shape[-1],
        )(inputs, inputs)

        x = tf.keras.layers.Add()([inputs, attention])

        x = tf.keras.layers.LayerNormalization()(x)

        ff = tf.keras.layers.Dense(
            self.ff_dim,
            activation="relu",
        )(x)

        ff = tf.keras.layers.Dense(self.input_shape[-1])(ff)

        x = tf.keras.layers.Add()([x, ff])

        x = tf.keras.layers.LayerNormalization()(x)

        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        outputs = tf.keras.layers.Dense(
            self.output_units,
            activation=self.output_activation,
        )(x)

        return tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="Transformer",
        )
