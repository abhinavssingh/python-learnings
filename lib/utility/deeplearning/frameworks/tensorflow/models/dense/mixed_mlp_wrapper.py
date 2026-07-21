import tensorflow as tf

from ..functional_wrapper import FunctionalWrapper


class MixedMLPWrapper(FunctionalWrapper):

    def __init__(
        self,
        input_dim,
        output_units=1,
        output_activation="sigmoid",
        dropout_rate=0.3
    ):
        self.input_dim = input_dim
        self.output_units = output_units
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate

        super().__init__()

    def build_model(self):

        inputs = tf.keras.Input(
            shape=(self.input_dim,),
            name="input_layer"
        )

        x1 = tf.keras.layers.Dense(
            128,
            activation="relu"
        )(inputs)

        x1 = tf.keras.layers.BatchNormalization()(x1)

        x1 = tf.keras.layers.Dropout(
            self.dropout_rate
        )(x1)

        x2 = tf.keras.layers.Dense(
            128,
            activation="relu"
        )(x1)

        residual = tf.keras.layers.Add()(
            [x1, x2]
        )

        x = tf.keras.layers.Dense(
            64,
            activation="relu"
        )(residual)

        outputs = tf.keras.layers.Dense(
            self.output_units,
            activation=self.output_activation
        )(x)

        return tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="MixedMLP"
        )
