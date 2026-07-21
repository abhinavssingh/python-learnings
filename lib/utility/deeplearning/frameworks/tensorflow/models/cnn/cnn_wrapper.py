import tensorflow as tf

from ..functional_wrapper import FunctionalWrapper


class CNNWrapper(FunctionalWrapper):
    def __init__(
        self,
        input_shape: tuple,
        num_classes: int,
        output_activation: str = "softmax",
        dropout_rate: float = 0.3,
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate

        super().__init__()

    def build_model(self) -> tf.keras.Model:

        inputs = tf.keras.Input(
            shape=self.input_shape,
            name="input_layer",
        )

        x = tf.keras.layers.Conv2D(
            32,
            3,
            activation="relu",
            padding="same",
        )(inputs)

        x = tf.keras.layers.MaxPooling2D()(x)

        x = tf.keras.layers.Conv2D(
            64,
            3,
            activation="relu",
            padding="same",
        )(x)

        x = tf.keras.layers.MaxPooling2D()(x)

        x = tf.keras.layers.Conv2D(
            128,
            3,
            activation="relu",
            padding="same",
        )(x)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        outputs = tf.keras.layers.Dense(
            self.num_classes,
            activation=self.output_activation,
        )(x)

        return tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="CNN",
        )
