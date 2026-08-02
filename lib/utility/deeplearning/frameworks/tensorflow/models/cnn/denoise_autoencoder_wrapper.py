import tensorflow as tf

from ..tensorflow_model_wrapper import TensorFlowModelWrapper


class DenoiseAutoencoder(tf.keras.Model):
    def __init__(self, input_shape: tuple[int, int, int]):
        super().__init__(name="DenoiseAutoencoder")

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Conv2D(
                    64,
                    (3, 3),
                    activation="relu",
                    padding="same",
                    strides=2,
                ),
                tf.keras.layers.Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    padding="same",
                    strides=2,
                ),
            ],
            name="encoder",
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2DTranspose(
                    32,
                    kernel_size=3,
                    strides=2,
                    activation="relu",
                    padding="same",
                ),
                tf.keras.layers.Conv2DTranspose(
                    64,
                    kernel_size=3,
                    strides=2,
                    activation="relu",
                    padding="same",
                ),
                tf.keras.layers.Conv2D(
                    1,
                    kernel_size=(3, 3),
                    activation="sigmoid",
                    padding="same",
                ),
            ],
            name="decoder",
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class DenoiseAutoencoderWrapper(TensorFlowModelWrapper):
    def __init__(self, input_shape: tuple[int, int, int]):
        self.input_shape = input_shape
        self.model = self.build_model()

    def build_model(self) -> tf.keras.Model:
        return DenoiseAutoencoder(input_shape=self.input_shape)
