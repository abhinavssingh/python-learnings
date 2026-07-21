import tensorflow as tf

from ..functional_wrapper import FunctionalWrapper


class ResNetWrapper(FunctionalWrapper):
    def __init__(
        self,
        input_shape: tuple,
        num_classes: int,
        trainable: bool = False,
        output_activation: str = "softmax",
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.trainable = trainable
        self.output_activation = output_activation

        super().__init__()

    def build_model(self) -> tf.keras.Model:

        inputs = tf.keras.Input(
            shape=self.input_shape,
            name="input_layer",
        )

        backbone = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
        )

        backbone.trainable = self.trainable

        x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)

        outputs = tf.keras.layers.Dense(
            self.num_classes,
            activation=self.output_activation,
        )(x)

        return tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="ResNet50",
        )
