import tensorflow as tf

from lib.utility.deeplearning.optimization.initializer_factory import (
    InitializerFactory,
)
from lib.utility.deeplearning.optimization.regularizer_factory import (
    RegularizerFactory,
)

from ..tensorflow_model_wrapper import TensorFlowModelWrapper


class MLPWrapper(TensorFlowModelWrapper):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list[int],
        activation: str = "relu",
        output_activation: str | None = None,
        dropout_rate: float = 0.0,
        initializer: str = "he_normal",
        regularizer: str | None = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers

        self.activation = activation
        self.output_activation = output_activation

        self.dropout_rate = dropout_rate
        self.initializer = initializer
        self.regularizer = regularizer

        self.model = self.build_model()

    def build_model(self):

        initializer = InitializerFactory.create(
            framework="tensorflow",
            initializer_name=self.initializer,
        )

        regularizer = RegularizerFactory.create(
            framework="tensorflow",
            regularizer_name=self.regularizer,
        )

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
                    activation=self.activation,
                    kernel_initializer=initializer,
                    kernel_regularizer=regularizer,
                )
            )

            if self.dropout_rate > 0:
                model.add(
                    tf.keras.layers.Dropout(
                        self.dropout_rate
                    )
                )

        model.add(
            tf.keras.layers.Dense(
                self.output_dim,
                activation=self.output_activation,
            )
        )

        return model
