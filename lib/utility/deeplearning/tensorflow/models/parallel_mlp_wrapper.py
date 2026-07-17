import tensorflow as tf

from .functional_wrapper import FunctionalWrapper


class ParallelMLPWrapper(FunctionalWrapper):

    def __init__(
        self,
        input_dim,
        output_units=1,
        output_activation="sigmoid"
    ):
        self.input_dim = input_dim
        self.output_units = output_units
        self.output_activation = output_activation

        super().__init__()

    def build_model(self):

        inputs = tf.keras.Input(
            shape=(self.input_dim,),
            name="input_layer"
        )

        # Branch 1
        branch1 = tf.keras.layers.Dense(
            64,
            activation="relu"
        )(inputs)

        branch1 = tf.keras.layers.Dense(
            32,
            activation="relu"
        )(branch1)

        # Branch 2
        branch2 = tf.keras.layers.Dense(
            128,
            activation="relu"
        )(inputs)

        branch2 = tf.keras.layers.Dense(
            64,
            activation="relu"
        )(branch2)

        # Merge
        merged = tf.keras.layers.Concatenate()(
            [branch1, branch2]
        )

        merged = tf.keras.layers.Dense(
            32,
            activation="relu"
        )(merged)

        outputs = tf.keras.layers.Dense(
            self.output_units,
            activation=self.output_activation
        )(merged)

        return tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="ParallelMLP"
        )
