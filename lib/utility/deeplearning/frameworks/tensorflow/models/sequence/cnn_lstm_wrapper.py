import tensorflow as tf

from ..functional_wrapper import FunctionalWrapper


class CNNLSTMWrapper(FunctionalWrapper):
    def __init__(
        self,
        max_nb_words: int,
        max_sequence_length: int,
        embedding_dim: int = 50,
        conv_filters: int = 64,
        conv_kernel_size: int = 5,
        pool_size: int = 5,
        lstm_units: int = 64,
        dropout_rate: float = 0.2,
        output_units: int = 2,
        output_activation: str = "softmax",
    ):
        self.max_nb_words = max_nb_words
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.pool_size = pool_size
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.output_units = output_units
        self.output_activation = output_activation

        super().__init__()

    def build_model(self) -> tf.keras.Model:
        inputs = tf.keras.Input(
            shape=(self.max_sequence_length,),
            dtype="int32",
            name="input_layer",
        )

        x = tf.keras.layers.Embedding(
            input_dim=self.max_nb_words,
            output_dim=self.embedding_dim,
        )(inputs)

        x = tf.keras.layers.Conv1D(
            filters=self.conv_filters,
            kernel_size=self.conv_kernel_size,
            activation="relu",
        )(x)

        x = tf.keras.layers.MaxPooling1D(pool_size=self.pool_size)(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        x = tf.keras.layers.Conv1D(
            filters=self.conv_filters,
            kernel_size=self.conv_kernel_size,
            activation="relu",
        )(x)

        x = tf.keras.layers.MaxPooling1D(pool_size=self.pool_size)(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        x = tf.keras.layers.LSTM(self.lstm_units)(x)

        outputs = tf.keras.layers.Dense(
            units=self.output_units,
            activation=self.output_activation,
        )(x)

        return tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="CNN_LSTM",
        )
