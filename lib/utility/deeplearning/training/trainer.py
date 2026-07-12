import tensorflow as tf


class Trainer:

    def __init__(self, model, config):
        self.model = model
        self.config = config

    def compile_model(self, loss, metrics=None):

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.learning_rate
        )

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics or ["accuracy"]
        )

    def fit(self, X_train, y_train):

        history = self.model.fit(
            X_train,
            y_train,
            validation_split=self.config.validation_split,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=self.config.verbose
        )

        return history

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(
            X_test,
            y_test,
            verbose=0
        )

    def predict(self, X):
        return self.model.predict(X)
