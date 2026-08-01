from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin


@dataclass
class KerasClassifierConfig:
    hidden_layers: tuple[int, ...] = (64, 32)
    activation: str = "relu"
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 25
    batch_size: int = 32
    validation_split: float = 0.2
    verbose: int = 0


class TFKerasClassifier(ClassifierMixin, BaseEstimator):
    """
    Scikit-learn compatible Keras classifier for tabular data.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        random_state: int = 42,
        hidden_layers: tuple[int, ...] = (64, 32),
        activation: str = "relu",
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 25,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 0,
    ):
        self.random_state = random_state
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.verbose = verbose

        self.model_: tf.keras.Model | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def _build_model(self, n_features: int, n_classes: int) -> tf.keras.Model:
        tf.keras.utils.set_random_seed(self.random_state)

        model = tf.keras.Sequential(name="KerasClassifier")
        model.add(tf.keras.layers.Input(shape=(n_features,)))

        for units in self.hidden_layers:
            model.add(
                tf.keras.layers.Dense(
                    units,
                    activation=self.activation,
                )
            )
            if self.dropout_rate > 0:
                model.add(tf.keras.layers.Dropout(self.dropout_rate))

        if n_classes == 2:
            model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
        else:
            model.add(tf.keras.layers.Dense(n_classes, activation="softmax"))
            loss = "sparse_categorical_crossentropy"

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=["accuracy"],
        )

        return model

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).ravel()

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        n_classes = len(self.classes_)

        class_to_index = {
            cls: idx
            for idx, cls in enumerate(self.classes_)
        }
        y_index = np.vectorize(class_to_index.get)(y)

        self.model_ = self._build_model(
            n_features=self.n_features_in_,
            n_classes=n_classes,
        )

        self.model_.fit(
            X,
            y_index,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=self.verbose,
        )

        return self

    def predict_proba(self, X):
        if self.model_ is None or self.classes_ is None:
            raise RuntimeError("Model is not fitted. Call fit first.")

        X = np.asarray(X, dtype=np.float32)
        prob = self.model_.predict(X, verbose=0)

        if prob.ndim == 1:
            prob = prob.reshape(-1, 1)

        if prob.shape[1] == 1:
            prob_pos = prob[:, 0]
            prob = np.column_stack([1.0 - prob_pos, prob_pos])

        return prob

    def predict(self, X):
        prob = self.predict_proba(X)
        class_indices = np.argmax(prob, axis=1)
        return self.classes_[class_indices]
