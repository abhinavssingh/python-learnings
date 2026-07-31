from typing import Any, cast

import tensorflow as tf

from lib.utility.deeplearning.abstractions.base_trainer import (
    BaseTrainer,
)
from lib.utility.deeplearning.frameworks.tensorflow.training.tensorflow_training_history import (
    TensorFlowTrainingHistory,
)
from lib.utility.deeplearning.optimization.callbacks_factory import (
    CallbacksFactory,
)
from lib.utility.deeplearning.optimization.loss_factory import (
    LossFactory,
)
from lib.utility.deeplearning.optimization.optimizer_factory import (
    OptimizerFactory,
)


class TensorFlowTrainer(BaseTrainer):
    """
    TensorFlow training engine.

    Responsibilities:
        - Compile model
        - Train model
        - Evaluate model
        - Predict
        - Manage callbacks
    """

    def __init__(
        self,
        model: tf.keras.Model,
        config: Any,
    ):
        self.model = model
        self.config = config

    # =====================================================
    # Compile
    # =====================================================

    def compile_model(
        self,
        metrics: list[str] | None = None,
    ) -> None:
        """
        Compile TensorFlow model using framework factories.
        """

        optimizer = OptimizerFactory.create(
            framework="tensorflow",
            optimizer_name=self.config.optimizer,
            learning_rate=self.config.learning_rate,
        )

        loss = LossFactory.create(
            framework="tensorflow",
            loss_name=self.config.loss,
        )

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics or ["accuracy"],
        )
    # =====================================================
    # Train
    # =====================================================

    def fit(
        self,
        X_train,
        y_train,
        validation_data=None,
    ) -> TensorFlowTrainingHistory:
        """
        Train model and return normalized training history.
        """

        callbacks = CallbacksFactory.create(
            framework="tensorflow",
            config=self.config,
        )

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=getattr(self.config, "verbose", 1),
        )

        return TensorFlowTrainingHistory.from_keras_history(history)

    # =====================================================
    # Evaluate
    # =====================================================

    def evaluate(
        self,
        X_test,
        y_test,
    ) -> dict[str, float]:

        result = self.model.evaluate(
            X_test,
            y_test,
            verbose=0,
            return_dict=True,
        )

        return cast(dict[str, float], result)

    # =====================================================
    # Predict
    # =====================================================

    def predict(
        self,
        X,
    ):
        """
        Generate predictions.
        """

        return self.model.predict(
            X,
            verbose=0,
        )

    # =====================================================
    # Utilities
    # =====================================================

    def summary(self) -> None:
        """
        Display model summary.
        """

        self.model.summary()

    def get_model(self) -> tf.keras.Model:
        """
        Return underlying model.
        """

        return self.model
