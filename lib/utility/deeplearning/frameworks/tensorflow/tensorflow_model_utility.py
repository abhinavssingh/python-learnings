import tensorflow as tf

from lib.utility.common.base_experiment_utility import (
    BaseExperimentUtility,
)
from lib.utility.deeplearning.frameworks.tensorflow.training.tensorflow_trainer import (
    TensorFlowTrainer,
)


class TensorFlowModelUtility(BaseExperimentUtility):
    """
    High-level TensorFlow model utility.

    Responsibilities:
        - Model orchestration
        - Training
        - Evaluation
        - Prediction
        - Persistence
        - Experiment tracking
    """

    def __init__(
        self,
        model_wrapper,
        config,
    ):
        super().__init__()

        self.model_wrapper = model_wrapper
        self.model = model_wrapper.get_model()

        self.config = config

        self.trainer = TensorFlowTrainer(
            model=self.model,
            config=config,
        )

        self.history = None

    # =====================================================
    # Model Access
    # =====================================================

    def get_model(self) -> tf.keras.Model:
        """
        Return underlying TensorFlow model.
        """
        return self.model

    def summary(self):
        """
        Display model summary.
        """
        self.model.summary()

    # =====================================================
    # Compilation
    # =====================================================

    def compile(
        self,
        metrics: list[str] | None = None,
    ) -> None:
        """
        Compile model using trainer factories.
        """

        self.trainer.compile_model(
            metrics=metrics
        )

    # =====================================================
    # Training
    # =====================================================

    def train(
        self,
        X_train,
        y_train,
        validation_data=None,
    ):
        """
        Train the model.
        """

        self.start_experiment(
            self.model_wrapper.model_name
        )

        self.history = self.trainer.fit(
            X_train=X_train,
            y_train=y_train,
            validation_data=validation_data,
        )

        self.end_experiment()

        return self.history

    # =====================================================
    # Evaluation
    # =====================================================

    def evaluate(
        self,
        X_test,
        y_test,
    ) -> dict:
        """
        Evaluate trained model.
        """

        metrics = self.trainer.evaluate(
            X_test=X_test,
            y_test=y_test,
        )

        self.add_metrics(metrics)

        return metrics

    # =====================================================
    # Prediction
    # =====================================================

    def predict(
        self,
        X,
    ):
        """
        Generate predictions.
        """

        return self.trainer.predict(X)

    # =====================================================
    # Persistence
    # =====================================================

    def save_model(
        self,
        filepath: str,
    ) -> None:
        """
        Save TensorFlow model.
        """

        self.model.save(filepath)

    def load_model(
        self,
        filepath: str,
    ) -> tf.keras.Model:
        """
        Load TensorFlow model.
        """

        self.model = tf.keras.models.load_model(
            filepath
        )

        self.trainer.model = self.model

        return self.model

    # =====================================================
    # Reporting
    # =====================================================

    def get_training_history(self):
        """
        Return training history.
        """

        return self.history

    def get_results(self) -> dict:
        """
        Return experiment report.
        """

        return self.generate_report()
