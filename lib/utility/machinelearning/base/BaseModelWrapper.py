from lib.utility.machinelearning.base.MLModelBase import MLModelBase


class BaseModelWrapper(MLModelBase):

    def __init__(self, model):
        self.model = model
        self.pipeline = None
        self.imbalance_handler = None  # ✅ Injected dynamically

    def set_imbalance_handler(self, handler):
        """
        Inject imbalance handler (e.g., SMOTE)
        """
        self.imbalance_handler = handler

    def get_pipeline(self):
        if self.pipeline is None:
            raise ValueError(
                f"Pipeline not built for model: {type(self.model).__name__}"
            )
        return self.pipeline

    def train(self, X, y):
        """
        Train model using constructed pipeline.
        SMOTE (if configured) runs inside pipeline safely.
        """
        pipeline = self.get_pipeline()
        pipeline.fit(X, y)

    def predict(self, X):
        pipeline = self.get_pipeline()
        return pipeline.predict(X)
