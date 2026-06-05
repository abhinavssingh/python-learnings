import pandas as pd
from sklearn.model_selection import train_test_split

from lib.utility.machinelearning.evaluation.Metrics import Metrics
from lib.utility.machinelearning.experiment.ExperimentRunner import ExperimentRunner
from lib.utility.machinelearning.pipeline.Preprocessor import Preprocessor
from lib.utility.machinelearning.registry.ModelRegistry import ModelRegistry


class ClassificationModelUtility:

    def __init__(self, df, target_col, imputer=None, outlier_handler=None):

        self.df = df
        self.target_col = target_col

        self.imputer = imputer
        self.outlier_handler = outlier_handler

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.results = []

        self.registry = ModelRegistry()
        self.runner = None
        self.preprocessor = None

    # ----------------------------
    def prepare_data(self, test_size=0.2, random_state=42):

        df = self.df.copy()

        if self.imputer:
            df = self.imputer.fit_transform(df)

        if self.outlier_handler:
            df = self.outlier_handler.fit_transform(df)

        X = df.drop(self.target_col, axis=1)
        y = df[self.target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

        self.preprocessor = Preprocessor(
            X,
            imputer=self.imputer,
            outlier_handler=self.outlier_handler
        ).build()

        self.runner = ExperimentRunner(self.preprocessor)

    # ----------------------------
    def run_experiment(self, model_name):

        if self.X_train is None:
            raise ValueError("Call prepare_data() before running experiments")

        wrapper = self.registry.get_model(model_name)

        wrapper.build_pipeline(self.preprocessor)
        wrapper.train(self.X_train, self.y_train)

        preds = wrapper.predict(self.X_test)

        metrics = Metrics.classification(self.y_test, preds)

        result = {
            "model": model_name,
            "experiment": f"{model_name} | classification",
            "mode": "train-test",
            "type": "baseline",
            **metrics
        }

        self.results.append(result)
        return result

    # ----------------------------
    def run_all_models(self):

        models = self.registry.get_models_by_task("classification")

        results = []

        for name in models.keys():
            results.append(self.run_experiment(name))

        return pd.DataFrame(results)
