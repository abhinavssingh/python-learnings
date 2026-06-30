import json
import os

import joblib

from .ClassificationInference import ClassificationInference
from .RegressionInference import RegressionInference
from .UnsupervisedInference import UnsupervisedInference


from lib.utility.machinelearning._logging import ExceptionLoggingMixin


class InferenceFactory(ExceptionLoggingMixin):

    @staticmethod
    def load(model_path):

        pipeline = joblib.load(os.path.join(model_path, "pipeline.pkl"))

        with open(os.path.join(model_path, "metadata.json")) as f:
            meta = json.load(f)

        task = meta.get("task")

        # ✅ ROUTING

        if task == "classification":
            return ClassificationInference(pipeline, meta)

        elif task == "multiclass_classification":
            return ClassificationInference(pipeline, meta)

        elif task == "regression":
            return RegressionInference(pipeline, meta)

        elif task == "unsupervised":
            return UnsupervisedInference(pipeline, meta, model_path)

        else:
            raise ValueError(f"Unsupported task: {task}")

