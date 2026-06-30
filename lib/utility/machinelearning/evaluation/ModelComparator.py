from .BaseComparator import BaseComparator
from .ClassificationModelComparator import ClassificationModelComparator
from .RegressionModelComparator import RegressionModelComparator
from .UnsupervisedModelComparator import UnsupervisedModelComparator


from lib.utility.machinelearning._logging import ExceptionLoggingMixin


class ModelComparator(ExceptionLoggingMixin):

    @staticmethod
    def get_comparator(results):

        if not results:
            raise ValueError("No results provided")

        task = results[0].get("task")

        if task == "classification":
            return ClassificationModelComparator(results)

        elif task == "regression":
            return RegressionModelComparator(results)

        elif task == "unsupervised":
            return UnsupervisedModelComparator(results)

        else:
            return BaseComparator(results)

