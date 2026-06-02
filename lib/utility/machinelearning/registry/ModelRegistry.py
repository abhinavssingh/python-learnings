from typing import Dict

# Import sklearn models
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

# Base wrapper type (IMPORTANT for typing ✅)
from lib.utility.machinelearning.base.BaseModelWrapper import BaseModelWrapper
from lib.utility.machinelearning.models.linear.ElasticNetWrapper import ElasticNetWrapper
from lib.utility.machinelearning.models.linear.LassoWrapper import LassoWrapper

# Import wrappers
from lib.utility.machinelearning.models.linear.LinearRegressionWrapper import LinearRegressionWrapper
from lib.utility.machinelearning.models.linear.RidgeWrapper import RidgeWrapper


class ModelRegistry:
    """
    Central registry for all available models.
    Responsible for providing model wrappers in a decoupled way.
    """

    def __init__(self):
        self._registry: Dict[str, BaseModelWrapper] = self._build_registry()

    def _build_registry(self) -> Dict[str, BaseModelWrapper]:
        """
        Initialize all model wrappers.
        Add new models here without touching the rest of the system.
        """
        return {
            "LinearRegression": LinearRegressionWrapper(LinearRegression()),
            "Ridge": RidgeWrapper(Ridge()),
            "Lasso": LassoWrapper(Lasso()),
            "ElasticNet": ElasticNetWrapper(ElasticNet())
        }

    def get_model(self, model_name: str) -> BaseModelWrapper:
        """
        Get a single model wrapper with correct typing.
        """
        if model_name not in self._registry:
            raise ValueError(f"Model '{model_name}' not found in registry.")
        return self._registry[model_name]

    def get_all_models(self) -> Dict[str, BaseModelWrapper]:
        """
        Return all registered models.
        """
        return self._registry

    def register_model(self, model_name: str, wrapper: BaseModelWrapper) -> None:
        """
        Dynamically add a new model.
        Enables extensibility without modifying core code.
        """
        self._registry[model_name] = wrapper
