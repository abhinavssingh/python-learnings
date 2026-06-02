from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import pandas as pd


class MLModelBase(ABC):
    """
    Abstract Base Class for ML utilities (orchestrator level).
    Responsible for:
    - data preparation
    - experiment execution
    - tuning orchestration
    - result tracking
    """

    @abstractmethod
    def preprocess(self, X: pd.DataFrame) -> Any:
        pass

    @abstractmethod
    def split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        pass

    @abstractmethod
    def run_experiment(self, model_name: str, **kwargs) -> Dict:
        """Run a single model experiment"""
        pass

    @abstractmethod
    def run_all_models(self) -> Dict:
        """Run all registered models"""
        pass

    @abstractmethod
    def tune_model(self, model_name: str, param_grid: Dict, **kwargs) -> Dict:
        """Run hyperparameter tuning"""
        pass

    @abstractmethod
    def get_best_model(self, metric: str):
        pass

    @abstractmethod
    def compare_models(self):
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        pass
