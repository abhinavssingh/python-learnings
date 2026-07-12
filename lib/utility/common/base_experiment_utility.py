from abc import ABC, abstractmethod
from datetime import datetime

import joblib
import pandas as pd


class BaseExperimentUtility(ABC):
    """
    Common functionality shared by
    Classification,
    Regression,
    Deep Learning utilities.
    """

    def __init__(self):

        self.model = None

        self.history = None

        self.metrics = {}

        self.results = []

        self.experiment_name = None

        self.start_time = None

        self.end_time = None

    # --------------------------------------------------
    # Experiment Tracking
    # --------------------------------------------------

    def start_experiment(
        self,
        experiment_name: str
    ):

        self.experiment_name = experiment_name

        self.start_time = datetime.now()

    def end_experiment(self):

        self.end_time = datetime.now()

    def get_experiment_duration(self):

        if (
            self.start_time is None
            or self.end_time is None
        ):
            return None

        return (
            self.end_time - self.start_time
        ).total_seconds()

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    def add_metrics(
        self,
        metrics: dict
    ):

        self.metrics.update(metrics)

    def get_metrics(self):

        return self.metrics

    # --------------------------------------------------
    # Results Tracking
    # --------------------------------------------------

    def add_result(
        self,
        result: dict
    ):
        self.results.append(result)

    def get_results_df(self):

        return pd.DataFrame(self.results)

    # --------------------------------------------------
    # Model Persistence
    # --------------------------------------------------

    def save_model(
        self,
        filepath: str
    ):

        joblib.dump(
            self.model,
            filepath
        )

    def load_model(
        self,
        filepath: str
    ):

        self.model = joblib.load(
            filepath
        )

        return self.model

    # --------------------------------------------------
    # Reporting
    # --------------------------------------------------

    def generate_report(self):

        report = {
            "experiment_name":
                self.experiment_name,

            "duration_seconds":
                self.get_experiment_duration(),

            "metrics":
                self.metrics
        }

        return report

    # --------------------------------------------------
    # Plotting Interface
    # --------------------------------------------------

    @abstractmethod
    def plot_results(self):
        pass
