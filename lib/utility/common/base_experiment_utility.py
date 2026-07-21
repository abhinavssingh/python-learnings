from abc import ABC
from datetime import datetime

import pandas as pd


class BaseExperimentUtility(ABC):

    def __init__(self):

        self.model = None

        self.history = None

        self.metrics = {}

        self.results = []

        self.experiment_name = None

        self.start_time = None

        self.end_time = None

    # =================================================
    # Experiment Tracking
    # =================================================

    def start_experiment(self, experiment_name: str):

        self.experiment_name = experiment_name

        self.start_time = datetime.now()

    def end_experiment(self):

        self.end_time = datetime.now()

    def get_experiment_duration(self):

        if not self.start_time or not self.end_time:
            return None

        return (
            self.end_time - self.start_time
        ).total_seconds()

    # =================================================
    # Metrics
    # =================================================

    def add_metrics(self, metrics: dict):

        self.metrics.update(metrics)

    def get_metrics(self):

        return self.metrics

    # =================================================
    # Results
    # =================================================

    def add_result(self, result: dict):

        self.results.append(result)

    def get_results_df(self):

        return pd.DataFrame(self.results)

    def clear_results(self):
        self.results = []

    def reset_metrics(self):
        self.metrics = {}
    # =================================================
    # Reporting
    # =================================================

    def generate_report(self):

        return {
            "experiment_name": self.experiment_name,
            "duration_seconds": self.get_experiment_duration(),
            "metrics": self.metrics
        }

    # =================================================
    # Plotting Hook
    # =================================================

    def plot_results(self):
        """
        Override only if plotting
        is supported.
        """
        pass
