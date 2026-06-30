import pandas as pd

from lib.utility.machinelearning.evaluation.MetricResolver import MetricResolver

from ..classification.ClassificationPlots import plot_all as classification_plots
from ..generic.ComparisonPlots import (
    plot_all_model_comparison,
    plot_best_model_highlight,
    plot_model_ranking,
)
from ..generic.DistributionPlots import plot_metric_distribution
from ..regression.RegressionPlots import plot_all as regression_plots
from ..unsupervised.ClusteringPlots import plot_all as clustering_plots


class VisualizerEngine:

    def __init__(self, results, artifacts=None):

        if not results:
            raise ValueError("No results available")

        self.results = results
        self.df = pd.DataFrame(results)

        if artifacts is None:
            self.artifacts = []
        elif isinstance(artifacts, pd.DataFrame):
            self.artifacts = artifacts.to_dict("records")
        else:
            self.artifacts = artifacts

        self.task = results[0].get("task")

    # ---------------------------------------------------
    # ✅ GENERIC COMPARISON
    # ---------------------------------------------------

    def plot_comparison(self, metric=None):

        metric = metric or MetricResolver.get_best_metric(self.task)

        return plot_all_model_comparison(
            self.df,
            metric1=metric,
            task=self.task
        )

    # ---------------------------------------------------
    # ✅ MODEL RANKING
    # ---------------------------------------------------

    def plot_ranking(self, metric=None):

        metric = metric or MetricResolver.get_best_metric(self.task)

        return plot_model_ranking(
            self.df,
            metric=metric,
            task=self.task
        )

    # ---------------------------------------------------
    # ✅ BEST MODEL HIGHLIGHT
    # ---------------------------------------------------

    def plot_best_model(self, metric=None):

        metric = metric or MetricResolver.get_best_metric(self.task)

        return plot_best_model_highlight(
            self.df,
            metric=metric,
            task=self.task
        )

    # ---------------------------------------------------
    # ✅ METRIC DISTRIBUTION
    # ---------------------------------------------------

    def plot_distribution(self, metric=None):

        metric = metric or MetricResolver.get_best_metric(self.task)

        return plot_metric_distribution(self.df, metric)

    # ---------------------------------------------------
    # ✅ TASK-SPECIFIC VISUALIZATION
    # ---------------------------------------------------

    def plot_task_specific(self):

        if self.task == "classification":

            return classification_plots(
                self.df,
                self.artifacts
            )

        elif self.task == "regression":

            return regression_plots(self.df)

        elif self.task == "unsupervised":
            return clustering_plots(
                self.df,
                self.artifacts
            )

        return {}

    # ---------------------------------------------------
    # ✅ COMPLETE DASHBOARD OUTPUT
    # ---------------------------------------------------

    def render_all(self):

        metric = MetricResolver.get_best_metric(self.task)

        return {
            "comparison": self.plot_comparison(metric),
            "ranking": self.plot_ranking(metric),
            "best_model": self.plot_best_model(metric),
            "distribution": self.plot_distribution(metric),

            # ✅ Always consistent structure
            "task_specific": self.plot_task_specific()
        }
