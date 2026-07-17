import pandas as pd


class ReportGenerator:

    @staticmethod
    def generate_summary(
        experiment_name,
        metrics,
        duration
    ):

        return {
            "Experiment Name": experiment_name,
            "Duration (sec)": duration,
            **metrics
        }

    @staticmethod
    def generate_results_df(results):

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)
