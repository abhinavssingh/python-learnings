import numpy as np
import plotly.graph_objects as go
from scipy.stats import chi2, f


class DistributionPlotHelper:
    """
    Helper class to generate statistical distribution plots
    (F-distribution and Chi-Square distribution) using Plotly.
    """

    @staticmethod
    def plot_f_distribution(
        observed_f,
        dfn,
        dfd,
        alpha=0.05,
        title="F‑Distribution"
    ):
        x = np.linspace(0, f.ppf(0.99, dfn, dfd), 500)
        y = f.pdf(x, dfn, dfd)

        critical_value = f.ppf(1 - alpha, dfn, dfd)

        fig = go.Figure()

        # F-distribution curve
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name="F Distribution"
        ))

        # Critical value
        fig.add_vline(
            x=critical_value,
            line_dash="dash",
            line_color="red",
            annotation_text="Critical Value",
        )

        # Observed F-statistic
        fig.add_vline(
            x=observed_f,
            line_color="green",
            annotation_text="Observed F",
        )

        fig.update_layout(
            title=title,
            xaxis_title="F value",
            yaxis_title="Density",
            template="plotly_white"
        )

        return fig

    @staticmethod
    def plot_chi_square_distribution(
        observed_chi2,
        dof,
        alpha=0.05,
        title="Chi‑Square Distribution"
    ):
        x = np.linspace(0, chi2.ppf(0.99, dof), 500)
        y = chi2.pdf(x, dof)

        critical_value = chi2.ppf(1 - alpha, dof)

        fig = go.Figure()

        # Chi-square distribution curve
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name="Chi‑Square Distribution"
        ))

        # Critical value
        fig.add_vline(
            x=critical_value,
            line_dash="dash",
            line_color="red",
            annotation_text="Critical Value",
        )

        # Observed chi-square
        fig.add_vline(
            x=observed_chi2,
            line_color="green",
            annotation_text="Observed χ²",
        )

        fig.update_layout(
            title=title,
            xaxis_title="Chi‑Square value",
            yaxis_title="Density",
            template="plotly_white"
        )

        return fig

    @staticmethod
    def plot_multiple_f_distributions(
        df_pairs,
        observed_f=None,
        alpha=0.05,
        title="F‑Distribution for Multiple Degrees of Freedom"
    ):
        """
        df_pairs: list of tuples -> [(dfn, dfd), ...]
        """

        fig = go.Figure()

        for dfn, dfd in df_pairs:
            x = np.linspace(0, f.ppf(0.99, dfn, dfd), 500)
            y = f.pdf(x, dfn, dfd)

            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"F(dfn={dfn}, dfd={dfd})"
            ))

            crit = f.ppf(1 - alpha, dfn, dfd)
            fig.add_vline(
                x=crit,
                line_dash="dot",
                line_color="red",
                opacity=0.4
            )

        if observed_f is not None:
            fig.add_vline(
                x=observed_f,
                line_color="green",
                line_width=3,
                annotation_text="Observed F"
            )

        fig.update_layout(
            title=title,
            xaxis_title="F value",
            yaxis_title="Density",
            template="plotly_white"
        )

        return fig

    @staticmethod
    def plot_multiple_chi_square_distributions(
        dof_list,
        observed_chi2=None,
        alpha=0.05,
        title="Chi‑Square Distribution Comparison"
    ):
        fig = go.Figure()

        for dof in dof_list:
            x = np.linspace(0, chi2.ppf(0.99, dof), 500)
            y = chi2.pdf(x, dof)

            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"χ²(dof={dof})"
            ))

            crit = chi2.ppf(1 - alpha, dof)
            fig.add_vline(
                x=crit,
                line_dash="dot",
                line_color="red",
                opacity=0.4
            )

        if observed_chi2 is not None:
            fig.add_vline(
                x=observed_chi2,
                line_color="green",
                annotation_text="Observed χ²"
            )

        fig.update_layout(
            title=title,
            xaxis_title="Chi‑Square value",
            yaxis_title="Density",
            template="plotly_white"
        )

        return fig
