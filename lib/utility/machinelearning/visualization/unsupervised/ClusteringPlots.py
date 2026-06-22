import pandas as pd
import plotly.express as px


def plot_metrics(results):

    df = pd.DataFrame(results)

    return px.bar(
        df,
        x="model",
        y="silhouette_score",
        title="Clustering Performance"
    )
