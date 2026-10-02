from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def ngram_frequency_bar(counts: Counter, top: int = 15, title: str = "Top n-grams") -> go.Figure:
    items = counts.most_common(top)
    labels = [" ".join(k) if isinstance(k, tuple) else str(k) for k, _ in items]
    fig = px.bar(x=[v for _, v in items][::-1], y=labels[::-1], orientation="h", title=title,
                 labels={"x": "count", "y": "n-gram"})
    fig.update_layout(height=450)
    return fig


def similarity_heatmap(matrix: np.ndarray | pd.DataFrame, labels: list[str] | None = None,
                       title: str = "Cosine similarity") -> go.Figure:
    values = matrix.values if isinstance(matrix, pd.DataFrame) else np.asarray(matrix)
    labels = labels or (list(matrix.index) if isinstance(matrix, pd.DataFrame) else [str(i) for i in range(len(values))])
    short = [label if len(label) <= 30 else label[:28] + "…" for label in labels]
    fig = go.Figure(go.Heatmap(z=values, x=short, y=short, colorscale="Viridis", text=np.round(values, 2),
                               texttemplate="%{text}", hovertext=[[f"{a} ↔ {b}" for b in labels] for a in labels]))
    fig.update_layout(title=title, height=500, yaxis_autorange="reversed")
    return fig


def attention_heatmap(weights: np.ndarray, tokens: list[str], title: str = "Attention weights") -> go.Figure:
    fig = go.Figure(go.Heatmap(z=weights, x=tokens, y=tokens, colorscale="Blues", text=np.round(weights, 2),
                               texttemplate="%{text}"))
    fig.update_layout(title=title, xaxis_title="key (attended to)", yaxis_title="query (attending)",
                      height=450, yaxis_autorange="reversed")
    return fig


def embedding_scatter(points: np.ndarray, labels: list[str], groups: list[str] | None = None,
                      title: str = "Embedding projection", show_text: bool = True) -> go.Figure:
    df = pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "label": labels, "group": groups or ["all"] * len(labels)})
    fig = px.scatter(df, x="x", y="y", color="group", hover_name="label", text="label" if show_text else None, title=title)
    if show_text:
        fig.update_traces(textposition="top center")
    fig.update_layout(height=500)
    return fig


def chunk_size_histogram(sizes_by_strategy: dict[str, list[int]], title: str = "Chunk size distribution") -> go.Figure:
    fig = go.Figure()
    for name, sizes in sizes_by_strategy.items():
        fig.add_trace(go.Box(y=sizes, name=name, boxpoints="all", jitter=0.4))
    fig.update_layout(title=title, yaxis_title="characters", height=450)
    return fig


def breakpoint_plot(distances: np.ndarray, threshold: float, title: str = "Semantic distance between sentences") -> go.Figure:
    fig = go.Figure(go.Scatter(y=distances, mode="lines+markers", name="cosine distance"))
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="breakpoint threshold")
    fig.update_layout(title=title, xaxis_title="sentence index", yaxis_title="distance", height=400)
    return fig


def metric_bar(df: pd.DataFrame, x: str, metrics: list[str], title: str = "Metrics") -> go.Figure:
    long = df.melt(id_vars=[x], value_vars=metrics, var_name="metric", value_name="value")
    fig = px.bar(long, x=x, y="value", color="metric", barmode="group", title=title)
    fig.update_layout(height=450)
    return fig


def line_plot(y: list[float], title: str, x_title: str = "step", y_title: str = "value") -> go.Figure:
    fig = go.Figure(go.Scatter(y=y, mode="lines+markers"))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, height=400)
    return fig


def positional_encoding_heatmap(pe: np.ndarray, title: str = "Sinusoidal positional encoding") -> go.Figure:
    fig = go.Figure(go.Heatmap(z=pe, colorscale="RdBu", zmid=0))
    fig.update_layout(title=title, xaxis_title="embedding dimension", yaxis_title="position", height=450)
    return fig


def mermaid_html(mermaid_source: str) -> str:
    """Render a Mermaid diagram (e.g. LangGraph `draw_mermaid()`) inside a report card."""
    import html
    return (
        f'<pre class="mermaid" style="background:white;padding:8px;border-radius:8px">{html.escape(mermaid_source)}</pre>'
        '<script type="module">import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";'
        'mermaid.initialize({startOnLoad:true});</script>'
    )
