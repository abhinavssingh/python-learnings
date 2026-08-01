import plotly.graph_objects as go


class ClassDistributionPlot:
    @staticmethod
    def create_plot(counts: dict, title: str = "Class Distribution"):
        labels = [str(k) for k in counts.keys()]
        values = list(counts.values())

        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=values,
                    marker_color=["#2E86AB", "#E74C3C"][: len(values)],
                    text=values,
                    textposition="outside",
                )
            ]
        )

        fig.update_layout(
            title=title,
            xaxis_title="TARGET",
            yaxis_title="Count",
            template="plotly_white",
            height=430,
            margin=dict(l=40, r=20, t=60, b=40),
        )

        return fig

    @staticmethod
    def to_html(counts: dict, title: str = "Class Distribution", include_plotlyjs: str | bool = False):
        fig = ClassDistributionPlot.create_plot(counts=counts, title=title)
        return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)
