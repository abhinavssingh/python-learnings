import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, kurtosis, skew

from lib.html import HtmlBuilder, PlotRenderer
from lib.mathshelper import FORMULA_REGISTRY
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running basic statistics operation report...")
    # ...


builder = HtmlBuilder()
plotRenderer = PlotRenderer()
content = []

# Example dataset
sample_data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 4, 3, 2, 1])


# Reproducibility
np.random.seed(42)

# Data
symmetric = np.random.normal(0, 1, 1000)
right_skewed = np.random.exponential(1, 1000)
left_skewed = -np.random.exponential(1, 1000)
heavy_tailed = np.random.standard_t(df=2, size=1000)

plots = [
    ("Symmetric (Normal)", symmetric),
    ("Right Skewed", right_skewed),
    ("Left Skewed", left_skewed),
    ("Heavy Tailed", heavy_tailed),
]

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[name for name, _ in plots]
)

for idx, (title, data) in enumerate(plots):
    row = idx // 2 + 1
    col = idx % 2 + 1

    # Histogram (density)
    fig.add_trace(
        go.Histogram(
            x=data, nbinsx=40,
            histnorm='probability density',
            opacity=0.6
        ),
        row=row, col=col
    )

    # KDE
    kde = gaussian_kde(data)
    x_vals = np.linspace(data.min(), data.max(), 400)
    y_vals = kde(x_vals)
    fig.add_trace(
        go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(width=3)),
        row=row, col=col
    )

    # Mean & Median
    fig.add_vline(x=np.mean(data), line=dict(dash='dash'), row=row, col=col)
    fig.add_vline(x=np.median(data), line=dict(dash='dot'), row=row, col=col)

    # Stats annotation
    sk = skew(data, bias=False)
    ku = kurtosis(data, bias=False)
    plot_index = (row - 1) * 2 + col
    xref = 'x domain' if plot_index == 1 else f'x{plot_index} domain'
    yref = 'y domain' if plot_index == 1 else f'y{plot_index} domain'
    fig.add_annotation(
        text=f"Skewness = {sk:.2f}<br>Kurtosis = {ku:.2f}<br>Mean (--) | Median (·)",
        xref=xref, yref=yref,
        x=0.05, y=0.95, showarrow=False, align='left'
    )

fig.update_layout(
    title="Skewness & Kurtosis with KDE, Mean & Median (Plotly)",
    bargap=0.05,
    showlegend=False
)

content.append(
    builder.grid([
        formula.render(builder)
        for formula in FORMULA_REGISTRY.by_category("Statistics")
    ]
    ))

content.append(
    builder.grid(
        [
            builder.math_card("Dataset is:", builder.render_array(np.array(sample_data), display=False)),
            builder.math_card("Population/Sample Mean for the data:", builder.render_array(np.mean(sample_data), display=False)),
            builder.math_card("Polulation/Sample Median:", builder.render_array(np.median(sample_data), display=False)),
            builder.math_card("Population Variance:", builder.render_array(np.var(sample_data), display=False)),
            builder.math_card("Sample Variance:", builder.render_array(np.var(sample_data, ddof=1), display=False)),
            builder.math_card("Polulation Standard Deviation:", builder.render_array(np.std(sample_data), display=False)),
            builder.math_card("Sample Standard Deviation:", builder.render_array(np.std(sample_data, ddof=1), display=False)),
            builder.math_card("Dataset X is:", builder.render_array(x, display=False)),
            builder.math_card("Dataset Y is:", builder.render_array(y, display=False)),
            builder.math_card("Poulation Covariance:", builder.render_array(np.mean((x - x.mean()) * (y - y.mean())), display=False)),
            builder.math_card("Sample Covariance:", builder.render_array(np.cov(x, y, ddof=1)[0, 1], display=False)),
            builder.math_card("Correlation:", builder.render_array(np.corrcoef(x, y)[0, 1], display=False)),
            builder.math_card("Polulation Skewness:", builder.render_array(skew(sample_data, bias=True), display=False)),
            builder.math_card("Sample Skewness:", builder.render_array(skew(sample_data, bias=False), display=False)),
            builder.math_card("Polulation Kurtosis:", builder.render_array(kurtosis(sample_data, bias=True), display=False)),
        ]))

content.append(builder.chart_grid([
    plotRenderer.plot_to_card(fig, " Skewness and Kurtosis"),
]))

html_doc = builder.build_page(
    "Basic Statistics Operation  Report",
    "\n".join(content)
)

# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "basic_statistics_operation_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")

if __name__ == "__main__":
    main()
    main()
    main()
