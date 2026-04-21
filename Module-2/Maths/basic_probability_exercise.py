import numpy as np
import plotly.graph_objects as go
from scipy.stats import binom, gaussian_kde, norm, poisson

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running Probability exercise  report...")
    # ...


builder = HtmlBuilder()
plotRenderer = PlotRenderer()
content = []

df, report = dl.read_dataset("Retail_Store_Data.csv", optimize=True, handle_unnamed="drop", return_report=True)

# Get information about the DataFrame
df_info_str = dfh.get_dataframe_info_str(df)


# Bernoulli Plot histogram
df["High_Time_Spent"] = (df["Visit_Duration"] > 30).astype(int)
bernoulli_data = df["High_Time_Spent"]
bernouli_fig = go.Figure()


bernouli_fig.add_histogram(
    x=bernoulli_data,
    histnorm="probability",
    xbins=dict(start=-0.5, end=1.5, size=1),
    name="Bernoulli Histogram",
    marker_color="#4C72B0"
)


# --- KDE (smoothed visualization) ---
# KDE needs samples (repeat Bernoulli draws)
kde = gaussian_kde(bernoulli_data)
x_kde = np.linspace(-0.5, 1.5, 200)
y_kde = kde(x_kde)

bernouli_fig.add_scatter(
    x=x_kde,
    y=y_kde,
    mode="lines",
    name="KDE (smoothed)",
    line=dict(color="#DD8452", width=3)
)

# Layout
bernouli_fig.update_layout(
    title="Bernoulli Histogram with KDE: High_Time_Spent",
    xaxis=dict(
        title="High_Time_Spent (0 = ≤30 mins, 1 = >30 mins)",
        tickmode="array",
        tickvals=[0, 1]
    ),
    yaxis_title="Probability / Density",
    bargap=0.4
)

# Binomial Plot

# 1. Create Bernoulli column using FULL dataset
df["High_Spend"] = (df["Purchase_Amount"] > 100).astype(int)

# 2. Estimate probability p from ALL rows
p_hat = df["High_Spend"].mean()

# 3. Binomial setup: 10 random visits
n = 10
k = np.arange(0, n + 1)

# Binomial PMF
pmf = binom.pmf(k, n=n, p=p_hat)

# 4. KDE via simulated binomial samples
samples = binom.rvs(n=n, p=p_hat, size=5000)
kde = gaussian_kde(samples)
x_kde = np.linspace(0, n, 400)
y_kde = kde(x_kde)

# 5. Plot
binomial_fig = go.Figure()

# Binomial PMF bars
binomial_fig.add_bar(
    x=k,
    y=pmf,
    name="Binomial PMF",
    marker_color="#4C72B0"
)

# KDE overlay
binomial_fig.add_scatter(
    x=x_kde,
    y=y_kde,
    mode="lines",
    name="KDE (smoothed – visual aid)",
    line=dict(color="#DD8452", width=3)
)

binomial_fig.update_layout(
    title="Binomial Distribution: High-Spend Visits (> $100) out of 10",
    xaxis_title="Number of High-Spend Visits (k)",
    yaxis_title="Probability / Density",
    bargap=0.25
)

# Poisson Plot

# Given average rate
lambda_per_hour = 15

# Possible customer counts per hour
k = np.arange(0, 40)

# Poisson PMF
pmf = poisson.pmf(k, mu=lambda_per_hour)


# --- KDE via simulation ---
# Generate large sample of Poisson observations
samples = poisson.rvs(mu=lambda_per_hour, size=5000)

kde = gaussian_kde(samples)
x_kde = np.linspace(0, 40, 400)
y_kde = kde(x_kde)


# Plot
poisson_fig = go.Figure()


# Poisson PMF bars
poisson_fig.add_bar(
    x=k,
    y=pmf,
    name="Poisson PMF",
    marker_color="#4C72B0"
)

# KDE overlay (smoothed visual aid)
poisson_fig.add_scatter(
    x=x_kde,
    y=y_kde,
    mode="lines",
    name="KDE (smoothed – visual aid)",
    line=dict(color="#DD8452", width=3)
)

poisson_fig.update_layout(
    title="Poisson Distribution: Customers Visiting the Store per Hour (λ = 15)",
    xaxis_title="Number of Customers in One Hour",
    yaxis_title="Probability / Density",
    bargap=0.15
)

# Normal Distribution
purchase = df["Purchase_Amount"]
mu = purchase.mean()
sigma = purchase.std()

# X-axis range
x = np.linspace(purchase.min(), purchase.max(), 500)

# Normal PDF
normal_pdf = norm.pdf(x, mu, sigma)

# KDE (empirical density)
kde = gaussian_kde(purchase)
kde_values = kde(x)

# Create figure
normal_fig = go.Figure()

# Histogram (density-normalized)
normal_fig.add_histogram(
    x=purchase,
    histnorm="probability density",
    nbinsx=40,
    name="Purchase Amount Histogram",
    opacity=0.6
)

# Normal PDF
normal_fig.add_scatter(
    x=x,
    y=normal_pdf,
    mode="lines",
    name="Normal PDF",
    line=dict(color="#4C72B0", width=3)
)

# KDE curve
normal_fig.add_scatter(
    x=x,
    y=kde_values,
    mode="lines",
    name="KDE (empirical)",
    line=dict(color="#DD8452", width=3, dash="dash")
)

# Layout
normal_fig.update_layout(
    title=f"Normal Distribution of Purchase Amount (μ = {mu:.2f}, σ = {sigma:.2f})",
    xaxis_title="Purchase Amount",
    yaxis_title="Density",
    bargap=0.05
)

# Uniform Distribution

# Use full Visit_Duration column
visit_duration = df["Visit_Duration"]
a = visit_duration.min()
b = visit_duration.max()


# X-axis range
x = np.linspace(a, b, 500)

# Uniform PDF
uniform_pdf = np.full_like(x, 1 / (b - a))

# KDE (empirical density)
kde = gaussian_kde(visit_duration)
kde_values = kde(x)

uniform_fig = go.Figure()

# Histogram (density-normalized)
uniform_fig.add_histogram(
    x=visit_duration,
    histnorm="probability density",
    nbinsx=40,
    name="Visit Duration Histogram",
    opacity=0.6
)

# Uniform PDF
uniform_fig.add_scatter(
    x=x,
    y=uniform_pdf,
    mode="lines",
    name="Uniform PDF",
    line=dict(color="#4C72B0", width=3)
)

# KDE curve
uniform_fig.add_scatter(
    x=x,
    y=kde_values,
    mode="lines",
    name="KDE (empirical)",
    line=dict(color="#DD8452", width=3, dash="dash")
)

uniform_fig.update_layout(
    title=f"Uniform Distribution with KDE for Visit Duration (a={a:.2f}, b={b:.2f})",
    xaxis_title="Visit Duration (minutes)",
    yaxis_title="Density",
    bargap=0.05
)


content.append(builder.full_width_card(
    "Numeric DataFrame",
    builder.render_dataframe_collapsible(df, initial_rows=10)
))

content.append(
    builder.grid([
        builder.card("dataframe Description:",
                     builder.render_dict(df.describe().to_dict())),
        builder.card("Information of the Housing Dataframe is:",
                     builder.render_pre(df_info_str)),
        builder.card("Optimized Dataframe report:",
                     builder.render_pre(report))
    ]))

content.append(builder.chart_grid([
    plotRenderer.plot_to_card(bernouli_fig, " Bernoulli Plot PMF + KDE"),
    plotRenderer.plot_to_card(binomial_fig, " Binomial Plot PMF + KDE"),
    plotRenderer.plot_to_card(poisson_fig, " Poisson Plot PMF + KDE"),
    plotRenderer.plot_to_card(normal_fig, " Normal Distribution Plot PDF + KDE"),
    plotRenderer.plot_to_card(uniform_fig, " Uniform Distribution Plot PDF + KDE"),
]))

html_doc = builder.build_page(
    "Basic probability exercise Report",
    "\n".join(content)
)


# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "basic_probability_exercise_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")


if __name__ == "__main__":
    main()
    main()
