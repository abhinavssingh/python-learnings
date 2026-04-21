import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import gaussian_kde

from lib.html import HtmlBuilder, PlotRenderer
from lib.mathshelper import FORMULA_REGISTRY
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running basic probability distribution operation report...")
    # ...


def kde_curve(data, x_grid):
    kde = gaussian_kde(data)
    return kde(x_grid)


builder = HtmlBuilder()
plotRenderer = PlotRenderer()
content = []


np.random.seed(42)

sample_data = pd.DataFrame({
    "X_discrete": np.random.randint(1, 7, size=1000),   # Dice outcomes (1–6)
    "Y_continuous": np.random.normal(50, 10, size=1000)  # Continuous variable
})


discrete_pmf = sample_data["X_discrete"].value_counts(normalize=True).sort_index()

x = np.linspace(1, 6, 300)
kde_y = kde_curve(sample_data["X_discrete"], x)

discrete_pmf_fig = go.Figure()

discrete_pmf_fig.add_bar(x=discrete_pmf.index, y=discrete_pmf.values, name="Empirical PMF")
discrete_pmf_fig.add_scatter(x=x, y=kde_y, mode="lines", name="KDE")

discrete_pmf_fig.update_layout(
    title="Discrete Uniform Distribution (PMF + KDE)",
    xaxis_title="X",
    yaxis_title="Probability / Density"
)


num = ((sample_data["X_discrete"] == 3) & (sample_data["Y_continuous"] > 55)).sum()
den = (sample_data["Y_continuous"] > 55).sum()

# bernouli distribution
bernoulli_data = (sample_data["X_discrete"] == 1).astype(int)

probs = bernoulli_data.value_counts(normalize=True).sort_index()

x_kde = np.linspace(0, 1, 200)
y_kde = kde_curve(bernoulli_data, x_kde)

bernoulli_fig = go.Figure()
bernoulli_fig.add_bar(x=probs.index, y=probs.values, name="Empirical PMF")
bernoulli_fig.add_scatter(x=x_kde, y=y_kde, mode="lines", name="KDE")

bernoulli_fig.update_layout(
    title="Bernoulli Distribution (PMF + KDE)",
    xaxis_title="X",
    yaxis_title="Probability / Density"
)


# binomial distribution
binomial_pretext = f"""
📘 Binomial Distribution Understanding the Setup

In this example, we construct a Binomial distribution using information
estimated directly from the dataset.

The Binomial distribution models:
→ how many times a specific event (called a "success")
→ occurs in a fixed number of independent trials
→ when the probability of success remains constant.

🔹 Defining the Success Event:
We define a success as:
    "X_discrete equals 3"

Each observation is treated as one trial, where the outcome is either:
- success (X = 3), or
- failure (X ≠ 3)

🔹 Estimating the Probability of Success (p̂):
We calculate the proportion of times X_discrete equals 3 in the dataset.
This gives us an empirical estimate of the success probability.

Estimated probability of success:
    p̂ ≈ {{p_hat:.4f}}

🔹 Choosing the Number of Trials:
We fix the number of trials per experiment as:
    n = {{n_trials}}

This means we are interested in how many successes occur
out of {{n_trials}} independent trials.

🔹 Creating the Binomial Model:
Using the estimated probability p̂ and the fixed number of trials n,
we create a Binomial random variable.

This model describes the probability of observing:
    0, 1, 2, ..., {{n_trials}} successes

🔹 Evaluating the PMF:
The Probability Mass Function (PMF) assigns a probability
to each possible number of successes.

This allows us to answer questions such as:
- What is the probability of exactly 2 successes?
- How likely are we to see no successes at all?

Key takeaway:
The Binomial distribution connects real data to probability theory
by modeling repeated experiments with a fixed number of trials
and a constant probability of success.
"""

p_hat = (sample_data["X_discrete"] == 3).mean()  # empirical success probability
n_trials = 10                                    # fixed number of trials
rv_binom = stats.binom(n=n_trials, p=p_hat)
k_binom = np.arange(0, n_trials + 1)
pmf_binom = rv_binom.pmf(k_binom)

fig_binom_pmf = go.Figure()

fig_binom_pmf.add_bar(x=k_binom, y=rv_binom.pmf(k_binom), name="Binomial PMF")
fig_binom_pmf.add_scatter(x=k_binom, y=kde_curve(rv_binom.rvs(1000), k_binom),
                          mode="lines", name="KDE")

fig_binom_pmf.update_layout(
    title="Binomial PMF + KDE",
    xaxis_title="k",
    yaxis_title="Probability / Density"
)


# poisson distribution

poisson_pretext = f"""
📘 Poisson Distribution Understanding the Setup

In this example, we build a Poisson distribution directly from data.

The Poisson distribution models how many times a specific event occurs
within a fixed interval, assuming the event is relatively rare and
happens independently.

🔹 Event Definition:
We define our event as:
    "X_discrete equals 6"

🔹 Estimating the Average Rate (λ):
First, we compute how often this event occurs in the dataset.
This gives us the probability of the event in one observation.

We then scale this probability to represent an interval of 10 observations.
This gives us the expected number of events per interval, called lambda (λ).

Estimated value:
    λ ≈ {{lambda_hat:.4f}}

🔹 Creating the Poisson Model:
Using this λ value, we create a Poisson random variable to model
the probability of observing 0, 1, 2, … occurrences of the event
within the chosen interval.

🔹 Evaluating the PMF:
We compute the Probability Mass Function (PMF) for small values of k,
where k represents how many times the event occurs.

This allows us to answer questions like:
- What is the probability that the event happens exactly once?
- How likely is it that the event does not occur at all?

Key takeaway:
We did not assume the Poisson distribution — we estimated it
using real data, which mirrors real-world statistical modeling.
"""

lambda_hat = (sample_data["X_discrete"] == 6).mean() * 10
rv_pois = stats.poisson(mu=lambda_hat)

k_pois = np.arange(0, 8)  # small range is typical for Poisson
pmf_pois = rv_pois.pmf(k_pois)

fig_poisson_pmf = go.Figure()

fig_poisson_pmf.add_bar(x=k_pois, y=pmf_pois, name="Poisson PMF")
fig_poisson_pmf.add_scatter(x=k_pois, y=kde_curve(rv_pois.rvs(1000), k_pois),
                            mode="lines", name="KDE")

fig_poisson_pmf.update_layout(
    title="Poisson PMF + KDE",
    xaxis_title="k",
    yaxis_title="Probability / Density"
)


# uniform distribution
a, b = sample_data["Y_continuous"].min(), sample_data["Y_continuous"].max()
rv_uniform = stats.uniform(loc=a, scale=b - a)


# x-grid for PDF
x = np.linspace(a, b, 400)
pdf_uniform = rv_uniform.pdf(x)

# Plot
fig_uniform = go.Figure()

# Empirical histogram (density)
fig_uniform.add_histogram(
    x=sample_data["Y_continuous"],
    histnorm="probability density",
    nbinsx=40,
    name="Empirical Density",
    opacity=0.6
)

# Uniform PDF line
fig_uniform.add_scatter(
    x=x,
    y=pdf_uniform,
    mode="lines",
    name="Uniform PDF",
    line=dict(width=3)
)

fig_uniform.update_layout(
    title="Continuous Uniform Distribution (Empirical Data + Uniform PDF)",
    xaxis_title="Y_continuous",
    yaxis_title="Density",
    bargap=0.05
)

# exponential distribution
y = sample_data["Y_continuous"]
rv_exp = stats.expon(scale=y.mean())

expo_fig = go.Figure()
expo_fig.add_scatter(
    x=x, y=rv_exp.pdf(x),
    mode="lines", name="Exponential PDF"
)
expo_fig.add_scatter(
    x=x, y=kde_curve(y.clip(lower=0), x),
    mode="lines", name="KDE"
)

expo_fig.update_layout(
    title="Exponential Distribution (PDF + KDE)",
    xaxis_title="x",
    yaxis_title="Density"
)


# normal distribution
normal_pretext = f"""
📘 Normal Distribution – Understanding the Setup

In this example, we model a continuous variable using the Normal
(Gaussian) distribution based on observed data.

The Normal distribution is one of the most important distributions
in statistics because many real‑world measurements naturally follow
its bell‑shaped pattern.

Examples include:
- heights of people,
- test scores,
- measurement errors,
- natural variations in data.

🔹 Working with Continuous Data:
Here, we focus on the continuous variable:
    Y_continuous

This variable takes a wide range of values and is suitable for
modeling with a continuous probability distribution.

🔹 Estimating the Parameters (μ and σ):
The Normal distribution is fully described by two parameters:
- μ (mu): the mean, which controls the center of the distribution
- σ (sigma): the standard deviation, which controls the spread

Instead of assuming these values, we estimate them directly from data.

Estimated values from the dataset:
    μ̂ (mean)  ≈ {{mu_hat:.4f}}
    σ̂ (std)   ≈ {{sigma_hat:.4f}}

These estimates summarize the central tendency and variability
present in the observed data.

🔹 Creating the Normal Model:
Using the estimated parameters μ̂ and σ̂, we construct a Normal
random variable that represents the probability distribution
of Y_continuous.

This model can now be used to:
- calculate probabilities,
- compute percentiles,
- compare empirical data with theoretical expectations.

Key takeaway:
The Normal distribution provides a smooth, mathematical description
of real data by capturing its average behavior and natural variation.
"""

y = sample_data["Y_continuous"]
mu_hat, sigma_hat = stats.norm.fit(y)
rv_norm = stats.norm(mu_hat, sigma_hat)

x = np.linspace(
    sample_data["Y_continuous"].min(),
    sample_data["Y_continuous"].max(),
    400
)
cont_hist_fig = go.Figure()

cont_hist_fig.add_histogram(
    x=y, histnorm="probability density", nbinsx=40, name="Histogram"
)
cont_hist_fig.add_scatter(
    x=x, y=stats.norm.pdf(x, mu_hat, sigma_hat),
    mode="lines", name="Normal PDF"
)
cont_hist_fig.add_scatter(
    x=x, y=kde_curve(y, x),
    mode="lines", name="KDE"
)

cont_hist_fig.update_layout(
    title="Continuous PDF: Histogram + Normal PDF + KDE",
    xaxis_title="Y",
    yaxis_title="Density"
)


# ------------ CDF-------------------
y = np.sort(sample_data["Y_continuous"])
ecdf = np.arange(1, len(y) + 1) / len(y)

scatter_fig = go.Figure()
scatter_fig.add_scatter(x=y, y=ecdf, mode="lines", name="Empirical CDF")
scatter_fig.add_scatter(
    x=x,
    y=stats.norm.cdf(x, mu_hat, sigma_hat),
    mode="lines",
    name="Normal CDF (Fit)"
)
scatter_fig.update_layout(
    title="CDF: Empirical vs Normal",
    xaxis_title="Y",
    yaxis_title="F(Y)"
)

# ---------- CONDITIONAL PMF ----------
filtered = sample_data[sample_data["Y_continuous"] > 55]
conditional_pmf = filtered["X_discrete"].value_counts(normalize=True).sort_index()
conditional_pmf_fig = go.Figure()
conditional_pmf_fig.add_bar(x=conditional_pmf.index, y=conditional_pmf.values)
conditional_pmf_fig.update_layout(
    title="Conditional PMF: P(X | Y > 55)",
    xaxis_title="X",
    yaxis_title="Conditional Probability"
)

# ---------- JOINT PROBABILITY HEATMAP ----------
y_bins = pd.cut(sample_data["Y_continuous"], bins=5)
joint = pd.crosstab(
    sample_data["X_discrete"],
    y_bins,
    normalize=True
)
joint_prob_fig = go.Figure(
    data=go.Heatmap(
        z=joint.values,
        x=[str(b) for b in joint.columns],
        y=joint.index,
        colorscale="Viridis"
    )
)
joint_prob_fig.update_layout(
    title="Joint Probability Heatmap: X_discrete vs Y_continuous",
    xaxis_title="Y bins",
    yaxis_title="X"
)


# z distribution
z_60 = (60 - mu_hat) / sigma_hat

# Standard Normal Distribution

z = (sample_data["Y_continuous"] -
     sample_data["Y_continuous"].mean()) / sample_data["Y_continuous"].std()

x = np.linspace(-4, 4, 400)

standard_normal_dist_fig = go.Figure()
standard_normal_dist_fig.add_scatter(
    x=x, y=stats.norm.pdf(x),
    mode="lines", name="Standard Normal PDF"
)
standard_normal_dist_fig.add_scatter(
    x=x, y=kde_curve(z, x),
    mode="lines", name="KDE"
)

standard_normal_dist_fig.update_layout(
    title="Standard Normal Distribution (PDF + KDE)",
    xaxis_title="z",
)


# approximate empirical joint CDF
joint_cdf_val = (
    (sample_data["X_discrete"] <= 3) &
    (sample_data["Y_continuous"] <= 55)
).mean()


content.append(builder.full_width_card("Probability Sample Dataframe", builder.render_dataframe_collapsible(sample_data)))

categories = ["Probability", "Distribution", "CDF"]
content.append(
    builder.grid([
        formula.render(builder)
        for category in categories
        for formula in FORMULA_REGISTRY.by_category(category)
    ]
    ))

content.append(
    builder.grid(
        [
            builder.math_card(
                "Simple Probability (Discrete) p_x_eq_3:",
                builder.render_array(
                    (sample_data["X_discrete"] == 3).mean(),
                    display=False)),
            builder.math_card("Simple Probability (Continuous) P(45 ≤ Y ≤ 60):", builder.render_array(
                ((sample_data["Y_continuous"] >= 45) & (sample_data["Y_continuous"] <= 60)).mean(), display=False)),
            builder.math_card("Joint Probability (Discrete) P(X = 3 AND Y > 55):", builder.render_array(
                ((sample_data["X_discrete"] == 3) & (sample_data["Y_continuous"] > 55)).mean(), display=False)),
            builder.math_card("Marginal Probability (Discrete) P(X=x):", builder.render_series(discrete_pmf)),
            builder.math_card("Conditional Probability (Discrete) P(X=3∣Y>55):", builder.render_array((num / den), display=False)),
            builder.math_card("Joint CDF Example P(X ≤ 3, Y ≤ 55):", builder.render_array(joint_cdf_val, display=False)),
            builder.math_card(
                "Empirical CDF at y = 60 (Continuous Case):",
                builder.render_array(
                    (sample_data["Y_continuous"] <= 60).mean(),
                    display=False)),
            builder.math_card(
                "Discrete Uniform Distribution Example P(X = 4):",
                builder.render_array(
                    (sample_data["X_discrete"] == 4).mean(),
                    display=False)),
            builder.math_card("Bernoulli Distribution Example P(X = 1):", builder.render_array(bernoulli_data.mean(), display=False)),
            builder.card("Noraml Distribution Understanding:", builder.render_pre(normal_pretext)),
            builder.card("Binomial Distribution Understanding:", builder.render_pre(binomial_pretext)),
            builder.card("Poisson Distribution Understanding:", builder.render_pre(poisson_pretext)),
            builder.math_card("Binomial Distribution Example P(X = 3):", builder.render_array(rv_binom.pmf(3), display=False)),
            builder.math_card("Poisson Distribution Example P(X = 6):", builder.render_array(rv_pois.pmf(6), display=False)),
            builder.math_card("Binomial Distribution CDF Example F(3):", builder.render_array(rv_binom.cdf(3), display=False)),
            builder.math_card("Poisson Distribution CDF Example F(6):", builder.render_array(rv_pois.cdf(6), display=False)),

            builder.math_card(
                "Continuous Uniform Distribution Example P(45 ≤ Y ≤ 60):",
                builder.render_array(
                    rv_uniform.cdf(60) -
                    rv_uniform.cdf(45),
                    display=False)),
            builder.math_card("Exponential Distribution Example P(Y ≤ 60):", builder.render_array(rv_exp.cdf(60), display=False)),
            builder.math_card("Normal Distribution Example P(45 ≤ Y ≤ 60):", builder.render_array(rv_norm.cdf(60) - rv_norm.cdf(45), display=False)),
            builder.math_card("Standard Normal Distribution Example (z):", builder.render_array(stats.norm.cdf(z_60), display=False)),
            builder.math_card("Continuous CDF (Empirical) F_Y(60):", builder.render_array((sample_data["Y_continuous"] <= 60).mean(), display=False)),
            builder.math_card("Continuous CDF (Normal Fit) F_Y(60):", builder.render_array(rv_norm.cdf(60), display=False)),
        ]))

content.append(builder.chart_grid([
    plotRenderer.plot_to_card(discrete_pmf_fig, " Discrete Uniform Distribution (PMF + KDE)"),
    plotRenderer.plot_to_card(conditional_pmf_fig, " Conditional  PMF"),
    plotRenderer.plot_to_card(bernoulli_fig, " Bernoulli Distribution (PMF + KDE)"),
    plotRenderer.plot_to_card(fig_binom_pmf, " Binomial PMF + KDE"),
    plotRenderer.plot_to_card(fig_poisson_pmf, " Poisson PMF + KDE"),
    plotRenderer.plot_to_card(cont_hist_fig, " Continuous PDF: Histogram + Normal PDF + KDE"),
    plotRenderer.plot_to_card(standard_normal_dist_fig, "Standard Normal Distribution (PDF + KDE)"),
    plotRenderer.plot_to_card(expo_fig, " Exponential Continuous Distribution (PDF + KDE)"),
    plotRenderer.plot_to_card(joint_prob_fig, " Joint Probability Heatmap"),
    plotRenderer.plot_to_card(fig_uniform, " Continuous Uniform Distribution (Empirical Data + Uniform PDF)"),
]))

html_doc = builder.build_page(
    "Basic probability distribution Operation  Report",
    "\n".join(content)
)

# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "basic_probability_distribution_operation_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")

if __name__ == "__main__":
    main()
    main()
