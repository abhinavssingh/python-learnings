import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import gaussian_kde

from lib.html import HtmlBuilder, PlotRenderer
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


# SIMPLE PROBABILITY
simple_prob_ltx = r"P(X = x) = p_X(x)"
simple_prob_cont_ltx = r"P(a \le X \le b) = \int_a^b f(x)\,dx"
joint_prob_ltx = r"P(X = x, Y = y) = p_{X,Y}(x,y)"
joint_prob_cont_ltx = r"P(a \le X \le b,\, c \le Y \le d) = \int_a^b \int_c^d f(x,y)\,dy\,dx"
independent_ltx = r"P(X = x, Y = y) = P(X = x)P(Y = y)"
joint_pmf_ltx = r"P(X = x, Y = y) = p_{X,Y}(x,y)"
marginal_disc_ltx = r"P(X = x) = \sum_y P(X = x, Y = y)"
marginal_cont_ltx = r"f_X(x) = \int_{-\infty}^{\infty} f(x,y)\,dy"
conditional_disc_ltx = r"P(X = x \mid Y = y) = \frac{P(X = x, Y = y)}{P(Y = y)}"
conditional_cont_ltx = r"f_{X|Y}(x|y) = \frac{f(x,y)}{f_Y(y)}"
joint_cdf_ltx = r"F_{X,Y}(x,y) = P(X \le x, Y \le y)"
joint_cdf_cont_ltx = r"F_{X,Y}(x,y) = \int_{-\infty}^{x}\int_{-\infty}^{y} f_{X,Y}(u,v)\,dv\,du"

# DISCRETE PROBABILITY DISTRIBUTIONS
uniform_disc_ltx = r"P(X = x) = \frac{1}{n}, \quad x \in \{x_1, x_2, \dots, x_n\}"
bernoulli_ltx = r"P(X = x) = p^x (1-p)^{1-x}, \quad x \in \{0,1\}"
binom_ltx = r"\text{Binomial: } P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k=0,1,\dots,n"
geom_ltx = r"\text{Geometric: } P(X=k) = (1-p)^{k-1}p, \quad k=1,2,\dots"
neg_binom_ltx = r"\text{Negative Binomial: } P(X=k) = \binom{k-1}{r-1} p^r (1-p)^{k-r}, \quad k=r,r+1,\dots"
poisson_ltx = r"\text{Poisson: } P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k=0,1,2,\dots"
cdf_disc_ltx = r"F_X(x) = P(X \le x) = \sum_{t \le x} P(X = t)"
cdf_binomial_ltx = r"F_X(k) = \sum_{i=0}^{k} \binom{n}{i} p^i (1-p)^{n-i}"
cdf_poisson_ltx = r"F_X(k) = \sum_{i=0}^{k} \frac{\lambda^i e^{-\lambda}}{i!}"

# CONTINUOUS PROBABILITY DISTRIBUTIONS
cu_ltx = r"\text{Continuous Uniform: } f(x)=\frac{1}{b-a}, \quad a \le x \le b"
normal_ltx = r"f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x-\mu)^2}{2\sigma^2} \right)"
std_normal_ltx = r"f(z) = \frac{1}{\sqrt{2\pi}} e^{-\frac{z^2}{2}}"
exp_ltx = r"\text{Exponential: } f(x)=\lambda e^{-\lambda x}, \quad x \ge 0"
gamma_ltx = r"\text{Gamma: } f(x)=\frac{1}{\Gamma(k)\theta^k} x^{k-1} e^{-x/\theta}, \quad x \ge 0"
beta_ltx = r"\text{Beta: } f(x)=\frac{1}{B(\alpha,\beta)} x^{\alpha-1}(1-x)^{\beta-1}, \quad 0 \le x \le 1"
weibull_ltx = r"\text{Weibull: } f(x)=\frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1} e^{-(x/\lambda)^k}, \quad x \ge 0"
chisq_ltx = r"\text{Chi-Square: } f(x)=\frac{1}{2^{k/2}\Gamma(k/2)} x^{k/2-1} e^{-x/2}, \quad x \ge 0"
t_ltx = r"\text{Student t: } f(x)=\frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\Gamma\left(\frac{\nu}{2}\right)} \left(1+\frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}"
lognorm_ltx = r"\text{Log-Normal: } f(x)=\frac{1}{x\sigma\sqrt{2\pi}} e^{-\frac{(\ln x - \mu)^2}{2\sigma^2}}, \quad x>0"
cdf_cont_ltx = r"F_X(x) = P(X \le x) = \int_{-\infty}^{x} f_X(t)\,dt"
cdf_uniform_cont_ltx = r"F_X(x) = \frac{x-a}{b-a}, \quad a \le x \le b"
cdf_exponential_ltx = r"F_X(x) = 1 - e^{-\lambda x}, \quad x \ge 0"
cdf_normal_ltx = r"F_X(x) = \int_{-\infty}^{x} \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(t-\mu)^2}{2\sigma^2}}\,dt"
cdf_std_normal_ltx = r"\Phi(z) = \int_{-\infty}^{z} \frac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}}\,dt"

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

content.append(
    builder.grid(
        [
            builder.math_card("Simple Probability (Discrete):", builder.render_latex_formula(simple_prob_ltx, display=True)),
            builder.math_card(
                "Simple Probability (Discrete) p_x_eq_3:",
                builder.render_array(
                    (sample_data["X_discrete"] == 3).mean(),
                    display=False)),
            builder.math_card("Simple Probability (Continuous):", builder.render_latex_formula(simple_prob_cont_ltx, display=True)),
            builder.math_card("Simple Probability (Continuous) P(45 ≤ Y ≤ 60):", builder.render_array(
                ((sample_data["Y_continuous"] >= 45) & (sample_data["Y_continuous"] <= 60)).mean(), display=False)),
            builder.math_card("Joint Probability (Discrete):", builder.render_latex_formula(joint_prob_ltx, display=True)),
            builder.math_card("Joint Probability (Discrete) P(X = 3 AND Y > 55):", builder.render_array(
                ((sample_data["X_discrete"] == 3) & (sample_data["Y_continuous"] > 55)).mean(), display=False)),
            builder.math_card("Joint Probability Mass Function:", builder.render_latex_formula(joint_pmf_ltx, display=True)),
            builder.math_card("Joint Probability (Continuous):", builder.render_latex_formula(joint_prob_cont_ltx, display=True)),
            builder.math_card("Independence of Random Variables:", builder.render_latex_formula(independent_ltx, display=True)),
            builder.math_card("Marginal Distribution (Discrete):", builder.render_latex_formula(marginal_disc_ltx, display=True)),
            builder.math_card("Marginal Probability (Discrete) P(X=x):", builder.render_series(discrete_pmf)),
            builder.math_card("Marginal Distribution (Continuous):", builder.render_latex_formula(marginal_cont_ltx, display=True)),
            builder.math_card("Conditional Probability (Discrete):", builder.render_latex_formula(conditional_disc_ltx, display=True)),
            builder.math_card("Conditional Probability (Discrete) P(X=3∣Y>55):", builder.render_array((num / den), display=False)),
            builder.math_card("Conditional Density (Continuous):", builder.render_latex_formula(conditional_cont_ltx, display=True)),
            builder.math_card("Joint Cumulative Distribution Function (CDF):", builder.render_latex_formula(joint_cdf_ltx, display=True)),
            builder.math_card("Joint CDF (Continuous Case):", builder.render_latex_formula(joint_cdf_cont_ltx, display=True)),
            builder.math_card("Joint CDF Example P(X ≤ 3, Y ≤ 55):", builder.render_array(joint_cdf_val, display=False)),
            builder.math_card(
                "Empirical CDF at y = 60 (Continuous Case):",
                builder.render_array(
                    (sample_data["Y_continuous"] <= 60).mean(),
                    display=False)),
            builder.math_card("Discrete Uniform Distribution (PMF):", builder.render_latex_formula(uniform_disc_ltx, display=True)),
            builder.math_card(
                "Discrete Uniform Distribution Example P(X = 4):",
                builder.render_array(
                    (sample_data["X_discrete"] == 4).mean(),
                    display=False)),
            builder.math_card("Bernoulli Distribution (PMF):", builder.render_latex_formula(bernoulli_ltx, display=True)),
            builder.math_card("Bernoulli Distribution Example P(X = 1):", builder.render_array(bernoulli_data.mean(), display=False)),
            builder.math_card("Binomial Distribution (PMF):", builder.render_latex_formula(binom_ltx, display=True)),
            builder.math_card("Binomial Distribution Example P(X = 2):", builder.render_array(rv_binom.pmf(2), display=False)),
            builder.math_card("Poisson Distribution (PMF):", builder.render_latex_formula(poisson_ltx, display=True)),
            builder.math_card("Poisson Distribution Example P(X = 1):", builder.render_array(rv_pois.pmf(1), display=False)),
            builder.math_card("Cumulative Distribution Function Discrete:", builder.render_latex_formula(cdf_disc_ltx, display=True)),
            builder.math_card("Binomial Distribution CDF:", builder.render_latex_formula(cdf_binomial_ltx, display=True)),
            builder.math_card("Binomial Distribution CDF Example F(2):", builder.render_array(rv_binom.cdf(2), display=False)),
            builder.math_card("Poisson Distribution CDF:", builder.render_latex_formula(cdf_poisson_ltx, display=True)),
            builder.math_card("Poisson Distribution CDF Example F(1):", builder.render_array(rv_pois.cdf(1), display=False)),
            builder.math_card("Continuous Uniform Distribution", builder.render_latex_formula(cu_ltx, display=True)),
            builder.math_card(
                "Continuous Uniform Distribution Example P(45 ≤ Y ≤ 60):",
                builder.render_array(
                    rv_uniform.cdf(60) -
                    rv_uniform.cdf(45),
                    display=False)),
            builder.math_card("Exponential Distribution (PDF):", builder.render_latex_formula(exp_ltx, display=True)),
            builder.math_card("Exponential Distribution Example P(Y ≤ 60):", builder.render_array(rv_exp.cdf(60), display=False)),
            builder.math_card("Normal Gaussian Distribution (PDF):", builder.render_latex_formula(normal_ltx, display=True)),
            builder.math_card("Normal Distribution Example P(45 ≤ Y ≤ 60):", builder.render_array(rv_norm.cdf(60) - rv_norm.cdf(45), display=False)),
            builder.math_card("Standard Normal Distribution (PDF):", builder.render_latex_formula(std_normal_ltx, display=True)),
            builder.math_card("Standard Normal Distribution Example (z):", builder.render_array(stats.norm.cdf(z_60), display=False)),
            builder.math_card("Gamma Distribution (PDF):", builder.render_latex_formula(gamma_ltx, display=True)),
            builder.math_card("Chi Square Distribution (PDF):", builder.render_latex_formula(chisq_ltx, display=True)),
            builder.math_card("Cumulative Distribution Function Continuous:", builder.render_latex_formula(cdf_cont_ltx, display=True)),
            builder.math_card("Uniform Distribution (Continuous) CDF:", builder.render_latex_formula(cdf_uniform_cont_ltx, display=True)),
            builder.math_card("Continuous CDF (Empirical) F_Y(60):", builder.render_array((sample_data["Y_continuous"] <= 60).mean(), display=False)),
            builder.math_card("Exponential Distribution CDF:", builder.render_latex_formula(cdf_exponential_ltx, display=True)),
            builder.math_card("Normal Distribution CDF:", builder.render_latex_formula(cdf_normal_ltx, display=True)),
            builder.math_card("Continuous CDF (Normal Fit) F_Y(60):", builder.render_array(rv_norm.cdf(60), display=False)),
            builder.math_card("Standard Normal Distribution CDF:", builder.render_latex_formula(cdf_std_normal_ltx, display=True))
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
