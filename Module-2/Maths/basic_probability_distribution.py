import numpy as np

from lib.html import HtmlBuilder
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running basic probability distribution operation report...")
    # ...


builder = HtmlBuilder()
content = []

# DISCRETE PROBABILITY DISTRIBUTIONS
uniform_disc_ltx = r"P(X = x) = \frac{1}{n}, \quad x \in \{x_1, x_2, \dots, x_n\}"
bernoulli_ltx = r"P(X = x) = p^x (1-p)^{1-x}, \quad x \in \{0,1\}"
binom_ltx = r"\text{Binomial: } P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k=0,1,\dots,n"
geom_ltx = r"\text{Geometric: } P(X=k) = (1-p)^{k-1}p, \quad k=1,2,\dots"
neg_binom_ltx = r"\text{Negative Binomial: } P(X=k) = \binom{k-1}{r-1} p^r (1-p)^{k-r}, \quad k=r,r+1,\dots"
poisson_ltx = r"\text{Poisson: } P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k=0,1,2,\dots"

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

content.append(
    builder.grid(
        [

            builder.math_card(
                "Discrete Uniform Distribution (PMF):",
                builder.render_latex_formula(uniform_disc_ltx, display=True)
            ),

            builder.math_card(
                "Bernoulli Distribution (PMF):",
                builder.render_latex_formula(bernoulli_ltx, display=True)
            ),
            builder.math_card(
                "Binomial Distribution (PMF):",
                builder.render_latex_formula(binom_ltx, display=True)),
            builder.math_card(
                "Poisson Distribution (PMF):",
                builder.render_latex_formula(poisson_ltx, display=True)
            ),
            builder.math_card(
                "Continuous Uniform Distribution", builder.render_latex_formula(cu_ltx, display=True)),
            builder.math_card(
                "Exponential Distribution (PDF):",
                builder.render_latex_formula(exp_ltx, display=True)
            ),
            builder.math_card(
                "Normal Gaussian Distribution (PDF):",
                builder.render_latex_formula(normal_ltx, display=True)
            ),
            builder.math_card(
                "Standard Normal Distribution (PDF):",
                builder.render_latex_formula(std_normal_ltx, display=True)
            ),
            builder.math_card(
                "Gamma Distribution (PDF):",
                builder.render_latex_formula(gamma_ltx, display=True)
            ),
            builder.math_card(
                "Chi Square Distribution (PDF):",
                builder.render_latex_formula(chisq_ltx, display=True)
            )


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
