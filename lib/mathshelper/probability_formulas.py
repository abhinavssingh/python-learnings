"""
All probability and distribution LaTeX formulas.
Raw LaTeX only. No math delimiters ($, \\[ \\]).
Naming convention matches statistics constants: lowercase + *_ltx
"""

# =======================
# SIMPLE PROBABILITY
# =======================

simple_prob_ltx = r"P(X = x) = p_X(x)"

simple_prob_cont_ltx = r"P(a \le X \le b) = \int_a^b f(x)\,dx"

joint_prob_ltx = r"P(X = x, Y = y) = p_{X,Y}(x,y)"

joint_prob_cont_ltx = (
    r"P(a \le X \le b,\, c \le Y \le d)"
    r" = \int_a^b \int_c^d f(x,y)\,dy\,dx"
)

independent_ltx = r"P(X = x, Y = y) = P(X = x)P(Y = y)"

joint_pmf_ltx = r"P(X = x, Y = y) = p_{X,Y}(x,y)"

marginal_disc_ltx = r"P(X = x) = \sum_y P(X = x, Y = y)"

marginal_cont_ltx = r"f_X(x) = \int_{-\infty}^{\infty} f(x,y)\,dy"

conditional_disc_ltx = (
    r"P(X = x \mid Y = y)"
    r" = \frac{P(X = x, Y = y)}{P(Y = y)}"
)

conditional_cont_ltx = (
    r"f_{X|Y}(x|y) = \frac{f(x,y)}{f_Y(y)}"
)

joint_cdf_ltx = r"F_{X,Y}(x,y) = P(X \le x, Y \le y)"

joint_cdf_cont_ltx = (
    r"F_{X,Y}(x,y)"
    r" = \int_{-\infty}^{x}\int_{-\infty}^{y}"
    r" f_{X,Y}(u,v)\,dv\,du"
)

# =======================
# DISCRETE DISTRIBUTIONS
# =======================

uniform_disc_ltx = (
    r"P(X = x) = \frac{1}{n},"
    r" \quad x \in \{x_1, x_2, \dots, x_n\}"
)

bernoulli_ltx = r"P(X = x) = p^x (1-p)^{1-x}, \quad x \in \{0,1\}"

binom_ltx = (
    r"\text{Binomial: }"
    r" P(X=k) = \binom{n}{k} p^k (1-p)^{n-k},"
    r" \quad k=0,1,\dots,n"
)

geom_ltx = (
    r"\text{Geometric: }"
    r" P(X=k) = (1-p)^{k-1}p,"
    r" \quad k=1,2,\dots"
)

neg_binom_ltx = (
    r"\text{Negative Binomial: }"
    r" P(X=k) = \binom{k-1}{r-1} p^r (1-p)^{k-r},"
    r" \quad k=r,r+1,\dots"
)

poisson_ltx = (
    r"\text{Poisson: }"
    r" P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!},"
    r" \quad k=0,1,2,\dots"
)

cdf_disc_ltx = r"F_X(x) = P(X \le x) = \sum_{t \le x} P(X = t)"

cdf_binom_ltx = (
    r"F_X(k) = \sum_{i=0}^{k}"
    r" \binom{n}{i} p^i (1-p)^{n-i}"
)

cdf_poisson_ltx = (
    r"F_X(k) = \sum_{i=0}^{k}"
    r" \frac{\lambda^i e^{-\lambda}}{i!}"
)

# =======================
# CONTINUOUS DISTRIBUTIONS
# =======================

cu_ltx = (
    r"\text{Continuous Uniform: }"
    r" f(x)=\frac{1}{b-a},"
    r" \quad a \le x \le b"
)

normal_ltx = (
    r"f(x) = \frac{1}{\sqrt{2\pi\sigma^2}}"
    r" \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)"
)

std_normal_ltx = r"f(z) = \frac{1}{\sqrt{2\pi}} e^{-\frac{z^2}{2}}"

exp_ltx = (
    r"\text{Exponential: }"
    r" f(x)=\lambda e^{-\lambda x},"
    r" \quad x \ge 0"
)

gamma_ltx = (
    r"\text{Gamma: }"
    r" f(x)=\frac{1}{\Gamma(k)\theta^k}"
    r" x^{k-1} e^{-x/\theta},"
    r" \quad x \ge 0"
)

beta_ltx = (
    r"\text{Beta: }"
    r" f(x)=\frac{1}{B(\alpha,\beta)}"
    r" x^{\alpha-1}(1-x)^{\beta-1},"
    r" \quad 0 \le x \le 1"
)

weibull_ltx = (
    r"\text{Weibull: }"
    r" f(x)=\frac{k}{\lambda}"
    r"\left(\frac{x}{\lambda}\right)^{k-1}"
    r" e^{-(x/\lambda)^k},"
    r" \quad x \ge 0"
)

chisq_ltx = (
    r"\text{Chi-Square: }"
    r" f(x)=\frac{1}{2^{k/2}\Gamma(k/2)}"
    r" x^{k/2-1} e^{-x/2},"
    r" \quad x \ge 0"
)

t_ltx = (
    r"\text{Student t: }"
    r" f(x)=\frac{\Gamma\left(\frac{\nu+1}{2}\right)}"
    r"{\sqrt{\nu\pi}\Gamma\left(\frac{\nu}{2}\right)}"
    r" \left(1+\frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}"
)

lognorm_ltx = (
    r"\text{Log-Normal: }"
    r" f(x)=\frac{1}{x\sigma\sqrt{2\pi}}"
    r" e^{-\frac{(\ln x - \mu)^2}{2\sigma^2}},"
    r" \quad x > 0"
)

cdf_cont_ltx = (
    r"F_X(x) = P(X \le x)"
    r" = \int_{-\infty}^{x} f_X(t)\,dt"
)

cdf_uniform_cont_ltx = r"F_X(x) = \frac{x-a}{b-a}, \quad a \le x \le b"

cdf_exponential_ltx = r"F_X(x) = 1 - e^{-\lambda x}, \quad x \ge 0"

cdf_normal_ltx = (
    r"F_X(x) = \int_{-\infty}^{x}"
    r" \frac{1}{\sqrt{2\pi\sigma^2}}"
    r" e^{-\frac{(t-\mu)^2}{2\sigma^2}}\,dt"
)

cdf_std_normal_ltx = (
    r"\Phi(z) = \int_{-\infty}^{z}"
    r" \frac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}}\,dt"
)
