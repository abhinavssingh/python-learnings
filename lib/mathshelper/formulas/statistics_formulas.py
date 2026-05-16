# ===============================
# MEASURES OF CENTRAL TENDENCY
# ===============================

mean_pop_ltx = r"\mu = \frac{1}{n}\sum_{i=1}^{n} x_i"

mean_sam_ltx = r"\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i"

median_ltx = (
    r"\text{Median} = "
    r"\begin{cases} "
    r"x_{\frac{n+1}{2}}, & \text{if } n \text{ is odd} \\ "
    r"\frac{x_{\frac{n}{2}} + x_{\frac{n}{2}+1}}{2}, & \text{if } n \text{ is even} "
    r"\end{cases}"
)

mode_ltx = r"\text{Mode} = \arg\max_x f(x)"


# ===============================
# MEASURES OF DISPERSION
# ===============================

var_pop_ltx = r"\sigma^2 = \frac{1}{n}\sum_{i=1}^{n} (x_i - \mu)^2"

var_sam_ltx = r"s^2 = \frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})^2"

std_pop_ltx = r"\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (x_i - \mu)^2}"

std_sam_ltx = r"s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})^2}"


# ===============================
# MEASURES OF ASSOCIATION
# ===============================

cov_pop_ltx = (
    r"\operatorname{Cov}(X,Y) = "
    r"\frac{1}{n}\sum_{i=1}^{n} (x_i - \mu_X)(y_i - \mu_Y)"
)

cov_sam_ltx = (
    r"s_{XY} = "
    r"\frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})"
)

corr_ltx = (
    r"\rho = "
    r"\frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}"
    r"{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2}"
    r"\sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}"
)


# ===============================
# MEASURES OF SHAPE
# ===============================

skew_pop_ltx = (
    r"\gamma_1 = "
    r"\frac{1}{n}\sum_{i=1}^{n}\left(\frac{x_i - \mu}{\sigma}\right)^3"
)

skew_sam_ltx = (
    r"g_1 = "
    r"\frac{1}{n}\sum_{i=1}^{n}\left(\frac{x_i - \bar{x}}{s}\right)^3"
)

kur_pop_ltx = (
    r"\gamma_2 = "
    r"\frac{1}{n}\sum_{i=1}^{n}\left(\frac{x_i - \mu}{\sigma}\right)^4"
)

kur_sam_ltx = (
    r"g_2 = "
    r"\frac{1}{n}\sum_{i=1}^{n}\left(\frac{x_i - \bar{x}}{s}\right)^4"
)
