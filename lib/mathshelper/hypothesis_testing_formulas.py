# ===============================
# Z & T DISTRIBUTIONS
# ===============================

z_score_ltx = r"Z = \frac{X - \mu}{\sigma}"
std_normal_pdf_ltx = r"f(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2 / 2}"

t_pdf_ltx = (
    r"f(t) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}"
    r"{\sqrt{\nu\pi}\,\Gamma\left(\frac{\nu}{2}\right)}"
    r"\left(1 + \frac{t^2}{\nu}\right)^{-\frac{\nu+1}{2}}"
)

# ===============================
# HYPOTHESIS TESTS
# ===============================

z_test_ltx = r"Z = \frac{\bar{X} - \mu_0}{\sigma / \sqrt{n}}"

t_test_one_sample_ltx = r"t = \frac{\bar{X} - \mu_0}{s / \sqrt{n}}"

t_test_two_sample_ltx = (
    r"t = \frac{\bar{X}_1 - \bar{X}_2}"
    r"{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}"
)

pooled_variance_ltx = (
    r"s_p^2 = \frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}"
    r"{n_1 + n_2 - 2}"
)

z_vs_t_ltx = (
    r"\begin{cases}"
    r"\text{Z-Test, } \sigma \text{ known, } n \ge 30 \\"
    r"\text{T-Test, } \sigma \text{ unknown, } n < 30"
    r"\end{cases}"
)

# ===============================
# P-VALUE
# ===============================

p_value_ltx = r"p = P(\text{Test Statistic} \mid H_0)"

p_value_decision_ltx = (
    r"\begin{cases}"
    r"p \le \alpha \Rightarrow \text{Reject } H_0 \\"
    r"p > \alpha \Rightarrow \text{Fail to reject } H_0"
    r"\end{cases}"
)

# ===============================
# CHI-SQUARE
# ===============================

chi_square_pdf_ltx = (
    r"f(x) = \frac{1}{2^{k/2}\Gamma(k/2)}"
    r"x^{k/2 - 1} e^{-x/2}"
)

chi_square_test_ltx = r"\chi^2 = \sum \frac{(O - E)^2}{E}"

chi_square_expected_ltx = (
    r"E_{ij} = \frac{(\text{Row Total}_i)(\text{Column Total}_j)}{\text{Grand Total}}"
)

# ===============================
# ANOVA
# ===============================

anova_f_ltx = r"F = \frac{\text{Between-Group Variance}}{\text{Within-Group Variance}}"

sst_ltx = r"SST = \sum (X_{ij} - \bar{X})^2"
ssb_ltx = r"SSB = \sum n_i (\bar{X}_i - \bar{X})^2"
ssw_ltx = r"SSW = \sum \sum (X_{ij} - \bar{X}_i)^2"

anova_f_stat_ltx = r"F = \frac{SSB/(k-1)}{SSW/(N-k)}"

anova_assumptions_ltx = r"\text{Normality, Independence, Equal Variances}"

anova_types_ltx = r"\text{One-Way, Two-Way, Repeated Measures}"

# ===============================
# F DISTRIBUTION & F TEST
# ===============================

f_pdf_ltx = (
    r"f(x) = \frac{\sqrt{\frac{(d_1 x)^{d_1} d_2^{d_2}}{(d_1 x + d_2)^{d_1 + d_2}}}}"
    r"{x\, B\left(\frac{d_1}{2}, \frac{d_2}{2}\right)}"
)

f_from_chi_ltx = r"F = \frac{\chi_1^2/d_1}{\chi_2^2/d_2}"

f_test_ltx = r"F = \frac{s_1^2}{s_2^2}"

f_test_assumptions_ltx = r"\text{Normal populations, Independent samples}"

f_test_hypothesis_ltx = (
    r"\begin{aligned}"
    r"H_0 &: \sigma_1^2 = \sigma_2^2 \\"
    r"H_1 &: \sigma_1^2 \ne \sigma_2^2"
    r"\end{aligned}"
)
