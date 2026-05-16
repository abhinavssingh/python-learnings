# =======================
# LINEAR REGRESSION
# =======================

# Hypothesis (Scalar)
linreg_hypothesis_ltx = r"\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n"

# Hypothesis (Vector Form)
linreg_vector_hypothesis_ltx = r"\hat{y} = \mathbf{w}^T \mathbf{x}"

# =======================
# COST FUNCTION
# =======================

linreg_cost_ltx = (
    r"J(\mathbf{w}) = \frac{1}{m} "
    r"\sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2"
)

linreg_expanded_cost_ltx = (
    r"J(\mathbf{w}) = \frac{1}{m} "
    r"\sum_{i=1}^{m} (\mathbf{w}^T \mathbf{x}^{(i)} - y^{(i)})^2"
)

# =======================
# GRADIENT DESCENT
# =======================

linreg_gd_update_ltx = (
    r"w_j := w_j - \alpha \frac{\partial J}{\partial w_j}"
)

linreg_gradient_ltx = (
    r"\frac{\partial J}{\partial w_j} = "
    r"\frac{2}{m} \sum_{i=1}^{m} "
    r"(\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}"
)

linreg_full_update_ltx = (
    r"w_j := w_j - \alpha \frac{2}{m} "
    r"\sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}"
)

# =======================
# MATRIX FORM
# =======================

linreg_matrix_prediction_ltx = r"\hat{\mathbf{y}} = X \mathbf{w}"

linreg_matrix_cost_ltx = (
    r"J(\mathbf{w}) = \frac{1}{m} "
    r"(X\mathbf{w} - \mathbf{y})^T (X\mathbf{w} - \mathbf{y})"
)

# =======================
# NORMAL EQUATION
# =======================

linreg_normal_equation_ltx = (
    r"\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}"
)

# =======================
# EVALUATION METRIC
# =======================

linreg_r2_ltx = (
    r"R^2 = 1 - "
    r"\frac{\sum (y_i - \hat{y}_i)^2}"
    r"{\sum (y_i - \bar{y})^2}"
)

# =======================
# BIAS-VARIANCE
# =======================

linreg_bias_variance_ltx = (
    r"\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}"
)

# =======================
# ASSUMPTIONS
# =======================

linreg_linearity_ltx = r"y = \mathbf{w}^T \mathbf{x}"

linreg_independence_ltx = r"\text{Cov}(\epsilon_i, \epsilon_j) = 0"

linreg_homoscedasticity_ltx = r"\text{Var}(\epsilon_i) = \sigma^2"


# Ridge Cost Function
ridge_cost_ltx = (
    r"J(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^{m} "
    r"(\hat{y}^{(i)} - y^{(i)})^2 + \lambda \sum_{j=1}^{n} w_j^2"
)

# Ridge Matrix Form
ridge_matrix_cost_ltx = (
    r"J(\mathbf{w}) = \frac{1}{m} (X\mathbf{w} - \mathbf{y})^T "
    r"(X\mathbf{w} - \mathbf{y}) + \lambda \mathbf{w}^T \mathbf{w}"
)

# Ridge Closed Form (Normal Equation)
ridge_normal_eq_ltx = (
    r"\mathbf{w} = (X^T X + \lambda I)^{-1} X^T \mathbf{y}"
)

# Lasso Cost Function
lasso_cost_ltx = (
    r"J(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^{m} "
    r"(\hat{y}^{(i)} - y^{(i)})^2 + \lambda \sum_{j=1}^{n} |w_j|"
)

# Lasso Matrix Form
lasso_matrix_cost_ltx = (
    r"J(\mathbf{w}) = \frac{1}{m} (X\mathbf{w} - \mathbf{y})^T "
    r"(X\mathbf{w} - \mathbf{y}) + \lambda \|\mathbf{w}\|_1"
)

# Elastic Net Cost Function
elasticnet_cost_ltx = (
    r"J(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^{m} "
    r"(\hat{y}^{(i)} - y^{(i)})^2 + "
    r"\lambda_1 \sum_{j=1}^{n} |w_j| + "
    r"\lambda_2 \sum_{j=1}^{n} w_j^2"
)

# Elastic Net Matrix Form
elasticnet_matrix_cost_ltx = (
    r"J(\mathbf{w}) = \frac{1}{m} (X\mathbf{w} - \mathbf{y})^T "
    r"(X\mathbf{w} - \mathbf{y}) + "
    r"\lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \|\mathbf{w}\|_2^2"
)
