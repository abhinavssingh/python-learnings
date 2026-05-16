# =======================
# LINEAR REGRESSION DERIVATION
# =======================

# Step 1: Define cost function
linreg_derivation_step1_ltx = (
    r"\text{Start with cost function: } "
    r"J(\mathbf{w}) = \frac{1}{m}(X\mathbf{w} - \mathbf{y})^T (X\mathbf{w} - \mathbf{y})"
)

# Step 2: Take derivative
linreg_derivation_step2_ltx = (
    r"\text{Take derivative w.r.t. } \mathbf{w}: "
    r"\frac{\partial J}{\partial \mathbf{w}} = \frac{2}{m} X^T (X\mathbf{w} - \mathbf{y})"
)

# Step 3: Set derivative to zero
linreg_derivation_step3_ltx = (
    r"\text{Set derivative to zero: } "
    r"X^T (X\mathbf{w} - \mathbf{y}) = 0"
)

# Step 4: Expand equation
linreg_derivation_step4_ltx = (
    r"\text{Rearrange: } X^T X \mathbf{w} = X^T \mathbf{y}"
)

# Step 5: Solve for weights
linreg_derivation_step5_ltx = (
    r"\text{Final solution: } "
    r"\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}"
)

# ===========================
# RIDGE REGRESSION DERIVATION
# ===========================
ridge_derivation_step1_ltx = (
    r"\text{Start with Ridge cost: } "
    r"J(\mathbf{w}) = (X\mathbf{w} - \mathbf{y})^T (X\mathbf{w} - \mathbf{y}) + \lambda \mathbf{w}^T \mathbf{w}"
)

ridge_derivation_step2_ltx = (
    r"\text{Take derivative: } "
    r"\frac{\partial J}{\partial \mathbf{w}} = 2X^T(X\mathbf{w} - \mathbf{y}) + 2\lambda \mathbf{w}"
)

ridge_derivation_step3_ltx = (
    r"\text{Set derivative = 0: } "
    r"X^T(X\mathbf{w} - \mathbf{y}) + \lambda \mathbf{w} = 0"
)

ridge_derivation_step4_ltx = (
    r"\text{Expand: } "
    r"X^T X \mathbf{w} + \lambda I \mathbf{w} = X^T \mathbf{y}"
)

ridge_derivation_step5_ltx = (
    r"\text{Final solution: } "
    r"\mathbf{w} = (X^T X + \lambda I)^{-1} X^T \mathbf{y}"
)

lasso_derivation_step1_ltx = (
    r"\text{Start with: } "
    r"J(\mathbf{w}) = \sum (y^{(i)} - \hat{y}^{(i)})^2 + \lambda \sum |w_j|"
)

# ===================================
# LASSO REGRESSION (KEY STEP INSIGHT)
# ===================================

lasso_derivation_step2_ltx = (
    r"\text{Derivative of } |w| \text{ is not smooth at } w=0"
)

lasso_derivation_step3_ltx = (
    r"\frac{d}{dw} |w| = "
    r"\begin{cases} 1 & w > 0 \\ -1 & w < 0 \\ \text{undefined} & w = 0 \end{cases}"
)

lasso_derivation_step4_ltx = (
    r"\text{Use subgradient methods instead of normal derivative}"
)

lasso_derivation_step5_ltx = (
    r"\text{Result: Many coefficients become exactly } 0"
)

# ======================
# ELASTIC NET DERIVATION
# ======================
elasticnet_derivation_step1_ltx = (
    r"\text{Start with: } "
    r"J(\mathbf{w}) = (X\mathbf{w} - \mathbf{y})^T (X\mathbf{w} - \mathbf{y}) "
    r"+ \lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \|\mathbf{w}\|_2^2"
)

elasticnet_derivation_step2_ltx = (
    r"\text{Split derivative into two parts (L1 + L2)}"
)

elasticnet_derivation_step3_ltx = (
    r"\text{L2 behaves like Ridge → smooth derivative}"
)

elasticnet_derivation_step4_ltx = (
    r"\text{L1 behaves like Lasso → subgradient}"
)

elasticnet_derivation_step5_ltx = (
    r"\text{Final model uses iterative optimization (e.g., coordinate descent)}"
)
