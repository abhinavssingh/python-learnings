# =======================
# LOGISTIC REGRESSION DERIVATION
# =======================

# Step 1: Define hypothesis using sigmoid
logreg_derivation_step1_ltx = (
    r"\text{Start with hypothesis: } "
    r"\hat{y} = \sigma(X\mathbf{w}) = \frac{1}{1 + e^{-X\mathbf{w}}}"
)

# Step 2: Define cost function (Log Loss)
logreg_derivation_step2_ltx = (
    r"\text{Define cost function: } "
    r"J(\mathbf{w}) = -\frac{1}{m} \sum_{i=1}^{m} "
    r"[ y^{(i)} \log(\hat{y}^{(i)}) + "
    r"(1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) ]"
)

# Step 3: Substitute hypothesis into cost
logreg_derivation_step3_ltx = (
    r"\text{Substitute } \hat{y} = \sigma(X\mathbf{w}): "
    r"J(\mathbf{w}) = -\frac{1}{m} \sum_{i=1}^{m} "
    r"[ y^{(i)} \log(\sigma(X\mathbf{w})) + "
    r"(1 - y^{(i)}) \log(1 - \sigma(X\mathbf{w})) ]"
)

# Step 4: Take derivative w.r.t. weights
logreg_derivation_step4_ltx = (
    r"\text{Take derivative: } "
    r"\frac{\partial J}{\partial \mathbf{w}} = "
    r"\frac{1}{m} X^T (\hat{\mathbf{y}} - \mathbf{y})"
)

# Step 5: Key simplification (IMPORTANT INSIGHT)
logreg_derivation_step5_ltx = (
    r"\text{Using } \frac{d}{dz}\sigma(z) = \sigma(z)(1 - \sigma(z)), "
    r"\text{gradient simplifies to: } "
    r"X^T (\hat{\mathbf{y}} - \mathbf{y})"
)

# Step 6: Gradient descent update rule
logreg_derivation_step6_ltx = (
    r"\text{Update weights: } "
    r"\mathbf{w} := \mathbf{w} - \alpha \frac{1}{m} X^T (\hat{\mathbf{y}} - \mathbf{y})"
)

# Step 7: No closed-form solution
logreg_derivation_step7_ltx = (
    r"\text{Unlike linear regression: no closed-form solution exists}"
)

# Step 8: Final conclusion
logreg_derivation_step8_ltx = (
    r"\text{Solution is obtained via iterative optimization (Gradient Descent)}"
)
