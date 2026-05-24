# =======================
# LOGISTIC REGRESSION
# =======================

# Hypothesis (Scalar)
logreg_hypothesis_ltx = (
    r"\hat{y} = \sigma(w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n)"
)

# Hypothesis (Vector Form)
logreg_vector_hypothesis_ltx = (
    r"\hat{y} = \sigma(\mathbf{w}^T \mathbf{x})"
)

# Sigmoid Function
sigmoid_function_ltx = (
    r"\sigma(z) = \frac{1}{1 + e^{-z}}"
)

# =======================
# COST FUNCTION (Binary Cross Entropy)
# =======================

# Basic cost
logreg_cost_ltx = (
    r"J(\mathbf{w}) = -\frac{1}{m} \sum_{i=1}^{m} "
    r"[ y^{(i)} \log(\hat{y}^{(i)}) + "
    r"(1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) ]"
)

# Expanded cost
logreg_expanded_cost_ltx = (
    r"J(\mathbf{w}) = -\frac{1}{m} \sum_{i=1}^{m} "
    r"[ y^{(i)} \log(\sigma(\mathbf{w}^T \mathbf{x}^{(i)})) + "
    r"(1 - y^{(i)}) \log(1 - \sigma(\mathbf{w}^T \mathbf{x}^{(i)})) ]"
)

# =======================
# GRADIENT DESCENT
# =======================

logreg_gd_update_ltx = (
    r"w_j := w_j - \alpha \frac{\partial J}{\partial w_j}"
)

logreg_gradient_ltx = (
    r"\frac{\partial J}{\partial w_j} = "
    r"\frac{1}{m} \sum_{i=1}^{m} "
    r"(\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}"
)

logreg_full_update_ltx = (
    r"w_j := w_j - \alpha \frac{1}{m} "
    r"\sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}"
)

# =======================
# MATRIX FORM
# =======================

logreg_matrix_prediction_ltx = (
    r"\hat{\mathbf{y}} = \sigma(X \mathbf{w})"
)

logreg_matrix_cost_ltx = (
    r"J(\mathbf{w}) = -\frac{1}{m} "
    r"[ \mathbf{y}^T \log(\hat{\mathbf{y}}) + "
    r"(1 - \mathbf{y})^T \log(1 - \hat{\mathbf{y}}) ]"
)

# =======================
# DECISION BOUNDARY
# =======================

logreg_decision_boundary_ltx = (
    r"\mathbf{w}^T \mathbf{x} = 0"
)

# =======================
# PREDICTION RULE
# =======================

logreg_prediction_ltx = (
    r"\hat{y} = \begin{cases} "
    r"1 & \text{if } \sigma(\mathbf{w}^T \mathbf{x}) \geq 0.5 \\ "
    r"0 & \text{otherwise} "
    r"\end{cases}"
)

# =======================
# EVALUATION METRICS
# =======================

logreg_accuracy_ltx = (
    r"\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}"
)

logreg_precision_ltx = (
    r"\text{Precision} = \frac{TP}{TP + FP}"
)

logreg_recall_ltx = (
    r"\text{Recall} = \frac{TP}{TP + FN}"
)

logreg_f1_ltx = (
    r"F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}"
)

# =======================
# REGULARIZATION
# =======================

# L2 Regularization
logreg_l2_cost_ltx = (
    r"J(\mathbf{w}) = -\frac{1}{m} \sum_{i=1}^{m} [ "
    r"y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) ] "
    r"+ \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2"
)

# L1 Regularization
logreg_l1_cost_ltx = (
    r"J(\mathbf{w}) = -\frac{1}{m} \sum_{i=1}^{m} [ "
    r"y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) ] "
    r"+ \frac{\lambda}{m} \sum_{j=1}^{n} |w_j|"
)
