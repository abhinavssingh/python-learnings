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

# == == == == == == == == == == == =
# K - NEAREST NEIGHBORS(KNN)
# == == == == == == == == == == == =

# Distance(Euclidean)
knn_distance_ltx = (
    r"d(\mathbf{x}, \mathbf{x}^{(i)}) = \sqrt{\sum_{j=1}^{n} (x_j - x_j^{(i)})^2}"
)

# Prediction(Majority Vote)
knn_prediction_ltx = (
    r"\hat{y} = \arg\max_{c \in {1,\dots,K}} "
    r"\sum_{i \in \mathcal{N}_k} \mathbf{1}(y^{(i)} = c)"
)

# Weighted KNN
knn_weighted_ltx = (
    r"\hat{y} = \arg\max_{c} \sum_{i \in \mathcal{N}_k} "
    r"\frac{1}{d(\mathbf{x}, \mathbf{x}^{(i)})} \mathbf{1}(y^{(i)} = c)"
)

# == == == == == == == == == == == =
# NAIVE BAYES
# == == == == == == == == == == == =
# Bayes Theorem
nb_bayes_ltx = (
    r"P(y \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y) P(y)}{P(\mathbf{x})}"
)
# Naive Independence Assumption
nb_naive_ltx = (
    r"P(\mathbf{x} \mid y) = \prod_{j=1}^{n} P(x_j \mid y)"
)
# Prediction Rule
nb_prediction_ltx = (
    r"\hat{y} = \arg\max_{y} P(y) \prod_{j=1}^{n} P(x_j \mid y)"
)
# Log Form(Numerical Stability)
nb_log_ltx = (
    r"\hat{y} = \arg\max_{y} \left( \log P(y) + \sum_{j=1}^{n} \log P(x_j \mid y) \right)"
)


# == == == == == == == == == == == =
# DECISION TREE
# == == == == == == == == == == == =
# Entropy
dt_entropy_ltx = (
    r"H(Y) = -\sum_{c} p(c) \log p(c)"
)
# Gini Impurity
dt_gini_ltx = (
    r"G = 1 - \sum_{c} p(c)^2"
)
# Information Gain
dt_information_gain_ltx = (
    r"IG = H(Y) - \sum_{k} \frac{|S_k|}{|S|} H(S_k)"
)
# Split Rule
dt_split_ltx = (
    r"\text{Choose split that maximizes } IG"
)

# == == == == == == == == == == == =
# RANDOM FOREST
# == == == == == == == == == == == =
# Ensemble Prediction(Majority Vote)
rf_prediction_ltx = (
    r"\hat{y} = \arg\max_{c} \sum_{t=1}^{T} \mathbf{1}(h_t(\mathbf{x}) = c)"
)
# Averaging(Regression Version)
rf_regression_ltx = (
    r"\hat{y} = \frac{1}{T} \sum_{t=1}^{T} h_t(\mathbf{x})"
)
# Bootstrap Sampling
rf_bootstrap_ltx = (
    r"S_t \sim \text{sampling with replacement from dataset}"
)

# == == == == == == == == == == == =
# SUPPORT VECTOR MACHINE(SVM)
# == == == == == == == == == == == =
# Decision Function
svm_decision_ltx = (
    r"f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b"
)
# Hinge Loss
svm_hinge_loss_ltx = (
    r"J(\mathbf{w}) = \sum_{i=1}^{m} \max(0, 1 - y^{(i)}(\mathbf{w}^T \mathbf{x}^{(i)} + b))"
)
# Soft Margin Optimization
svm_soft_margin_ltx = (
    r"\min_{\mathbf{w}, b} \ \frac{1}{2}||\mathbf{w}||^2 + C \sum_{i=1}^{m} \xi_i"
)

# == == == == == == == == == == == =
# BINARY SEARCH(Algorithm, Not ML but Added)
# == == == == == == == == == == =
# Mid index
binary_mid_ltx = (
    r"\text{mid} = \lfloor \frac{\text{low} + \text{high}}{2} \rfloor"
)
# Update rule
binary_update_ltx = (
    r"\begin{cases} "
    r"\text{low} = \text{mid} + 1 & \text{if } x > A[\text{mid}] \\ "
    r"\text{high} = \text{mid} - 1 & \text{if } x < A[\text{mid}] "
    r"\end{cases}"
)


# Time Complexity
binary_complexity_ltx = (
    r"T(n) = O(\log n)"
)

# == == == == == == == == == == == =
# EXTRA: MULTICLASS(SOFTMAX REGRESSION)
# == == == == == == == == == == == =
# Softmax Function
softmax_ltx = (
    r"P(y = c \mid \mathbf{x}) = \frac{e^{\mathbf{w}c^T \mathbf{x}}}{\sum{k} e^{\mathbf{w}_k^T \mathbf{x}}}"
)
# Cross - Entropy Loss
softmax_loss_ltx = (
    r"J = - \sum_{i=1}^{m} \sum_{c} y_c^{(i)} \log \hat{y}_c^{(i)}"
)
