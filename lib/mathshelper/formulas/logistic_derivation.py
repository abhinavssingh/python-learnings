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

# == == == == == == == == == == == =
# NAIVE BAYES DERIVATION
# == == == == == == == == == == == =

# Step 1: Start with Bayes theorem
nb_derivation_step1_ltx = (
    r"\text{Start with Bayes theorem: } P(y|\mathbf{x}) = \frac{P(\mathbf{x}|y)P(y)}{P(\mathbf{x})}"
)

# Step 2: Drop denominator
nb_derivation_step2_ltx = (
    r"\text{Ignore } P(\mathbf{x}) \text{ since it is constant across classes}"
)

# Step 3: Apply naive assumption
nb_derivation_step3_ltx = (
    r"P(\mathbf{x}|y) = \prod_{j=1}^{n} P(x_j|y)"
)

# Step 4: Combine terms
nb_derivation_step4_ltx = (
    r"P(y|\mathbf{x}) \propto P(y) \prod_{j=1}^{n} P(x_j|y)"
)

# Step 5: Take log
nb_derivation_step5_ltx = (
    r"\log P(y|\mathbf{x}) = \log P(y) + \sum \log P(x_j|y)"
)

# Step 6: Final prediction

nb_derivation_step6_ltx = (
    r"\hat{y} = \arg\max_y \left( \log P(y) + \sum_{j=1}^{n} \log P(x_j \mid y) \right)"
)

# == == == == == == == == == == == =
# KNN DERIVATION(INTUITIVE)
# == =====================
# Step 1: Compute distance
knn_derivation_step1_ltx = (
    r"\text{Compute distance: } d(\mathbf{x}, \mathbf{x}^{(i)}) = \sqrt{\sum (x_j - x_j^{(i)})^2}"
)

# Step 2: Select neighbors
knn_derivation_step2_ltx = (
    r"\text{Select } k \text{ nearest neighbors}"
)

# Step 3: Voting
knn_derivation_step3_ltx = (
    r"\text{Count class frequency among neighbors}"
)

# Step 4: Final prediction
knn_derivation_step4_ltx = (
    r"\hat{y} = \arg\max_c \sum \mathbf{1}(y^{(i)} = c)"
)

# == == == == == == == == == == == =
# DECISION TREE DERIVATION
# == == == == == == == == == == == =
# Step 1: Define impurity
dt_derivation_step1_ltx = (
    r"\text{Compute impurity using entropy: } H(Y) = -\sum p(c)\log p(c)"
)

# Step 2: Split dataset
dt_derivation_step2_ltx = (
    r"\text{Split dataset into subsets } S_k"
)

# Step 3: Compute weighted entropy
dt_derivation_step3_ltx = (
    r"\sum \frac{|S_k|}{|S|} H(S_k)"
)

# Step 4: Compute information gain
dt_derivation_step4_ltx = (
    r"IG = H(Y) - \sum \frac{|S_k|}{|S|} H(S_k)"
)

# Step 5: Choose best split
dt_derivation_step5_ltx = (
    r"\text{Choose split that maximizes } IG"
)

# == == == == == == == == == == == =
# RANDOM FOREST DERIVATION
# == =====================
# Step 1: Bootstrap sampling
rf_derivation_step1_ltx = (
    r"\text{Create bootstrap samples from dataset}"
)

# Step 2: Train trees
rf_derivation_step2_ltx = (
    r"\text{Train decision tree } h_t(\mathbf{x}) \text{ on each sample}"
)

# Step 3: Reduce correlation
rf_derivation_step3_ltx = (
    r"\text{Select random subset of features at each split}"
)

# Step 4: Aggregate predictions
rf_derivation_step4_ltx = (
    r"\hat{y} = \arg\max_c \sum_{t=1}^{T} \mathbf{1}(h_t(\mathbf{x}) = c)"
)
# == == == == == == == == == == == =
# SVM DERIVATION
# == =====================
# Step 1: Define hyperplane
svm_derivation_step1_ltx = (
    r"\text{Define hyperplane: } \mathbf{w}^T \mathbf{x} + b = 0"
)

# Step 2: Define margin
svm_derivation_step2_ltx = (
    r"\text{Maximize margin between classes}"
)

# Step 3: Add constraints
svm_derivation_step3_ltx = (
    r"y^{(i)}(\mathbf{w}^T \mathbf{x}^{(i)} + b) \geq 1"
)

# Step 4: Introduce hinge loss
svm_derivation_step4_ltx = (
    r"\max(0, 1 - y(\mathbf{w}^T \mathbf{x}))"
)

# Step 5: Optimization
svm_derivation_step5_ltx = (
    r"\min \frac{1}{2}||\mathbf{w}||^2 + C \sum \max(0, 1 - y(\mathbf{w}^T \mathbf{x}))"
)

# == == == == == == == == == == == =
# BINARY SEARCH DERIVATION
# == =====================
# Step 1: Define bounds
binary_derivation_step1_ltx = (
    r"\text{Initialize low and high indices}"
)

# Step 2: Compute mid
binary_derivation_step2_ltx = (
    r"\text{mid} = \lfloor \frac{\text{low} + \text{high}}{2} \rfloor"
)

# Step 3: Compare values
binary_derivation_step3_ltx = (
    r"\text{Compare } x \text{ with } A[\text{mid}]"
)

# Step 4: Reduce search space
binary_derivation_step4_ltx = (
    r"\text{Eliminate half of the search space}"
)

# Step 5: Repeat
binary_derivation_step5_ltx = (
    r"\text{Repeat until element is found or range collapses}"
)
