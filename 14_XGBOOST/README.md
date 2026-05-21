# XGBoost (Extreme Gradient Boosting) - From Scratch Implementation

A complete implementation of XGBoost - one of the most powerful machine learning algorithms for structured/tabular data. XGBoost extends gradient boosting with advanced regularization, efficient tree construction, and novel optimization techniques.

## Table of Contents
1. [Features](#features)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Usage](#usage)
6. [Examples](#examples)
7. [XGBoost vs Gradient Boosting](#xgboost-vs-gradient-boosting)
8. [Advantages and Limitations](#advantages-and-limitations)

## Features

- **Regularized Objective**: L1 (Lasso) + L2 (Ridge) regularization
- **Second-Order Optimization**: Uses both gradients and Hessians
- **Tree Pruning**: Gamma parameter for complexity control
- **Column Subsampling**: Random feature selection per tree
- **Row Subsampling**: Stochastic sampling for variance reduction
- **Exact Greedy Split Finding**: Optimal split point search
- **Binary Classification & Regression**: Full support for both tasks
- **sklearn-compatible API**: Familiar `fit()`, `predict()`, `predict_proba()` methods

## Mathematical Foundation

### 1. Overview: What is XGBoost?

XGBoost (Extreme Gradient Boosting) is an **optimized** gradient boosting algorithm that:
1. Adds **regularization** to prevent overfitting
2. Uses **second-order** Taylor approximation for better optimization
3. Includes **tree pruning** with complexity penalty
4. Employs **column/row subsampling** for variance reduction
5. Implements **efficient** split finding algorithms

**Key Innovation**: Regularized objective with second-order optimization.

### 2. The Objective Function

XGBoost minimizes a **regularized** objective:

```math
\mathcal{L}(\phi) = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)
```

Where:
- $l(y_i, \hat{y}_i)$ = Loss function (e.g., log loss, squared error)
- $\Omega(f_k)$ = Regularization term for tree $k$
- $K$ = Number of trees
- $\hat{y}_i = \sum_{k=1}^{K} f_k(x_i)$ = Ensemble prediction

**Regularization** (unique to XGBoost):

```math
\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2 + \alpha \sum_{j=1}^{T} |w_j|
```

Where:
- $T$ = Number of leaves in tree
- $w_j$ = Weight (value) of leaf $j$
- $\gamma$ = Complexity penalty (min gain for split)
- $\lambda$ = L2 regularization (Ridge)
- $\alpha$ = L1 regularization (Lasso)

**Interpretation**:
- $\gamma T$ penalizes tree complexity (number of leaves)
- $\lambda \sum w_j^2$ prevents large leaf weights (L2)
- $\alpha \sum |w_j|$ encourages sparsity (L1)

### 3. Additive Training

XGBoost builds trees sequentially. At iteration $t$:

```math
\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta \cdot f_t(x_i)
```

Where:
- $\eta$ = Learning rate (shrinkage)
- $f_t$ = New tree added at step $t$

The objective at step $t$:

```math
\mathcal{L}^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)
```

### 4. Second-Order Taylor Approximation

**Key Innovation**: XGBoost uses second-order Taylor expansion.

Approximate the loss around $\hat{y}_i^{(t-1)}$:

```math
l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) \approx l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)
```

Where:
- $g_i = \frac{\partial l(y_i, \hat{y})}{\partial \hat{y}}\Big|_{\hat{y} = \hat{y}_i^{(t-1)}}$ = **First-order gradient**
- $h_i = \frac{\partial^2 l(y_i, \hat{y})}{\partial \hat{y}^2}\Big|_{\hat{y} = \hat{y}_i^{(t-1)}}$ = **Second-order gradient (Hessian)**

Removing constant terms:

```math
\tilde{\mathcal{L}}^{(t)} = \sum_{i=1}^{n} \left[g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)\right] + \Omega(f_t)
```

**Why second-order?**
- More accurate optimization (Newton's method vs gradient descent)
- Faster convergence
- Better handling of different loss functions

### 5. Tree Structure and Leaf Weights

Define instance set of leaf $j$:

```math
I_j = \{i | q(x_i) = j\}
```

Where $q(x_i)$ = leaf index for sample $i$.

Rewrite objective:

```math
\tilde{\mathcal{L}}^{(t)} = \sum_{j=1}^{T} \left[\left(\sum_{i \in I_j} g_i\right) w_j + \frac{1}{2} \left(\sum_{i \in I_j} h_i + \lambda\right) w_j^2\right] + \gamma T + \alpha \sum_{j=1}^{T} |w_j|
```

Let:
- $G_j = \sum_{i \in I_j} g_i$ = Sum of gradients in leaf $j$
- $H_j = \sum_{i \in I_j} h_i$ = Sum of hessians in leaf $j$

### 6. Optimal Leaf Weight

For a **fixed tree structure**, the optimal weight for leaf $j$ is:

```math
w_j^* = -\frac{G_j}{H_j + \lambda}
```

(Assuming $\alpha = 0$ for simplicity; with L1, use soft thresholding)

**With L1 Regularization** (soft thresholding):

```math
w_j^* = \begin{cases}
-\frac{G_j - \alpha}{H_j + \lambda} & \text{if } G_j > \alpha \\
-\frac{G_j + \alpha}{H_j + \lambda} & \text{if } G_j < -\alpha \\
0 & \text{otherwise}
\end{cases}
```

### 7. Objective Value (Quality Score)

Substituting optimal weights back:

```math
\tilde{\mathcal{L}}^{(t)}(q) = -\frac{1}{2} \sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma T
```

This is the **quality score** for tree structure $q$.

**Lower is better** (minimization).

### 8. Split Finding Algorithm

To find the best split, consider splitting leaf $j$ into left (L) and right (R):

**Gain from split**:

```math
\text{Gain} = \frac{1}{2} \left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma
```

Where:
- $G_L, H_L$ = Sum of gradients/hessians in left child
- $G_R, H_R$ = Sum of gradients/hessians in right child
- $\gamma$ = Complexity penalty

**Split if**: Gain > 0

**Pruning**: The $-\gamma$ term prunes splits that don't improve enough.

### 9. Exact Greedy Algorithm

For each leaf to split:
1. For each feature:
   - Sort samples by feature value
   - Scan from left to right
   - For each split point:
     - Calculate $G_L, H_L, G_R, H_R$
     - Compute Gain
   - Track best split
2. Choose feature and split with maximum Gain
3. If Gain > 0, split; otherwise, make leaf

**Complexity**: $O(n \cdot d \cdot \log n \cdot K)$ where:
- $n$ = samples, $d$ = features, $K$ = trees

### 10. Column (Feature) Subsampling

**Technique borrowed from Random Forests**:

At each tree (or each split), randomly sample a subset of features.

**Parameter**: `colsample_bytree` ∈ (0, 1]

**Benefits**:
- Reduces overfitting
- Faster training
- Decorrelates trees
- Similar to Random Forest's feature randomness

### 11. Row (Sample) Subsampling

**Stochastic Gradient Boosting**:

For each tree, sample a random subset of training data.

**Parameter**: `subsample` ∈ (0, 1]

**Benefits**:
- Reduces variance
- Faster training
- Prevents overfitting

### 12. Loss Functions

#### Binary Classification (Log Loss)

```math
l(y, \hat{y}) = y \log(1 + e^{-\hat{y}}) + (1 - y) \log(1 + e^{\hat{y}})
```

**Gradients**:
- $g_i = p_i - y_i$ where $p_i = \frac{1}{1 + e^{-\hat{y}_i}}$
- $h_i = p_i (1 - p_i)$

#### Regression (Squared Error)

```math
l(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2
```

**Gradients**:
- $g_i = \hat{y}_i - y_i$
- $h_i = 1$

### 13. XGBoost vs Standard Gradient Boosting

| Aspect | Gradient Boosting | XGBoost |
|--------|-------------------|---------|
| **Regularization** | None | L1 + L2 + Gamma |
| **Optimization** | First-order (gradient) | Second-order (gradient + Hessian) |
| **Tree Pruning** | Pre-pruning (max_depth) | Pre-pruning + Gain-based pruning |
| **Split Finding** | Greedy | Exact Greedy + Approximate |
| **Column Sampling** | ❌ No | ✅ Yes |
| **Leaf Weights** | Mean of residuals | Optimal via Newton step |
| **Speed** | ⭐⭐ Slower | ⭐⭐⭐ Faster |
| **Overfitting** | ⭐⭐ More prone | ⭐⭐⭐ Less prone |

### 14. Hyperparameters

#### Tree Parameters
- **max_depth**: Maximum tree depth (prevents overfitting)
- **min_child_weight**: Minimum sum of Hessian in child (prevents overfitting)
- **gamma**: Minimum gain for split (complexity penalty)

#### Regularization
- **reg_lambda** (λ): L2 regularization on leaf weights
- **reg_alpha** (α): L1 regularization on leaf weights

#### Sampling
- **subsample**: Fraction of samples for each tree
- **colsample_bytree**: Fraction of features for each tree

#### Learning
- **learning_rate** (η): Shrinkage (lower = more conservative)
- **n_estimators**: Number of trees

### 15. Practical Tips

**Preventing Overfitting:**
1. Increase regularization (lambda, alpha, gamma)
2. Reduce max_depth
3. Increase min_child_weight
4. Use subsampling (subsample < 1.0)
5. Use column sampling (colsample_bytree < 1.0)
6. Lower learning rate + more trees

**Improving Speed:**
1. Reduce max_depth
2. Increase min_child_weight
3. Use subsampling
4. Use column sampling
5. Reduce n_estimators

**Tuning Priority:**
1. Start with: max_depth=6, learning_rate=0.3, n_estimators=100
2. Tune max_depth (try 3-10)
3. Tune min_child_weight (try 1-10)
4. Add regularization (lambda, alpha)
5. Add sampling (subsample, colsample_bytree)
6. Fine-tune learning_rate and n_estimators

## Installation

No installation required beyond NumPy:
```bash
pip install numpy scikit-learn  # sklearn only for testing/comparison
```

## Quick Start

```python
from xgboost import XGBoostClassifier, XGBoostRegressor
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split

# Classification
X, y = load_iris(return_X_y=True)
y_binary = (y != 0).astype(int)  # Binary classification
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

xgb_clf = XGBoostClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    reg_lambda=1.0,
    random_state=42
)
xgb_clf.fit(X_train, y_train)

y_pred = xgb_clf.predict(X_test)
y_proba = xgb_clf.predict_proba(X_test)
accuracy = xgb_clf.score(X_test, y_test)

print(f"Classification Accuracy: {accuracy:.3f}")

# Regression
X_reg, y_reg = fetch_california_housing(return_X_y=True)
X_reg = X_reg[:500]  # Subset for speed
y_reg = y_reg[:500]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

xgb_reg = XGBoostRegressor(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
xgb_reg.fit(X_train_r, y_train_r)

r2 = xgb_reg.score(X_test_r, y_test_r)
print(f"Regression R²: {r2:.3f}")
```

## Usage

### Classification with Regularization

```python
from xgboost import XGBoostClassifier
import numpy as np

# Generate data
np.random.seed(42)
X = np.random.randn(200, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# XGBoost with L1 and L2 regularization
xgb = XGBoostClassifier(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    reg_lambda=1.0,    # L2 regularization
    reg_alpha=0.5,     # L1 regularization
    gamma=0.1,         # Complexity penalty
    random_state=42
)

xgb.fit(X[:150], y[:150])
acc = xgb.score(X[150:], y[150:])
print(f"Accuracy with regularization: {acc:.3f}")
```

### Regression with Subsampling

```python
from xgboost import XGBoostRegressor
import numpy as np

# Generate non-linear data
np.random.seed(42)
X = np.random.rand(300, 1) * 10 - 5
y = np.sin(X[:, 0]) + np.random.randn(300) * 0.1

# XGBoost with row and column subsampling
xgb = XGBoostRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,           # Use 80% of samples per tree
    colsample_bytree=0.8,    # Use 80% of features per tree
    random_state=42
)

xgb.fit(X[:200], y[:200])
r2 = xgb.score(X[200:], y[200:])
print(f"R² with subsampling: {r2:.3f}")
```

### Tuning min_child_weight

```python
from xgboost import XGBoostClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
y_binary = (y != 0).astype(int)

weights = [1.0, 5.0, 10.0]
for weight in weights:
    xgb = XGBoostClassifier(
        n_estimators=30,
        min_child_weight=weight,
        random_state=42
    )
    xgb.fit(X[:100], y_binary[:100])
    acc = xgb.score(X[100:], y_binary[100:])
    print(f"min_child_weight={weight}: accuracy={acc:.3f}")
```

## Examples

Run the comprehensive test suite:

```bash
python test_xgboost.py
```

**Test Results:**
- Test 1: Basic Binary Classification ✓
- Test 2: Iris Dataset (Binary Classification) ✓
- Test 3: Probability Predictions ✓
- Test 4: Basic Regression ✓
- Test 5: Regression on California Housing ✓
- Test 6: L2 Regularization (lambda) ✓
- Test 7: L1 Regularization (alpha) ✓
- Test 8: Gamma (Complexity Control) ✓
- Test 9: Row Subsampling ✓
- Test 10: Feature Subsampling ✓
- Test 11: Max Depth ✓
- Test 12: Learning Rate Effect ✓
- Test 13: Min Child Weight ✓
- Test 14: Number of Trees ✓
- Test 15: Non-linear Regression ✓

## XGBoost vs Gradient Boosting

### Key Differences

1. **Regularization**
   - GB: Only implicit (via learning rate, tree depth)
   - XGB: Explicit L1, L2, and gamma penalties

2. **Optimization**
   - GB: First-order (gradient descent)
   - XGB: Second-order (Newton's method with Hessian)

3. **Leaf Weights**
   - GB: Mean of residuals in leaf
   - XGB: Optimal weight via $w^* = -G/(H + \lambda)$

4. **Split Criterion**
   - GB: Variance reduction / MSE reduction
   - XGB: Gain with regularization penalty

5. **Sampling**
   - GB: Row subsampling only
   - XGB: Row + Column subsampling

### Performance Comparison

| Metric | Gradient Boosting | XGBoost |
|--------|-------------------|---------|
| **Overfitting Prevention** | ⭐⭐ Good | ⭐⭐⭐ Excellent |
| **Training Speed** | ⭐⭐ Moderate | ⭐⭐⭐ Fast |
| **Prediction Speed** | ⭐⭐⭐ Fast | ⭐⭐⭐ Fast |
| **Hyperparameter Tuning** | ⭐⭐ Moderate | ⭐⭐ Moderate |
| **Out-of-the-box Performance** | ⭐⭐ Good | ⭐⭐⭐ Excellent |

## Advantages and Limitations

### Advantages

1. **Regularization**: L1 + L2 + Gamma prevent overfitting
2. **Second-Order Optimization**: Faster convergence, better accuracy
3. **Flexibility**: Works for classification and regression
4. **Handles Non-linearity**: Captures complex patterns
5. **Feature Importance**: Implicit feature selection
6. **Sampling**: Row and column sampling reduce variance
7. **State-of-the-art**: Dominates Kaggle competitions
8. **Missing Values**: Can handle (not in this implementation)

### Limitations

1. **Black Box**: Less interpretable than linear models or single trees
2. **Hyperparameters**: Many parameters to tune
3. **Training Time**: Slower than simple models
4. **Sequential**: Cannot parallelize tree building (unlike Random Forest)
5. **Structured Data Only**: Not suitable for images, text (use deep learning)
6. **Overfitting Risk**: Can overfit without proper regularization

### When to Use XGBoost

**Use XGBoost When:**
- Working with **structured/tabular** data
- Need **high accuracy** (competitions, production)
- Have **sufficient data** (100+ samples)
- Can afford **tuning time**
- Non-linear relationships exist
- Feature interactions matter

**Avoid XGBoost When:**
- Need **interpretability** (use linear models, single trees)
- Have **very small data** (< 50 samples, use simpler models)
- Working with **images, text, audio** (use CNNs, RNNs, Transformers)
- Need **real-time predictions** (use simpler models)
- Computational resources limited

## Comparison: XGBoost vs Other Methods

### XGBoost vs Random Forest

| Aspect | XGBoost | Random Forest |
|--------|---------|---------------|
| **Training** | Sequential | Parallel |
| **Trees** | Weak learners (shallow) | Strong learners (deep) |
| **Regularization** | ✅ Explicit | ❌ Implicit only |
| **Accuracy** | ⭐⭐⭐ Higher | ⭐⭐ Good |
| **Speed** | ⭐⭐ Slower training | ⭐⭐⭐ Faster training |
| **Tuning** | ⭐⭐ More complex | ⭐⭐⭐ Simpler |

### XGBoost vs Neural Networks

| Aspect | XGBoost | Neural Networks |
|--------|---------|-----------------|
| **Data Type** | Tabular | Images, Text, Audio |
| **Data Size** | ⭐⭐⭐ Works with less | Needs large datasets |
| **Training Time** | ⭐⭐⭐ Faster | ⭐ Slower |
| **Interpretability** | ⭐⭐ Moderate | ⭐ Low |
| **Hyperparameter Tuning** | ⭐⭐ Moderate | ⭐ Complex |

## References

1. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." KDD 2016.
2. Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine."
3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning."

## License

This implementation is for educational purposes.

---

**Note**: This is a from-scratch implementation for learning. For production use, consider the official XGBoost library which includes additional optimizations: parallel tree construction, GPU support, approximate split finding, handling missing values, and more.
