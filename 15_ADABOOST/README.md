# AdaBoost (Adaptive Boosting) - From Scratch Implementation

A complete implementation of AdaBoost for binary classification using only NumPy. AdaBoost combines multiple weak learners (decision stumps) into a strong classifier by adaptively reweighting training samples after each round.

## Table of Contents
1. [Features](#features)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Usage](#usage)
6. [Examples](#examples)
7. [AdaBoost vs Gradient Boosting](#adaboost-vs-gradient-boosting)
8. [Advantages and Limitations](#advantages-and-limitations)

## Features

- **Decision Stumps**: Single-feature threshold classifiers as weak learners
- **Adaptive Weighting**: Misclassified samples receive higher weights each round
- **Weighted Voting**: Each stump votes with weight proportional to its accuracy
- **Probability Estimates**: `predict_proba()` via sigmoid of decision function
- **Staged Predictions**: `staged_predict()` and `staged_score()` for convergence analysis
- **Feature Importances**: Gain-based importance (sum of |alpha| per feature)
- **sklearn-compatible API**: Familiar `fit()`, `predict()`, `predict_proba()` methods

## Mathematical Foundation

### 1. Overview: What is AdaBoost?

AdaBoost (Adaptive Boosting) is an **ensemble method** that:
1. Trains weak learners **sequentially**, each focusing on the mistakes of the previous
2. Assigns **adaptive sample weights** — misclassified samples become more important
3. Combines weak learners via **weighted majority vote**
4. Converts many weak classifiers (slightly better than random) into a strong one

**Key Insight**: A weak learner only needs to be better than random (> 50% accuracy). By focusing on hard examples, AdaBoost dramatically improves overall performance.

### 2. Weak Learner: Decision Stump

A **decision stump** is the simplest tree: a single split on one feature.

For feature $j$ and threshold $\theta$:

```math
h(x) = \begin{cases} +1 & \text{if } x_j \geq \theta \\ -1 & \text{if } x_j < \theta \end{cases}
```

Or with reversed polarity ($p = -1$):

```math
h(x) = \begin{cases} -1 & \text{if } x_j \geq \theta \\ +1 & \text{if } x_j < \theta \end{cases}
```

The best stump minimizes **weighted error**:

```math
\epsilon_t = \sum_{i=1}^{n} w_i^{(t)} \cdot \mathbf{1}[y_i \neq h_t(x_i)]
```

Where $w_i^{(t)}$ are the sample weights at round $t$.

### 3. The AdaBoost Algorithm

**Input**: Training data $(x_1, y_1), \ldots, (x_n, y_n)$ with $y_i \in \{-1, +1\}$

**Step 1 — Initialize** uniform weights:

```math
w_i^{(1)} = \frac{1}{n}, \quad i = 1, \ldots, n
```

**Step 2 — For each round** $t = 1, \ldots, T$:

**(a)** Train weak learner $h_t$ to minimize weighted error:
```math
\epsilon_t = \sum_{i=1}^{n} w_i^{(t)} \cdot \mathbf{1}[y_i \neq h_t(x_i)]
```

**(b)** Compute stump weight (higher = more accurate):
```math
\alpha_t = \frac{\eta}{2} \ln \frac{1 - \epsilon_t}{\epsilon_t}
```
Where $\eta$ is the learning rate (default 1.0).

**(c)** Update sample weights:
```math
w_i^{(t+1)} = w_i^{(t)} \cdot e^{-\alpha_t \, y_i \, h_t(x_i)}
```

**(d)** Normalize weights so they sum to 1:
```math
w_i^{(t+1)} \leftarrow \frac{w_i^{(t+1)}}{\sum_{j=1}^{n} w_j^{(t+1)}}
```

**Step 3 — Final prediction** (weighted majority vote):
```math
H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t \, h_t(x)\right)
```

### 4. Understanding the Weight Update

The weight update rule has an elegant interpretation:

```math
w_i^{(t+1)} \propto w_i^{(t)} \cdot e^{-\alpha_t \, y_i \, h_t(x_i)}
```

- If **correctly classified** ($y_i = h_t(x_i)$): exponent is $-\alpha_t < 0$ → weight **decreases**
- If **misclassified** ($y_i \neq h_t(x_i)$): exponent is $+\alpha_t > 0$ → weight **increases**

Result: the next stump focuses more on previously misclassified samples.

### 5. The Stump Weight Alpha

The formula $\alpha_t = \frac{1}{2} \ln \frac{1 - \epsilon_t}{\epsilon_t}$ has key properties:

| Error $\epsilon_t$ | Alpha $\alpha_t$ | Meaning |
|--------------------|-----------------|---------|
| $\epsilon_t \to 0$ | $\alpha_t \to +\infty$ | Perfect stump → very high weight |
| $\epsilon_t = 0.5$ | $\alpha_t = 0$ | Random stump → zero weight |
| $\epsilon_t > 0.5$ | $\alpha_t < 0$ | Worse than random → flips prediction |

A stump with error > 0.5 still contributes: AdaBoost flips its vote (negative alpha).

### 6. Probability Estimation

AdaBoost produces a **decision function** (raw score):

```math
F(x) = \sum_{t=1}^{T} \alpha_t \, h_t(x)
```

Probabilities via **sigmoid** (Platt scaling):

```math
P(y = +1 \mid x) = \frac{1}{1 + e^{-2F(x)}}
```

### 7. Forward Stagewise Additive Modeling (FSAM)

AdaBoost is a **special case of FSAM** — the formal framework that explains WHY the algorithm and its formulas work.

**General FSAM problem**: minimize a loss $\mathcal{L}$ by adding weak learners one at a time without adjusting previous ones:

```math
F_t(x) = F_{t-1}(x) + \alpha_t h_t(x)
```

At each step $t$, find the optimal $(\alpha_t, h_t)$:

```math
(\alpha_t, h_t) = \arg\min_{\alpha, h} \sum_{i=1}^{n} \mathcal{L}\bigl(y_i,\; F_{t-1}(x_i) + \alpha \, h(x_i)\bigr)
```

For AdaBoost, the loss is **exponential**: $\mathcal{L}(y, F) = e^{-yF}$. Substituting:

```math
\sum_{i=1}^{n} e^{-y_i(F_{t-1}(x_i) + \alpha h(x_i))}
= \sum_{i=1}^{n} w_i^{(t)} \cdot e^{-\alpha y_i h(x_i)}
```

where $w_i^{(t)} = e^{-y_i F_{t-1}(x_i)}$ are exactly the **AdaBoost sample weights**.

Since $y_i h(x_i) \in \{-1, +1\}$, split the sum into correctly and incorrectly classified:

```math
\sum_{i=1}^{n} w_i^{(t)} e^{-\alpha y_i h(x_i)}
= e^{-\alpha}(W - \epsilon) + e^{\alpha} \epsilon
```

where $W = \sum_i w_i^{(t)}$ and $\epsilon = \sum_{i: y_i \neq h(x_i)} w_i^{(t)}$ is the **weighted error**.

### 8. Derivation of Alpha

Differentiating the FSAM objective with respect to $\alpha$ and setting to zero:

```math
\frac{d}{d\alpha}\bigl[e^{-\alpha}(W - \epsilon) + e^{\alpha}\epsilon\bigr] = 0
```

```math
-e^{-\alpha}(W - \epsilon) + e^{\alpha}\epsilon = 0
```

```math
e^{2\alpha} = \frac{W - \epsilon}{\epsilon}
\quad \Longrightarrow \quad
\boxed{\alpha_t = \frac{1}{2} \ln \frac{W - \epsilon}{\epsilon} = \frac{1}{2} \ln \frac{1 - \epsilon_t}{\epsilon_t}}
```

(after normalizing weights so $W = 1$). This is **not arbitrary** — it is the exact minimizer of the exponential loss at each step.

### 9. Derivation of the Weight Update

After fitting stump $t$ with weight $\alpha_t$, update the sample weights:

```math
w_i^{(t+1)} = e^{-y_i F_t(x_i)} = e^{-y_i (F_{t-1}(x_i) + \alpha_t h_t(x_i))}
= w_i^{(t)} \cdot e^{-\alpha_t y_i h_t(x_i)}
```

This **directly follows** from the FSAM framework — the weights are the exponential loss contributions. The formula is derived, not assumed.

**Intuition**:
- Correct prediction: $y_i h_t(x_i) = +1$ → multiply by $e^{-\alpha_t} < 1$ → weight **decreases**
- Wrong prediction: $y_i h_t(x_i) = -1$ → multiply by $e^{+\alpha_t} > 1$ → weight **increases**
- The ratio between wrong and correct weight updates: $e^{2\alpha_t} = (1-\epsilon_t)/\epsilon_t$

### 10. Exponential Loss

AdaBoost **implicitly minimizes** the exponential loss:

```math
\mathcal{L}(F) = \sum_{i=1}^{n} e^{-y_i F(x_i)}
```

**Why exponential loss?**
- Differentiable everywhere (unlike 0-1 loss)
- Penalizes misclassifications exponentially → hard focus on errors
- The negative gradient is: $-\frac{\partial \mathcal{L}}{\partial F(x_i)} = y_i e^{-y_i F(x_i)} = y_i w_i^{(t)}$

This connects AdaBoost to **Gradient Boosting**: AdaBoost is gradient boosting in function space with exponential loss.

**Comparison of loss functions:**

| Loss | Formula | Gradient | Used by |
|------|---------|----------|---------|
| 0-1 loss | $\mathbf{1}[y \neq \hat{y}]$ | Not differentiable | — |
| Exponential | $e^{-yF}$ | $-y e^{-yF}$ | **AdaBoost** |
| Log loss | $\log(1 + e^{-yF})$ | $-y \sigma(-yF)$ | Gradient Boosting |
| Hinge | $\max(0, 1-yF)$ | $-y \cdot \mathbf{1}[yF < 1]$ | SVM |
| Squared | $(y - F)^2$ | $2(F - y)$ | Regression GBM |

### 11. Margin Theory

The **functional margin** of sample $i$ is:

```math
\rho_i = y_i \cdot \frac{F(x_i)}{\sum_{t=1}^{T} \alpha_t} = y_i \cdot \frac{\sum_t \alpha_t h_t(x_i)}{\sum_t \alpha_t}
```

- $\rho_i > 0$: correctly classified
- $\rho_i < 0$: misclassified
- $|\rho_i|$: confidence of prediction

**Key theorem** (Schapire et al., 1998): The generalization error is bounded by:

```math
P(\text{test error}) \leq P(\rho_i \leq \theta) + O\!\left(\sqrt{\frac{T \log n}{n \theta^2}}\right)
```

For any margin threshold $\theta > 0$. AdaBoost continues to improve the **margin distribution** even after training error reaches 0, explaining why it rarely overfits on clean data.

**Minimum margin**: AdaBoost maximizes the minimum margin across all samples — analogous to SVM maximizing the geometric margin with respect to the decision boundary.

### 13. Feature Importance

Feature importance is based on the **total absolute alpha** from splits using each feature:

```math
\text{importance}(j) = \frac{\sum_{t: \, \text{stump}_t \text{ uses feature } j} |\alpha_t|}{\sum_{t=1}^{T} |\alpha_t|}
```

Features used by high-weight stumps receive higher importance.

### 14. Bias-Variance Analysis

AdaBoost primarily reduces **bias** (unlike Random Forest which reduces variance):

- Each additional stump reduces training error
- Too many rounds can increase variance (overfitting on noisy data)
- Learning rate $\eta < 1$ acts as regularization (shrinkage)
- Optimal number of estimators found via cross-validation

### 15. Convergence Guarantee

Under mild conditions, AdaBoost's training error decreases exponentially:

```math
\text{Train Error} \leq \exp\left(-2 \sum_{t=1}^{T} \gamma_t^2\right)
```

Where $\gamma_t = 0.5 - \epsilon_t$ is the **edge** of stump $t$ over random. If each stump has edge $\gamma > 0$, the error converges to 0.

### 16. AdaBoost vs Gradient Boosting

| Aspect | AdaBoost | Gradient Boosting |
|--------|---------|-------------------|
| **Loss function** | Exponential (implicit) | Any differentiable loss |
| **Weak learner** | Decision stumps (depth=1) | Shallow trees (depth 3-6) |
| **Weight update** | Sample reweighting | Residual fitting |
| **Sensitivity to noise** | High (exponential loss) | Lower (log/MSE loss) |
| **Speed** | Fast (stumps) | Slower (deeper trees) |
| **Interpretability** | High | Moderate |

### 17. Hyperparameters

| Parameter | Description | Typical range |
|-----------|-------------|---------------|
| `n_estimators` | Number of stumps | 50–500 |
| `learning_rate` | Shrinkage on alpha | 0.01–2.0 |

**Tuning tips:**
- Lower `learning_rate` + more `n_estimators` → better generalization
- Noisy data → reduce `n_estimators` to avoid overfitting

## Installation

No installation required beyond NumPy:
```bash
pip install numpy scikit-learn matplotlib  # sklearn only for testing
```

## Quick Start

```python
from adaboost import AdaBoostClassifier
import numpy as np

# Generate binary data
np.random.seed(42)
X = np.random.randn(200, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Train
clf = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
clf.fit(X[:150], y[:150])

# Predict
y_pred = clf.predict(X[150:])
y_proba = clf.predict_proba(X[150:])
acc = clf.score(X[150:], y[150:])

print(f"Accuracy: {acc:.3f}")
print(f"Feature importances: {clf.feature_importances_}")
```

## Usage

### Classification with Iris

```python
from adaboost import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
y_bin = (y != 0).astype(int)  # Binary: setosa vs rest

X_train, X_test, y_train, y_test = train_test_split(
    X, y_bin, test_size=0.3, random_state=42
)

clf = AdaBoostClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)

print(f"Test accuracy:         {clf.score(X_test, y_test):.3f}")
print(f"Feature importances:   {clf.feature_importances_}")
```

### Staged Score (convergence monitoring)

```python
from adaboost import AdaBoostClassifier
import numpy as np

clf = AdaBoostClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Monitor accuracy after each boosting round
scores = list(clf.staged_score(X_test, y_test))
print(f"Score after  1 round: {scores[0]:.3f}")
print(f"Score after 50 rounds: {scores[49]:.3f}")
print(f"Score after 100 rounds: {scores[99]:.3f}")
```

### Learning Rate Tuning

```python
from adaboost import AdaBoostClassifier

for lr in [0.1, 0.5, 1.0, 2.0]:
    clf = AdaBoostClassifier(n_estimators=50, learning_rate=lr, random_state=42)
    clf.fit(X_train, y_train)
    print(f"lr={lr}  acc={clf.score(X_test, y_test):.3f}")
```

## Examples

Run the test suite and generate visualizations:

```bash
cd 15_ADABOOST
python test_adaboost.py      # 12 tests, all passing
python generate_images.py    # 8 visualizations
```

**Test Results:**
- Test 1:  Decision Stump Predictions ✓
- Test 2:  Basic Binary Classification ✓
- Test 3:  Iris Dataset (Binary) ✓
- Test 4:  Predict Probabilities ✓
- Test 5:  Feature Importances ✓
- Test 6:  N Estimators Effect ✓
- Test 7:  Learning Rate Effect ✓
- Test 8:  Staged Score ✓
- Test 9:  Training Errors ✓
- Test 10: Decision Function ✓
- Test 11: Custom Labels {-1, +1} ✓
- Test 12: Reproducibility ✓

**Visualizations:**

| Image | Description |
|-------|-------------|
| `01_boundary_evolution.png` | Decision boundary over boosting rounds |
| `02_adaboost_vs_stump.png` | AdaBoost vs single stump comparison |
| `03_weight_evolution.png` | Sample weight updates per round |
| `04_convergence.png` | Train/test accuracy convergence |
| `05_learning_rate.png` | Learning rate vs n_estimators |
| `06_feature_importance.png` | Feature importances |
| `07_error_alpha.png` | Weighted error and alpha per round |
| `08_probability_contours.png` | Probability contours |

## AdaBoost vs Gradient Boosting

AdaBoost is a **special case** of Gradient Boosting with exponential loss.

### Key Differences

1. **Weak learner depth**
   - AdaBoost: stumps (depth=1)
   - GBM: shallow trees (depth 3-6)

2. **Focusing mechanism**
   - AdaBoost: reweights samples
   - GBM: fits to negative gradient (residuals)

3. **Noise sensitivity**
   - AdaBoost: exponential loss → sensitive to outliers
   - GBM: log/MSE loss → more robust

4. **Regularization**
   - AdaBoost: only `learning_rate`
   - GBM: `learning_rate`, subsampling, `max_depth`, `min_samples_split`

## Advantages and Limitations

### Advantages

1. **Simple**: only 2 hyperparameters (`n_estimators`, `learning_rate`)
2. **No preprocessing**: works directly on raw features
3. **Feature importance**: identifies most useful features
4. **Interpretable**: each stump is a simple rule
5. **Theoretical guarantees**: convergence proven under mild conditions
6. **Versatile**: works with any weak learner, not just stumps

### Limitations

1. **Binary only**: standard AdaBoost handles only 2 classes (SAMME for multi-class)
2. **Noise sensitive**: exponential loss amplifies outlier influence
3. **Sequential**: cannot be parallelized (unlike Random Forest)
4. **Weak learners only**: each stump is very limited — needs many rounds
5. **No built-in regularization**: beyond `learning_rate`

### When to Use AdaBoost

**Use AdaBoost when:**
- Binary classification problem
- Clean data (not too many outliers)
- Need fast training + simple tuning
- Interpretability matters (stumps are human-readable)
- Baseline ensemble before trying XGBoost/GBM

**Prefer other methods when:**
- Multi-class classification → use SAMME or GBM
- Noisy data → use Gradient Boosting (robust losses)
- High accuracy needed → use XGBoost (second-order optimization)
- Regression → use Gradient Boosting

## References

1. Freund, Y., & Schapire, R. E. (1997). "A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting." Journal of Computer and System Sciences, 55(1), 119-139.
2. Schapire, R. E. (1990). "The Strength of Weak Learnability." Machine Learning, 5(2), 197-227.
3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning." Chapter 10.
4. Friedman, J., Hastie, T., & Tibshirani, R. (2000). "Additive Logistic Regression: A Statistical View of Boosting."

## License

This implementation is for educational purposes.

---

**Note**: This is a from-scratch implementation for learning. For production use, consider `sklearn.ensemble.AdaBoostClassifier` which includes SAMME.R (multi-class), sample weighting optimizations, and additional base estimators.
