# CatBoost (Categorical Boosting) — From Scratch Implementation

A complete from-scratch implementation of CatBoost — Yandex's gradient boosting library
(Prokhorenkova et al., NeurIPS 2018) that introduces three fundamental innovations over
XGBoost and LightGBM: **symmetric (oblivious) trees**, **ordered boosting**, and
**ordered target statistics** for categorical features.

---

## Table of Contents

1. [Features](#features)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Hyperparameters Guide](#hyperparameters-guide)
6. [Visualizations](#visualizations)
7. [References](#references)

---

## Features

- **Oblivious Trees**: same split at every node of a given depth — $2^d$ leaves, $O(d)$ prediction
- **Ordered Boosting**: cross-fold unbiased gradient estimation — eliminates prediction shift
- **Ordered Target Statistics**: leak-free categorical feature encoding
- **L2 Regularization**: on leaf weights via $\lambda$ parameter
- **Row Subsampling**: `subsample` for variance reduction
- **Feature Importances**: gain-based, normalized to sum to 1
- **sklearn-compatible API**: `fit()`, `predict()`, `predict_proba()`, `score()`

---

## Mathematical Foundation

### 1. Gradient Boosting Foundation

CatBoost builds on the standard GBDT framework. Given a differentiable loss $L(y, F(x))$, the model is constructed additively:

**Initialize**:

```math
F_0(x) = \arg\min_{c} \sum_{i=1}^{n} L(y_i, c)
```

**Iterate** for $m = 1, \ldots, M$:

```math
F_m(x) = F_{m-1}(x) + \eta \cdot T_m(x)
```

where $T_m$ is a regression tree and $\eta \in (0, 1]$ is the learning rate.

**Second-order approximation**: using Taylor expansion around $F_{m-1}(x_i)$, define per-sample:

```math
g_i = \frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)}, \qquad
h_i = \frac{\partial^2 L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)^2}
```

The objective for tree $T_m$ simplifies to:

```math
\tilde{L}_m = \sum_{i=1}^{n} \left[ g_i \cdot T_m(x_i) + \frac{1}{2}\, h_i \cdot T_m(x_i)^2 \right] + \Omega(T_m)
```

---

### 2. Innovation 1 — Symmetric (Oblivious) Trees

#### 2.1 Definition

A **symmetric** or **oblivious** decision tree enforces that all nodes at the same depth $d$ share an identical split condition $(j, \theta)$. A depth-$d$ oblivious tree has exactly $2^d$ leaves.

Standard trees use different splits at each node; oblivious trees use the **same** $(j, \theta)$ at every node of a given level:

```math
\text{At depth } d: \quad \text{all nodes split on } (j_d,\, \theta_d)
```

#### 2.2 Prediction in O(depth)

Given the $d$ split conditions $(j_1, \theta_1), \ldots, (j_d, \theta_d)$, a sample $x$ reaches leaf index:

```math
\text{leaf}(x) = \sum_{k=1}^{d} \mathbf{1}[x_{j_k} > \theta_k] \cdot 2^{k-1}
```

This is a binary number of $d$ bits — each bit says whether the sample goes right at depth $k$.

**Complexity**: $O(d)$ per sample vs $O(2^d)$ node traversals for asymmetric trees.

#### 2.3 Greedy Level-Wise Growth

The tree is grown one level at a time. At depth $d$ with $K = 2^d$ current nodes, we find the single $(j, \theta)$ that maximises the **total gain summed across all nodes**:

```math
(j_d^*, \theta_d^*) = \arg\max_{j,\, \theta} \sum_{k=1}^{K} \text{Gain}_k(j, \theta)
```

where $\text{Gain}_k(j, \theta)$ is the standard split gain for node $k$:

```math
\text{Gain}_k(j, \theta) = \frac{G_{L,k}^2}{H_{L,k} + \lambda} + \frac{G_{R,k}^2}{H_{R,k} + \lambda} - \frac{G_k^2}{H_k + \lambda}
```

with $G_{L,k} = \sum_{i \in S_{L,k}} g_i$, $H_{L,k} = \sum_{i \in S_{L,k}} h_i$, and likewise for the right child and parent.

#### 2.4 Optimal Leaf Weight

Once the tree structure is fixed, the optimal weight for leaf $\ell$ with sample set $S_\ell$ is:

```math
w_\ell^* = -\frac{G_\ell}{H_\ell + \lambda}, \qquad G_\ell = \sum_{i \in S_\ell} g_i, \quad H_\ell = \sum_{i \in S_\ell} h_i
```

**Derivation**: the per-leaf objective $\tilde{L}_\ell = G_\ell w + \frac{1}{2}(H_\ell + \lambda) w^2$ is minimised by setting $d\tilde{L}_\ell / dw = 0$:

```math
G_\ell + (H_\ell + \lambda)\, w = 0 \quad \Longrightarrow \quad \boxed{w_\ell^* = -\frac{G_\ell}{H_\ell + \lambda}}
```

#### 2.5 Split Gain Formula Derivation

Substituting $w^*$ into $\tilde{L}$ gives the minimum value of the leaf objective:

```math
\tilde{L}^*_\ell = -\frac{G_\ell^2}{2(H_\ell + \lambda)}
```

The gain from splitting leaf $S$ into $S_L$ and $S_R$ is the decrease in the total objective:

```math
\text{Gain} = \tilde{L}^*(S) - \tilde{L}^*(S_L) - \tilde{L}^*(S_R)
```

```math
= \left(-\frac{G^2}{2(H+\lambda)}\right) - \left(-\frac{G_L^2}{2(H_L+\lambda)}\right) - \left(-\frac{G_R^2}{2(H_R+\lambda)}\right)
```

```math
\boxed{\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda}\right]}
```

#### 2.6 Regularization Effect

The $\lambda$ parameter smooths the denominator of $w^*$ and $\text{Gain}$:

- $\lambda \to 0$: no regularization, model can overfit
- $\lambda \to \infty$: all leaf weights $\to 0$, no learning

The symmetric constraint itself acts as a regularizer: by forcing all nodes at depth $d$ to use the same split, the tree has far fewer effective parameters than an asymmetric tree of the same depth. A depth-6 asymmetric tree has up to $2^6 - 1 = 63$ independent splits; the oblivious version has only $6$.

---

### 3. Innovation 2 — Ordered Boosting

#### 3.1 The Prediction Shift Problem

In standard GBDT, the gradient for sample $i$ at round $m$ is:

```math
g_i^{(m)} = \frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)}
```

The model $F_{m-1}$ was trained on **all** $n$ samples including sample $i$ itself. This means that $F_{m-1}(x_i)$ has already "memorised" $y_i$ to some degree, so the gradient $g_i^{(m)}$ is biased — it underestimates the true error on new data.

This bias is called **prediction shift** and leads to over-optimistic training loss estimates.

#### 3.2 Ordered Boosting Solution

CatBoost generates a random permutation $\sigma$ of $\{1, \ldots, n\}$ at each boosting round. For sample $\sigma(i)$, the gradient is computed using a model $M_{\sigma(i)}$ trained **only on the preceding samples** $\{\sigma(1), \ldots, \sigma(i-1)\}$:

```math
g_{\sigma(i)}^{(m)} = \frac{\partial L\!\left(y_{\sigma(i)},\; M_{\sigma(i-1)}(x_{\sigma(i)})\right)}{\partial M_{\sigma(i-1)}(x_{\sigma(i)})}
```

Since $M_{\sigma(i-1)}$ was never trained on $\sigma(i)$, the gradient estimate is **conditionally unbiased**:

```math
\mathbb{E}\!\left[g_{\sigma(i)}^{(m)} \,\Big|\, x_{\sigma(i)}, y_{\sigma(i)}\right] = \frac{\partial L(y_{\sigma(i)},\; F^*(x_{\sigma(i)}))}{\partial F^*}
```

where the expectation is over the permutation and $F^*$ is the true optimal function.

#### 3.3 Practical Implementation (Cross-Fold Approximation)

The exact ordered boosting requires maintaining $n$ separate models — computationally prohibitive. CatBoost uses a practical approximation with $K$ ordered folds:

**Step 1** — Generate random permutation $\pi$ of $\{1, \ldots, n\}$, divide into $K$ equal folds $I_1, \ldots, I_K$.

**Step 2** — For fold $k$ (samples $I_k$), compute gradients using the model built on folds $I_1, \ldots, I_{k-1}$:

```math
g_i^{(m)} = \frac{\partial L\!\left(y_i,\; F_{m-1}^{(-I_k)}(x_i)\right)}{\partial F_{m-1}^{(-I_k)}(x_i)}, \quad i \in I_k
```

where $F_{m-1}^{(-I_k)}$ denotes the model trained before seeing fold $k$ in this round.

**Step 3** — Build the tree using all samples $\{1, \ldots, n\}$ with these cross-fold gradients.

This gives $K$ approximately unbiased gradient estimates at a cost of $K$ model evaluations per round (not $K$ new model fits, since $F_{m-1}$ is reused).

#### 3.4 Bias–Variance Tradeoff in Ordered Boosting

| Property | Plain GBDT | Ordered Boosting |
|----------|-----------|-----------------|
| Gradient bias | High (prediction shift) | Low (approximately unbiased) |
| Gradient variance | Low (full dataset) | Higher (partial dataset per fold) |
| Generalisation | Can overfit | Better test performance |
| Computational cost | $O(n)$ per round | $O(K \cdot n)$ per round |

---

### 4. Innovation 3 — Ordered Target Statistics

#### 4.1 The Problem with Plain Mean Encoding

For a categorical feature with category $c$, plain mean encoding computes:

```math
\hat{x}_i = \frac{\sum_{j=1}^{n} y_j \cdot \mathbf{1}[\text{cat}_j = c]}{\sum_{j=1}^{n} \mathbf{1}[\text{cat}_j = c]}
```

This uses $y_i$ in the computation of $\hat{x}_i$ — introducing **target leakage**: the encoding "knows" the answer for the sample it is encoding. The model trained on these features will overfit.

#### 4.2 Ordered Target Statistics

CatBoost fixes this by computing $\hat{x}_i$ using only the samples that appear **before** $i$ in a random permutation $\pi$:

```math
\hat{x}_{\pi(i)} = \frac{\displaystyle\sum_{j < i} y_{\pi(j)} \cdot \mathbf{1}[\text{cat}_{\pi(j)} = c] \;+\; p \cdot s}{\displaystyle\sum_{j < i} \mathbf{1}[\text{cat}_{\pi(j)} = c] \;+\; s}
```

where $p$ is a prior value (e.g., the global mean of $y$) and $s > 0$ is the prior strength.

**Key property**: $y_{\pi(i)}$ is never used in computing $\hat{x}_{\pi(i)}$, so there is no target leakage:

```math
\mathbb{E}\!\left[\hat{x}_{\pi(i)} \,\Big|\, x_{\pi(i)}\right] \approx \mathbb{E}[y \mid \text{cat} = c]
```

in an unbiased sense, whereas plain encoding gives a biased (leaking) estimate.

#### 4.3 Smoothed Estimate

For rare categories (few preceding samples), the estimate reverts to the prior:

```math
\hat{x}_{\pi(i)} \;\xrightarrow{0 \text{ history}}\; p, \qquad \hat{x}_{\pi(i)} \;\xrightarrow{\infty \text{ history}}\; \mathbb{E}[y \mid \text{cat} = c]
```

The prior strength $s$ controls how quickly the estimate transitions from the prior to the empirical mean.

---

### 5. Loss Functions and Gradients

#### 5.1 Regression — Squared Error

```math
L(y, F) = \frac{1}{2}(y - F)^2
```

```math
g_i = F_i - y_i, \qquad h_i = 1
```

Base score: $F_0 = \bar{y}$

#### 5.2 Binary Classification — Log Loss

```math
L(y, F) = -\left[y \log \sigma(F) + (1-y) \log(1 - \sigma(F))\right]
```

where $\sigma(F) = \dfrac{1}{1 + e^{-F}}$.

```math
g_i = \sigma(F_i) - y_i, \qquad h_i = \sigma(F_i)\bigl(1 - \sigma(F_i)\bigr)
```

Base score: $F_0 = \log(n_+ / n_-)$

#### 5.3 Summary Table

| Loss | $L(y, F)$ | $g_i$ | $h_i$ |
|------|-----------|--------|--------|
| Squared error | $\frac{1}{2}(y-F)^2$ | $F_i - y_i$ | $1$ |
| Log loss | $-[y\log\sigma(F) + (1-y)\log(1-\sigma(F))]$ | $\sigma(F_i) - y_i$ | $\sigma(F_i)(1-\sigma(F_i))$ |
| Poisson | $F - y\log F$ | $e^{F_i} - y_i$ | $e^{F_i}$ |

---

### 6. Feature Importance (Gain-Based)

For each split condition at depth $d$ of tree $T_m$, the gain $\text{Gain}_d^{(m)}$ for feature $j_d$ is recorded. The importance of feature $j$ is:

```math
I_j = \sum_{m=1}^{M} \sum_{d=1}^{\text{depth}(T_m)} \text{Gain}_d^{(m)} \cdot \mathbf{1}[j_d^{(m)} = j]
```

Normalised to sum to 1:

```math
\text{importance}(j) = \frac{I_j}{\sum_{k=1}^{p} I_k}
```

Note: in an oblivious tree, the same feature $j_d$ is used by ALL nodes at depth $d$, so its contribution is the sum of gains across all $2^{d-1}$ nodes at that level.

---

### 7. Comparison: CatBoost vs XGBoost vs LightGBM

| Property | XGBoost | LightGBM | CatBoost |
|----------|---------|----------|----------|
| Tree type | Asymmetric | Asymmetric (leaf-wise) | Symmetric (oblivious) |
| Gradient estimation | Standard (biased) | Standard (biased) | Ordered (approximately unbiased) |
| Categorical features | Manual encoding | Manual encoding | Built-in ordered target statistics |
| Split finding | Exact or approx | Histogram $O(B)$ | Histogram or exact per level |
| Memory per sample | float64 | uint8 (binned) | float32 |
| Prediction speed | Fast | Fast | Very fast ($O(d)$ per sample) |
| Regularization | L1 + L2 + gamma | L1 + L2 + gamma | L2 + symmetric constraint |
| Best for | General tabular | Large datasets | Categorical-heavy data |

---

### 8. Full Training Algorithm

**Input**: $(X, y)$, parameters $M$, $\eta$, `depth`, $\lambda$, `subsample`
**Output**: ensemble of $M$ oblivious trees, base score

```
Initialize:
    F_0  <-  base_score(y)

For m = 1 to M:

  (1) Gradient computation:
        Plain mode:
            g_i  <-  dL(y_i, F_{m-1}(x_i)) / dF
            h_i  <-  d2L(y_i, F_{m-1}(x_i)) / dF2

        Ordered mode (K folds):
            pi  <-  random_permutation(n)
            For fold k = 1..K:
                F_fold  <-  F_{m-1} evaluated on fold k
                g_i, h_i  <-  dL / dF  for i in fold k

  (2) Row subsampling (if subsample < 1):
        S  <-  random_sample(n, subsample * n)

  (3) Build oblivious tree T_m on sample set S:
        node_assignments  <-  zeros(|S|)   [all samples in root]

        For depth d = 1, ..., max_depth:
            For each candidate (feature j, threshold theta):
                For each current node k:
                    Compute Gain_k(j, theta)
                total_gain  <-  sum_k Gain_k(j, theta)

            (j_d*, theta_d*)  <-  argmax total_gain
            Update node_assignments:
                leaf_index  <-  leaf_index * 2 + (x_{j_d*} > theta_d*)

        For each leaf l:
            w_l  <-  -G_l / (H_l + lambda)

  (4) Update predictions:
        F_m(x_i)  <-  F_{m-1}(x_i)  +  eta * T_m(x_i)

Return all trees, F_0
```

---

## Installation

```bash
pip install numpy scikit-learn matplotlib
```

---

## Quick Start

```python
from catboost_scratch import CatBoostClassifier, CatBoostRegressor
from catboost_scratch import ordered_target_statistics

# --- Binary Classification ---
clf = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    reg_lambda=3.0,
    use_ordered_boosting=True,
    n_ordered_folds=4,
    random_state=42
)
clf.fit(X_train, y_train)
print(f"Accuracy:    {clf.score(X_test, y_test):.3f}")
print(f"Importances: {clf.feature_importances_}")

proba = clf.predict_proba(X_test)   # shape (n_test, 2)

# --- Regression ---
reg = CatBoostRegressor(
    iterations=200,
    learning_rate=0.05,
    depth=6,
    reg_lambda=3.0,
    random_state=42
)
reg.fit(X_train, y_train)
print(f"R2: {reg.score(X_test, y_test):.3f}")

# --- Ordered Target Statistics for categorical features ---
# Encode a categorical column before training
encoded = ordered_target_statistics(
    categories=X_train[:, cat_col].astype(int),
    targets=y_train,
    prior=y_train.mean(),
    prior_strength=1.0
)
```

---

## Hyperparameters Guide

### Core Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `iterations` | 100 | Number of boosting rounds $M$ |
| `learning_rate` | 0.1 | Shrinkage $\eta$ — smaller needs more iterations |
| `depth` | 6 | Oblivious tree depth — tree has $2^d$ leaves |

### Regularization

| Parameter | Default | Effect |
|-----------|---------|--------|
| `reg_lambda` | 3.0 | L2 coefficient $\lambda$: shrinks leaf weights |
| `min_samples_leaf` | 1 | Minimum $\|S\|$ per leaf to allow a split |

### Sampling

| Parameter | Default | Effect |
|-----------|---------|--------|
| `subsample` | 1.0 | Row subsampling fraction |
| `colsample_bylevel` | 1.0 | Feature subsampling per depth level |

### Ordered Boosting

| Parameter | Default | Effect |
|-----------|---------|--------|
| `use_ordered_boosting` | False | Enable cross-fold ordered gradient estimation |
| `n_ordered_folds` | 4 | Number of folds $K$ for ordered boosting |

### Recommended Tuning

```
1. Start: iterations=500, learning_rate=0.05, depth=6
2. Tune depth in [4, 6, 8] — shallower = more regularized
3. Tune reg_lambda in [1, 3, 10] — higher = stronger regularization
4. Enable ordered boosting for small/medium datasets
5. Final: lower learning_rate + more iterations
```

---

## Visualizations

| Image | Description |
|-------|-------------|
| `01_oblivious_tree_structure.png` | Symmetric tree diagram vs standard asymmetric tree |
| `02_ordered_target_statistics.png` | Plain mean encoding vs ordered target statistics |
| `03_ordered_boosting.png` | Plain vs ordered boosting on a regression task |
| `04_decision_boundary.png` | Decision boundary evolution over boosting rounds |
| `05_regression.png` | Nonlinear function approximation at different iteration counts |
| `06_feature_importances.png` | Gain-based feature importances at different depths |
| `07_depth_effect.png` | Effect of tree depth ($2^d$ leaves) on model complexity |
| `08_regularization.png` | Effect of $\lambda$ (L2) on regression fit |

---

## References

1. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., Gulin, A. (2018).
   **CatBoost: unbiased boosting with categorical features.**
   *NeurIPS 2018*.

2. Dorogush, A. V., Ershov, V., Gulin, A. (2018).
   **CatBoost: gradient boosting with categorical features support.**
   *arXiv:1810.11363*.

3. Chen, T., Guestrin, C. (2016). **XGBoost: A Scalable Tree Boosting System.** *KDD 2016*.

4. Ke, G., et al. (2017). **LightGBM: A Highly Efficient Gradient Boosting Decision Tree.** *NeurIPS 2017*.

5. Friedman, J. H. (2001). **Greedy Function Approximation: A Gradient Boosting Machine.**
   *Annals of Statistics, 29(5)*, 1189–1232.

---

*Implementation: ML Algorithms From Scratch — Module 17*
