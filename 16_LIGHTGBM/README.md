# LightGBM (Light Gradient Boosting Machine) — From Scratch Implementation

A complete from-scratch implementation of LightGBM — Microsoft's highly efficient gradient boosting
framework (Ke et al., NeurIPS 2017) that achieves state-of-the-art accuracy with significantly lower
training time than XGBoost through three core innovations:
**histogram-based split finding**, **leaf-wise tree growth**, and **GOSS sampling**.

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

- **Histogram-Based Splits**: Discretize features into at most $B$ bins — split scan is $O(B)$ instead of $O(n \log n)$
- **Leaf-Wise Growth**: Always splits the leaf with maximum gain — more accurate per tree than level-wise
- **GOSS Sampling**: Keeps all large-gradient samples; randomly samples small-gradient ones with compensating weights
- **L1 + L2 + Gamma Regularization**: Soft-thresholding leaf weights, L2 shrinkage, minimum gain pruning
- **Feature & Row Subsampling**: `colsample_bytree` and `subsample` for variance reduction
- **Feature Importances**: Normalized gain-based importances across all trees
- **sklearn-compatible API**: `fit()`, `predict()`, `predict_proba()`, `score()`

---

## Mathematical Foundation

### 1. Gradient Boosting Foundation

LightGBM is a **gradient boosted decision tree (GBDT)** ensemble. Given a loss function $L(y, F(x))$, the model is built additively:

**Initialize** with the best constant prediction:

```math
F_0(x) = \arg\min_{c} \sum_{i=1}^{n} L(y_i,\, c)
```

**Iterate** for $m = 1, \ldots, M$:

```math
F_m(x) = F_{m-1}(x) + \eta \cdot T_m(x)
```

where $T_m$ is a regression tree fitted at round $m$ and $\eta \in (0, 1]$ is the learning rate (shrinkage).

---

### 2. Second-Order Taylor Approximation

At round $m$, define for each sample $i$ the **first-order gradient** and **second-order gradient (hessian)**:

```math
g_i = \frac{\partial L(y_i,\, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)}
```

```math
h_i = \frac{\partial^2 L(y_i,\, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)^2}
```

A second-order Taylor expansion of the loss around $F_{m-1}(x_i)$ gives:

```math
L\bigl(y_i,\ F_{m-1}(x_i) + T_m(x_i)\bigr)
\approx
L\bigl(y_i,\ F_{m-1}(x_i)\bigr)
+ g_i \cdot T_m(x_i)
+ \frac{1}{2}\, h_i \cdot T_m(x_i)^2
```

Dropping the constant term (no effect on the optimization), the objective for tree $T_m$ becomes:

```math
\tilde{L}_m = \sum_{i=1}^{n} \left[ g_i \cdot T_m(x_i) + \frac{1}{2}\, h_i \cdot T_m(x_i)^2 \right] + \Omega(T_m)
```

where $\Omega(T_m)$ is the regularization term on the tree structure.

---

### 3. Innovation 1 — Histogram-Based Split Finding

#### 3.1 Why Histograms?

Standard GBDT finds the best split for feature $j$ by:
1. Sorting all $n$ samples by $x_j$: costs $O(n \log n)$
2. Scanning all $n-1$ candidate thresholds: costs $O(n)$

For $p$ features and $M$ rounds this is $O(M \cdot p \cdot n \log n)$ — too slow for large $n$.

#### 3.2 Bin Construction (done once before training)

LightGBM **discretizes** each continuous feature $j$ into at most $B$ bins (default $B = 255$).

**Step 1 — Compute percentile-based bin edges** from the full training data:

```math
\text{edges}_j = \Bigl\{ \text{percentile}\!\bigl(X_{:,j},\ k \cdot \tfrac{100}{B}\bigr) \;:\; k = 0, 1, \ldots, B \Bigr\}
```

**Step 2 — Map each continuous value to its bin index** (integer in $\{0, \ldots, B-1\}$):

```math
\tilde{x}_{i,j} = \text{searchsorted}(\text{edges}_j,\; x_{i,j}) - 1
```

All subsequent training operations use integer bin indices instead of floating-point values — halving memory and enabling cache-friendly access.

#### 3.3 Histogram Accumulation for a Leaf

For a leaf holding sample set $S$, build three arrays for each feature $j$:

```math
G_{j,b} = \sum_{i \in S,\ \tilde{x}_{i,j} = b} g_i
\qquad
H_{j,b} = \sum_{i \in S,\ \tilde{x}_{i,j} = b} h_i
\qquad
c_{j,b} = \bigl|\{ i \in S : \tilde{x}_{i,j} = b \}\bigr|
```

for $b = 0, 1, \ldots, B-1$. This requires a single pass over $|S|$ samples: cost $O(|S|)$.

#### 3.4 Scanning All $B$ Split Thresholds

With the histogram ready, scan all $B - 1$ possible splits in $O(B)$ using prefix sums:

```math
G_L^{(b)} = \sum_{k=0}^{b} G_{j,k}, \qquad
H_L^{(b)} = \sum_{k=0}^{b} H_{j,k}, \qquad
G_R^{(b)} = G_S - G_L^{(b)}, \qquad
H_R^{(b)} = H_S - H_L^{(b)}
```

where $G_S = \sum_{i \in S} g_i$ and $H_S = \sum_{i \in S} h_i$ are the leaf totals.

The gain at threshold $b$ is then computed in $O(1)$ (see Section 6).

**Total cost per leaf per feature**: $O(|S|)$ histogram build $+$ $O(B)$ scan.  
**Overall training**: $O(M \cdot p \cdot n)$ — the $\log n$ factor of exact split finding is eliminated.

#### 3.5 Histogram Subtraction Trick

After splitting a parent node into left and right children, build only the **smaller child's** histogram in $O(|S_{\text{small}}|)$, then recover the larger one in $O(B)$:

```math
\text{Hist}(S_R) = \text{Hist}(S_{\text{parent}}) - \text{Hist}(S_L)
```

This is a $O(B)$ subtraction regardless of how many samples are in $S_R$ — a major speedup for nearly balanced splits.

---

### 4. Innovation 2 — Leaf-Wise Tree Growth

#### 4.1 Level-Wise Growth (XGBoost style)

In level-wise (breadth-first) growth, **all** leaves at the same depth are split simultaneously:

| Round | Action | Leaves |
|-------|--------|--------|
| 0 | Split root | 2 |
| 1 | Split all 2 leaves | 4 |
| 2 | Split all 4 leaves | 8 |

After $d$ rounds: $2^d$ leaves. Many of these splits have negligible gain — wasted computation.

#### 4.2 Leaf-Wise Growth (LightGBM style)

LightGBM maintains a pool of current leaves and always splits the **single leaf with the highest gain**:

```math
\ell^* = \arg\max_{\ell \in \text{leaves}} \; \text{Gain}(\ell)
```

Repeat until the number of leaves reaches `num_leaves` or $\text{Gain}(\ell^*) \leq \gamma$.

**Formal guarantee**: for the same leaf budget $L$, leaf-wise growth achieves at least as much total gain as level-wise growth, because it is a greedy optimum over the same budget:

```math
\sum_{\ell} \text{Gain}_{\text{leaf-wise}}(\ell) \;\geq\; \sum_{\ell} \text{Gain}_{\text{level-wise}}(\ell)
```

#### 4.3 Overfitting and Mitigation

Leaf-wise trees grow **asymmetric** and can become very deep on small datasets, causing overfitting. Key regularization levers:

| Parameter | Role |
|-----------|------|
| `num_leaves` | Hard cap on total number of leaves |
| `min_data_in_leaf` | Minimum $\|S\|$ required to attempt a split |
| `min_child_weight` | Minimum $H_S$ required to attempt a split |
| `gamma` | Minimum gain required to accept a split |

---

### 5. Innovation 3 — GOSS Sampling

#### 5.1 Motivation

Samples that are **already well-predicted** have small $|g_i|$ and contribute little to the next tree. GOSS exploits this: keep all informative (large-gradient) samples, sub-sample the rest.

#### 5.2 GOSS Algorithm

Given gradients $g_1, \ldots, g_n$ at round $m$:

**Step 1** — Sort samples by $|g_i|$ in descending order:

```math
|g_{\pi(1)}| \geq |g_{\pi(2)}| \geq \cdots \geq |g_{\pi(n)}|
```

**Step 2** — Keep the top fraction $a$ (large-gradient set $A$):

```math
A = \bigl\{\pi(1),\; \pi(2),\; \ldots,\; \pi(\lfloor a \cdot n \rfloor)\bigr\}
```

**Step 3** — Randomly sample fraction $b$ from the remaining (small-gradient set $B$):

```math
B \subseteq \bigl\{\pi(\lfloor a n \rfloor + 1),\; \ldots,\; \pi(n)\bigr\}, \quad |B| = \lfloor b \cdot n \rfloor
```

**Step 4** — Scale the gradients of samples in $B$ to preserve the distribution:

```math
\tilde{g}_i = \begin{cases} g_i & \text{if } i \in A \\[4pt] \dfrac{1 - a}{b} \cdot g_i & \text{if } i \in B \end{cases}
```

**Step 5** — Build histograms and grow the tree using only $A \cup B$ with scaled gradients $\tilde{g}_i$.

#### 5.3 Why the Scaling Factor?

Without scaling, removing small-gradient samples would introduce bias because their total gradient contribution would be missing. The scale factor:

```math
s = \frac{1 - a}{b}
```

ensures that $B$ represents the full small-gradient pool:

```math
\sum_{i \in B} s \cdot g_i \approx \sum_{i \notin A} g_i
```

This is **importance sampling**: each element of $B$ is a proxy for $1/b$ elements from its pool, scaled by the fraction $(1-a)$ that was excluded from $A$.

#### 5.4 Theoretical Guarantee

The approximation error variance of GOSS versus using the full dataset satisfies:

```math
\mathcal{E}(\tilde{V}_j) = O\!\left(\frac{1}{|A| + s^2 \cdot |B|}\right)
```

For typical values $a = 0.2$, $b = 0.1$, the scale is $s = 8$ and the effective sample count is:

```math
|A| + s^2 \cdot |B| = 0.2n + 64 \times 0.1n = 6.6n > n
```

GOSS can achieve **lower variance** than uniform subsampling of the same size because it always retains the most informative samples (largest $|g_i|$).

---

### 6. Split Gain Formula — Full Derivation

For a leaf $S$ with aggregate gradient $G_S = \sum_{i \in S} g_i$ and hessian $H_S = \sum_{i \in S} h_i$, the leaf's contribution to the objective is:

```math
\tilde{L}(S) = G_S \cdot w + \frac{1}{2}(H_S + \lambda)\, w^2
```

Differentiating with respect to the leaf weight $w$ and setting to zero:

```math
\frac{\partial \tilde{L}(S)}{\partial w} = G_S + (H_S + \lambda)\, w = 0
\quad\Longrightarrow\quad
w^* = -\frac{G_S}{H_S + \lambda}
```

Substituting $w^*$ back to get the **minimum loss value** (score) of the leaf:

```math
\tilde{L}^*(S)
= G_S \cdot \left(-\frac{G_S}{H_S + \lambda}\right) + \frac{1}{2}(H_S + \lambda) \cdot \frac{G_S^2}{(H_S + \lambda)^2}
= -\frac{G_S^2}{2(H_S + \lambda)}
```

Define the **leaf score**:

```math
\text{score}(G, H) = -\frac{G^2}{2(H + \lambda)}
```

The **gain** from splitting $S$ into left child $S_L$ and right child $S_R$ is the improvement in objective:

```math
\text{Gain} = \tilde{L}^*(S) - \tilde{L}^*(S_L) - \tilde{L}^*(S_R) - \gamma
```

```math
= \left(-\frac{G_S^2}{2(H_S + \lambda)}\right)
  - \left(-\frac{G_L^2}{2(H_L + \lambda)}\right)
  - \left(-\frac{G_R^2}{2(H_R + \lambda)}\right)
  - \gamma
```

```math
\boxed{
\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G_S^2}{H_S + \lambda}\right] - \gamma
}
```

A split is accepted only when $\text{Gain} > 0$. The $-\gamma$ term penalizes creating an additional leaf.

---

### 7. Optimal Leaf Weight — Full Derivation

#### 7.1 L2 Regularization Only ($\alpha = 0$)

The per-leaf objective with L2 regularization:

```math
\tilde{L}(S, w) = G_S \cdot w + \frac{1}{2}(H_S + \lambda)\, w^2
```

Setting the derivative to zero:

```math
\frac{d\tilde{L}}{dw} = G_S + (H_S + \lambda)\, w = 0
```

```math
\boxed{w^* = -\frac{G_S}{H_S + \lambda}}
```

**Intuition**: $\lambda$ smooths the denominator. As $\lambda \to \infty$, $w^* \to 0$ (full regularization). As $\lambda \to 0$, $w^* = -G_S / H_S$ (no regularization).

#### 7.2 L1 + L2 Regularization ($\alpha > 0$)

Adding the L1 term $\alpha |w|$, the subgradient optimality condition gives the **soft-thresholding** rule:

```math
w^* = -\frac{\text{sign}(G_S) \cdot \max(|G_S| - \alpha,\; 0)}{H_S + \lambda}
```

Equivalently, define the clipped gradient:

```math
G_S^{\text{reg}} = \begin{cases}
G_S - \alpha & \text{if } G_S > +\alpha \\
G_S + \alpha & \text{if } G_S < -\alpha \\
0            & \text{if } |G_S| \leq \alpha
\end{cases}
```

then:

```math
\boxed{w^* = -\frac{G_S^{\text{reg}}}{H_S + \lambda}}
```

**Effect**: leaf weights with $|G_S| \leq \alpha$ are exactly zero — producing **sparse trees** where uninformative leaves are pruned.

---

### 8. Regularization Terms

The full regularization on a tree $T$ with $K$ leaves and weights $w_1, \ldots, w_K$ is:

```math
\Omega(T) = \gamma K + \frac{\lambda}{2} \sum_{j=1}^{K} w_j^2 + \alpha \sum_{j=1}^{K} |w_j|
```

| Term | Role |
|------|------|
| $\gamma K$ | Penalizes number of leaves; controls tree complexity |
| $\frac{\lambda}{2}\sum_j w_j^2$ | L2 (Ridge): shrinks all leaf weights toward zero |
| $\alpha \sum_j |w_j|$ | L1 (Lasso): drives small leaf weights exactly to zero |

**Effect on the gain formula**: as $\lambda \to \infty$, all gains $\to 0$ and no splits occur. As $\lambda \to 0$, the tree overfits. The $\gamma$ term acts as a pre-pruning threshold.

---

### 9. Loss Functions and Gradients

#### 9.1 Regression — Squared Error

```math
L(y, F) = \frac{1}{2}(y - F)^2
```

```math
g_i = \frac{\partial L}{\partial F}\bigg|_{F = F_{m-1}(x_i)} = F_{m-1}(x_i) - y_i
\qquad \text{(residual)}
```

```math
h_i = \frac{\partial^2 L}{\partial F^2} = 1
\qquad \text{(constant hessian)}
```

Base score: $F_0 = \bar{y} = \dfrac{1}{n}\displaystyle\sum_{i=1}^n y_i$

#### 9.2 Binary Classification — Log Loss

```math
L(y, F) = -\bigl[y \log \sigma(F) + (1 - y) \log(1 - \sigma(F))\bigr]
```

where the sigmoid is $\sigma(F) = \dfrac{1}{1 + e^{-F}}$.

```math
g_i = \sigma(F_{m-1}(x_i)) - y_i
```

```math
h_i = \sigma(F_{m-1}(x_i))\,\bigl(1 - \sigma(F_{m-1}(x_i))\bigr)
```

Base score: $F_0 = \log\!\left(\dfrac{n_+}{n_-}\right)$ (log-odds of class prevalence).

Final probability: $\hat{p} = \sigma(F_M(x))$; class label $= \mathbf{1}[\hat{p} \geq 0.5]$.

#### 9.3 Summary Table

| Loss | $L(y, F)$ | $g_i$ | $h_i$ |
|------|-----------|--------|--------|
| Squared error | $\frac{1}{2}(y-F)^2$ | $F_i - y_i$ | $1$ |
| Log loss (binary) | $-[y\log\sigma(F) + (1-y)\log(1-\sigma(F))]$ | $\sigma(F_i) - y_i$ | $\sigma(F_i)(1-\sigma(F_i))$ |
| Poisson regression | $F - y\log F$ | $e^{F_i} - y_i$ | $e^{F_i}$ |

---

### 10. Feature Importance (Gain-Based)

For each internal node $v$ in tree $T_m$, let $j(v)$ be the feature used and $\text{Gain}(v)$ the split gain achieved. The **gain importance** of feature $j$ is:

```math
I_j = \sum_{m=1}^{M} \sum_{\substack{v \in T_m \\ j(v) = j}} \text{Gain}(v)
```

**Normalized** to sum to 1:

```math
\text{importance}(j) = \frac{I_j}{\sum_{k=1}^{p} I_k}
```

**Interpretation**: feature $j$ is important when splits on it produce the largest total reduction in the regularized objective, summed across all trees and all nodes.

---

### 11. Exclusive Feature Bundling (EFB)

EFB is LightGBM's fourth innovation (not implemented in this from-scratch version) for **sparse high-dimensional data** such as one-hot encoded categoricals.

#### 11.1 Exclusivity Condition

Features $A$ and $B$ are **exclusive** if they are never simultaneously non-zero:

```math
x_{i,A} \neq 0 \;\Rightarrow\; x_{i,B} = 0 \qquad \forall\, i
```

Such features carry redundant histogram bins and can be merged without loss of information.

#### 11.2 Bundling Procedure

1. Build a **conflict graph**: edge $(A, B)$ exists when the fraction of samples where both $A \neq 0$ and $B \neq 0$ exceeds a tolerance $\epsilon$.
2. Apply **greedy graph coloring** to partition features into bundles with few conflicts.
3. **Merge** each bundle by offsetting bin indices so features occupy non-overlapping ranges:

```math
\tilde{x}_{i,\,\text{bundle}} = \tilde{x}_{i,A}\cdot\mathbf{1}[x_{i,A} \neq 0] + (\tilde{x}_{i,B} + B_A)\cdot\mathbf{1}[x_{i,B} \neq 0] + \cdots
```

where $B_A$ is the bin offset allocated to feature $B$ within the bundle.

**Result**: reduces $p$ features to $p' \ll p$ bundles — all histogram builds scale as $O(n \cdot p')$.

---

### 12. Complexity Analysis

#### 12.1 Training Complexity

| Phase | Cost |
|-------|------|
| Histogram binning (once, before training) | $O(n \cdot p)$ |
| Gradient and hessian computation per round | $O(n)$ |
| Histogram build per leaf, per feature | $O(\|S\|)$ |
| Histogram scan per leaf, per feature | $O(B)$ |
| Total per boosting round | $O(n \cdot p)$ |
| **Total training** | $O(M \cdot n \cdot p)$ |

Compared to XGBoost exact: $O(M \cdot p \cdot n \log n)$ — LightGBM removes the $\log n$ factor.

#### 12.2 Prediction Complexity

For each test sample, traverse $M$ trees of depth at most $\log_2(\text{num\_leaves})$:

```math
O\!\left(M \cdot \log_2(\text{num\_leaves})\right) \text{ per sample}
```

#### 12.3 Memory

| Component | Size |
|-----------|------|
| Binned data $\tilde{X}$ | $O(n \cdot p)$ integers (1 byte each if $B \leq 255$, vs 8 bytes for float64) |
| Histograms (reused each round) | $O(B \cdot p)$ floats |
| All trees | $O(M \cdot \text{num\_leaves})$ nodes |

---

### 13. Full Training Algorithm

**Input**: $(X, y)$, parameters $M$, $\eta$, `num_leaves`, $\lambda$, $\alpha$, $\gamma$  
**Output**: ensemble of $M$ trees, base score, histogram bin edges

```
=== Preprocessing (once) ===
For each feature j = 1, ..., p:
    edges_j  <-  percentile-based bin edges from X[:,j]
    X_binned[:,j]  <-  searchsorted(edges_j, X[:,j]) - 1

=== Initialization ===
F_0  <-  mean(y)              [regression]
F_0  <-  log(n+ / n-)        [classification, log-odds]

=== Boosting loop ===
For m = 1 to M:

  (1) Compute per-sample gradients g_i and hessians h_i

  (2) Row sampling:
        [GOSS]      A = top (top_rate * n) by |g_i|
                    B = random_sample(rest, other_rate * n)
                    Scale g_i by (1 - top_rate) / other_rate  for i in B
        [Subsample] S = random_sample(n, subsample * n)
        [None]      S = {1, ..., n}

  (3) Column sampling:
        sel_feats = random_sample(p, colsample_bytree * p)

  (4) Leaf-wise tree growth:
        root  <-  single leaf covering all samples in S
        w_root  <-  -G_S / (H_S + lambda)
        pool = [root]

        While |leaves| < num_leaves:
            For each leaf in pool:
                Build histograms for sel_feats  [O(|leaf| * |sel_feats|)]
                Scan histograms for best split  [O(B * |sel_feats|)]

            leaf*  <-  argmax_{leaf in pool} best_gain(leaf)
            If best_gain(leaf*) <= gamma:  break

            Split leaf* -> (left, right):
                w_left   <-  -G_left_reg  / (H_left  + lambda)
                w_right  <-  -G_right_reg / (H_right + lambda)
            Add (left, right) to pool

  (5) Update predictions:
        F_m(x_i)  <-  F_{m-1}(x_i)  +  eta * T_m(x_i)

Return trees, F_0, {edges_j}
```

---

### 14. LightGBM vs XGBoost

| Property | XGBoost | LightGBM |
|----------|---------|----------|
| Split finding | Exact $O(n \log n)$ or approximate | Histogram $O(B)$, $B \ll n$ |
| Tree growth strategy | Level-wise (breadth-first) | Leaf-wise (best-first) |
| Data sampling | Uniform subsample | GOSS (gradient-guided) |
| Feature reduction | None | EFB for sparse features |
| Memory per sample | float64 (8 bytes) | uint8 (1 byte, after binning) |
| Training speed (large $n$) | Baseline | 10 – 20 times faster |
| Accuracy | Very high | Comparable or higher |
| Overfitting risk | Lower (level-wise is conservative) | Higher — mitigate with `min_data_in_leaf` |
| Best use case | Moderate size, many hyperparameter options | Large datasets, speed-critical applications |

---

## Installation

```bash
pip install numpy scikit-learn matplotlib
```

---

## Quick Start

```python
from lightgbm_scratch import LightGBMClassifier, LightGBMRegressor

# --- Binary Classification ---
clf = LightGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=20,
    reg_lambda=1.0,
    use_goss=True,
    random_state=42
)
clf.fit(X_train, y_train)
print(f"Accuracy:    {clf.score(X_test, y_test):.3f}")
print(f"Importances: {clf.feature_importances_}")

proba = clf.predict_proba(X_test)   # shape (n_test, 2)

# --- Regression ---
reg = LightGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=63,
    reg_lambda=1.0,
    random_state=42
)
reg.fit(X_train, y_train)
print(f"R2: {reg.score(X_test, y_test):.3f}")
```

---

## Hyperparameters Guide

### Core Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_estimators` | 100 | Number of boosting rounds $M$ |
| `learning_rate` | 0.1 | Shrinkage $\eta$ — smaller requires more trees |
| `num_leaves` | 31 | Maximum leaves per tree (key complexity control) |
| `max_depth` | -1 | Hard depth limit ($-1$ = unlimited) |
| `n_bins` | 255 | Histogram resolution $B$ |

### Regularization

| Parameter | Default | Effect |
|-----------|---------|--------|
| `reg_lambda` | 1.0 | L2 coefficient $\lambda$: shrinks leaf weights |
| `reg_alpha` | 0.0 | L1 coefficient $\alpha$: sparsifies leaf weights |
| `gamma` | 0.0 | Minimum gain $\gamma$ to accept a split |
| `min_data_in_leaf` | 20 | Minimum samples $\|S\|$ per leaf |
| `min_child_weight` | 1e-3 | Minimum hessian sum $H_S$ per leaf |

### Sampling

| Parameter | Default | Effect |
|-----------|---------|--------|
| `subsample` | 1.0 | Row subsampling fraction |
| `colsample_bytree` | 1.0 | Feature subsampling fraction |
| `use_goss` | False | Enable GOSS sampling |
| `goss_top_rate` | 0.2 | Fraction $a$ of large-gradient samples kept |
| `goss_other_rate` | 0.1 | Fraction $b$ of small-gradient samples sampled |

### Recommended Tuning

```
1. Start: n_estimators=1000, learning_rate=0.05, num_leaves=31
2. Tune num_leaves in [15, 31, 63, 127] — increase for complex data
3. Regularize: min_data_in_leaf=20..100, reg_lambda=1..10
4. Subsample large datasets: subsample=0.8, colsample_bytree=0.8
5. Very large datasets: use_goss=True
6. Final model: lower learning_rate + more n_estimators
```

---

## Visualizations

| Image | Description |
|-------|-------------|
| `01_histogram_binning.png` | How continuous values are discretized into $B$ bins |
| `02_leafwise_vs_levelwise.png` | Leaf-wise vs level-wise tree growth diagram |
| `03_goss_sampling.png` | GOSS: large-gradient samples kept, small ones subsampled |
| `04_split_gain.png` | Gain at each bin threshold; $O(B)$ vs $O(n \log n)$ comparison |
| `05_decision_boundary.png` | Decision boundary evolution over boosting rounds |
| `06_regression.png` | Nonlinear function approximation at different $M$ |
| `07_feature_importances.png` | Gain-based feature importances on Iris |
| `08_regularization.png` | Effect of $\lambda$ on regression fit |

---

## References

1. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.-Y. (2017).
   **LightGBM: A Highly Efficient Gradient Boosting Decision Tree.**
   *NeurIPS 2017*.

2. Chen, T., Guestrin, C. (2016). **XGBoost: A Scalable Tree Boosting System.** *KDD 2016*.

3. Friedman, J. H. (2001). **Greedy Function Approximation: A Gradient Boosting Machine.**
   *Annals of Statistics, 29(5)*, 1189–1232.

4. Hastie, T., Tibshirani, R., Friedman, J. (2009).
   **The Elements of Statistical Learning**, 2nd ed. Springer.

---

*Implementation: ML Algorithms From Scratch — Module 16*
