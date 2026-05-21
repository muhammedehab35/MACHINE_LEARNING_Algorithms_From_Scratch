# Machine Learning Algorithms from Scratch

A comprehensive collection of **26 Machine Learning algorithms** implemented from scratch in pure NumPy, with complete mathematical documentation, test suites, and visualisations.

## Project Overview

| Metric | Value |
|--------|-------|
| Total modules | 26 |
| Tests per module | 15 |
| Visualisations per module | 8 |
| Core dependency | NumPy only |

Each module follows the same structure:

```
NN_ALGORITHM/
├── README.md              # Full mathematical derivation (KaTeX LaTeX)
├── algorithm_scratch.py   # From-scratch implementation
├── test_algorithm.py      # 15 passing tests
├── generate_images.py     # 8 visualisations
├── __init__.py
└── images/                # Generated plots
```

---

## Algorithms

### Supervised Learning — Regression

| # | Module | Key Concepts |
|---|--------|-------------|
| 01 | [Linear Regression](01_LINEAR_REGRESSION/) | OLS, gradient descent, Ridge, learning rate decay |
| 02 | [Polynomial Regression](02_POLYNOMIAL_REGRESSION/) | Feature expansion, bias-variance tradeoff |
| 03 | [Elastic Net](03_ELASTIC_NET/) | L1 + L2 penalty, soft-thresholding proximal operator |
| 04 | [Gaussian Processes](04_GAUSSIAN_PROCESSES/) | RBF/Matern/polynomial kernels, posterior predictive, marginal log-likelihood |

### Supervised Learning — Classification

| # | Module | Key Concepts |
|---|--------|-------------|
| 05 | [Logistic Regression](05_LOGISTIC_REGRESSION/) | Sigmoid, cross-entropy, L1/L2 regularisation |
| 06 | [K-Nearest Neighbours](06_KNN/) | Euclidean/Manhattan/Minkowski distance, weighted voting |
| 07 | [Support Vector Machine](07_SVM/) | Max-margin hyperplane, SMO, RBF/poly/linear kernels |
| 08 | [Naive Bayes](08_NAIVE_BAYES/) | Gaussian/Bernoulli/Multinomial, Laplace smoothing |
| 09 | [Decision Tree](09_DECISION_TREE/) | Gini impurity, information gain, pruning, CART |
| 12 | [Linear Discriminant Analysis](12_LDA/) | Fisher criterion, within/between scatter, dimensionality reduction |
| 13 | [Quadratic Discriminant Analysis](13_QDA/) | Class-specific covariance, Mahalanobis distance |

### Ensemble Methods

| # | Module | Key Concepts |
|---|--------|-------------|
| 10 | [Random Forest](10_RANDOM_FOREST/) | Bootstrap aggregation, feature subsampling, OOB error |
| 11 | [Gradient Boosting](11_GRADIENT_BOOSTING/) | Pseudo-residuals, shrinkage, tree-based learners |
| 14 | [XGBoost](14_XGBOOST/) | 2nd-order Taylor expansion, L1/L2 leaf regularisation, exact greedy split |
| 15 | [AdaBoost](15_ADABOOST/) | Exponential loss, adaptive sample weights, weak learners |
| 16 | [LightGBM](16_LIGHTGBM/) | Histogram-based splits, GOSS, EFB, leaf-wise tree growth |
| 17 | [CatBoost](17_CATBOOST/) | Ordered boosting, target encoding, symmetric trees |
| 18 | [Bagging](18_BAGGING/) | Bootstrap sampling, variance reduction, base estimator agnostic |

### Unsupervised Learning — Clustering

| # | Module | Key Concepts |
|---|--------|-------------|
| 19 | [K-Means](19_KMEANS/) | Lloyd's algorithm, k-means++, inertia, elbow method |
| 20 | [K-Medoids](20_KMEDOIDS/) | PAM, swap phase, arbitrary distance metrics |
| 21 | [DBSCAN](21_DBSCAN/) | Core/border/noise points, eps-neighbourhood, any-shape clusters |
| 22 | [Hierarchical Clustering](22_HIERARCHICAL_CLUSTERING/) | Lance-Williams recurrence, 6 linkage methods, cophenetic correlation |
| 23 | [Gaussian Mixture Models](23_GMM/) | EM algorithm, E/M steps, 4 covariance types, BIC/AIC |

### Dimensionality Reduction

| # | Module | Key Concepts |
|---|--------|-------------|
| 24 | [PCA](24_PCA/) | Economy SVD, Eckart-Young theorem, whitening, PPCA noise variance |
| 25 | [t-SNE](25_TSNE/) | Gaussian bandwidth binary search, Student-t kernel, KL divergence, early exaggeration |
| 26 | [UMAP](26_UMAP/) | Fuzzy simplicial sets, binary search for sigma, fuzzy union, cross-entropy SGD |

---

## Quick Start

```bash
git clone https://github.com/muhammedehab35/ML_algorithms_from_scratch.git
cd ML_algorithms_from_scratch
pip install numpy scipy matplotlib scikit-learn
```

### Run any module

```python
# Dimensionality reduction example
import numpy as np
import sys
sys.path.insert(0, '24_PCA')
sys.path.insert(0, '25_TSNE')
sys.path.insert(0, '26_UMAP')

from pca_scratch import PCA
from tsne_scratch import TSNE
from umap_scratch import UMAP

X = np.random.randn(200, 50)   # 200 points in 50D

Z_pca  = PCA(n_components=2).fit_transform(X)
Z_tsne = TSNE(n_components=2, perplexity=30, n_iter=500).fit_transform(X)
Z_umap = UMAP(n_components=2, n_neighbors=15).fit_transform(X)
```

```python
# Clustering example
import sys
sys.path.insert(0, '19_KMEANS')
sys.path.insert(0, '23_GMM')

from kmeans_scratch import KMeans
from gmm_scratch import GaussianMixture

km  = KMeans(n_clusters=3, init='kmeans++').fit(X)
gmm = GaussianMixture(n_components=3, covariance_type='full').fit(X)
labels_km  = km.labels_
labels_gmm = gmm.predict(X)
```

### Run tests

```bash
cd 26_UMAP && python test_umap.py
cd 25_TSNE && python test_tsne.py
cd 24_PCA  && python test_pca.py
```

### Generate visualisations

```bash
cd 26_UMAP && python generate_images.py
```

---

## Mathematical Highlights

### Dimensionality Reduction Progression

| Method | Cost function | Optimisation | High-dim kernel | Low-dim kernel |
|--------|--------------|-------------|----------------|----------------|
| PCA | $\min \|X - ZW\|_F^2$ | SVD | Linear | Linear |
| t-SNE | $\text{KL}(P \| Q)$ | Gradient descent + momentum | Gaussian | Student-t |
| UMAP | $\mathcal{C}(W \| Q)$ cross-entropy | Mini-batch SGD | Fuzzy Gaussian | $(1+ad^{2b})^{-1}$ |

### Ensemble Gradient Derivations

| Method | Loss | Weak learner update |
|--------|------|---------------------|
| AdaBoost | Exponential $e^{-y\hat{y}}$ | Sample reweighting |
| Gradient Boosting | Any differentiable $L$ | Fit pseudo-residuals $-\partial L/\partial \hat{y}$ |
| XGBoost | $L$ + Taylor 2nd order | Closed-form leaf weights $w^* = -G/(H+\lambda)$ |

### EM for GMM

**E-step:** $r_{ik} = \frac{\pi_k\,\mathcal{N}(x_i|\mu_k,\Sigma_k)}{\sum_j \pi_j\,\mathcal{N}(x_i|\mu_j,\Sigma_j)}$

**M-step:** $\mu_k = \frac{\sum_i r_{ik} x_i}{N_k}$, $\quad\Sigma_k = \frac{\sum_i r_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top}{N_k}$, $\quad\pi_k = \frac{N_k}{n}$

---

## Requirements

```
numpy >= 1.22
scipy >= 1.7          # UMAP curve_fit, SVM QP
matplotlib >= 3.4     # visualisations
scikit-learn >= 0.24  # datasets and validation in tests
```

---

## Repository

[github.com/muhammedehab35/ML_algorithms_from_scratch](https://github.com/muhammedehab35/ML_algorithms_from_scratch)

**26 modules — 390 tests — 208 visualisations — pure NumPy**
