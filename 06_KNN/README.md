# K-Nearest Neighbors (KNN) - Complete Implementation

A comprehensive implementation of **K-Nearest Neighbors** for both classification and regression from scratch using only NumPy.

## Table of Contents

- [Overview](#overview)
- [Mathematical Foundation](#mathematical-foundation)
- [Features](#features)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Visualizations](#visualizations)
- [Performance](#performance)

---

## Overview

**K-Nearest Neighbors (KNN)** is a simple yet powerful non-parametric, instance-based learning algorithm used for both classification and regression tasks.

### Key Characteristics

- **Model Type**: Instance-based, Lazy Learning
- **Algorithm**: Non-parametric
- **Decision**: Based on k nearest training examples
- **No Training Phase**: Stores all training data
- **Prediction**: Computed at query time

---

## Mathematical Foundation

### 1. Distance Metrics

KNN relies on distance metrics to find nearest neighbors.

#### Euclidean Distance (L2)


```math
d(x, x') = \sqrt{\sum_{i=1}^{n} (x_i - x'_i)^2}
```



Most common metric. Sensitive to feature scaling.

#### Manhattan Distance (L1)


```math
d(x, x') = \sum_{i=1}^{n} |x_i - x'_i|
```



Less sensitive to outliers than Euclidean.

#### Minkowski Distance


```math
d(x, x') = \left(\sum_{i=1}^{n} |x_i - x'_i|^p\right)^{1/p}
```



Where:
- $p = 1$: Manhattan distance
- $p = 2$: Euclidean distance
- $p = \infty$: Chebyshev distance

#### Cosine Distance


```math
d(x, x') = 1 - \frac{x \cdot x'}{\|x\| \|x'\|}
```



Cosine similarity:


```math
\text{similarity}(x, x') = \frac{x \cdot x'}{\|x\| \|x'\|}
```



Useful for high-dimensional sparse data.

### 2. K-Nearest Neighbors Selection

For a query point $x_q$:

1. Compute distances to all training points:
   
```math
D = \{d(x_q, x_i) : x_i \in \text{Training Set}\}
```


2. Select k smallest distances:


```math
\mathcal{N}_k(x_q) = \{\text{k points with smallest } d(x_q, x_i)\}
```

### 3. Classification

#### Uniform Voting

Majority vote among k neighbors:


```math
\hat{y} = \arg\max_{c} \sum_{x_i \in \mathcal{N}_k(x_q)} \mathbb{1}(y_i = c)
```



Where $\mathbb{1}$ is the indicator function.

#### Distance-Weighted Voting

Weight votes by inverse distance:


```math
\hat{y} = \arg\max_{c} \sum_{x_i \in \mathcal{N}_k(x_q)} w_i \cdot \mathbb{1}(y_i = c)
```



Where:


```math
w_i = \frac{1}{d(x_q, x_i) + \epsilon}
```



$\epsilon$ prevents division by zero.

#### Probability Estimates

For class $c$:

**Uniform weights**:


```math
P(y = c | x_q) = \frac{1}{k} \sum_{x_i \in \mathcal{N}_k(x_q)} \mathbb{1}(y_i = c)
```



**Distance weights**:


```math
P(y = c | x_q) = \frac{\sum_{x_i \in \mathcal{N}_k(x_q)} w_i \cdot \mathbb{1}(y_i = c)}{\sum_{x_i \in \mathcal{N}_k(x_q)} w_i}
```


### 4. Regression

#### Uniform Weights

Simple average of k neighbors:


```math
\hat{y} = \frac{1}{k} \sum_{x_i \in \mathcal{N}_k(x_q)} y_i
```


#### Distance-Weighted Average


```math
\hat{y} = \frac{\sum_{x_i \in \mathcal{N}_k(x_q)} w_i \cdot y_i}{\sum_{x_i \in \mathcal{N}_k(x_q)} w_i}
```



Where $w_i = \frac{1}{d(x_q, x_i) + \epsilon}$.

### 5. Choosing K

The value of k is crucial:

#### Small k (k=1)

- **Pros**: Flexible decision boundaries, low bias
- **Cons**: High variance, sensitive to noise, overfitting

#### Large k

- **Pros**: Smooth decision boundaries, low variance, more robust
- **Cons**: High bias, underfitting, expensive computation

#### Rule of Thumb


```math
k = \sqrt{n}
```



Where $n$ is the number of training samples.

#### Cross-Validation

Use k-fold cross-validation to find optimal k:

1. For each candidate $k \in \{1, 3, 5, ..., k_{max}\}$
2. Compute average validation accuracy
3. Select k with highest validation score

### 6. Bias-Variance Tradeoff

**Variance** (as function of k):


```math
\text{Var}[\hat{y}] \propto \frac{1}{k}
```



**Bias** (as function of k):


```math
\text{Bias}[\hat{y}] \propto k
```



**Total Error**:


```math
\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
```



Optimal k minimizes total error.

### 7. Decision Boundary

For 2D classification with k=1:

The decision boundary is the **Voronoi diagram** of training points.

For k>1:

More complex, piecewise linear boundaries.

### 8. Curse of Dimensionality

In high dimensions:


```math
d(x, x') \approx \text{constant}
```



All points become equidistant, making nearest neighbors meaningless.

**Volume of hypersphere**:


```math
V_d(r) = \frac{\pi^{d/2}}{\Gamma(d/2 + 1)} r^d
```



As $d \to \infty$, most volume concentrates near the surface.

**Mitigation**:
- Feature selection
- Dimensionality reduction (PCA, t-SNE)
- Use appropriate distance metrics (cosine for sparse data)

---

## Features

### Core Capabilities

- ✅ **Binary & Multi-class Classification**
- ✅ **Regression**
- ✅ **4 Distance Metrics**: Euclidean, Manhattan, Minkowski, Cosine
- ✅ **2 Weighting Schemes**: Uniform and Distance-based
- ✅ **Probability Predictions** for uncertainty estimation
- ✅ **Brute Force Search** (future: Ball Tree, KD-Tree)

### Metrics (9 Classification Metrics)

1. **Confusion Matrix**
2. **Accuracy**
3. **Precision**
4. **Recall**
5. **F1 Score**
6. **Specificity**
7. **ROC AUC**
8. **Log Loss**
9. **Matthews Correlation Coefficient**

### Utilities

- `StandardScaler` - Z-score normalization
- `MinMaxScaler` - [0, 1] scaling
- `train_test_split` - Data splitting
- `polynomial_features` - Feature engineering

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ML_ALGORITHMS_FROM_SCRATCH.git
cd ML_ALGORITHMS_FROM_SCRATCH/05_KNN

# No installation needed - pure NumPy implementation
```

**Requirements**:
- Python 3.7+
- NumPy
- Matplotlib (for visualizations)

---

## Usage Examples

### Example 1: Basic Classification

```python
import numpy as np
from knn import KNNClassifier
from utils import train_test_split, StandardScaler

# Generate data
np.random.seed(42)
X = np.random.randn(200, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (important for KNN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
knn = KNNClassifier(n_neighbors=5, weights='uniform')
knn.fit(X_train_scaled, y_train)

# Predict
predictions = knn.predict(X_test_scaled)
accuracy = knn.score(X_test_scaled, y_test)

print(f"Accuracy: {accuracy:.4f}")
```

### Example 2: Distance-Weighted Voting

```python
# Better for imbalanced or noisy data
knn = KNNClassifier(n_neighbors=10, weights='distance')
knn.fit(X_train_scaled, y_train)

# Get probability predictions
proba = knn.predict_proba(X_test_scaled)
print(f"Probabilities shape: {proba.shape}")  # (n_samples, n_classes)
```

### Example 3: Different Distance Metrics

```python
# Try different metrics
for metric in ['euclidean', 'manhattan', 'cosine']:
    knn = KNNClassifier(n_neighbors=5, metric=metric)
    knn.fit(X_train_scaled, y_train)
    acc = knn.score(X_test_scaled, y_test)
    print(f"{metric:12} - Accuracy: {acc:.4f}")
```

### Example 4: Regression

```python
from knn import KNNRegressor

# Generate regression data
X = np.sort(np.random.rand(100, 1) * 10, axis=0)
y = np.sin(X).ravel() + 0.1 * np.random.randn(100)

# Train regressor
knn_reg = KNNRegressor(n_neighbors=5, weights='distance')
knn_reg.fit(X, y)

# Predict
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = knn_reg.predict(X_test)

# Evaluate
r2 = knn_reg.score(X, y)
print(f"R² Score: {r2:.4f}")
```

### Example 5: Finding Optimal K

```python
# Cross-validation to find best k
from utils import train_test_split

k_range = range(1, 31)
scores = []

for k in k_range:
    knn = KNNClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    scores.append(knn.score(X_test_scaled, y_test))

best_k = k_range[np.argmax(scores)]
print(f"Best k: {best_k}")
```

---

## API Reference

### KNNClassifier

```python
class KNNClassifier:
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: Literal['uniform', 'distance'] = 'uniform',
        metric: Literal['euclidean', 'manhattan', 'minkowski', 'cosine'] = 'euclidean',
        p: int = 2,
        algorithm: Literal['brute'] = 'brute'
    )
```

**Parameters**:
- `n_neighbors` (int): Number of neighbors (default: 5)
- `weights` (str): Weighting scheme - 'uniform' or 'distance' (default: 'uniform')
- `metric` (str): Distance metric (default: 'euclidean')
- `p` (int): Power for Minkowski metric (default: 2)
- `algorithm` (str): Algorithm for neighbor search (default: 'brute')

**Methods**:
- `fit(X, y)` - Store training data
- `predict(X)` - Predict class labels
- `predict_proba(X)` - Predict class probabilities
- `score(X, y)` - Return accuracy

**Attributes**:
- `X_train_` - Stored training features
- `y_train_` - Stored training labels
- `classes_` - Unique class labels
- `n_classes_` - Number of classes

### KNNRegressor

```python
class KNNRegressor:
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: Literal['uniform', 'distance'] = 'uniform',
        metric: Literal['euclidean', 'manhattan', 'minkowski', 'cosine'] = 'euclidean',
        p: int = 2,
        algorithm: Literal['brute'] = 'brute'
    )
```

**Methods**:
- `fit(X, y)` - Store training data
- `predict(X)` - Predict target values
- `score(X, y)` - Return R² score

---

## Visualizations

The implementation includes 6 comprehensive visualizations:

### 1. Decision Boundaries
- Shows how boundaries change with k
- Compares k=1, 5, 15, 50

### 2. Distance Metrics
- Euclidean, Manhattan, Minkowski, Cosine
- Circular data to show differences

### 3. Weighted Voting
- Uniform vs Distance-weighted
- Training vs Test performance

### 4. K Optimization
- Accuracy vs k curve
- Bias-variance tradeoff visualization

### 5. KNN Regression
- Nonlinear function approximation
- Different k values comparison

### 6. Multi-class Classification
- 3-class problem
- Confusion matrix
- Probability predictions

**Generate visualizations**:

```bash
python examples.py
```

---

## Performance

### Computational Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Training | $O(1)$ | $O(n \cdot d)$ |
| Prediction (1 sample) | $O(n \cdot d)$ | $O(1)$ |
| Prediction (m samples) | $O(m \cdot n \cdot d)$ | $O(m)$ |
| Distance Computation | $O(d)$ | $O(1)$ |

Where:
- $n$ = number of training samples
- $d$ = number of features
- $m$ = number of test samples

### Optimization Strategies

1. **Feature Scaling**: Always scale features!
   ```python
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

2. **Distance Metric**: Choose appropriate metric
   - Euclidean: General purpose
   - Manhattan: Robust to outliers
   - Cosine: High-dimensional sparse data

3. **K Selection**: Use cross-validation

4. **Dimensionality Reduction**: For high-dimensional data
   ```python
   from sklearn.decomposition import PCA
   pca = PCA(n_components=10)
   X_reduced = pca.fit_transform(X)
   ```

5. **Distance-Weighted Voting**: Better for varying densities

### Comparison with Scikit-Learn

```python
from sklearn.neighbors import KNeighborsClassifier

# Our implementation
knn_ours = KNNClassifier(n_neighbors=5)
knn_ours.fit(X_train, y_train)
acc_ours = knn_ours.score(X_test, y_test)

# Scikit-learn
knn_sklearn = KNeighborsClassifier(n_neighbors=5)
knn_sklearn.fit(X_train, y_train)
acc_sklearn = knn_sklearn.score(X_test, y_test)

print(f"Our implementation: {acc_ours:.4f}")
print(f"Scikit-learn: {acc_sklearn:.4f}")
# Typically identical results
```

---

## Advantages and Disadvantages

### Advantages

✅ **Simple to understand and implement**
✅ **No training phase** (instance-based)
✅ **Naturally handles multi-class**
✅ **Non-parametric** (no assumptions about data distribution)
✅ **Versatile** (classification and regression)
✅ **Interpretable** decisions

### Disadvantages

❌ **Slow prediction** ($O(nd)$ per query)
❌ **High memory requirements** (stores all training data)
❌ **Sensitive to irrelevant features**
❌ **Curse of dimensionality**
❌ **Requires feature scaling**
❌ **No model to interpret** (just stored data)

---

## When to Use KNN

### Good For:

- Small to medium datasets
- Low-dimensional feature spaces
- Multi-class problems
- Baseline models
- Non-linear decision boundaries
- Online learning (streaming data)

### Not Good For:

- Large datasets (millions of samples)
- High-dimensional data (hundreds of features)
- Real-time predictions
- Interpretability requirements
- Memory-constrained environments

---

## Testing

Run the comprehensive test suite:

```bash
python test_knn.py
```

**Test Coverage**:
1. Basic classification
2. Distance metrics
3. Weighted voting
4. Probability predictions
5. Different k values
6. Regression
7. Classification metrics
8. Utility functions
9. Edge cases
10. Multi-class classification

All tests use reproducible random seeds.

---

## Common Issues and Solutions

### Issue 1: Poor Performance

**Symptoms**: Low accuracy on test data

**Solutions**:
- Scale features using `StandardScaler`
- Try different k values (cross-validation)
- Try different distance metrics
- Use distance-weighted voting
- Remove irrelevant features

### Issue 2: Slow Predictions

**Symptoms**: Long prediction times

**Solutions**:
- Reduce training set size
- Use dimensionality reduction
- Consider approximate nearest neighbors (future)
- Switch to parametric models for large datasets

### Issue 3: Memory Issues

**Symptoms**: Out of memory errors

**Solutions**:
- Reduce training set size
- Use lower precision (float32 instead of float64)
- Consider online learning variants

### Issue 4: Curse of Dimensionality

**Symptoms**: All points equally distant in high dimensions

**Solutions**:
- Feature selection
- PCA or t-SNE
- Use cosine distance for sparse data
- Consider alternative algorithms (Random Forest, Neural Networks)

---

## References

1. Cover, T., & Hart, P. (1967). *Nearest neighbor pattern classification*. IEEE Transactions on Information Theory.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
4. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.

---

## License

This implementation is part of the ML Algorithms from Scratch project.

## Author

ML Algorithms from Scratch

---

## Changelog

### Version 1.0.0 (2025)
- Initial implementation
- KNNClassifier and KNNRegressor
- 4 distance metrics
- Uniform and distance-weighted voting
- Probability predictions
- Comprehensive test suite
- 6 visualizations
- Complete documentation
