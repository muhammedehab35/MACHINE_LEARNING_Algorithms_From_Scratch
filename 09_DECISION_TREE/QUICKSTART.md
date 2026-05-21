# Decision Tree - Quick Start Guide

Get started with the Decision Tree classifier in 5 minutes!

## Installation

No installation needed! Just ensure you have NumPy:

```bash
pip install numpy matplotlib
```

## Basic Usage

### 1. Simple Classification

```python
from decision_tree import DecisionTreeClassifier
import numpy as np

# Create simple dataset
X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
y = np.array([0, 1, 1, 0])

# Train classifier
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

# Make predictions
predictions = clf.predict(X)
print(f"Predictions: {predictions}")

# Get accuracy
accuracy = clf.score(X, y)
print(f"Accuracy: {accuracy:.2%}")
```

### 2. Multi-class Classification

```python
# Three classes
X = np.random.randn(150, 2)
y = np.array([0]*50 + [1]*50 + [2]*50)

clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X, y)

print(f"Number of classes: {clf.n_classes_}")
print(f"Class labels: {clf.classes_}")
```

### 3. Get Probabilities

```python
# Predict class probabilities
proba = clf.predict_proba(X)

# Show probabilities for first 3 samples
for i in range(3):
    print(f"Sample {i}: {proba[i]}")
```

### 4. Feature Importance

```python
# Train model
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X, y)

# Get importance scores
importances = clf.feature_importances_

# Display
for i, imp in enumerate(importances):
    print(f"Feature {i}: {imp:.4f}")
```

### 5. Visualize Tree Structure

```python
# Print tree
clf.print_tree()

# Output:
# Root: X[0] <= 0.5 (samples=100, impurity=0.5)
#   |- True:  Predict class 0 (samples=50, impurity=0.0)
#   |_ False: Predict class 1 (samples=50, impurity=0.0)
```

### 6. Control Overfitting

```python
# Pre-pruning
clf = DecisionTreeClassifier(
    max_depth=5,              # Limit tree depth
    min_samples_split=10,     # Need 10+ samples to split
    min_samples_leaf=5        # Each leaf needs 5+ samples
)
clf.fit(X, y)

# Cost-complexity pruning
clf = DecisionTreeClassifier(
    ccp_alpha=0.01           # Higher = more pruning
)
clf.fit(X, y)
```

### 7. Compare Split Criteria

```python
# Gini (default, fast)
clf_gini = DecisionTreeClassifier(criterion='gini')
clf_gini.fit(X, y)

# Entropy (information gain)
clf_entropy = DecisionTreeClassifier(criterion='entropy')
clf_entropy.fit(X, y)

# Compare
print(f"Gini accuracy: {clf_gini.score(X, y):.4f}")
print(f"Entropy accuracy: {clf_entropy.score(X, y):.4f}")
```

### 8. Tree Statistics

```python
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X, y)

print(f"Tree depth: {clf.get_depth()}")
print(f"Number of leaves: {clf.get_n_leaves()}")
print(f"Number of features: {clf.n_features_}")
print(f"Number of classes: {clf.n_classes_}")
```

## Common Patterns

### Pattern 1: Train-Test Split

```python
# Split data
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)

# Evaluate
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)

print(f"Train: {train_acc:.4f}, Test: {test_acc:.4f}")
print(f"Overfitting gap: {train_acc - test_acc:.4f}")
```

### Pattern 2: Find Optimal Depth

```python
depths = range(1, 11)
test_scores = []

for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train, y_train)
    test_scores.append(clf.score(X_test, y_test))

optimal_depth = depths[np.argmax(test_scores)]
print(f"Optimal depth: {optimal_depth}")
```

### Pattern 3: Feature Selection

```python
# Train model
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X, y)

# Get importance
importances = clf.feature_importances_

# Select top features
threshold = 0.1
important_features = np.where(importances > threshold)[0]
print(f"Important features: {important_features}")

# Train with selected features
X_selected = X[:, important_features]
clf_reduced = DecisionTreeClassifier(max_depth=5)
clf_reduced.fit(X_selected, y)
```

## Running Examples

### Binary Classification Example
```bash
cd examples
python example_binary.py
```

### Multi-class Classification Example
```bash
python example_multiclass.py
```

### Compare Tree Depths
```bash
python example_depths.py
```

### Compare Split Criteria
```bash
python example_criteria.py
```

### Pruning Techniques
```bash
python example_pruning.py
```

### Feature Importance Analysis
```bash
python example_feature_importance.py
```

## Common Parameters

| Parameter | Default | Description | When to Use |
|-----------|---------|-------------|-------------|
| `max_depth` | None | Maximum tree depth | Start with 3-5 to prevent overfitting |
| `min_samples_split` | 2 | Min samples to split | Increase (10-50) for noisy data |
| `min_samples_leaf` | 1 | Min samples in leaf | Increase (5-20) for smoother boundaries |
| `criterion` | 'gini' | Split quality measure | 'gini' is faster, 'entropy' is more balanced |
| `max_features` | None | Features per split | Use 'sqrt' or 'log2' for high dimensions |
| `ccp_alpha` | 0.0 | Pruning strength | Try 0.001-0.1 for pruning |
| `random_state` | None | Random seed | Set for reproducibility |

## Tips for Better Models

1. **Start Simple**: Begin with `max_depth=3` and increase if needed
2. **Monitor Overfitting**: Compare train vs test accuracy
3. **Use Gini**: It's faster and usually works as well as entropy
4. **Prune Carefully**: Start with pre-pruning before trying cost-complexity
5. **Check Feature Importance**: Remove low-importance features
6. **Visualize**: Use `print_tree()` to understand decisions
7. **Cross-validate**: Test different parameters systematically

## Troubleshooting

### Problem: Model overfits training data
**Solution**:
- Decrease `max_depth` (try 3-5)
- Increase `min_samples_split` and `min_samples_leaf`
- Use `ccp_alpha > 0`

### Problem: Model too simple (underfits)
**Solution**:
- Increase `max_depth`
- Decrease `min_samples_split` and `min_samples_leaf`
- Check if data has predictive features

### Problem: Training is slow
**Solution**:
- Reduce `max_depth`
- Use `max_features='sqrt'` for high dimensions
- Use 'gini' instead of 'entropy'

### Problem: Unstable predictions
**Solution**:
- Use ensemble methods (Random Forest, Gradient Boosting)
- Increase minimum sample constraints
- Collect more training data

## What's Next?

1. Read the full [README.md](README.md) for mathematical details
2. Run the example scripts to see visualizations
3. Try the test suite: `python test_decision_tree.py`
4. Experiment with your own datasets
5. Learn about ensemble methods (Random Forests, Gradient Boosting)

## Getting Help

- Check the [README.md](README.md) for detailed documentation
- Run examples to see working code
- Look at test cases in `test_decision_tree.py` for usage patterns

---

Happy tree building!
