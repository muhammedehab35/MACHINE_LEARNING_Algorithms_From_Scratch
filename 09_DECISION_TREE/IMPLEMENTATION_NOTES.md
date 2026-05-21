# Decision Tree Implementation Notes

## Overview

This is a complete, production-quality implementation of a Decision Tree classifier from scratch using only NumPy. The implementation follows the CART (Classification and Regression Trees) algorithm and provides an API compatible with scikit-learn's `DecisionTreeClassifier`.

## Implementation Highlights

### 1. Core Algorithm (CART)

The implementation uses recursive binary splitting:

1. **Node Creation**: Each node stores:
   - Split feature and threshold (internal nodes)
   - Prediction and class distribution (leaf nodes)
   - Impurity, sample count, and depth

2. **Best Split Selection**: At each node:
   - Consider all features (or subset if `max_features` specified)
   - Try all possible thresholds (midpoints between unique values)
   - Calculate information gain for each split
   - Select split with maximum gain

3. **Recursive Building**:
   - Create root node with all data
   - Find best split
   - Recursively build left and right subtrees
   - Stop when criteria met

### 2. Split Criteria

#### Gini Impurity (Default)
```python
def _gini_impurity(self, y):
    _, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)
    return 1.0 - np.sum(proportions ** 2)
```

**Why it's good:**
- Computationally efficient (no logarithms)
- Ranges from 0 (pure) to 0.5 (balanced, binary case)
- Default in sklearn and most implementations

#### Entropy (Information Gain)
```python
def _entropy(self, y):
    _, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)
    proportions = proportions[proportions > 0]
    return -np.sum(proportions * np.log2(proportions))
```

**Why it's good:**
- Theoretically grounded in information theory
- May produce more balanced trees
- Better when interpretability matters

#### Misclassification Error
```python
def _misclassification_error(self, y):
    _, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)
    return 1.0 - np.max(proportions)
```

**Why it's rarely used:**
- Less sensitive to probability changes
- Doesn't penalize impure splits as much
- Gini and Entropy are superior in practice

### 3. Stopping Criteria

The tree stops growing when:

1. **Maximum depth reached**: `depth == max_depth`
2. **Too few samples**: `n_samples < min_samples_split`
3. **Pure node**: All samples same class
4. **Zero impurity**: `impurity == 0`
5. **Child too small**: Would violate `min_samples_leaf`
6. **No valid split**: No split improves impurity

This prevents overfitting and reduces computational cost.

### 4. Pruning Mechanisms

#### Pre-pruning (Implemented)
Applied during tree construction:
- `max_depth`: Hard limit on tree depth
- `min_samples_split`: Minimum samples to attempt split
- `min_samples_leaf`: Minimum samples required in leaf

**Advantages:**
- Fast (stops early)
- Easy to understand
- Effective for most cases

#### Cost-Complexity Pruning (Simplified Implementation)
Applied after tree is built:
- Uses `ccp_alpha` parameter
- Removes nodes where impurity decrease < alpha
- More sophisticated than pre-pruning

**Advantages:**
- Can find optimal tree size
- Works with cross-validation
- Better for fine-tuning

### 5. Feature Importance

Calculated as weighted impurity decrease:

```python
Importance(f) = Σ [n_node/n_total * impurity_decrease]
```

For each node using feature `f`:
- Weight by fraction of samples at node
- Sum impurity decrease from split
- Normalize to sum to 1.0

**Properties:**
- Features used higher in tree have more importance
- Features used in more nodes accumulate importance
- Normalized scores sum to 1.0

### 6. Probability Prediction

For `predict_proba()`:
- Traverse to leaf node
- Return class distribution in leaf
- Normalized by number of samples in leaf

Example:
```
Leaf with [30 class 0, 20 class 1] samples
→ P(class 0) = 30/50 = 0.6
→ P(class 1) = 20/50 = 0.4
```

### 7. Tree Traversal

Prediction for sample **x**:

```python
def _predict_sample(self, x, node):
    if node.is_leaf():
        return node.value

    if x[node.feature_index] <= node.threshold:
        return _predict_sample(x, node.left)
    else:
        return _predict_sample(x, node.right)
```

Time complexity: O(log n) for balanced trees, O(n) worst case

## Design Decisions

### 1. Why NumPy Only?

**Pros:**
- Educational value (understand internals)
- No hidden optimizations
- Transparent implementation
- Minimal dependencies

**Cons:**
- Slower than sklearn (uses Cython)
- No advanced features (sample weights, etc.)
- Limited to basic CART

### 2. Why Recursive Implementation?

**Pros:**
- Clean, readable code
- Natural representation of tree structure
- Easy to understand algorithm

**Cons:**
- Stack overflow for very deep trees (Python limit ~1000)
- More memory usage than iterative
- Harder to parallelize

**Alternative:** Could implement iteratively with stack, but recursion is clearer.

### 3. Class Structure

**Node Class:**
- Separate class for nodes
- Stores all node information
- Clean separation of concerns

**DecisionTreeClassifier Class:**
- Main interface
- Manages training and prediction
- Holds tree and metadata

**Benefits:**
- Modular design
- Easy to extend
- Matches sklearn API

### 4. Threshold Selection

Uses midpoints between consecutive unique values:

```python
threshold = (unique_values[i] + unique_values[i+1]) / 2
```

**Why:**
- Mathematically equivalent to all values in range
- Reduces number of splits to try
- Standard in most implementations

**Alternative:** Could try all unique values, but midpoints are sufficient.

### 5. Feature Importance Tracking

Accumulated during tree building:

```python
self._impurity_importance[feature] += weight * decrease
```

**Why:**
- Efficient (single pass)
- Accurate (uses actual tree)
- Normalized automatically

**Alternative:** Could traverse tree after building, but this is cleaner.

## Comparison with sklearn

### What's Implemented

✅ Binary and multi-class classification
✅ Gini and Entropy criteria
✅ Pre-pruning (max_depth, min_samples_split, min_samples_leaf)
✅ Cost-complexity pruning (simplified)
✅ Feature importance
✅ predict_proba()
✅ max_features parameter
✅ Random state for reproducibility

### What's Missing

❌ Multi-output classification
❌ Sample weights
❌ Class weights
❌ min_impurity_decrease
❌ max_leaf_nodes
❌ Regression (DecisionTreeRegressor)
❌ Cython optimizations
❌ Advanced pruning algorithms

### Performance Comparison

**Speed:**
- sklearn: ~10-100x faster (Cython)
- This implementation: Pure Python/NumPy

**Memory:**
- sklearn: More optimized storage
- This implementation: More readable structure

**Accuracy:**
- Both produce similar results
- Same CART algorithm

## Testing Strategy

The test suite covers:

1. **Basic functionality**: Binary and multi-class
2. **Different criteria**: Gini, Entropy, Error
3. **Pruning**: Pre-pruning and cost-complexity
4. **Feature importance**: Calculation and normalization
5. **Probability prediction**: Shape and properties
6. **Tree structure**: Depth and leaf count
7. **Edge cases**: Minimal data, pure nodes, high dimensions

All tests pass successfully!

## Common Issues and Solutions

### Issue 1: Overfitting

**Symptoms:**
- Perfect training accuracy
- Poor test accuracy
- Large train-test gap

**Solutions:**
- Reduce `max_depth` (try 3-5)
- Increase `min_samples_split` (try 10-50)
- Increase `min_samples_leaf` (try 5-20)
- Use `ccp_alpha > 0`

### Issue 2: Underfitting

**Symptoms:**
- Poor training accuracy
- Test similar to train
- Very shallow tree

**Solutions:**
- Increase `max_depth`
- Decrease minimum sample constraints
- Check data quality

### Issue 3: Training Slow

**Symptoms:**
- Takes long time to fit
- Many features or samples

**Solutions:**
- Reduce `max_depth`
- Use `max_features='sqrt'`
- Use 'gini' instead of 'entropy'
- Reduce number of features

### Issue 4: Unstable Predictions

**Symptoms:**
- Different trees on similar data
- High variance

**Solutions:**
- Increase sample size
- Use ensemble methods
- Increase minimum sample constraints

## Extensions and Future Work

### Possible Extensions

1. **Regression Trees**
   - Use MSE or MAE instead of impurity
   - Predict continuous values
   - Similar structure to classifier

2. **Random Forest**
   - Ensemble of decision trees
   - Bootstrap sampling
   - Random feature subsets

3. **Gradient Boosting**
   - Sequential tree building
   - Fit residuals
   - More powerful but complex

4. **Multi-output**
   - Predict multiple targets
   - Useful for structured prediction

5. **Missing Value Handling**
   - Surrogate splits
   - Common in real data

6. **Categorical Features**
   - Direct handling without encoding
   - More efficient splits

### Performance Optimizations

1. **Cython Implementation**
   - 10-100x speedup
   - Static typing
   - More complex code

2. **Parallel Tree Building**
   - Parallelize feature search
   - Use multiprocessing
   - Better for large datasets

3. **Memory Optimization**
   - Store tree more efficiently
   - Use arrays instead of objects
   - Trade readability for speed

## Learning Resources

### Original Papers

1. **Breiman et al. (1984)**: "Classification and Regression Trees"
   - Original CART algorithm
   - Foundation for modern implementations

2. **Quinlan (1986)**: "Induction of Decision Trees"
   - ID3 algorithm (entropy-based)
   - Information gain concept

3. **Quinlan (1993)**: "C4.5: Programs for Machine Learning"
   - Improved ID3
   - Pruning and continuous features

### Modern References

1. **sklearn Documentation**: Excellent overview and examples
2. **Elements of Statistical Learning**: Chapter 9 on trees
3. **Introduction to Statistical Learning**: More accessible tree introduction
4. **Hands-On Machine Learning**: Practical sklearn usage

## Conclusion

This implementation provides:

✅ **Educational Value**: Learn how trees work internally
✅ **Production Quality**: Well-tested, documented code
✅ **sklearn Compatible**: Familiar API and behavior
✅ **Extensible**: Easy to modify and extend

**Use for:**
- Learning algorithms
- Teaching machine learning
- Understanding tree internals
- Prototyping tree-based methods

**Use sklearn for:**
- Production systems
- Large-scale applications
- Performance-critical tasks
- Advanced features

---

**Author**: ML Algorithms from Scratch
**Date**: 2026
**Version**: 1.0.0
