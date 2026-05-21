# Support Vector Machine (SVM)

A comprehensive implementation of Support Vector Machines for binary classification from scratch using only NumPy.

## Overview

Support Vector Machines (SVM) are powerful supervised learning algorithms used for classification and regression. This implementation focuses on binary classification and demonstrates the core concepts of margin maximization, kernel methods, and the soft margin approach.

## Key Concepts

### 1. Hyperplane and Margin

SVM finds the optimal hyperplane that separates two classes while maximizing the margin (distance between the hyperplane and the closest points from each class).

**Decision function**: `f(x) = w^T x + b`

- Positive class: `f(x) > 0`
- Negative class: `f(x) < 0`
- Decision boundary: `f(x) = 0`

### 2. Support Vectors

Support vectors are the data points closest to the decision boundary. They are the most difficult to classify and define the margin. Removing non-support vectors doesn't change the decision boundary.

### 3. Margin Maximization

The margin is the distance between the hyperplane and the nearest data point from either class. SVM aims to maximize this margin:

**Margin width**: `2 / ||w||`

To maximize the margin, we minimize `||w||^2`.

### 4. Soft Margin (C Parameter)

Real-world data is often not linearly separable. The soft margin approach allows some misclassifications controlled by parameter C:

- **Large C**: Hard margin, fewer violations, risk of overfitting
- **Small C**: Soft margin, more violations, better generalization

**Objective function**:

```
minimize: (1/2)||w||^2 + C * sum(max(0, 1 - y_i(w^T x_i + b)))
```

### 5. Kernel Trick

For non-linearly separable data, kernels implicitly map data to higher dimensions where it becomes linearly separable.

**Supported kernels**:
