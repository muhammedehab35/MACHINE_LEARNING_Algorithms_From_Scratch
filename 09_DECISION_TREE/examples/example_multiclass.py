"""
Example: Multi-class Classification with Decision Tree
======================================================

Demonstrates Decision Tree classifier on a multi-class classification problem
similar to the Iris dataset (3 classes).

Shows:
- Multi-class classification (3 classes)
- Decision boundaries in 2D
- Confusion matrix
- Class probabilities
- Feature importance

This example demonstrates that Decision Trees naturally handle
multi-class problems without modifications.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from decision_tree import DecisionTreeClassifier


def make_multiclass_dataset(n_samples=300, n_classes=3, random_state=42):
    """
    Create a synthetic multi-class classification dataset.

    Creates multiple clusters representing different classes.

    Parameters:
    -----------
    n_samples : int
        Total number of samples
    n_classes : int
        Number of classes
    random_state : int
        Random seed

    Returns:
    --------
    X : ndarray of shape (n_samples, 2)
        Features
    y : ndarray of shape (n_samples,)
        Labels (0 to n_classes-1)
    """
    np.random.seed(random_state)

    samples_per_class = n_samples // n_classes
    X_list = []
    y_list = []

    # Create clusters for each class
    centers = [
        [-3, -3],   # Class 0
        [3, -3],    # Class 1
        [0, 3]      # Class 2
    ]

    for i in range(n_classes):
        if i < len(centers):
            center = centers[i]
        else:
            # Random center for additional classes
            center = np.random.randn(2) * 4

        X_class = np.random.randn(samples_per_class, 2) * 1.2 + center
        y_class = np.full(samples_per_class, i)

        X_list.append(X_class)
        y_list.append(y_class)

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    # Shuffle
    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]

    return X, y.astype(int)


def plot_multiclass_boundary(X, y, model, n_classes=3):
    """
    Plot decision boundaries for multi-class classification.

    Parameters:
    -----------
    X : ndarray
        Features
    y : ndarray
        Labels
    model : DecisionTreeClassifier
        Fitted model
    n_classes : int
        Number of classes
    """
    # Create mesh
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Decision regions
    ax = axes[0]
    ax.contourf(xx, yy, Z, alpha=0.3, levels=n_classes-1, cmap='viridis')
    scatter = ax.scatter(
        X[:, 0], X[:, 1], c=y, cmap='viridis',
        edgecolors='black', s=50, alpha=0.8
    )
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title('Decision Boundaries (Multi-class)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Class')

    # Right plot: Probability contours for each class
    ax = axes[1]
    proba = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])

    # Plot probability of class 0
    Z_proba = proba[:, 0].reshape(xx.shape)
    contour = ax.contourf(xx, yy, Z_proba, alpha=0.6, levels=20, cmap='RdYlGn')
    ax.contour(xx, yy, Z_proba, colors='black', linewidths=0.5, alpha=0.3, levels=10)

    scatter = ax.scatter(
        X[:, 0], X[:, 1], c=y, cmap='viridis',
        edgecolors='black', s=50, alpha=0.8
    )
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title('Probability of Class 0', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(contour, ax=ax, label='P(Class 0)')

    plt.tight_layout()


def compute_confusion_matrix(y_true, y_pred, n_classes):
    """
    Compute confusion matrix.

    Parameters:
    -----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    n_classes : int
        Number of classes

    Returns:
    --------
    cm : ndarray of shape (n_classes, n_classes)
        Confusion matrix
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    return cm


def plot_confusion_matrix(cm, class_names=None):
    """
    Plot confusion matrix as heatmap.

    Parameters:
    -----------
    cm : ndarray
        Confusion matrix
    class_names : list or None
        Class names for labels
    """
    plt.figure(figsize=(8, 6))

    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]

    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()


def main():
    """Run multi-class classification example."""
    print("=" * 70)
    print("Decision Tree - Multi-class Classification Example")
    print("=" * 70)

    # Generate data
    print("\n1. Generating synthetic multi-class data...")
    n_classes = 3
    X, y = make_multiclass_dataset(n_samples=300, n_classes=n_classes, random_state=42)
    print(f"   Dataset shape: X={X.shape}, y={y.shape}")
    print(f"   Number of classes: {n_classes}")
    print(f"   Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # Train Decision Tree
    print("\n2. Training Decision Tree classifier...")
    print("   Parameters:")
    print("   - criterion='gini'")
    print("   - max_depth=6")
    print("   - min_samples_split=5")

    clf = DecisionTreeClassifier(
        criterion='gini',
        max_depth=6,
        min_samples_split=5,
        random_state=42
    )
    clf.fit(X_train, y_train)
    print("   Training complete!")

    # Evaluate
    print("\n3. Model Evaluation:")
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"   Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"   Test Accuracy:     {test_acc:.4f} ({test_acc*100:.2f}%)")

    # Tree statistics
    print("\n4. Tree Structure:")
    print(f"   Tree depth: {clf.get_depth()}")
    print(f"   Number of leaves: {clf.get_n_leaves()}")

    # Feature importance
    print("\n5. Feature Importance:")
    for i, importance in enumerate(clf.feature_importances_):
        print(f"   Feature {i}: {importance:.4f}")

    # Confusion matrix
    print("\n6. Confusion Matrix:")
    y_pred = clf.predict(X_test)
    cm = compute_confusion_matrix(y_test, y_pred, n_classes)
    print("\n   True\\Pred  ", end="")
    for i in range(n_classes):
        print(f"Class {i:2d}  ", end="")
    print()
    for i in range(n_classes):
        print(f"   Class {i}:  ", end="")
        for j in range(n_classes):
            print(f"{cm[i, j]:7d}  ", end="")
        print()

    # Per-class metrics
    print("\n7. Per-class Metrics:")
    for i in range(n_classes):
        true_positives = cm[i, i]
        false_positives = cm[:, i].sum() - true_positives
        false_negatives = cm[i, :].sum() - true_positives

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n   Class {i}:")
        print(f"   - Precision: {precision:.4f}")
        print(f"   - Recall:    {recall:.4f}")
        print(f"   - F1-Score:  {f1:.4f}")

    # Sample predictions
    print("\n8. Sample Predictions:")
    test_samples = X_test[:5]
    predictions = clf.predict(test_samples)
    probabilities = clf.predict_proba(test_samples)

    for i, (sample, pred, proba, true_label) in enumerate(
        zip(test_samples, predictions, probabilities, y_test[:5])
    ):
        print(f"\n   Sample {i+1}: {sample}")
        print(f"   Predicted class: {pred}")
        print(f"   True class: {true_label}")
        print(f"   Probabilities:")
        for cls in range(n_classes):
            print(f"     Class {cls}: {proba[cls]:.4f}")
        print(f"   Correct: {'✓' if pred == true_label else '✗'}")

    # Visualize
    print("\n9. Creating visualizations...")

    # Decision boundaries
    plot_multiclass_boundary(X_train, y_train, clf, n_classes)
    plt.savefig('decision_tree_multiclass_boundaries.png', dpi=150, bbox_inches='tight')
    print("   Saved boundaries to: decision_tree_multiclass_boundaries.png")

    # Confusion matrix
    plot_confusion_matrix(cm)
    plt.savefig('decision_tree_multiclass_confusion.png', dpi=150, bbox_inches='tight')
    print("   Saved confusion matrix to: decision_tree_multiclass_confusion.png")

    plt.show()

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
