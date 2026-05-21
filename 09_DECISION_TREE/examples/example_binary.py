"""
Example: Binary Classification with Decision Tree
=================================================

Demonstrates Decision Tree classifier on a binary classification problem.
Creates a synthetic 2D dataset with two classes and visualizes:
- Decision boundary
- Training accuracy
- Tree structure
- Feature importance

This example shows how the tree learns to separate two classes.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from decision_tree import DecisionTreeClassifier


def make_binary_classification(n_samples=200, random_state=42):
    """
    Create a synthetic binary classification dataset.

    Creates two clusters with some overlap.

    Parameters:
    -----------
    n_samples : int
        Total number of samples
    random_state : int
        Random seed

    Returns:
    --------
    X : ndarray of shape (n_samples, 2)
        Features
    y : ndarray of shape (n_samples,)
        Labels (0 or 1)
    """
    np.random.seed(random_state)

    # Class 0: centered at (-2, -2) with some spread
    n_class_0 = n_samples // 2
    X_0 = np.random.randn(n_class_0, 2) * 1.5 + np.array([-2, -2])
    y_0 = np.zeros(n_class_0)

    # Class 1: centered at (2, 2) with some spread
    n_class_1 = n_samples - n_class_0
    X_1 = np.random.randn(n_class_1, 2) * 1.5 + np.array([2, 2])
    y_1 = np.ones(n_class_1)

    # Combine
    X = np.vstack([X_0, X_1])
    y = np.hstack([y_0, y_1])

    # Shuffle
    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]

    return X, y.astype(int)


def plot_decision_boundary(X, y, model, title="Decision Tree - Binary Classification"):
    """
    Plot decision boundary for 2D binary classification.

    Parameters:
    -----------
    X : ndarray
        Features
    y : ndarray
        Labels
    model : DecisionTreeClassifier
        Fitted model
    title : str
        Plot title
    """
    # Create mesh
    h = 0.1  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 8))

    # Decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.contour(xx, yy, Z, colors='black', linewidths=0.5, alpha=0.5)

    # Training points
    scatter = plt.scatter(
        X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
        edgecolors='black', s=50, alpha=0.8
    )

    plt.colorbar(scatter, label='Class')
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def main():
    """Run binary classification example."""
    print("=" * 70)
    print("Decision Tree - Binary Classification Example")
    print("=" * 70)

    # Generate data
    print("\n1. Generating synthetic binary classification data...")
    X, y = make_binary_classification(n_samples=200, random_state=42)
    print(f"   Dataset shape: X={X.shape}, y={y.shape}")
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
    print("   - max_depth=5")
    print("   - min_samples_split=2")
    print("   - min_samples_leaf=1")

    clf = DecisionTreeClassifier(
        criterion='gini',
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
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

    # Print tree structure
    print("\n6. Tree Structure (text representation):")
    print("-" * 70)
    clf.print_tree()
    print("-" * 70)

    # Test predictions
    print("\n7. Sample Predictions:")
    test_samples = X_test[:5]
    predictions = clf.predict(test_samples)
    probabilities = clf.predict_proba(test_samples)

    for i, (sample, pred, proba, true_label) in enumerate(
        zip(test_samples, predictions, probabilities, y_test[:5])
    ):
        print(f"\n   Sample {i+1}: {sample}")
        print(f"   Predicted class: {pred}")
        print(f"   True class: {true_label}")
        print(f"   Probabilities: Class 0={proba[0]:.4f}, Class 1={proba[1]:.4f}")
        print(f"   Correct: {'✓' if pred == true_label else '✗'}")

    # Visualize
    print("\n8. Creating visualization...")
    plot_decision_boundary(X_train, y_train, clf)
    plt.savefig('decision_tree_binary.png', dpi=150, bbox_inches='tight')
    print("   Saved to: decision_tree_binary.png")

    plt.show()

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
