"""
Example: Comparing Split Criteria
==================================

Compares different split criteria for Decision Trees:
- Gini Impurity
- Entropy (Information Gain)
- Misclassification Error

Shows:
- Decision boundaries for each criterion
- Performance comparison
- Computational differences
- When to use each criterion

This example demonstrates that while theoretically different,
these criteria often produce similar results in practice.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('..')
from decision_tree import DecisionTreeClassifier


def make_classification_dataset(n_samples=300, random_state=42):
    """
    Create a synthetic dataset with clear patterns.

    Parameters:
    -----------
    n_samples : int
        Number of samples
    random_state : int
        Random seed

    Returns:
    --------
    X : ndarray of shape (n_samples, 2)
        Features
    y : ndarray of shape (n_samples,)
        Labels
    """
    np.random.seed(random_state)

    # Create two overlapping clusters
    n_class_0 = n_samples // 2

    # Class 0: lower left region
    X_0 = np.random.randn(n_class_0, 2) * 1.2 + np.array([-1.5, -1.5])
    y_0 = np.zeros(n_class_0)

    # Class 1: upper right region
    X_1 = np.random.randn(n_samples - n_class_0, 2) * 1.2 + np.array([1.5, 1.5])
    y_1 = np.ones(n_samples - n_class_0)

    # Combine
    X = np.vstack([X_0, X_1])
    y = np.hstack([y_0, y_1])

    # Shuffle
    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]

    return X, y.astype(int)


def compare_criteria_visual(X_train, y_train, X_test, y_test, criteria):
    """
    Visualize decision boundaries for different criteria.

    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    criteria : list of str
        Criteria to compare
    """
    n_criteria = len(criteria)
    fig, axes = plt.subplots(2, n_criteria, figsize=(6*n_criteria, 12))

    if n_criteria == 1:
        axes = axes.reshape(-1, 1)

    # Create mesh
    h = 0.02
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    criterion_names = {
        'gini': 'Gini Impurity',
        'entropy': 'Entropy (Info Gain)',
        'error': 'Misclassification Error'
    }

    for idx, criterion in enumerate(criteria):
        # Train model
        start_time = time.time()
        clf = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=5,
            random_state=42
        )
        clf.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Predictions
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        # Training data plot
        ax = axes[0, idx]
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        ax.contour(xx, yy, Z, colors='black', linewidths=0.5, alpha=0.5)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                  cmap='RdYlBu', edgecolors='black', s=40, alpha=0.8)
        ax.set_title(f'{criterion_names[criterion]}\n'
                    f'Train Acc={train_acc:.3f}, Depth={clf.get_depth()}',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)

        # Test data plot
        ax = axes[1, idx]
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        ax.contour(xx, yy, Z, colors='black', linewidths=0.5, alpha=0.5)
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                  cmap='RdYlBu', edgecolors='black', s=40, alpha=0.8)
        ax.set_title(f'Test Acc={test_acc:.3f}\n'
                    f'Time={train_time:.4f}s, Leaves={clf.get_n_leaves()}',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()


def compare_criteria_metrics(X_train, y_train, X_test, y_test, criteria, max_depths):
    """
    Compare criteria across different tree depths.

    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    criteria : list of str
        Criteria to compare
    max_depths : list of int
        Tree depths to test
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    criterion_names = {
        'gini': 'Gini',
        'entropy': 'Entropy',
        'error': 'Error'
    }

    colors = {'gini': 'blue', 'entropy': 'green', 'error': 'red'}
    markers = {'gini': 'o', 'entropy': 's', 'error': '^'}

    results = {criterion: {'train': [], 'test': [], 'time': [], 'leaves': []}
               for criterion in criteria}

    # Collect results
    for criterion in criteria:
        for depth in max_depths:
            clf = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=depth,
                random_state=42
            )

            start_time = time.time()
            clf.fit(X_train, y_train)
            train_time = time.time() - start_time

            train_acc = clf.score(X_train, y_train)
            test_acc = clf.score(X_test, y_test)
            n_leaves = clf.get_n_leaves()

            results[criterion]['train'].append(train_acc)
            results[criterion]['test'].append(test_acc)
            results[criterion]['time'].append(train_time)
            results[criterion]['leaves'].append(n_leaves)

    # Plot training accuracy
    ax = axes[0, 0]
    for criterion in criteria:
        ax.plot(max_depths, results[criterion]['train'],
               marker=markers[criterion], color=colors[criterion],
               label=criterion_names[criterion], linewidth=2, markersize=6)
    ax.set_xlabel('Max Depth', fontsize=12)
    ax.set_ylabel('Training Accuracy', fontsize=12)
    ax.set_title('Training Accuracy vs Depth', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot test accuracy
    ax = axes[0, 1]
    for criterion in criteria:
        ax.plot(max_depths, results[criterion]['test'],
               marker=markers[criterion], color=colors[criterion],
               label=criterion_names[criterion], linewidth=2, markersize=6)
    ax.set_xlabel('Max Depth', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Test Accuracy vs Depth', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot training time
    ax = axes[1, 0]
    for criterion in criteria:
        ax.plot(max_depths, results[criterion]['time'],
               marker=markers[criterion], color=colors[criterion],
               label=criterion_names[criterion], linewidth=2, markersize=6)
    ax.set_xlabel('Max Depth', fontsize=12)
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Training Time vs Depth', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot number of leaves
    ax = axes[1, 1]
    for criterion in criteria:
        ax.plot(max_depths, results[criterion]['leaves'],
               marker=markers[criterion], color=colors[criterion],
               label=criterion_names[criterion], linewidth=2, markersize=6)
    ax.set_xlabel('Max Depth', fontsize=12)
    ax.set_ylabel('Number of Leaves', fontsize=12)
    ax.set_title('Tree Complexity vs Depth', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return results


def demonstrate_impurity_calculation():
    """
    Demonstrate how each criterion calculates impurity.
    """
    print("\n" + "=" * 70)
    print("Impurity Calculation Demonstration")
    print("=" * 70)

    # Example label distribution
    y_examples = [
        np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),  # Perfect balance
        np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1]),  # Imbalanced
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # Pure node
    ]

    descriptions = [
        "Balanced (50-50)",
        "Imbalanced (70-30)",
        "Pure (100-0)"
    ]

    for y, desc in zip(y_examples, descriptions):
        print(f"\n{desc}:")
        print(f"Labels: {y}")

        # Count classes
        unique, counts = np.unique(y, return_counts=True)
        n = len(y)
        proportions = counts / n

        print(f"Class distribution: {dict(zip(unique, counts))}")
        print(f"Proportions: {dict(zip(unique, proportions))}")

        # Calculate Gini
        gini = 1.0 - np.sum(proportions ** 2)
        print(f"\nGini Impurity = 1 - Σ(p²)")
        print(f"  = 1 - ({' + '.join([f'{p:.2f}²' for p in proportions])})")
        print(f"  = 1 - {np.sum(proportions ** 2):.4f}")
        print(f"  = {gini:.4f}")

        # Calculate Entropy
        proportions_nonzero = proportions[proportions > 0]
        entropy = -np.sum(proportions_nonzero * np.log2(proportions_nonzero))
        print(f"\nEntropy = -Σ(p * log₂(p))")
        terms = [f'{p:.2f} * log₂({p:.2f})' for p in proportions_nonzero]
        print(f"  = -({' + '.join(terms)})")
        print(f"  = {entropy:.4f}")

        # Calculate Error
        error = 1.0 - np.max(proportions)
        print(f"\nMisclassification Error = 1 - max(p)")
        print(f"  = 1 - {np.max(proportions):.2f}")
        print(f"  = {error:.4f}")

    print("\n" + "=" * 70)


def main():
    """Run criteria comparison example."""
    print("=" * 70)
    print("Decision Tree - Split Criteria Comparison")
    print("=" * 70)

    # Demonstrate impurity calculations
    demonstrate_impurity_calculation()

    # Generate data
    print("\n1. Generating synthetic dataset...")
    X, y = make_classification_dataset(n_samples=300, random_state=42)
    print(f"   Dataset shape: X={X.shape}, y={y.shape}")

    # Split data
    split_idx = int(0.75 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # Compare criteria
    print("\n2. Training trees with different criteria...")
    criteria = ['gini', 'entropy', 'error']

    results_single = {}
    for criterion in criteria:
        clf = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=5,
            random_state=42
        )

        start_time = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start_time

        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        results_single[criterion] = {
            'model': clf,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'time': train_time,
            'depth': clf.get_depth(),
            'leaves': clf.get_n_leaves()
        }

        print(f"\n   Criterion: {criterion}")
        print(f"   - Training Accuracy: {train_acc:.4f}")
        print(f"   - Test Accuracy:     {test_acc:.4f}")
        print(f"   - Training Time:     {train_time:.4f}s")
        print(f"   - Tree Depth:        {clf.get_depth()}")
        print(f"   - Number of Leaves:  {clf.get_n_leaves()}")

    # Compare across depths
    print("\n3. Comparing criteria across different depths...")
    max_depths = list(range(1, 11))
    results = compare_criteria_metrics(
        X_train, y_train, X_test, y_test,
        criteria, max_depths
    )

    # Analysis
    print("\n4. Analysis:")
    print("\n   Key Observations:")
    print("   - Gini and Entropy usually produce very similar results")
    print("   - Gini is slightly faster to compute (no logarithm)")
    print("   - Entropy may produce slightly more balanced trees")
    print("   - Error is less commonly used in practice")
    print("   - All criteria become similar as trees grow deeper")

    print("\n   When to use each criterion:")
    print("   - Gini: Default choice, fast and effective")
    print("   - Entropy: When you want more balanced splits")
    print("   - Error: Rarely used, less sensitive to probability changes")

    # Visualize
    print("\n5. Creating visualizations...")

    # Visual comparison
    compare_criteria_visual(X_train, y_train, X_test, y_test, criteria)
    plt.savefig('decision_tree_criteria_visual.png', dpi=150, bbox_inches='tight')
    print("   Saved visual comparison to: decision_tree_criteria_visual.png")

    # Already created in compare_criteria_metrics
    plt.savefig('decision_tree_criteria_metrics.png', dpi=150, bbox_inches='tight')
    print("   Saved metrics comparison to: decision_tree_criteria_metrics.png")

    plt.show()

    # Summary
    print("\n6. Summary:")
    print("   " + "=" * 66)
    print("   Mathematical Properties:")
    print("   - Gini: Σ(pᵢ * (1-pᵢ)), range [0, 0.5] for binary")
    print("   - Entropy: -Σ(pᵢ * log₂(pᵢ)), range [0, 1] for binary")
    print("   - Error: 1 - max(pᵢ), range [0, 0.5] for binary")
    print("\n   Practical Recommendations:")
    print("   - Use Gini for most applications (sklearn default)")
    print("   - Use Entropy when interpretability matters")
    print("   - Differences are usually small in practice")
    print("   " + "=" * 66)

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
