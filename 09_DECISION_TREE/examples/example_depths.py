"""
Example: Comparing Different Tree Depths
=========================================

Demonstrates the effect of tree depth on model performance and overfitting.
Compares Decision Trees with different max_depth values to show:
- Underfitting (shallow trees)
- Good fit (appropriate depth)
- Overfitting (very deep trees)

This example illustrates the bias-variance tradeoff in Decision Trees.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from decision_tree import DecisionTreeClassifier


def make_noisy_circles(n_samples=400, noise=0.15, random_state=42):
    """
    Create concentric circles dataset with noise.

    Parameters:
    -----------
    n_samples : int
        Number of samples
    noise : float
        Standard deviation of Gaussian noise
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

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # Outer circle
    linspace_out = np.linspace(0, 2 * np.pi, n_samples_out)
    outer_circ_x = np.cos(linspace_out)
    outer_circ_y = np.sin(linspace_out)
    X_out = np.vstack([outer_circ_x, outer_circ_y]).T * 2
    y_out = np.ones(n_samples_out)

    # Inner circle
    linspace_in = np.linspace(0, 2 * np.pi, n_samples_in)
    inner_circ_x = np.cos(linspace_in)
    inner_circ_y = np.sin(linspace_in)
    X_in = np.vstack([inner_circ_x, inner_circ_y]).T
    y_in = np.zeros(n_samples_in)

    # Combine
    X = np.vstack([X_in, X_out])
    y = np.hstack([y_in, y_out])

    # Add noise
    X += np.random.randn(n_samples, 2) * noise

    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    return X, y.astype(int)


def plot_tree_comparison(X_train, y_train, X_test, y_test, depths):
    """
    Plot decision boundaries for trees with different depths.

    Parameters:
    -----------
    X_train : ndarray
        Training features
    y_train : ndarray
        Training labels
    X_test : ndarray
        Test features
    y_test : ndarray
        Test labels
    depths : list of int
        Tree depths to compare
    """
    n_depths = len(depths)
    fig, axes = plt.subplots(2, n_depths, figsize=(5*n_depths, 10))

    if n_depths == 1:
        axes = axes.reshape(-1, 1)

    # Create mesh for decision boundary
    h = 0.02
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    for idx, depth in enumerate(depths):
        # Train model
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)

        # Predictions
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        # Plot training data
        ax = axes[0, idx]
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                  cmap='RdYlBu', edgecolors='black', s=30, alpha=0.8)
        ax.set_title(f'Depth={depth} (Train)\n'
                    f'Acc={train_acc:.3f}, Leaves={clf.get_n_leaves()}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)

        # Plot test data
        ax = axes[1, idx]
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                  cmap='RdYlBu', edgecolors='black', s=30, alpha=0.8)
        ax.set_title(f'Depth={depth} (Test)\n'
                    f'Acc={test_acc:.3f}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()


def plot_depth_vs_accuracy(X_train, y_train, X_test, y_test, max_depth_range):
    """
    Plot accuracy vs tree depth.

    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    max_depth_range : list
        Range of depths to test
    """
    train_accs = []
    test_accs = []
    n_leaves_list = []

    for depth in max_depth_range:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)

        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        train_accs.append(train_acc)
        test_accs.append(test_acc)
        n_leaves_list.append(clf.get_n_leaves())

    # Plot accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(max_depth_range, train_accs, 'o-', label='Training', linewidth=2, markersize=6)
    ax1.plot(max_depth_range, test_accs, 's-', label='Test', linewidth=2, markersize=6)
    ax1.set_xlabel('Max Depth', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy vs Tree Depth', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Highlight regions
    optimal_idx = np.argmax(test_accs)
    optimal_depth = max_depth_range[optimal_idx]
    ax1.axvline(optimal_depth, color='green', linestyle='--', alpha=0.5, label=f'Optimal depth={optimal_depth}')
    ax1.legend(fontsize=11)

    # Plot number of leaves
    ax2.plot(max_depth_range, n_leaves_list, 'o-', color='purple', linewidth=2, markersize=6)
    ax2.set_xlabel('Max Depth', fontsize=12)
    ax2.set_ylabel('Number of Leaves', fontsize=12)
    ax2.set_title('Tree Complexity vs Depth', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    return optimal_depth, train_accs, test_accs


def main():
    """Run depth comparison example."""
    print("=" * 70)
    print("Decision Tree - Depth Comparison (Overfitting Analysis)")
    print("=" * 70)

    # Generate data
    print("\n1. Generating noisy circles dataset...")
    X, y = make_noisy_circles(n_samples=400, noise=0.15, random_state=42)
    print(f"   Dataset shape: X={X.shape}, y={y.shape}")

    # Split data
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # Compare specific depths
    print("\n2. Training trees with different depths...")
    depths_to_compare = [1, 3, 5, 10]

    results = []
    for depth in depths_to_compare:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)

        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        n_leaves = clf.get_n_leaves()

        results.append({
            'depth': depth,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'n_leaves': n_leaves
        })

        print(f"\n   Depth={depth:2d}:")
        print(f"   - Training Accuracy: {train_acc:.4f}")
        print(f"   - Test Accuracy:     {test_acc:.4f}")
        print(f"   - Number of leaves:  {n_leaves}")
        print(f"   - Overfitting gap:   {(train_acc - test_acc):.4f}")

    # Analysis
    print("\n3. Analysis:")
    print("\n   Interpretation:")

    # Find underfitting, good fit, and overfitting
    for r in results:
        gap = r['train_acc'] - r['test_acc']
        if r['depth'] <= 2:
            status = "UNDERFITTING"
            desc = "Too simple, high bias"
        elif gap < 0.05:
            status = "GOOD FIT"
            desc = "Balanced bias-variance"
        else:
            status = "OVERFITTING"
            desc = "Too complex, high variance"

        print(f"\n   Depth {r['depth']:2d}: {status}")
        print(f"   - {desc}")
        print(f"   - Train-Test gap: {gap:.4f}")

    # Find optimal depth
    print("\n4. Finding optimal depth...")
    max_depth_range = list(range(1, 16))
    optimal_depth, train_accs, test_accs = plot_depth_vs_accuracy(
        X_train, y_train, X_test, y_test, max_depth_range
    )

    optimal_idx = np.argmax(test_accs)
    print(f"\n   Optimal depth: {optimal_depth}")
    print(f"   Best test accuracy: {test_accs[optimal_idx]:.4f}")
    print(f"   Training accuracy at optimal: {train_accs[optimal_idx]:.4f}")

    # Visualize
    print("\n5. Creating visualizations...")

    # Depth comparison
    plot_tree_comparison(X_train, y_train, X_test, y_test, depths_to_compare)
    plt.savefig('decision_tree_depth_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved comparison to: decision_tree_depth_comparison.png")

    # Already created in plot_depth_vs_accuracy
    plt.savefig('decision_tree_depth_vs_accuracy.png', dpi=150, bbox_inches='tight')
    print("   Saved accuracy plot to: decision_tree_depth_vs_accuracy.png")

    plt.show()

    # Key takeaways
    print("\n6. Key Takeaways:")
    print("   " + "=" * 66)
    print("   - Shallow trees (depth 1-2): Underfit, cannot capture patterns")
    print("   - Medium trees (depth 3-5): Often optimal, good generalization")
    print("   - Deep trees (depth >10): Overfit, memorize training data")
    print("   - Test accuracy peaks at moderate depth, then decreases")
    print("   - Training accuracy keeps increasing with depth")
    print("   - The gap between train and test accuracy indicates overfitting")
    print("   " + "=" * 66)

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
