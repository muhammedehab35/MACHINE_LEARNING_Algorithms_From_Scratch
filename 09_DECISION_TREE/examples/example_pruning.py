"""
Example: Tree Pruning Techniques
=================================

Demonstrates different pruning techniques to prevent overfitting:
1. Pre-pruning: Stop growing early based on constraints
   - max_depth
   - min_samples_split
   - min_samples_leaf

2. Cost-complexity pruning: Remove branches based on complexity
   - ccp_alpha parameter

Shows how pruning affects:
- Model complexity (number of nodes/leaves)
- Training vs test accuracy
- Generalization

This example illustrates the importance of pruning in Decision Trees.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from decision_tree import DecisionTreeClassifier


def make_noisy_moons(n_samples=400, noise=0.2, random_state=42):
    """
    Create two interleaving half circles with noise.

    Parameters:
    -----------
    n_samples : int
        Number of samples
    noise : float
        Standard deviation of noise
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

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # Outer semicircle
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    X_out = np.vstack([outer_circ_x, outer_circ_y]).T

    # Inner semicircle (shifted)
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5
    X_in = np.vstack([inner_circ_x, inner_circ_y]).T

    # Combine
    X = np.vstack([X_in, X_out])
    y = np.hstack([np.zeros(n_samples_in), np.ones(n_samples_out)])

    # Add noise
    X += np.random.randn(n_samples, 2) * noise

    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    return X, y.astype(int)


def compare_prepruning(X_train, y_train, X_test, y_test):
    """
    Compare different pre-pruning strategies.

    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Create mesh for decision boundary
    h = 0.02
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    configs = [
        {'name': 'No Pruning', 'params': {}},
        {'name': 'max_depth=3', 'params': {'max_depth': 3}},
        {'name': 'min_samples_split=50', 'params': {'min_samples_split': 50}},
        {'name': 'min_samples_leaf=20', 'params': {'min_samples_leaf': 20}},
    ]

    results = []

    for idx, config in enumerate(configs):
        # Train model
        clf = DecisionTreeClassifier(random_state=42, **config['params'])
        clf.fit(X_train, y_train)

        # Predictions
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        depth = clf.get_depth()
        n_leaves = clf.get_n_leaves()

        results.append({
            'name': config['name'],
            'train_acc': train_acc,
            'test_acc': test_acc,
            'depth': depth,
            'leaves': n_leaves,
            'gap': train_acc - test_acc
        })

        # Plot training
        ax = axes[0, idx]
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                  cmap='RdYlBu', edgecolors='black', s=30, alpha=0.8)
        ax.set_title(f'{config["name"]}\nTrain Acc={train_acc:.3f}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)

        # Plot test
        ax = axes[1, idx]
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                  cmap='RdYlBu', edgecolors='black', s=30, alpha=0.8)
        ax.set_title(f'Test Acc={test_acc:.3f}\nDepth={depth}, Leaves={n_leaves}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return results


def compare_ccp_alpha(X_train, y_train, X_test, y_test):
    """
    Compare different ccp_alpha values for cost-complexity pruning.

    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    """
    ccp_alphas = [0.0, 0.01, 0.02, 0.05]

    fig, axes = plt.subplots(2, len(ccp_alphas), figsize=(5*len(ccp_alphas), 10))

    # Create mesh
    h = 0.02
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    results = []

    for idx, alpha in enumerate(ccp_alphas):
        # Train model
        clf = DecisionTreeClassifier(
            max_depth=10,  # Allow deep tree initially
            ccp_alpha=alpha,
            random_state=42
        )
        clf.fit(X_train, y_train)

        # Predictions
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        depth = clf.get_depth()
        n_leaves = clf.get_n_leaves()

        results.append({
            'alpha': alpha,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'depth': depth,
            'leaves': n_leaves
        })

        # Plot training
        ax = axes[0, idx]
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                  cmap='RdYlBu', edgecolors='black', s=30, alpha=0.8)
        ax.set_title(f'ccp_alpha={alpha}\nTrain Acc={train_acc:.3f}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)

        # Plot test
        ax = axes[1, idx]
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                  cmap='RdYlBu', edgecolors='black', s=30, alpha=0.8)
        ax.set_title(f'Test Acc={test_acc:.3f}\nDepth={depth}, Leaves={n_leaves}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return results


def plot_pruning_comparison(X_train, y_train, X_test, y_test):
    """
    Plot comprehensive comparison of pruning effects.

    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    """
    # Test different pruning parameters
    max_depths = list(range(1, 16))
    min_samples_splits = [2, 5, 10, 20, 50, 100]
    min_samples_leafs = [1, 5, 10, 20, 40]
    ccp_alphas = np.linspace(0, 0.1, 20)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. max_depth
    ax = axes[0, 0]
    train_accs, test_accs, leaves = [], [], []
    for depth in max_depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        train_accs.append(clf.score(X_train, y_train))
        test_accs.append(clf.score(X_test, y_test))
        leaves.append(clf.get_n_leaves())

    ax.plot(max_depths, train_accs, 'o-', label='Train', linewidth=2)
    ax.plot(max_depths, test_accs, 's-', label='Test', linewidth=2)
    ax.set_xlabel('max_depth', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Effect of max_depth', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 2. min_samples_split
    ax = axes[0, 1]
    train_accs, test_accs, leaves = [], [], []
    for min_split in min_samples_splits:
        clf = DecisionTreeClassifier(min_samples_split=min_split, random_state=42)
        clf.fit(X_train, y_train)
        train_accs.append(clf.score(X_train, y_train))
        test_accs.append(clf.score(X_test, y_test))
        leaves.append(clf.get_n_leaves())

    ax.plot(min_samples_splits, train_accs, 'o-', label='Train', linewidth=2)
    ax.plot(min_samples_splits, test_accs, 's-', label='Test', linewidth=2)
    ax.set_xlabel('min_samples_split', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Effect of min_samples_split', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 3. min_samples_leaf
    ax = axes[1, 0]
    train_accs, test_accs, leaves = [], [], []
    for min_leaf in min_samples_leafs:
        clf = DecisionTreeClassifier(min_samples_leaf=min_leaf, random_state=42)
        clf.fit(X_train, y_train)
        train_accs.append(clf.score(X_train, y_train))
        test_accs.append(clf.score(X_test, y_test))
        leaves.append(clf.get_n_leaves())

    ax.plot(min_samples_leafs, train_accs, 'o-', label='Train', linewidth=2)
    ax.plot(min_samples_leafs, test_accs, 's-', label='Test', linewidth=2)
    ax.set_xlabel('min_samples_leaf', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Effect of min_samples_leaf', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 4. ccp_alpha
    ax = axes[1, 1]
    train_accs, test_accs, leaves = [], [], []
    for alpha in ccp_alphas:
        clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
        clf.fit(X_train, y_train)
        train_accs.append(clf.score(X_train, y_train))
        test_accs.append(clf.score(X_test, y_test))
        leaves.append(clf.get_n_leaves())

    ax.plot(ccp_alphas, train_accs, 'o-', label='Train', linewidth=2, markersize=4)
    ax.plot(ccp_alphas, test_accs, 's-', label='Test', linewidth=2, markersize=4)
    ax.set_xlabel('ccp_alpha', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Effect of ccp_alpha (Cost-Complexity)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()


def main():
    """Run pruning comparison example."""
    print("=" * 70)
    print("Decision Tree - Pruning Techniques")
    print("=" * 70)

    # Generate data
    print("\n1. Generating noisy moons dataset...")
    X, y = make_noisy_moons(n_samples=400, noise=0.2, random_state=42)
    print(f"   Dataset shape: X={X.shape}, y={y.shape}")

    # Split data
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # Pre-pruning comparison
    print("\n2. Comparing Pre-pruning Strategies...")
    prepruning_results = compare_prepruning(X_train, y_train, X_test, y_test)

    print("\n   Pre-pruning Results:")
    print("   " + "-" * 66)
    print(f"   {'Strategy':<25} {'Train':<8} {'Test':<8} {'Gap':<8} {'Depth':<7} {'Leaves':<8}")
    print("   " + "-" * 66)
    for r in prepruning_results:
        print(f"   {r['name']:<25} {r['train_acc']:.4f}   {r['test_acc']:.4f}   "
              f"{r['gap']:.4f}   {r['depth']:<7} {r['leaves']:<8}")
    print("   " + "-" * 66)

    # Cost-complexity pruning
    print("\n3. Comparing Cost-Complexity Pruning (ccp_alpha)...")
    ccp_results = compare_ccp_alpha(X_train, y_train, X_test, y_test)

    print("\n   Cost-Complexity Pruning Results:")
    print("   " + "-" * 66)
    print(f"   {'ccp_alpha':<12} {'Train Acc':<12} {'Test Acc':<12} {'Depth':<8} {'Leaves':<8}")
    print("   " + "-" * 66)
    for r in ccp_results:
        print(f"   {r['alpha']:<12.3f} {r['train_acc']:<12.4f} {r['test_acc']:<12.4f} "
              f"{r['depth']:<8} {r['leaves']:<8}")
    print("   " + "-" * 66)

    # Analysis
    print("\n4. Analysis:")
    print("\n   Pre-pruning (Early Stopping):")
    print("   - max_depth: Most direct control over tree complexity")
    print("   - min_samples_split: Prevents splitting small nodes")
    print("   - min_samples_leaf: Ensures leaves have enough samples")
    print("   - Applied during tree building")

    print("\n   Cost-Complexity Pruning:")
    print("   - Applied after tree is fully grown")
    print("   - Removes branches with low impurity decrease")
    print("   - ccp_alpha controls pruning strength")
    print("   - Higher alpha = more aggressive pruning")

    print("\n   Key Observations:")
    best_prepruning = max(prepruning_results, key=lambda x: x['test_acc'])
    print(f"   - Best pre-pruning: {best_prepruning['name']}")
    print(f"     Test Acc: {best_prepruning['test_acc']:.4f}")
    print(f"     Complexity: {best_prepruning['leaves']} leaves")

    best_ccp = max(ccp_results, key=lambda x: x['test_acc'])
    print(f"\n   - Best ccp_alpha: {best_ccp['alpha']:.3f}")
    print(f"     Test Acc: {best_ccp['test_acc']:.4f}")
    print(f"     Complexity: {best_ccp['leaves']} leaves")

    # Visualizations
    print("\n5. Creating visualizations...")

    # Pre-pruning comparison is already created
    plt.savefig('decision_tree_prepruning.png', dpi=150, bbox_inches='tight')
    print("   Saved pre-pruning to: decision_tree_prepruning.png")

    # CCP comparison is already created
    plt.savefig('decision_tree_ccp_pruning.png', dpi=150, bbox_inches='tight')
    print("   Saved ccp pruning to: decision_tree_ccp_pruning.png")

    # Comprehensive comparison
    plot_pruning_comparison(X_train, y_train, X_test, y_test)
    plt.savefig('decision_tree_pruning_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved comparison to: decision_tree_pruning_comparison.png")

    plt.show()

    # Recommendations
    print("\n6. Pruning Recommendations:")
    print("   " + "=" * 66)
    print("   When to use Pre-pruning:")
    print("   - When you know the desired tree complexity")
    print("   - For faster training (stops early)")
    print("   - When interpretability is important")
    print("   - Recommended: Start with max_depth=3-5")

    print("\n   When to use Cost-Complexity Pruning:")
    print("   - When optimal complexity is unknown")
    print("   - For fine-tuning a fully-grown tree")
    print("   - When you want minimal validation error")
    print("   - Can be combined with cross-validation")

    print("\n   Combining Both:")
    print("   - Use pre-pruning to limit initial growth")
    print("   - Apply cost-complexity for fine-tuning")
    print("   - Best of both worlds: fast + optimal")
    print("   " + "=" * 66)

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
