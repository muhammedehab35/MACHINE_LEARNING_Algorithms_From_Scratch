"""
Example: Feature Importance Analysis
====================================

Demonstrates how Decision Trees calculate and interpret feature importance.

Shows:
- Feature importance calculation based on impurity decrease
- Visualization of feature importance
- Relationship between importance and decision boundaries
- Feature selection using importance scores
- Comparison with permutation importance

Feature importance in Decision Trees is calculated as:
    Importance(f) = Σ (n_node/n_total) * impurity_decrease
where impurity_decrease is weighted by the number of samples at each node.

This example helps understand which features the tree considers most important.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from decision_tree import DecisionTreeClassifier


def make_feature_importance_dataset(n_samples=500, random_state=42):
    """
    Create dataset with features of varying importance.

    Features:
    - Feature 0 & 1: Strong predictors (directly related to class)
    - Feature 2 & 3: Weak predictors (noisy relationship)
    - Feature 4 & 5: Noise features (no relationship)

    Parameters:
    -----------
    n_samples : int
        Number of samples
    random_state : int
        Random seed

    Returns:
    --------
    X : ndarray of shape (n_samples, 6)
        Features
    y : ndarray of shape (n_samples,)
        Labels
    feature_names : list
        Feature names with descriptions
    """
    np.random.seed(random_state)

    # Class 0: x0 < 0 and x1 < 0
    n_class_0 = n_samples // 2
    X_0_important = np.random.randn(n_class_0, 2) - 1.5  # Features 0, 1
    X_0_weak = np.random.randn(n_class_0, 2) * 0.5       # Features 2, 3
    X_0_noise = np.random.randn(n_class_0, 2)             # Features 4, 5
    X_0 = np.hstack([X_0_important, X_0_weak, X_0_noise])
    y_0 = np.zeros(n_class_0)

    # Class 1: x0 > 0 and x1 > 0
    n_class_1 = n_samples - n_class_0
    X_1_important = np.random.randn(n_class_1, 2) + 1.5  # Features 0, 1
    X_1_weak = np.random.randn(n_class_1, 2) * 0.5       # Features 2, 3
    X_1_noise = np.random.randn(n_class_1, 2)             # Features 4, 5
    X_1 = np.hstack([X_1_important, X_1_weak, X_1_noise])
    y_1 = np.ones(n_class_1)

    # Combine
    X = np.vstack([X_0, X_1])
    y = np.hstack([y_0, y_1])

    # Shuffle
    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]

    feature_names = [
        'Feature 0 (Strong)',
        'Feature 1 (Strong)',
        'Feature 2 (Weak)',
        'Feature 3 (Weak)',
        'Feature 4 (Noise)',
        'Feature 5 (Noise)'
    ]

    return X, y.astype(int), feature_names


def plot_feature_importance(importances, feature_names, title="Feature Importance"):
    """
    Plot feature importance as horizontal bar chart.

    Parameters:
    -----------
    importances : ndarray
        Feature importance scores
    feature_names : list
        Feature names
    title : str
        Plot title
    """
    # Sort features by importance
    indices = np.argsort(importances)

    plt.figure(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(importances)))

    plt.barh(range(len(importances)), importances[indices],
            color=colors[indices], edgecolor='black', linewidth=1.5)

    plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (idx, imp) in enumerate(zip(indices, importances[indices])):
        plt.text(imp + 0.01, i, f'{imp:.4f}',
                va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()


def plot_pairwise_features(X, y, feature_importances, feature_names):
    """
    Plot pairwise scatter plots colored by importance.

    Parameters:
    -----------
    X : ndarray
        Features
    y : ndarray
        Labels
    feature_importances : ndarray
        Feature importance scores
    feature_names : list
        Feature names
    """
    # Select top 4 most important features
    top_indices = np.argsort(feature_importances)[-4:][::-1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, feat_idx in enumerate(top_indices):
        ax = axes[idx]

        # Plot feature vs class
        for class_label in np.unique(y):
            mask = y == class_label
            ax.scatter(X[mask, feat_idx], np.random.randn(np.sum(mask)) * 0.1,
                      label=f'Class {class_label}', alpha=0.6, s=30)

        ax.set_xlabel(feature_names[feat_idx], fontsize=11)
        ax.set_ylabel('Random jitter', fontsize=11)
        ax.set_title(f'{feature_names[feat_idx]}\nImportance: {feature_importances[feat_idx]:.4f}',
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()


def compare_with_feature_selection(X, y, feature_names):
    """
    Compare model performance with different feature subsets.

    Parameters:
    -----------
    X : ndarray
        All features
    y : ndarray
        Labels
    feature_names : list
        Feature names
    """
    # Split data
    split_idx = int(0.75 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train on all features
    clf_all = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf_all.fit(X_train, y_train)
    acc_all = clf_all.score(X_test, y_test)
    importances = clf_all.feature_importances_

    # Sort features by importance
    sorted_indices = np.argsort(importances)[::-1]

    # Test with top-k features
    results = []
    for k in range(1, len(feature_names) + 1):
        top_k_features = sorted_indices[:k]

        clf_k = DecisionTreeClassifier(max_depth=5, random_state=42)
        clf_k.fit(X_train[:, top_k_features], y_train)
        acc_k = clf_k.score(X_test[:, top_k_features], y_test)

        results.append({
            'n_features': k,
            'features': [feature_names[i] for i in top_k_features],
            'accuracy': acc_k
        })

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy vs number of features
    n_features = [r['n_features'] for r in results]
    accuracies = [r['accuracy'] for r in results]

    ax1.plot(n_features, accuracies, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.axhline(acc_all, color='red', linestyle='--', label='All features', linewidth=2)
    ax1.set_xlabel('Number of Top Features Used', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Accuracy vs Number of Features', fontsize=14, fontweight='bold')
    ax1.set_xticks(n_features)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Feature cumulative importance
    sorted_importances = importances[sorted_indices]
    cumulative_importances = np.cumsum(sorted_importances)

    ax2.plot(n_features, cumulative_importances, 'o-', linewidth=2, markersize=8, color='green')
    ax2.axhline(0.95, color='red', linestyle='--', label='95% threshold', linewidth=2)
    ax2.set_xlabel('Number of Top Features', fontsize=12)
    ax2.set_ylabel('Cumulative Importance', fontsize=12)
    ax2.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
    ax2.set_xticks(n_features)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    return results, importances


def calculate_permutation_importance(clf, X, y, n_repeats=10):
    """
    Calculate permutation importance by shuffling features.

    Parameters:
    -----------
    clf : DecisionTreeClassifier
        Fitted model
    X : ndarray
        Test features
    y : ndarray
        Test labels
    n_repeats : int
        Number of times to permute each feature

    Returns:
    --------
    importances : ndarray
        Mean importance for each feature
    importances_std : ndarray
        Standard deviation of importance
    """
    baseline_score = clf.score(X, y)

    importances = []
    for feature_idx in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, feature_idx])
            score = clf.score(X_permuted, y)
            scores.append(baseline_score - score)  # Drop in accuracy

        importances.append(scores)

    importances = np.array(importances)
    return importances.mean(axis=1), importances.std(axis=1)


def main():
    """Run feature importance analysis example."""
    print("=" * 70)
    print("Decision Tree - Feature Importance Analysis")
    print("=" * 70)

    # Generate data
    print("\n1. Generating dataset with varying feature importance...")
    X, y, feature_names = make_feature_importance_dataset(n_samples=500, random_state=42)
    print(f"   Dataset shape: X={X.shape}, y={y.shape}")
    print(f"   Features: {len(feature_names)}")
    print("\n   Feature Design:")
    for i, name in enumerate(feature_names):
        print(f"   - {name}")

    # Split data
    split_idx = int(0.75 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"\n   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # Train model
    print("\n2. Training Decision Tree...")
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"   Training Accuracy: {train_acc:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Tree Depth: {clf.get_depth()}")
    print(f"   Number of Leaves: {clf.get_n_leaves()}")

    # Feature importance
    print("\n3. Feature Importance (Impurity-based):")
    importances = clf.feature_importances_
    print("\n   " + "-" * 62)
    print(f"   {'Feature':<30} {'Importance':<15} {'Rank':<10}")
    print("   " + "-" * 62)

    # Sort by importance
    sorted_indices = np.argsort(importances)[::-1]
    for rank, idx in enumerate(sorted_indices, 1):
        print(f"   {feature_names[idx]:<30} {importances[idx]:<15.6f} {rank:<10}")
    print("   " + "-" * 62)

    # Interpretation
    print("\n4. Interpretation:")
    print("\n   Expected behavior:")
    print("   - Strong features (0, 1) should have HIGH importance")
    print("   - Weak features (2, 3) should have MEDIUM importance")
    print("   - Noise features (4, 5) should have LOW/ZERO importance")

    print("\n   Actual results:")
    strong_importance = importances[0] + importances[1]
    weak_importance = importances[2] + importances[3]
    noise_importance = importances[4] + importances[5]

    print(f"   - Strong features total: {strong_importance:.4f}")
    print(f"   - Weak features total: {weak_importance:.4f}")
    print(f"   - Noise features total: {noise_importance:.4f}")

    # Permutation importance
    print("\n5. Calculating Permutation Importance...")
    perm_importances, perm_std = calculate_permutation_importance(
        clf, X_test, y_test, n_repeats=10
    )

    print("\n   Permutation Importance (mean ± std):")
    print("   " + "-" * 62)
    for idx in sorted_indices:
        print(f"   {feature_names[idx]:<30} {perm_importances[idx]:.6f} ± {perm_std[idx]:.6f}")
    print("   " + "-" * 62)

    # Compare methods
    print("\n6. Comparing Importance Methods:")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Impurity-based
    ax1.barh(range(len(importances)), importances[sorted_indices],
            color='skyblue', edgecolor='black')
    ax1.set_yticks(range(len(importances)))
    ax1.set_yticklabels([feature_names[i] for i in sorted_indices])
    ax1.set_xlabel('Importance', fontsize=12)
    ax1.set_title('Impurity-based Importance', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # Permutation-based
    ax2.barh(range(len(perm_importances)), perm_importances[sorted_indices],
            xerr=perm_std[sorted_indices], color='lightcoral', edgecolor='black')
    ax2.set_yticks(range(len(perm_importances)))
    ax2.set_yticklabels([feature_names[i] for i in sorted_indices])
    ax2.set_xlabel('Importance (Accuracy Drop)', fontsize=12)
    ax2.set_title('Permutation Importance', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('decision_tree_importance_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved comparison to: decision_tree_importance_comparison.png")

    # Feature selection
    print("\n7. Testing Feature Selection...")
    selection_results, _ = compare_with_feature_selection(X, y, feature_names)

    print("\n   Results with top-k features:")
    print("   " + "-" * 62)
    print(f"   {'k':<5} {'Accuracy':<12} {'Features':<45}")
    print("   " + "-" * 62)
    for r in selection_results:
        features_str = ', '.join([f.split()[0] + f.split()[1] for f in r['features'][:3]])
        if r['n_features'] > 3:
            features_str += f", ... (+{r['n_features']-3})"
        print(f"   {r['n_features']:<5} {r['accuracy']:<12.4f} {features_str}")
    print("   " + "-" * 62)

    # Find optimal number of features
    accuracies = [r['accuracy'] for r in selection_results]
    optimal_k = np.argmax(accuracies) + 1
    print(f"\n   Optimal number of features: {optimal_k}")
    print(f"   Best accuracy: {max(accuracies):.4f}")

    # Already created in compare_with_feature_selection
    plt.savefig('decision_tree_feature_selection.png', dpi=150, bbox_inches='tight')
    print("   Saved feature selection to: decision_tree_feature_selection.png")

    # Visualize all features
    print("\n8. Creating visualizations...")
    plot_feature_importance(importances, feature_names)
    plt.savefig('decision_tree_feature_importance.png', dpi=150, bbox_inches='tight')
    print("   Saved importance plot to: decision_tree_feature_importance.png")

    plot_pairwise_features(X, y, importances, feature_names)
    plt.savefig('decision_tree_feature_distributions.png', dpi=150, bbox_inches='tight')
    print("   Saved distributions to: decision_tree_feature_distributions.png")

    plt.show()

    # Key takeaways
    print("\n9. Key Takeaways:")
    print("   " + "=" * 66)
    print("   Feature Importance Calculation:")
    print("   - Based on weighted impurity decrease at each split")
    print("   - Features used higher in tree have more importance")
    print("   - Features used in more nodes accumulate importance")
    print("   - Normalized to sum to 1.0")

    print("\n   Impurity-based vs Permutation-based:")
    print("   - Impurity: Fast, based on training data splits")
    print("   - Permutation: Slower, based on prediction degradation")
    print("   - Permutation is more robust to overfitting")
    print("   - Both should agree on important features")

    print("\n   Using Feature Importance:")
    print("   - Feature selection: Remove low-importance features")
    print("   - Feature engineering: Focus on important features")
    print("   - Model interpretation: Understand key predictors")
    print("   - Dimensionality reduction: Keep top features")
    print("   " + "=" * 66)

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
