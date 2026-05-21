"""
Visualize Random Forest Structure

Creates detailed visualizations showing how random forests combine multiple trees
and how individual trees differ due to bootstrap sampling and feature randomness.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_moons
from sklearn.model_selection import train_test_split
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from random_forest import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '09_DECISION_TREE'))
from decision_tree import DecisionTreeClassifier

# Create images directory
os.makedirs('images', exist_ok=True)

print("Generating Random Forest structure visualizations...")
print("=" * 70)


def count_tree_nodes(node, depth=0):
    """Count nodes at each depth"""
    if node is None:
        return {}

    counts = {depth: 1}

    if node.value is None:  # Not a leaf
        left_counts = count_tree_nodes(node.left, depth + 1)
        right_counts = count_tree_nodes(node.right, depth + 1)

        for d in left_counts:
            counts[d] = counts.get(d, 0) + left_counts[d]
        for d in right_counts:
            counts[d] = counts.get(d, 0) + right_counts[d]

    return counts


# 1. Individual Trees in Forest
print("\n1. Generating individual tree visualizations...")
np.random.seed(42)

X_moons, y_moons = make_moons(n_samples=200, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_moons, y_moons, test_size=0.3, random_state=42)

# Train random forest
rf = RandomForestClassifier(n_estimators=6, max_depth=4, random_state=42)
rf.fit(X_train, y_train)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx in range(6):
    ax = axes[idx]
    tree = rf.trees_[idx]

    # Plot decision boundary for this tree
    h = 0.02
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')

    # Plot training points
    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
              c='red', s=20, alpha=0.6, edgecolors='black')
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
              c='blue', s=20, alpha=0.6, edgecolors='black')

    # Calculate accuracy
    acc = tree.score(X_test, y_test)

    ax.set_xlabel('Feature 1', fontsize=10)
    ax.set_ylabel('Feature 2', fontsize=10)
    ax.set_title(f'Tree {idx + 1}\nTest Acc: {acc:.3f}',
                fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.suptitle('Individual Trees in Random Forest (n_estimators=6)\nEach tree trained on different bootstrap sample',
            fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('images/forest_individual_trees.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/forest_individual_trees.png")

# 2. Ensemble Prediction vs Individual Trees
print("\n2. Generating ensemble vs individual predictions...")
np.random.seed(42)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Train with fewer trees for clarity
rf_small = RandomForestClassifier(n_estimators=5, max_depth=5, random_state=42)
rf_small.fit(X_train, y_train)

h = 0.02
x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Plot first 5 individual trees
for idx in range(5):
    ax = axes[0, idx] if idx < 3 else axes[1, idx - 3]
    tree = rf_small.trees_[idx]

    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
    ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
              c='red', s=15, alpha=0.6, edgecolors='black')
    ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
              c='blue', s=15, alpha=0.6, edgecolors='black')

    acc = tree.score(X_test, y_test)
    ax.set_title(f'Tree {idx + 1}: {acc:.3f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature 1', fontsize=9)
    ax.set_ylabel('Feature 2', fontsize=9)
    ax.grid(True, alpha=0.3)

# Plot ensemble
ax = axes[1, 2]
Z_ensemble = rf_small.predict(np.c_[xx.ravel(), yy.ravel()])
Z_ensemble = Z_ensemble.reshape(xx.shape)

ax.contourf(xx, yy, Z_ensemble, alpha=0.4, cmap='RdBu')
ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
          c='red', s=15, alpha=0.8, edgecolors='black', linewidths=1.5)
ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
          c='blue', s=15, alpha=0.8, edgecolors='black', linewidths=1.5)

ensemble_acc = rf_small.score(X_test, y_test)
ax.set_title(f'ENSEMBLE: {ensemble_acc:.3f}\n(Majority Vote)', fontsize=11, fontweight='bold',
            color='green')
ax.set_xlabel('Feature 1', fontsize=9)
ax.set_ylabel('Feature 2', fontsize=9)
ax.grid(True, alpha=0.3)

# Add border to ensemble plot
for spine in ax.spines.values():
    spine.set_edgecolor('green')
    spine.set_linewidth(3)

plt.suptitle('Random Forest Ensemble: Combining Individual Tree Predictions',
            fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('images/forest_ensemble_voting.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/forest_ensemble_voting.png")

# 3. Tree Diversity Analysis
print("\n3. Generating tree diversity analysis...")
np.random.seed(42)

# Train forest with more trees
rf_div = RandomForestClassifier(n_estimators=20, max_depth=6, random_state=42)
rf_div.fit(X_train, y_train)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Tree depths distribution
depths = []
for tree in rf_div.trees_:
    def get_depth(node, current_depth=0):
        if node is None or node.value is not None:
            return current_depth
        return max(get_depth(node.left, current_depth + 1),
                  get_depth(node.right, current_depth + 1))

    depths.append(get_depth(tree.tree_))

ax1.hist(depths, bins=range(min(depths), max(depths) + 2), alpha=0.7,
        color='#2E86AB', edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Tree Depth', fontsize=12)
ax1.set_ylabel('Number of Trees', fontsize=12)
ax1.set_title('Distribution of Tree Depths', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.axvline(np.mean(depths), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {np.mean(depths):.1f}')
ax1.legend(fontsize=11)

# Plot 2: Feature usage across trees
feature_usage = np.zeros(X_train.shape[1])
for tree in rf_div.trees_:
    if hasattr(tree, 'feature_importances_') and tree.feature_importances_ is not None:
        feature_usage += (tree.feature_importances_ > 0).astype(int)

ax2.bar(['Feature 1', 'Feature 2'], feature_usage, color=['#2E86AB', '#F18F01'],
       alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Number of Trees Using Feature', fontsize=12)
ax2.set_title('Feature Usage Across Trees', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

for i, v in enumerate(feature_usage):
    ax2.text(i, v + 0.3, f'{int(v)}/{len(rf_div.trees_)}',
            ha='center', fontsize=11, fontweight='bold')

# Plot 3: Prediction agreement
# Get predictions from all trees
all_preds = np.array([tree.predict(X_test) for tree in rf_div.trees_])
agreement = np.mean(all_preds == rf_div.predict(X_test), axis=0)

scatter = ax3.scatter(X_test[:, 0], X_test[:, 1], c=agreement,
                     cmap='RdYlGn', s=80, alpha=0.8,
                     edgecolors='black', linewidths=1)
ax3.set_xlabel('Feature 1', fontsize=12)
ax3.set_ylabel('Feature 2', fontsize=12)
ax3.set_title('Tree Agreement on Predictions\n(Green = high agreement)',
             fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Agreement Ratio', fontsize=11)

# Plot 4: Individual tree accuracy distribution
tree_accuracies = [tree.score(X_test, y_test) for tree in rf_div.trees_]

ax4.hist(tree_accuracies, bins=15, alpha=0.7, color='#2E86AB',
        edgecolor='black', linewidth=1.5)
ax4.axvline(np.mean(tree_accuracies), color='red', linestyle='--',
           linewidth=2, label=f'Mean: {np.mean(tree_accuracies):.3f}')
ax4.axvline(rf_div.score(X_test, y_test), color='green', linestyle='--',
           linewidth=2.5, label=f'Ensemble: {rf_div.score(X_test, y_test):.3f}')
ax4.set_xlabel('Test Accuracy', fontsize=12)
ax4.set_ylabel('Number of Trees', fontsize=12)
ax4.set_title('Individual Tree Accuracies vs Ensemble', fontsize=13, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('images/forest_tree_diversity.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/forest_tree_diversity.png")

# 4. Bootstrap Sampling Effect
print("\n4. Generating bootstrap sampling effect...")
np.random.seed(42)

iris = load_iris()
X_iris = iris.data[:, [0, 2]]  # sepal length, petal length
y_iris = iris.target

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Train forest
rf_iris = RandomForestClassifier(n_estimators=6, max_depth=4, bootstrap=True, random_state=42)
rf_iris.fit(X_iris, y_iris)

# For each tree, show what samples were used (simulated)
rng = np.random.default_rng(42)

for idx in range(6):
    ax = axes[idx]

    # Simulate bootstrap sample
    n_samples = len(X_iris)
    bootstrap_indices = rng.choice(n_samples, size=n_samples, replace=True)
    oob_mask = np.ones(n_samples, dtype=bool)
    oob_mask[bootstrap_indices] = False

    # Plot in-bag samples (larger, opaque)
    for i in range(3):
        class_mask = (y_iris == i) & ~oob_mask
        ax.scatter(X_iris[class_mask, 0], X_iris[class_mask, 1],
                  s=80, alpha=0.8, edgecolors='black', linewidths=1.5,
                  label=f'Class {i} (in-bag)' if i == 0 else '')

    # Plot OOB samples (smaller, transparent)
    for i in range(3):
        class_mask = (y_iris == i) & oob_mask
        ax.scatter(X_iris[class_mask, 0], X_iris[class_mask, 1],
                  s=30, alpha=0.3, edgecolors='gray', linewidths=1,
                  marker='x')

    # Count in-bag and OOB
    n_in_bag = np.sum(~oob_mask)
    n_oob = np.sum(oob_mask)

    ax.set_xlabel('Sepal Length', fontsize=10)
    ax.set_ylabel('Petal Length', fontsize=10)
    ax.set_title(f'Tree {idx + 1}: {n_in_bag} in-bag, {n_oob} OOB',
                fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if idx == 0:
        # Add legend only to first plot
        ax.plot([], [], 'o', color='gray', markersize=10, alpha=0.8,
               markeredgecolor='black', label='In-bag samples')
        ax.plot([], [], 'x', color='gray', markersize=8, alpha=0.3,
               label='Out-of-bag samples')
        ax.legend(fontsize=9, loc='upper left')

plt.suptitle('Bootstrap Sampling: Different Training Sets for Each Tree\n' +
            'Large dots = in-bag (used for training), Small X = out-of-bag (not used)',
            fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('images/forest_bootstrap_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/forest_bootstrap_effect.png")

print("\n" + "=" * 70)
print("All Random Forest structure visualizations generated successfully!")
print("Images saved in: 10_RANDOM_FOREST/images/")
print("=" * 70)
