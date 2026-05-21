"""
Generate all visualization images for Decision Tree README
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, load_iris
from sklearn.model_selection import train_test_split
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decision_tree import DecisionTreeClassifier

# Create images directory
os.makedirs('images', exist_ok=True)

print("Generating Decision Tree visualizations...")
print("=" * 70)

# 1. Gini vs Entropy Comparison
print("\n1. Generating Gini vs Entropy comparison...")

p = np.linspace(0.001, 0.999, 1000)
gini = 2 * p * (1 - p)
entropy = -(p * np.log2(p) + (1-p) * np.log2(1-p))
error = 1 - np.maximum(p, 1-p)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Impurity measures
ax1.plot(p, gini, label='Gini Impurity', linewidth=2.5, color='blue')
ax1.plot(p, entropy, label='Entropy', linewidth=2.5, color='red', linestyle='--')
ax1.plot(p, error, label='Misclassification Error', linewidth=2.5, color='green', linestyle=':')
ax1.set_xlabel('Probability of Class 1 (p)', fontsize=12)
ax1.set_ylabel('Impurity', fontsize=12)
ax1.set_title('Impurity Measures Comparison\n(Binary Classification)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Maximum')

# Sensitivity comparison
p_range = np.linspace(0.3, 0.7, 100)
gini_range = 2 * p_range * (1 - p_range)
entropy_range = -(p_range * np.log2(p_range) + (1-p_range) * np.log2(1-p_range))

# Normalize for comparison
gini_norm = (gini_range - gini_range.min()) / (gini_range.max() - gini_range.min())
entropy_norm = (entropy_range - entropy_range.min()) / (entropy_range.max() - entropy_range.min())

ax2.plot(p_range, gini_norm, label='Gini (normalized)', linewidth=2.5, color='blue')
ax2.plot(p_range, entropy_norm, label='Entropy (normalized)', linewidth=2.5, color='red', linestyle='--')
ax2.set_xlabel('Probability of Class 1 (p)', fontsize=12)
ax2.set_ylabel('Normalized Impurity', fontsize=12)
ax2.set_title('Sensitivity to Probability Changes\n(Near p=0.5)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.text(0.5, 0.95, 'Entropy is more sensitive', ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('images/gini_vs_entropy.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/gini_vs_entropy.png")

# 2. Split Example
print("\n2. Generating split example...")

np.random.seed(42)
X_split = np.random.randn(50, 2)
X_split[:25, 0] += 2
X_split[25:, 0] -= 2
y_split = np.array([0]*25 + [1]*25)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

# Before split
ax1.scatter(X_split[y_split==0, 0], X_split[y_split==0, 1], c='red', s=100,
           alpha=0.6, label='Class A', edgecolors='black')
ax1.scatter(X_split[y_split==1, 0], X_split[y_split==1, 1], c='blue', s=100,
           alpha=0.6, label='Class B', edgecolors='black')
ax1.set_xlabel('Feature x1', fontsize=12)
ax1.set_ylabel('Feature x2', fontsize=12)
ax1.set_title('Before Split\nGini = 0.50 (mixed)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# After split - left node
threshold = 0.0
left_mask = X_split[:, 0] <= threshold
ax2.scatter(X_split[left_mask & (y_split==0), 0], X_split[left_mask & (y_split==0), 1],
           c='red', s=100, alpha=0.6, label='Class A', edgecolors='black')
ax2.scatter(X_split[left_mask & (y_split==1), 0], X_split[left_mask & (y_split==1), 1],
           c='blue', s=100, alpha=0.6, label='Class B', edgecolors='black')
ax2.axvline(x=threshold, color='green', linestyle='--', linewidth=3, label='Split')
ax2.set_xlabel('Feature x1', fontsize=12)
ax2.set_ylabel('Feature x2', fontsize=12)

left_gini = 1 - ((np.sum(left_mask & (y_split==0)) / np.sum(left_mask))**2 +
                 (np.sum(left_mask & (y_split==1)) / np.sum(left_mask))**2)
ax2.set_title(f'Left Node (x1 <= {threshold})\nGini = {left_gini:.3f}',
             fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# After split - right node
right_mask = X_split[:, 0] > threshold
ax3.scatter(X_split[right_mask & (y_split==0), 0], X_split[right_mask & (y_split==0), 1],
           c='red', s=100, alpha=0.6, label='Class A', edgecolors='black')
ax3.scatter(X_split[right_mask & (y_split==1), 0], X_split[right_mask & (y_split==1), 1],
           c='blue', s=100, alpha=0.6, label='Class B', edgecolors='black')
ax3.axvline(x=threshold, color='green', linestyle='--', linewidth=3, label='Split')
ax3.set_xlabel('Feature x1', fontsize=12)
ax3.set_ylabel('Feature x2', fontsize=12)

right_gini = 1 - ((np.sum(right_mask & (y_split==0)) / np.sum(right_mask))**2 +
                  (np.sum(right_mask & (y_split==1)) / np.sum(right_mask))**2)
ax3.set_title(f'Right Node (x1 > {threshold})\nGini = {right_gini:.3f}',
             fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Information gain visualization
ax4.axis('off')

info_gain = 0.5 - (np.sum(left_mask)/len(X_split) * left_gini +
                   np.sum(right_mask)/len(X_split) * right_gini)

text = f"""
Information Gain Calculation:

Parent Gini = 0.500

Left Child:
  - Samples: {np.sum(left_mask)}
  - Gini: {left_gini:.3f}
  - Weight: {np.sum(left_mask)/len(X_split):.3f}

Right Child:
  - Samples: {np.sum(right_mask)}
  - Gini: {right_gini:.3f}
  - Weight: {np.sum(right_mask)/len(X_split):.3f}

Information Gain:
  IG = 0.500 - ({np.sum(left_mask)/len(X_split):.3f} × {left_gini:.3f} + {np.sum(right_mask)/len(X_split):.3f} × {right_gini:.3f})
  IG = {info_gain:.3f}

Reduction: {info_gain/0.5*100:.1f}%
"""

ax4.text(0.1, 0.5, text, fontsize=11, family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
        verticalalignment='center')
ax4.set_title('Information Gain from Split', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('images/split_example.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/split_example.png")

# 3. Tree Depth Comparison
print("\n3. Generating tree depth comparison...")

X_depth, y_depth = make_moons(n_samples=200, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_depth, y_depth, test_size=0.3, random_state=42)

depths = [1, 2, 3, 5, 10, None]
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, depth in enumerate(depths):
    ax = axes[idx // 3, idx % 3]

    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)

    # Create mesh
    h = 0.02
    x_min, x_max = X_depth[:, 0].min() - 0.5, X_depth[:, 0].max() + 0.5
    y_min, y_max = X_depth[:, 1].min() - 0.5, X_depth[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu',
              edgecolors='black', s=50, alpha=0.7)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    actual_depth = clf.get_depth()
    n_leaves = clf.get_n_leaves()

    depth_str = str(depth) if depth is not None else 'None'
    ax.set_title(f'max_depth={depth_str} (actual={actual_depth})\n' +
                f'Train: {train_acc:.3f}, Test: {test_acc:.3f}\n' +
                f'Leaves: {n_leaves}',
                fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature 1', fontsize=10)
    ax.set_ylabel('Feature 2', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('Effect of max_depth on Decision Boundaries and Performance',
            fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('images/tree_depth_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/tree_depth_comparison.png")

# 4. Pre-pruning Effect
print("\n4. Generating pre-pruning effect...")

X_prune, y_prune = make_classification(n_samples=300, n_features=2, n_redundant=0,
                                       n_informative=2, n_clusters_per_class=1,
                                       random_state=42, flip_y=0.1)
X_train, X_test, y_train, y_test = train_test_split(X_prune, y_prune, test_size=0.3, random_state=42)

configs = [
    {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'title': 'No Pruning'},
    {'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 1, 'title': 'max_depth=5'},
    {'max_depth': None, 'min_samples_split': 20, 'min_samples_leaf': 1, 'title': 'min_samples_split=20'},
    {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 10, 'title': 'min_samples_leaf=10'}
]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for idx, config in enumerate(configs):
    ax = axes[idx // 2, idx % 2]

    clf = DecisionTreeClassifier(
        max_depth=config['max_depth'],
        min_samples_split=config['min_samples_split'],
        min_samples_leaf=config['min_samples_leaf'],
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Create mesh
    h = 0.02
    x_min, x_max = X_prune[:, 0].min() - 1, X_prune[:, 0].max() + 1
    y_min, y_max = X_prune[:, 1].min() - 1, X_prune[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm',
              edgecolors='black', s=50, alpha=0.6)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    depth = clf.get_depth()
    leaves = clf.get_n_leaves()

    ax.set_title(f'{config["title"]}\n' +
                f'Depth: {depth}, Leaves: {leaves}\n' +
                f'Train: {train_acc:.3f}, Test: {test_acc:.3f}',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1', fontsize=10)
    ax.set_ylabel('Feature 2', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('Effect of Pre-pruning Parameters on Model Complexity',
            fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('images/prepruning_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/prepruning_effect.png")

# 5. Pruning Comparison
print("\n5. Generating pruning comparison...")

X_comp, y_comp = make_moons(n_samples=300, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_comp, y_comp, test_size=0.3, random_state=42)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

# Unpruned tree
clf_unpruned = DecisionTreeClassifier(random_state=42)
clf_unpruned.fit(X_train, y_train)

h = 0.02
x_min, x_max = X_comp[:, 0].min() - 0.5, X_comp[:, 0].max() + 0.5
y_min, y_max = X_comp[:, 1].min() - 0.5, X_comp[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z_unpruned = clf_unpruned.predict(np.c_[xx.ravel(), yy.ravel()])
Z_unpruned = Z_unpruned.reshape(xx.shape)

ax1.contourf(xx, yy, Z_unpruned, alpha=0.3, cmap='RdYlBu')
ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu',
           edgecolors='black', s=50, alpha=0.6)
ax1.set_title(f'Unpruned Tree\nDepth: {clf_unpruned.get_depth()}, ' +
             f'Leaves: {clf_unpruned.get_n_leaves()}\n' +
             f'Train: {clf_unpruned.score(X_train, y_train):.3f}, ' +
             f'Test: {clf_unpruned.score(X_test, y_test):.3f}',
             fontsize=12, fontweight='bold')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.grid(True, alpha=0.3)

# Pre-pruned tree
clf_prepruned = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_prepruned.fit(X_train, y_train)

Z_prepruned = clf_prepruned.predict(np.c_[xx.ravel(), yy.ravel()])
Z_prepruned = Z_prepruned.reshape(xx.shape)

ax2.contourf(xx, yy, Z_prepruned, alpha=0.3, cmap='RdYlBu')
ax2.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu',
           edgecolors='black', s=50, alpha=0.6)
ax2.set_title(f'Pre-pruned (max_depth=3)\nDepth: {clf_prepruned.get_depth()}, ' +
             f'Leaves: {clf_prepruned.get_n_leaves()}\n' +
             f'Train: {clf_prepruned.score(X_train, y_train):.3f}, ' +
             f'Test: {clf_prepruned.score(X_test, y_test):.3f}',
             fontsize=12, fontweight='bold')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.grid(True, alpha=0.3)

# Cost-complexity pruned tree
clf_ccp = DecisionTreeClassifier(ccp_alpha=0.02, random_state=42)
clf_ccp.fit(X_train, y_train)

Z_ccp = clf_ccp.predict(np.c_[xx.ravel(), yy.ravel()])
Z_ccp = Z_ccp.reshape(xx.shape)

ax3.contourf(xx, yy, Z_ccp, alpha=0.3, cmap='RdYlBu')
ax3.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu',
           edgecolors='black', s=50, alpha=0.6)
ax3.set_title(f'Cost-Complexity Pruned (ccp_alpha=0.02)\nDepth: {clf_ccp.get_depth()}, ' +
             f'Leaves: {clf_ccp.get_n_leaves()}\n' +
             f'Train: {clf_ccp.score(X_train, y_train):.3f}, ' +
             f'Test: {clf_ccp.score(X_test, y_test):.3f}',
             fontsize=12, fontweight='bold')
ax3.set_xlabel('Feature 1')
ax3.set_ylabel('Feature 2')
ax3.grid(True, alpha=0.3)

# Comparison metrics
models = ['Unpruned', 'Pre-pruned\n(max_depth=3)', 'CCP\n(alpha=0.02)']
train_scores = [
    clf_unpruned.score(X_train, y_train) * 100,
    clf_prepruned.score(X_train, y_train) * 100,
    clf_ccp.score(X_train, y_train) * 100
]
test_scores = [
    clf_unpruned.score(X_test, y_test) * 100,
    clf_prepruned.score(X_test, y_test) * 100,
    clf_ccp.score(X_test, y_test) * 100
]

x_pos = np.arange(len(models))
width = 0.35

ax4.bar(x_pos - width/2, train_scores, width, label='Training Accuracy',
       alpha=0.8, color='lightgreen')
ax4.bar(x_pos + width/2, test_scores, width, label='Test Accuracy',
       alpha=0.8, color='lightblue')

ax4.set_xlabel('Pruning Strategy', fontsize=12)
ax4.set_ylabel('Accuracy (%)', fontsize=12)
ax4.set_title('Pruning Strategies Performance Comparison', fontsize=14, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(models)
ax4.legend()
ax4.set_ylim([0, 105])
ax4.grid(True, alpha=0.3, axis='y')

# Add values on bars
for i, (train, test) in enumerate(zip(train_scores, test_scores)):
    ax4.text(i - width/2, train + 1, f'{train:.1f}%', ha='center', fontsize=9)
    ax4.text(i + width/2, test + 1, f'{test:.1f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('images/pruning_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/pruning_comparison.png")

# 6. Feature Importance
print("\n6. Generating feature importance...")

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

# Feature importances
feature_names = iris.feature_names
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

colors = ['coral', 'lightblue', 'lightgreen', 'lightyellow']
ax1.bar(range(len(importances)), importances[indices],
       color=[colors[i] for i in indices], alpha=0.8, edgecolor='black')
ax1.set_xticks(range(len(importances)))
ax1.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
ax1.set_ylabel('Importance', fontsize=12)
ax1.set_title('Feature Importance Rankings', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

for i, imp in enumerate(importances[indices]):
    ax1.text(i, imp + 0.01, f'{imp:.3f}', ha='center', fontsize=10, fontweight='bold')

# Cumulative importance
cumulative_importance = np.cumsum(importances[indices])

ax2.plot(range(1, len(importances)+1), cumulative_importance, marker='o',
        linewidth=2.5, markersize=8, color='darkblue')
ax2.fill_between(range(1, len(importances)+1), cumulative_importance, alpha=0.3, color='lightblue')
ax2.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% Threshold')
ax2.set_xlabel('Number of Features', fontsize=12)
ax2.set_ylabel('Cumulative Importance', fontsize=12)
ax2.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
ax2.set_xticks(range(1, len(importances)+1))
ax2.set_yticks(np.arange(0, 1.1, 0.1))
ax2.legend()
ax2.grid(True, alpha=0.3)

# Feature pairs scatter
most_important = indices[0]
second_important = indices[1]

scatter = ax3.scatter(X_train[:, most_important], X_train[:, second_important],
                     c=y_train, cmap='viridis', s=100, alpha=0.6, edgecolors='black')
ax3.set_xlabel(f'{feature_names[most_important]} (Most Important)', fontsize=11)
ax3.set_ylabel(f'{feature_names[second_important]} (2nd Important)', fontsize=11)
ax3.set_title('Decision Boundary Using Top 2 Features', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Class', fontsize=11)

# Performance with feature subsets
n_features_list = range(1, len(importances)+1)
test_accuracies = []

for n in n_features_list:
    top_features = indices[:n]
    clf_subset = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf_subset.fit(X_train[:, top_features], y_train)
    test_accuracies.append(clf_subset.score(X_test[:, top_features], y_test) * 100)

ax4.plot(n_features_list, test_accuracies, marker='o', linewidth=2.5,
        markersize=8, color='green')
ax4.fill_between(n_features_list, test_accuracies, alpha=0.3, color='lightgreen')
ax4.axvline(x=2, color='red', linestyle='--', linewidth=2, alpha=0.7,
           label='Optimal: 2 features')
ax4.set_xlabel('Number of Top Features Used', fontsize=12)
ax4.set_ylabel('Test Accuracy (%)', fontsize=12)
ax4.set_title('Performance vs Number of Features', fontsize=14, fontweight='bold')
ax4.set_xticks(n_features_list)
ax4.legend()
ax4.set_ylim([min(test_accuracies)-5, 105])
ax4.grid(True, alpha=0.3)

for i, acc in enumerate(test_accuracies):
    if i in [0, 1, len(test_accuracies)-1]:
        ax4.text(i+1, acc + 1, f'{acc:.1f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('images/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/feature_importance.png")

# 7. Multi-class Boundaries
print("\n7. Generating multi-class boundaries...")

iris = load_iris()
X = iris.data[:, :2]  # Use first 2 features
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

# Decision boundaries
h = 0.02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax1.contourf(xx, yy, Z, alpha=0.4, cmap='viridis', levels=2)
scatter = ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis',
                     edgecolors='black', s=100, alpha=0.8)
ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis',
           edgecolors='red', s=100, linewidths=2, alpha=0.8, marker='s')
ax1.set_xlabel('Sepal Length (cm)', fontsize=12)
ax1.set_ylabel('Sepal Width (cm)', fontsize=12)
ax1.set_title('3-Class Decision Boundaries\nIris Dataset', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

cbar1 = plt.colorbar(scatter, ax=ax1)
cbar1.set_label('Class', fontsize=11)
cbar1.set_ticks([0, 1, 2])
cbar1.set_ticklabels(['Setosa', 'Versicolor', 'Virginica'])

# Probability heatmaps for each class
class_names = ['Setosa', 'Versicolor', 'Virginica']

for idx, (ax, class_idx, class_name) in enumerate([(ax2, 0, class_names[0]),
                                                     (ax3, 1, class_names[1]),
                                                     (ax4, 2, class_names[2])]):
    proba = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z_proba = proba[:, class_idx].reshape(xx.shape)

    im = ax.contourf(xx, yy, Z_proba, levels=20, cmap='RdYlGn', alpha=0.8)
    ax.scatter(X_train[y_train==class_idx, 0], X_train[y_train==class_idx, 1],
              c='darkgreen', s=100, alpha=0.8, edgecolors='black', label=f'{class_name} samples')
    ax.scatter(X_train[y_train!=class_idx, 0], X_train[y_train!=class_idx, 1],
              c='lightgray', s=30, alpha=0.3, edgecolors='none')
    ax.set_xlabel('Sepal Length (cm)', fontsize=11)
    ax.set_ylabel('Sepal Width (cm)', fontsize=11)
    ax.set_title(f'P(y = {class_name} | X)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Probability', fontsize=10)

plt.suptitle('Multi-class Classification with Probability Distributions',
            fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('images/multiclass_boundaries.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/multiclass_boundaries.png")

# 8. Bias-Variance Tradeoff
print("\n8. Generating bias-variance tradeoff...")

np.random.seed(42)
X_bv, y_bv = make_moons(n_samples=200, noise=0.25, random_state=42)

depths = range(1, 16)
train_scores = []
test_scores = []

for depth in depths:
    scores_train = []
    scores_test = []

    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X_bv, y_bv, test_size=0.3, random_state=i)
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        scores_train.append(clf.score(X_train, y_train))
        scores_test.append(clf.score(X_test, y_test))

    train_scores.append(np.mean(scores_train))
    test_scores.append(np.mean(scores_test))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

# Accuracy vs depth
ax1.plot(depths, train_scores, 'o-', linewidth=2.5, markersize=6,
        label='Training Accuracy', color='blue')
ax1.plot(depths, test_scores, 's-', linewidth=2.5, markersize=6,
        label='Test Accuracy', color='red')
ax1.fill_between(depths, train_scores, test_scores, alpha=0.2, color='yellow',
                where=[train_scores[i] > test_scores[i] for i in range(len(depths))],
                label='Overfitting Gap')
ax1.set_xlabel('Tree Depth', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Training vs Test Accuracy\n(Overfitting Detection)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(depths[::2])

optimal_depth = depths[np.argmax(test_scores)]
ax1.axvline(x=optimal_depth, color='green', linestyle='--', linewidth=2,
           label=f'Optimal Depth: {optimal_depth}')

# Bias-variance illustration
bias = [(1 - ts) for ts in test_scores]  # Simplified
variance = [abs(tr - ts) for tr, ts in zip(train_scores, test_scores)]

ax2.plot(depths, bias, 'o-', linewidth=2.5, markersize=6,
        label='Bias (Test Error)', color='red')
ax2.plot(depths, variance, 's-', linewidth=2.5, markersize=6,
        label='Variance (Train-Test Gap)', color='blue')
ax2.fill_between(depths, 0, bias, alpha=0.2, color='red')
ax2.fill_between(depths, 0, variance, alpha=0.2, color='blue')
ax2.set_xlabel('Tree Depth', fontsize=12)
ax2.set_ylabel('Error', fontsize=12)
ax2.set_title('Bias-Variance Tradeoff', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(depths[::2])

# Visualization for three depths
example_depths = [2, 5, 15]
axes_examples = [ax3, ax4]

for i, depth in enumerate([2, 15]):
    ax = axes_examples[i]

    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_bv, y_bv)

    h = 0.02
    x_min, x_max = X_bv[:, 0].min() - 0.5, X_bv[:, 0].max() + 0.5
    y_min, y_max = X_bv[:, 1].min() - 0.5, X_bv[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X_bv[:, 0], X_bv[:, 1], c=y_bv, cmap='RdYlBu',
              edgecolors='black', s=50, alpha=0.7)

    bias_label = 'High Bias' if depth == 2 else 'Low Bias'
    var_label = 'Low Variance' if depth == 2 else 'High Variance'

    ax.set_title(f'Depth = {depth}\n{bias_label}, {var_label}',
                fontsize=13, fontweight='bold')
    ax.set_xlabel('Feature 1', fontsize=11)
    ax.set_ylabel('Feature 2', fontsize=11)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/bias_variance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/bias_variance.png")

print("\n" + "=" * 70)
print("All Decision Tree visualizations generated successfully!")
print("Images saved in: 09_DECISION_TREE/images/")
print("=" * 70)
