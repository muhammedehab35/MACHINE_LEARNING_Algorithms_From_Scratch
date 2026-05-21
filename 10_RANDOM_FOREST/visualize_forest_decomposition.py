"""
Visualize Random Forest Decomposition - Step by Step

Shows how each tree in the random forest decomposes the feature space,
and how the ensemble combines these decompositions for final predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, load_iris
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from random_forest import RandomForestClassifier

# Create images directory
os.makedirs('images', exist_ok=True)

print("Generating Random Forest decomposition visualizations...")
print("=" * 70)


def plot_tree_decomposition_levels(tree, X, y, ax, max_depth=3, title=''):
    """Plot a single tree's decomposition at different depth levels"""
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']

    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict
    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision regions
    n_classes = len(np.unique(y))
    ax.contourf(xx, yy, Z, alpha=0.3, colors=colors[:n_classes],
                levels=np.arange(n_classes + 1) - 0.5)

    # Plot training points
    for idx in range(n_classes):
        mask = y == idx
        ax.scatter(X[mask, 0], X[mask, 1],
                  c=[colors[idx]], s=40, edgecolors='black',
                  linewidths=1, alpha=0.7)

    # Draw split lines
    def draw_splits(node, x_min, x_max, y_min, y_max, current_depth=0):
        if node is None or node.value is not None or current_depth >= max_depth:
            return

        feature = node.feature_index
        threshold = node.threshold

        if feature == 0:
            ax.plot([threshold, threshold], [y_min, y_max],
                   'k-', linewidth=2, alpha=0.7)
            if node.left:
                draw_splits(node.left, x_min, threshold, y_min, y_max, current_depth + 1)
            if node.right:
                draw_splits(node.right, threshold, x_max, y_min, y_max, current_depth + 1)
        else:
            ax.plot([x_min, x_max], [threshold, threshold],
                   'k-', linewidth=2, alpha=0.7)
            if node.left:
                draw_splits(node.left, x_min, x_max, y_min, threshold, current_depth + 1)
            if node.right:
                draw_splits(node.right, x_min, x_max, threshold, y_max, current_depth + 1)

    draw_splits(tree.tree_, x_min, x_max, y_min, y_max)

    ax.set_xlabel('Feature 1', fontsize=10)
    ax.set_ylabel('Feature 2', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)


# 1. Progressive Depth Levels for Single Tree
print("\n1. Generating progressive depth levels for single tree...")
np.random.seed(42)

X_moons, y_moons = make_moons(n_samples=200, noise=0.2, random_state=42)

from decision_tree import DecisionTreeClassifier

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for depth in range(4):
    tree = DecisionTreeClassifier(max_depth=depth if depth > 0 else None, random_state=42)
    tree.fit(X_moons, y_moons)

    if depth == 0:
        title = f'Profondeur {depth}\n(Racine seulement)'
    else:
        title = f'Profondeur {depth}\n(≤{2**depth} régions)'

    plot_tree_decomposition_levels(tree, X_moons, y_moons, axes[depth],
                                   max_depth=depth if depth > 0 else 0, title=title)

plt.suptitle('Décomposition Progressive d\'un Arbre Unique\nComment l\'arbre divise l\'espace à chaque niveau',
            fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('images/forest_single_tree_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/forest_single_tree_decomposition.png")

# 2. Individual Trees Decomposition in Forest
print("\n2. Generating individual trees decomposition in forest...")
np.random.seed(42)

rf = RandomForestClassifier(n_estimators=6, max_depth=3, random_state=42)
rf.fit(X_moons, y_moons)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx in range(6):
    tree = rf.trees_[idx]
    plot_tree_decomposition_levels(tree, X_moons, y_moons, axes[idx],
                                   max_depth=3,
                                   title=f'Arbre {idx + 1}\n(échantillon bootstrap {idx + 1})')

plt.suptitle('Décomposition de 6 Arbres dans la Forêt\nChaque arbre décompose l\'espace différemment grâce au bootstrap',
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('images/forest_multiple_trees_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/forest_multiple_trees_decomposition.png")

# 3. Comparing Tree Depths in Forest
print("\n3. Generating depth comparison in forest...")
np.random.seed(42)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

depths = [1, 2, 3, 4, 5, 6]

for idx, depth in enumerate(depths):
    ax = axes[idx // 3, idx % 3]

    rf_depth = RandomForestClassifier(n_estimators=50, max_depth=depth, random_state=42)
    rf_depth.fit(X_moons, y_moons)

    # Plot ensemble prediction
    h = 0.02
    x_min, x_max = X_moons[:, 0].min() - 0.5, X_moons[:, 0].max() + 0.5
    y_min, y_max = X_moons[:, 1].min() - 0.5, X_moons[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = rf_depth.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, colors=['#FF6B6B', '#4ECDC4'],
                levels=[-0.5, 0.5, 1.5])

    ax.scatter(X_moons[y_moons == 0, 0], X_moons[y_moons == 0, 1],
              c='#FF6B6B', s=30, edgecolors='black', linewidths=1, alpha=0.6)
    ax.scatter(X_moons[y_moons == 1, 0], X_moons[y_moons == 1, 1],
              c='#4ECDC4', s=30, edgecolors='black', linewidths=1, alpha=0.6)

    acc = rf_depth.score(X_moons, y_moons)
    ax.set_title(f'Profondeur Max = {depth}\nPrécision: {acc:.3f}',
                fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature 1', fontsize=9)
    ax.set_ylabel('Feature 2', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Effet de la Profondeur sur la Décomposition de la Forêt\nPlus profond = frontières plus complexes',
            fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('images/forest_depth_effect_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/forest_depth_effect_decomposition.png")

# 4. Ensemble Combination - Progressive
print("\n4. Generating progressive ensemble combination...")
np.random.seed(42)

rf_full = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=42)
rf_full.fit(X_moons, y_moons)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

n_trees_list = [1, 3, 5, 10, 15, 20]

for idx, n_trees in enumerate(n_trees_list):
    ax = axes[idx]

    # Create a forest with limited trees
    rf_limited = RandomForestClassifier(n_estimators=n_trees, max_depth=4, random_state=42)
    rf_limited.fit(X_moons, y_moons)

    h = 0.02
    x_min, x_max = X_moons[:, 0].min() - 0.5, X_moons[:, 0].max() + 0.5
    y_min, y_max = X_moons[:, 1].min() - 0.5, X_moons[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = rf_limited.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, colors=['#FF6B6B', '#4ECDC4'],
                levels=[-0.5, 0.5, 1.5])

    ax.scatter(X_moons[y_moons == 0, 0], X_moons[y_moons == 0, 1],
              c='#FF6B6B', s=30, edgecolors='black', linewidths=1, alpha=0.6)
    ax.scatter(X_moons[y_moons == 1, 0], X_moons[y_moons == 1, 1],
              c='#4ECDC4', s=30, edgecolors='black', linewidths=1, alpha=0.6)

    acc = rf_limited.score(X_moons, y_moons)
    ax.set_title(f'{n_trees} Arbres\nPrécision: {acc:.3f}',
                fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature 1', fontsize=9)
    ax.set_ylabel('Feature 2', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Combinaison Progressive de l\'Ensemble\nPlus d\'arbres = frontières plus lisses et stables',
            fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('images/forest_progressive_ensemble.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/forest_progressive_ensemble.png")

# 5. Decomposition with Annotations
print("\n5. Generating annotated forest decomposition...")
np.random.seed(42)

# Simple dataset
X_simple = np.random.randn(150, 2)
y_simple = ((X_simple[:, 0] > 0) & (X_simple[:, 1] > 0)).astype(int)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Individual trees
for i in range(3):
    ax = axes[0, 0] if i == 0 else (axes[0, 1] if i == 1 else axes[1, 0])

    tree = DecisionTreeClassifier(max_depth=3, random_state=42 + i)
    tree.fit(X_simple, y_simple)

    plot_tree_decomposition_levels(tree, X_simple, y_simple, ax,
                                   max_depth=3,
                                   title=f'Arbre {i + 1} - Décomposition Individuelle')

    # Add annotation for first tree
    if i == 0:
        ax.text(0.05, 0.95, 'Chaque arbre\ndivise l\'espace\nindépendamment',
               transform=ax.transAxes, fontsize=10, fontweight='bold',
               verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

# Ensemble
ax = axes[1, 1]
rf_ens = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
rf_ens.fit(X_simple, y_simple)

h = 0.02
x_min, x_max = X_simple[:, 0].min() - 0.5, X_simple[:, 0].max() + 0.5
y_min, y_max = X_simple[:, 1].min() - 0.5, X_simple[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = rf_ens.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, alpha=0.3, colors=['#FF6B6B', '#4ECDC4'],
            levels=[-0.5, 0.5, 1.5])

ax.scatter(X_simple[y_simple == 0, 0], X_simple[y_simple == 0, 1],
          c='#FF6B6B', s=40, edgecolors='black', linewidths=1.5, alpha=0.7)
ax.scatter(X_simple[y_simple == 1, 0], X_simple[y_simple == 1, 1],
          c='#4ECDC4', s=40, edgecolors='black', linewidths=1.5, alpha=0.7)

ax.set_title('Ensemble Combiné (10 arbres)\nVote Majoritaire', fontsize=12, fontweight='bold', color='green')
ax.set_xlabel('Feature 1', fontsize=10)
ax.set_ylabel('Feature 2', fontsize=10)
ax.grid(True, alpha=0.3)

# Add border
for spine in ax.spines.values():
    spine.set_edgecolor('green')
    spine.set_linewidth(3)

ax.text(0.05, 0.95, 'La forêt combine\ntoutes les\ndécompositions\npar vote',
       transform=ax.transAxes, fontsize=10, fontweight='bold',
       verticalalignment='top',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

plt.suptitle('Comment Random Forest Combine les Décompositions\nChaque arbre vote → Prédiction finale par majorité',
            fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('images/forest_annotated_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/forest_annotated_decomposition.png")

# 6. Iris Dataset - Forest Decomposition
print("\n6. Generating Iris dataset forest decomposition...")
np.random.seed(42)

iris = load_iris()
X_iris_2d = iris.data[:, [0, 2]]  # sepal length, petal length
y_iris = iris.target

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Train forest
rf_iris = RandomForestClassifier(n_estimators=6, max_depth=4, random_state=42)
rf_iris.fit(X_iris_2d, y_iris)

colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']

for idx in range(5):
    ax = axes[idx]
    tree = rf_iris.trees_[idx]

    h = 0.02
    x_min, x_max = X_iris_2d[:, 0].min() - 0.5, X_iris_2d[:, 0].max() + 0.5
    y_min, y_max = X_iris_2d[:, 1].min() - 0.5, X_iris_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, colors=colors, levels=[-.5, .5, 1.5, 2.5])

    for i in range(3):
        mask = y_iris == i
        ax.scatter(X_iris_2d[mask, 0], X_iris_2d[mask, 1],
                  c=[colors[i]], s=30, edgecolors='black', linewidths=1, alpha=0.6,
                  label=['Setosa', 'Versicolor', 'Virginica'][i] if idx == 0 else '')

    ax.set_title(f'Arbre {idx + 1}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Sepal Length', fontsize=9)
    ax.set_ylabel('Petal Length', fontsize=9)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(loc='upper left', fontsize=8)

# Ensemble plot
ax = axes[5]
Z_ensemble = rf_iris.predict(np.c_[xx.ravel(), yy.ravel()])
Z_ensemble = Z_ensemble.reshape(xx.shape)

ax.contourf(xx, yy, Z_ensemble, alpha=0.3, colors=colors, levels=[-.5, .5, 1.5, 2.5])

for i in range(3):
    mask = y_iris == i
    ax.scatter(X_iris_2d[mask, 0], X_iris_2d[mask, 1],
              c=[colors[i]], s=40, edgecolors='black', linewidths=1.5, alpha=0.8)

ax.set_title('ENSEMBLE\n(Vote Majoritaire)', fontsize=11, fontweight='bold', color='green')
ax.set_xlabel('Sepal Length', fontsize=9)
ax.set_ylabel('Petal Length', fontsize=9)
ax.grid(True, alpha=0.3)

for spine in ax.spines.values():
    spine.set_edgecolor('green')
    spine.set_linewidth(3)

plt.suptitle('Random Forest sur Iris - Décomposition Multi-Classes\nChaque arbre décompose l\'espace pour 3 classes',
            fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('images/forest_iris_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/forest_iris_decomposition.png")

print("\n" + "=" * 70)
print("All Random Forest decomposition visualizations generated successfully!")
print("Images saved in: 10_RANDOM_FOREST/images/")
print("=" * 70)
