"""
Visualize Tree Decomposition - Step by Step

Shows how a decision tree splits the feature space progressively,
from the root node down to the leaves, with clear step-by-step visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_moons
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decision_tree import DecisionTreeClassifier

# Create images directory
os.makedirs('images', exist_ok=True)

print("Generating tree decomposition visualizations...")
print("=" * 70)


def plot_step_by_step_splits(tree, X, y, feature_names=None, class_names=None, max_depth=3):
    """
    Plot the tree splits step by step, showing how the feature space
    is progressively divided from root to leaves.
    """
    if feature_names is None:
        feature_names = [f'X{i}' for i in range(X.shape[1])]
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(np.unique(y)))]

    # We'll show splits at different depths
    depths_to_show = min(max_depth, 4)

    fig, axes = plt.subplots(1, depths_to_show + 1, figsize=(4 * (depths_to_show + 1), 4))
    if depths_to_show == 0:
        axes = [axes]

    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))

    def plot_at_depth(ax, depth_limit, title):
        """Plot tree splits up to a certain depth"""
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # Create a tree limited to this depth
        tree_limited = DecisionTreeClassifier(max_depth=depth_limit, random_state=42)
        tree_limited.fit(X, y)

        # Predict
        Z = tree_limited.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision regions
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='Set3', levels=np.arange(len(class_names) + 1) - 0.5)

        # Plot training points
        for idx, class_name in enumerate(class_names):
            mask = y == idx
            ax.scatter(X[mask, 0], X[mask, 1],
                      c=[colors[idx]], label=class_name,
                      s=60, edgecolors='black', linewidths=1.5, alpha=0.8)

        # Draw split lines recursively
        def draw_splits(node, x_min, x_max, y_min, y_max, current_depth=0):
            if node is None or node.value is not None or current_depth >= depth_limit:
                return

            feature = node.feature_index
            threshold = node.threshold

            if feature == 0:  # Split on first feature
                ax.plot([threshold, threshold], [y_min, y_max],
                       'k-', linewidth=2.5, alpha=0.8)
                ax.text(threshold, y_max - 0.3, f'{feature_names[0]}={threshold:.2f}',
                       fontsize=8, ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

                # Recurse
                if node.left:
                    draw_splits(node.left, x_min, threshold, y_min, y_max, current_depth + 1)
                if node.right:
                    draw_splits(node.right, threshold, x_max, y_min, y_max, current_depth + 1)
            else:  # Split on second feature
                ax.plot([x_min, x_max], [threshold, threshold],
                       'k-', linewidth=2.5, alpha=0.8)
                ax.text(x_min + 0.3, threshold, f'{feature_names[1]}={threshold:.2f}',
                       fontsize=8, va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

                # Recurse
                if node.left:
                    draw_splits(node.left, x_min, x_max, y_min, threshold, current_depth + 1)
                if node.right:
                    draw_splits(node.right, x_min, x_max, threshold, y_max, current_depth + 1)

        draw_splits(tree_limited.tree_, x_min, x_max, y_min, y_max)

        ax.set_xlabel(feature_names[0], fontsize=11, fontweight='bold')
        ax.set_ylabel(feature_names[1], fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if depth_limit == 0:
            ax.legend(loc='upper right', fontsize=9)

    # Plot at different depths
    plot_at_depth(axes[0], 0, 'Profondeur 0\n(Racine - Pas de division)')

    for i in range(1, depths_to_show + 1):
        plot_at_depth(axes[i], i, f'Profondeur {i}\n({2**i if i < 3 else "+"} régions)')

    return fig


# 1. Simple 2D Dataset - Progressive Splits
print("\n1. Generating progressive splits on 2D dataset...")
np.random.seed(42)

X_simple = np.random.randn(150, 2)
y_simple = ((X_simple[:, 0] > 0) & (X_simple[:, 1] > 0)).astype(int)
y_simple[((X_simple[:, 0] < -0.5) & (X_simple[:, 1] < -0.5))] = 2

tree_simple = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_simple.fit(X_simple, y_simple)

fig = plot_step_by_step_splits(tree_simple, X_simple, y_simple,
                                feature_names=['Feature 1', 'Feature 2'],
                                class_names=['Classe 0', 'Classe 1', 'Classe 2'],
                                max_depth=3)

plt.suptitle('Décomposition Progressive de l\'Arbre - Étape par Étape\nChaque division crée deux nouvelles régions',
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('images/tree_decomposition_progressive.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/tree_decomposition_progressive.png")

# 2. Iris Dataset - First 3 Levels
print("\n2. Generating Iris dataset decomposition...")
np.random.seed(42)

iris = load_iris()
# Use only 2 features for visualization
X_iris_2d = iris.data[:, [0, 2]]  # sepal length, petal length
y_iris = iris.target

tree_iris = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_iris.fit(X_iris_2d, y_iris)

fig = plot_step_by_step_splits(tree_iris, X_iris_2d, y_iris,
                                feature_names=['Sepal Length', 'Petal Length'],
                                class_names=['Setosa', 'Versicolor', 'Virginica'],
                                max_depth=3)

plt.suptitle('Dataset Iris - Décomposition de l\'Arbre par Profondeur\nComment l\'arbre divise progressivement l\'espace',
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('images/tree_decomposition_iris.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/tree_decomposition_iris.png")

# 3. Moons Dataset - Complex Boundaries
print("\n3. Generating moons dataset decomposition...")
np.random.seed(42)

X_moons, y_moons = make_moons(n_samples=200, noise=0.2, random_state=42)

tree_moons = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_moons.fit(X_moons, y_moons)

fig = plot_step_by_step_splits(tree_moons, X_moons, y_moons,
                                feature_names=['X1', 'X2'],
                                class_names=['Lune 1', 'Lune 2'],
                                max_depth=3)

plt.suptitle('Dataset Moons - Décomposition pour Frontière Non-Linéaire\nLes arbres approximent les courbes avec des divisions rectangulaires',
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('images/tree_decomposition_moons.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/tree_decomposition_moons.png")

# 4. Detailed Step-by-Step with Annotations
print("\n4. Generating annotated step-by-step decomposition...")
np.random.seed(42)

# Simple dataset for clarity
X_ann = np.random.randn(100, 2)
y_ann = ((X_ann[:, 0] > 0) & (X_ann[:, 1] > 0)).astype(int)

tree_ann = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_ann.fit(X_ann, y_ann)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

colors = ['#FF6B6B', '#4ECDC4']

for depth_idx in range(3):
    ax = axes[depth_idx]

    h = 0.02
    x_min, x_max = X_ann[:, 0].min() - 0.5, X_ann[:, 0].max() + 0.5
    y_min, y_max = X_ann[:, 1].min() - 0.5, X_ann[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Create limited depth tree
    tree_limited = DecisionTreeClassifier(max_depth=depth_idx, random_state=42)
    tree_limited.fit(X_ann, y_ann)

    Z = tree_limited.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, colors=colors, levels=[-0.5, 0.5, 1.5])

    # Plot points
    ax.scatter(X_ann[y_ann == 0, 0], X_ann[y_ann == 0, 1],
              c='#FF6B6B', s=50, edgecolors='black', linewidths=1, alpha=0.7, label='Classe 0')
    ax.scatter(X_ann[y_ann == 1, 0], X_ann[y_ann == 1, 1],
              c='#4ECDC4', s=50, edgecolors='black', linewidths=1, alpha=0.7, label='Classe 1')

    # Draw splits with annotations
    def annotate_splits(node, x_min, x_max, y_min, y_max, current_depth=0, position=''):
        if node is None or node.value is not None or current_depth >= depth_idx:
            return

        feature = node.feature_index
        threshold = node.threshold

        if feature == 0:
            ax.plot([threshold, threshold], [y_min, y_max],
                   'k-', linewidth=3, alpha=0.8)

            # Add arrow annotation
            mid_y = (y_min + y_max) / 2
            ax.annotate(f'Division {current_depth + 1}:\nX1 ≤ {threshold:.2f}?',
                       xy=(threshold, mid_y), xytext=(threshold + 0.8, mid_y),
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))

            if node.left:
                annotate_splits(node.left, x_min, threshold, y_min, y_max,
                              current_depth + 1, 'L')
            if node.right:
                annotate_splits(node.right, threshold, x_max, y_min, y_max,
                              current_depth + 1, 'R')
        else:
            ax.plot([x_min, x_max], [threshold, threshold],
                   'k-', linewidth=3, alpha=0.8)

            mid_x = (x_min + x_max) / 2
            ax.annotate(f'Division {current_depth + 1}:\nX2 ≤ {threshold:.2f}?',
                       xy=(mid_x, threshold), xytext=(mid_x, threshold + 0.8),
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))

            if node.left:
                annotate_splits(node.left, x_min, x_max, y_min, threshold,
                              current_depth + 1, 'L')
            if node.right:
                annotate_splits(node.right, x_min, x_max, threshold, y_max,
                              current_depth + 1, 'R')

    annotate_splits(tree_limited.tree_, x_min, x_max, y_min, y_max)

    ax.set_xlabel('Feature 1 (X1)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Feature 2 (X2)', fontsize=11, fontweight='bold')

    if depth_idx == 0:
        title = 'Étape 1: Nœud Racine\n(Aucune division)'
        ax.legend(loc='upper right', fontsize=9)
    elif depth_idx == 1:
        title = 'Étape 2: Première Division\n(2 régions)'
    else:
        title = 'Étape 3: Divisions Suivantes\n(4 régions)'

    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('Décomposition Annotée de l\'Arbre - Processus de Division Progressif\nChaque division teste une condition et crée deux sous-régions',
            fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('images/tree_decomposition_annotated.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/tree_decomposition_annotated.png")

print("\n" + "=" * 70)
print("All tree decomposition visualizations generated successfully!")
print("Images saved in: 09_DECISION_TREE/images/")
print("=" * 70)
