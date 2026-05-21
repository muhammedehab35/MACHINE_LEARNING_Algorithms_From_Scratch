"""
Draw Beautiful Random Forest Tree Diagrams

Creates hierarchical tree diagrams showing individual trees within the forest
and how they differ due to bootstrap sampling and feature randomness.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.datasets import load_iris
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from random_forest import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '09_DECISION_TREE'))

# Create images directory
os.makedirs('images', exist_ok=True)

print("Generating Random Forest tree diagrams...")
print("=" * 70)


class TreeDiagramDrawer:
    """Draw beautiful tree diagrams"""

    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.axis('off')

    def draw_node(self, x, y, text, is_leaf=False, color=None):
        """Draw a single node"""
        if color is None:
            color = '#E8F5E9' if is_leaf else '#E3F2FD'

        # Node box
        width = 1.6
        height = 0.7

        if is_leaf:
            # Rounded rectangle for leaf
            box = FancyBboxPatch(
                (x - width/2, y - height/2), width, height,
                boxstyle="round,pad=0.1",
                facecolor=color,
                edgecolor='#2E7D32' if is_leaf else '#1976D2',
                linewidth=2
            )
        else:
            # Ellipse for decision node
            box = mpatches.Ellipse(
                (x, y), width, height,
                facecolor=color,
                edgecolor='#1976D2',
                linewidth=2
            )

        self.ax.add_patch(box)

        # Text
        self.ax.text(x, y, text, ha='center', va='center',
                    fontsize=7, fontweight='bold',
                    color='#1B5E20' if is_leaf else '#0D47A1')

    def draw_edge(self, x1, y1, x2, y2, label='', label_color='black'):
        """Draw edge between nodes"""
        # Arrow
        arrow = FancyArrowPatch(
            (x1, y1 - 0.35), (x2, y2 + 0.35),
            arrowstyle='->,head_width=0.3,head_length=0.3',
            color='#424242',
            linewidth=1.5,
            connectionstyle="arc3,rad=0"
        )
        self.ax.add_patch(arrow)

        # Label
        if label:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2

            # Background for label
            bbox = dict(boxstyle='round,pad=0.2',
                       facecolor='white',
                       edgecolor=label_color,
                       linewidth=1)

            self.ax.text(mid_x, mid_y, label,
                        ha='center', va='center',
                        fontsize=6, fontweight='bold',
                        color=label_color,
                        bbox=bbox)


def draw_tree_recursive(drawer, node, x, y, dx, dy, depth=0, max_depth=None,
                       feature_names=None, class_names=None):
    """Recursively draw tree structure"""
    if node is None or (max_depth is not None and depth >= max_depth):
        return

    # Format node text
    if node.value is not None:
        # Leaf node
        if class_names is not None:
            class_name = class_names[int(node.value)]
        else:
            class_name = f"C{int(node.value)}"

        text = f"{class_name}\nn={node.n_samples}"
        drawer.draw_node(x, y, text, is_leaf=True)
        return
    else:
        # Decision node
        if feature_names is not None:
            feature_name = feature_names[node.feature_index]
        else:
            feature_name = f"X{node.feature_index}"

        text = f"{feature_name}≤{node.threshold:.1f}\nn={node.n_samples}"
        drawer.draw_node(x, y, text, is_leaf=False)

    # Draw children
    if node.left is not None:
        child_x = x - dx
        child_y = y - dy
        drawer.draw_edge(x, y, child_x, child_y, 'Y', '#2E7D32')
        draw_tree_recursive(drawer, node.left, child_x, child_y, dx * 0.5, dy,
                          depth + 1, max_depth, feature_names, class_names)

    if node.right is not None:
        child_x = x + dx
        child_y = y - dy
        drawer.draw_edge(x, y, child_x, child_y, 'N', '#C62828')
        draw_tree_recursive(drawer, node.right, child_x, child_y, dx * 0.5, dy,
                          depth + 1, max_depth, feature_names, class_names)


# 1. Individual Trees in Forest
print("\n1. Generating individual tree diagrams from forest...")
np.random.seed(42)

iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# Train forest with few trees
rf = RandomForestClassifier(n_estimators=4, max_depth=3, random_state=42)
rf.fit(X_iris, y_iris)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

feature_names_short = ['SL', 'SW', 'PL', 'PW']
class_names = ['Set', 'Ver', 'Vir']

for idx in range(4):
    drawer = TreeDiagramDrawer(fig, axes[idx])
    tree = rf.trees_[idx]

    draw_tree_recursive(drawer, tree.tree_, x=5, y=8.5, dx=1.5, dy=2,
                       max_depth=3,
                       feature_names=feature_names_short,
                       class_names=class_names)

    axes[idx].set_title(f'Tree {idx + 1} - Random Forest\n(Different bootstrap sample)',
                       fontsize=12, fontweight='bold')

plt.suptitle('Individual Trees in Random Forest\nEach tree sees different data and uses random feature subsets',
            fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('images/forest_individual_tree_diagrams.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/forest_individual_tree_diagrams.png")

# 2. Comparing Trees with Different Bootstrap Samples
print("\n2. Generating bootstrap effect comparison...")
np.random.seed(42)

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

# Train 3 separate forests to show variability
for idx in range(3):
    rf_single = RandomForestClassifier(n_estimators=1, max_depth=3, random_state=42+idx)
    rf_single.fit(X_iris, y_iris)

    drawer = TreeDiagramDrawer(fig, axes[idx])
    tree = rf_single.trees_[0]

    draw_tree_recursive(drawer, tree.tree_, x=5, y=8.5, dx=1.8, dy=2,
                       feature_names=feature_names_short,
                       class_names=class_names)

    axes[idx].set_title(f'Bootstrap Sample {idx + 1}\n(Different random seed)',
                       fontsize=12, fontweight='bold')

plt.suptitle('Effect of Bootstrap Sampling on Tree Structure\nSame dataset, different bootstrap samples produce different trees',
            fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('images/forest_bootstrap_tree_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/forest_bootstrap_tree_comparison.png")

# 3. Simple Dataset - Forest Trees
print("\n3. Generating simple dataset forest trees...")
np.random.seed(42)

# Create simple 2D dataset
X_simple = np.random.randn(100, 2)
y_simple = ((X_simple[:, 0] + X_simple[:, 1]) > 0).astype(int)

rf_simple = RandomForestClassifier(n_estimators=6, max_depth=2, random_state=42)
rf_simple.fit(X_simple, y_simple)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx in range(6):
    drawer = TreeDiagramDrawer(fig, axes[idx])
    tree = rf_simple.trees_[idx]

    draw_tree_recursive(drawer, tree.tree_, x=5, y=8.5, dx=2, dy=2.5,
                       feature_names=['X1', 'X2'],
                       class_names=['0', '1'])

    axes[idx].set_title(f'Tree {idx + 1}',
                       fontsize=11, fontweight='bold')

plt.suptitle('Random Forest with 6 Trees (max_depth=2)\nSimple binary classification - each tree contributes one vote',
            fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('images/forest_simple_trees.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/forest_simple_trees.png")

# 4. Annotated Forest Diagram
print("\n4. Generating annotated forest diagram...")
np.random.seed(42)

X_ann = np.random.randn(80, 2)
y_ann = ((X_ann[:, 0] + X_ann[:, 1]) > 0).astype(int)

rf_ann = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=42)
rf_ann.fit(X_ann, y_ann)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

for idx in range(3):
    drawer = TreeDiagramDrawer(fig, axes[idx])
    tree = rf_ann.trees_[idx]

    draw_tree_recursive(drawer, tree.tree_, x=5, y=8, dx=2, dy=2.5,
                       feature_names=['A', 'B'],
                       class_names=['-', '+'])

    axes[idx].set_title(f'Tree {idx + 1}',
                       fontsize=11, fontweight='bold',
                       color='#1976D2')

# Add forest annotation
fig.text(0.5, 0.92, 'Random Forest = Ensemble of Decision Trees',
        ha='center', fontsize=14, fontweight='bold')
fig.text(0.5, 0.88, 'Each tree votes → Majority vote determines final prediction',
        ha='center', fontsize=11, style='italic')

# Add legend
legend_elements = [
    mpatches.Patch(facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2,
                  label='Decision node (split condition)'),
    mpatches.Patch(facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2,
                  label='Leaf node (predicted class)'),
    mpatches.Patch(facecolor='white', edgecolor='#2E7D32', linewidth=1.5,
                  label='Branch "Yes" (≤ threshold)'),
    mpatches.Patch(facecolor='white', edgecolor='#C62828', linewidth=1.5,
                  label='Branch "No" (> threshold)'),
]

fig.legend(handles=legend_elements, loc='lower center',
          ncol=2, fontsize=9, frameon=True,
          bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig('images/forest_annotated_diagram.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/forest_annotated_diagram.png")

print("\n" + "=" * 70)
print("All Random Forest tree diagrams generated successfully!")
print("Images saved in: 10_RANDOM_FOREST/images/")
print("=" * 70)
