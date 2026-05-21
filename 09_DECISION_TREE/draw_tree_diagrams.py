"""
Draw Beautiful Decision Tree Diagrams

Creates clean, professional tree diagrams showing the hierarchical structure
similar to academic papers and textbooks.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.datasets import load_iris
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decision_tree import DecisionTreeClassifier

# Create images directory
os.makedirs('images', exist_ok=True)

print("Generating beautiful decision tree diagrams...")
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
        width = 1.8
        height = 0.8

        if is_leaf:
            # Rounded rectangle for leaf
            box = FancyBboxPatch(
                (x - width/2, y - height/2), width, height,
                boxstyle="round,pad=0.1",
                facecolor=color,
                edgecolor='#2E7D32' if is_leaf else '#1976D2',
                linewidth=2.5
            )
        else:
            # Ellipse for decision node
            box = mpatches.Ellipse(
                (x, y), width, height,
                facecolor=color,
                edgecolor='#1976D2',
                linewidth=2.5
            )

        self.ax.add_patch(box)

        # Text
        self.ax.text(x, y, text, ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='#1B5E20' if is_leaf else '#0D47A1')

    def draw_edge(self, x1, y1, x2, y2, label='', label_color='black'):
        """Draw edge between nodes"""
        # Arrow
        arrow = FancyArrowPatch(
            (x1, y1 - 0.4), (x2, y2 + 0.4),
            arrowstyle='->,head_width=0.4,head_length=0.4',
            color='#424242',
            linewidth=2,
            connectionstyle="arc3,rad=0"
        )
        self.ax.add_patch(arrow)

        # Label
        if label:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2

            # Background for label
            bbox = dict(boxstyle='round,pad=0.3',
                       facecolor='white',
                       edgecolor=label_color,
                       linewidth=1.5)

            self.ax.text(mid_x, mid_y, label,
                        ha='center', va='center',
                        fontsize=8, fontweight='bold',
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
            class_name = f"Class {int(node.value)}"

        text = f"{class_name}\nn={node.n_samples}"
        drawer.draw_node(x, y, text, is_leaf=True)
        return
    else:
        # Decision node
        if feature_names is not None:
            feature_name = feature_names[node.feature_index]
        else:
            feature_name = f"X[{node.feature_index}]"

        text = f"{feature_name} ≤ {node.threshold:.2f}\ngini={node.impurity:.3f}\nn={node.n_samples}"
        drawer.draw_node(x, y, text, is_leaf=False)

    # Draw children
    if node.left is not None:
        child_x = x - dx
        child_y = y - dy
        drawer.draw_edge(x, y, child_x, child_y, 'Oui', '#2E7D32')
        draw_tree_recursive(drawer, node.left, child_x, child_y, dx * 0.5, dy,
                          depth + 1, max_depth, feature_names, class_names)

    if node.right is not None:
        child_x = x + dx
        child_y = y - dy
        drawer.draw_edge(x, y, child_x, child_y, 'Non', '#C62828')
        draw_tree_recursive(drawer, node.right, child_x, child_y, dx * 0.5, dy,
                          depth + 1, max_depth, feature_names, class_names)


# 1. Simple Binary Tree
print("\n1. Generating simple binary tree diagram...")
np.random.seed(42)

# Create simple dataset
X_simple = np.random.randn(100, 2)
y_simple = ((X_simple[:, 0] > 0) & (X_simple[:, 1] > 0)).astype(int)

tree_simple = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_simple.fit(X_simple, y_simple)

fig, ax = plt.subplots(figsize=(12, 8))
drawer = TreeDiagramDrawer(fig, ax)

draw_tree_recursive(drawer, tree_simple.tree_, x=5, y=8.5, dx=2, dy=2.5,
                   feature_names=['Feature 1', 'Feature 2'],
                   class_names=['Non', 'Oui'])

ax.set_title('Arbre de Décision Simple (Profondeur Max = 2)',
            fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('images/tree_diagram_simple.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/tree_diagram_simple.png")

# 2. Iris Dataset Tree
print("\n2. Generating Iris dataset tree diagram...")
np.random.seed(42)

iris = load_iris()
X_iris = iris.data
y_iris = iris.target

tree_iris = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_iris.fit(X_iris, y_iris)

fig, ax = plt.subplots(figsize=(16, 10))
drawer = TreeDiagramDrawer(fig, ax)

draw_tree_recursive(drawer, tree_iris.tree_, x=5, y=9, dx=1.5, dy=2,
                   feature_names=iris.feature_names,
                   class_names=iris.target_names)

ax.set_title('Arbre de Décision - Dataset Iris (Profondeur Max = 3)',
            fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('images/tree_diagram_iris.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/tree_diagram_iris.png")

# 3. Shallow vs Deep Tree Comparison
print("\n3. Generating shallow vs deep tree comparison...")
np.random.seed(42)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Shallow tree
tree_shallow = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_shallow.fit(X_iris, y_iris)

drawer_shallow = TreeDiagramDrawer(fig, axes[0])
draw_tree_recursive(drawer_shallow, tree_shallow.tree_, x=5, y=8.5, dx=2, dy=2.5,
                   feature_names=['SL', 'SW', 'PL', 'PW'],
                   class_names=['Setosa', 'Versicolor', 'Virginica'])
axes[0].set_title('Arbre Peu Profond (depth=2)\nPlus simple, moins précis',
                 fontsize=14, fontweight='bold')

# Deep tree
tree_deep = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_deep.fit(X_iris, y_iris)

drawer_deep = TreeDiagramDrawer(fig, axes[1])
draw_tree_recursive(drawer_deep, tree_deep.tree_, x=5, y=8.5, dx=1.2, dy=1.8,
                   max_depth=3,  # Show only first 3 levels for clarity
                   feature_names=['SL', 'SW', 'PL', 'PW'],
                   class_names=['Setosa', 'Versicolor', 'Virginica'])
axes[1].set_title('Arbre Profond (depth=4, montré=3)\nPlus complexe, plus précis',
                 fontsize=14, fontweight='bold')

plt.suptitle('Comparaison: Arbres Peu Profonds vs Profonds',
            fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('images/tree_diagram_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/tree_diagram_comparison.png")

# 4. Annotated Tree with Explanations
print("\n4. Generating annotated tree with explanations...")
np.random.seed(42)

# Simple tree for annotation
X_ann = np.random.randn(80, 2)
y_ann = ((X_ann[:, 0] + X_ann[:, 1]) > 0).astype(int)

tree_ann = DecisionTreeClassifier(max_depth=2, criterion='gini', random_state=42)
tree_ann.fit(X_ann, y_ann)

fig = plt.figure(figsize=(14, 10))
ax = plt.subplot(111)
drawer = TreeDiagramDrawer(fig, ax)

draw_tree_recursive(drawer, tree_ann.tree_, x=5, y=8, dx=2, dy=2.5,
                   feature_names=['Caractéristique A', 'Caractéristique B'],
                   class_names=['Négatif', 'Positif'])

# Add annotations
annotations = [
    {'xy': (5, 9.3), 'text': 'Nœud racine\n(teste toutes les données)',
     'xytext': (2, 9.8), 'color': '#1976D2'},

    {'xy': (3, 5.5), 'text': 'Nœud de décision\n(subdivision des données)',
     'xytext': (0.5, 4.5), 'color': '#1976D2'},

    {'xy': (7, 3), 'text': 'Nœud feuille\n(décision finale)',
     'xytext': (8.5, 1.5), 'color': '#2E7D32'},
]

for ann in annotations:
    ax.annotate(ann['text'],
               xy=ann['xy'], xytext=ann['xytext'],
               fontsize=10, fontweight='bold',
               color=ann['color'],
               bbox=dict(boxstyle='round,pad=0.5',
                        facecolor='lightyellow',
                        edgecolor=ann['color'],
                        linewidth=2),
               arrowprops=dict(arrowstyle='->',
                             color=ann['color'],
                             lw=2,
                             connectionstyle='arc3,rad=0.3'))

# Add legend
legend_elements = [
    mpatches.Patch(facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2,
                  label='Nœud de décision (test condition)'),
    mpatches.Patch(facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2,
                  label='Nœud feuille (classe prédite)'),
    mpatches.Patch(facecolor='white', edgecolor='#2E7D32', linewidth=1.5,
                  label='Branche "Oui" (condition vraie)'),
    mpatches.Patch(facecolor='white', edgecolor='#C62828', linewidth=1.5,
                  label='Branche "Non" (condition fausse)'),
]

ax.legend(handles=legend_elements, loc='lower center',
         ncol=2, fontsize=9, frameon=True,
         bbox_to_anchor=(0.5, -0.05))

ax.set_title('Arbre de Décision Annoté - Structure et Composants',
            fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('images/tree_diagram_annotated.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/tree_diagram_annotated.png")

print("\n" + "=" * 70)
print("All beautiful tree diagrams generated successfully!")
print("Images saved in: 09_DECISION_TREE/images/")
print("=" * 70)
