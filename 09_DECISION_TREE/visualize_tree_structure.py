"""
Visualize Decision Tree Structure

Creates detailed visualizations showing how decision trees split and make decisions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import load_iris, make_moons
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decision_tree import DecisionTreeClassifier

# Create images directory
os.makedirs('images', exist_ok=True)

print("Generating Decision Tree structure visualizations...")
print("=" * 70)


def plot_tree_splits_2d(ax, tree, X, y, depth=0, xlim=None, ylim=None,
                        node=None, parent_box=None):
    """
    Recursively plot decision tree splits in 2D feature space
    """
    if node is None:
        node = tree.tree_

    if xlim is None:
        xlim = [X[:, 0].min() - 0.5, X[:, 0].max() + 0.5]
    if ylim is None:
        ylim = [X[:, 1].min() - 0.5, X[:, 1].max() + 0.5]

    if parent_box is None:
        parent_box = [xlim[0], xlim[1], ylim[0], ylim[1]]

    # If leaf node, color the region
    if node.value is not None:
        x_min, x_max, y_min, y_max = parent_box
        colors = ['#FFB6C1', '#ADD8E6', '#90EE90']  # Light colors for classes
        ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                   facecolor=colors[int(node.value)], alpha=0.3,
                                   edgecolor='black', linewidth=1))
        return

    # Draw split line
    feature = node.feature_index
    threshold = node.threshold

    x_min, x_max, y_min, y_max = parent_box

    if feature == 0:  # Split on x-axis
        ax.plot([threshold, threshold], [y_min, y_max], 'k-', linewidth=2)
        # Recurse on left and right
        if node.left:
            plot_tree_splits_2d(ax, tree, X, y, depth + 1, xlim, ylim,
                              node.left, [x_min, threshold, y_min, y_max])
        if node.right:
            plot_tree_splits_2d(ax, tree, X, y, depth + 1, xlim, ylim,
                              node.right, [threshold, x_max, y_min, y_max])
    else:  # Split on y-axis
        ax.plot([x_min, x_max], [threshold, threshold], 'k-', linewidth=2)
        # Recurse on left and right
        if node.left:
            plot_tree_splits_2d(ax, tree, X, y, depth + 1, xlim, ylim,
                              node.left, [x_min, x_max, y_min, threshold])
        if node.right:
            plot_tree_splits_2d(ax, tree, X, y, depth + 1, xlim, ylim,
                              node.right, [x_min, x_max, threshold, y_max])


def draw_tree_diagram(ax, node, x=0.5, y=1.0, dx=0.25, dy=0.15, depth=0, max_depth=None):
    """
    Draw a tree diagram showing the hierarchical structure
    """
    if node is None or (max_depth is not None and depth >= max_depth):
        return

    # Node properties
    if node.value is not None:
        # Leaf node
        color = '#90EE90'
        label = f'Class: {int(node.value)}\nn={node.n_samples}'
        shape = 'round'
    else:
        # Internal node
        color = '#ADD8E6'
        label = f'X[{node.feature_index}] <= {node.threshold:.2f}\ngini={node.impurity:.3f}\nn={node.n_samples}'
        shape = 'box'

    # Draw node
    if shape == 'box':
        bbox = dict(boxstyle='round,pad=0.5', facecolor=color, edgecolor='black', linewidth=2)
    else:
        bbox = dict(boxstyle='round,pad=0.3', facecolor=color, edgecolor='black', linewidth=2)

    ax.text(x, y, label, ha='center', va='center', fontsize=8,
           bbox=bbox, fontweight='bold')

    # Draw children
    if node.left is not None:
        # Left child
        child_x = x - dx
        child_y = y - dy

        # Draw edge
        ax.plot([x, child_x], [y - 0.03, child_y + 0.03], 'k-', linewidth=1.5)
        ax.text((x + child_x) / 2 - 0.02, (y + child_y) / 2, 'True',
               fontsize=7, color='green', fontweight='bold')

        draw_tree_diagram(ax, node.left, child_x, child_y, dx * 0.6, dy, depth + 1, max_depth)

    if node.right is not None:
        # Right child
        child_x = x + dx
        child_y = y - dy

        # Draw edge
        ax.plot([x, child_x], [y - 0.03, child_y + 0.03], 'k-', linewidth=1.5)
        ax.text((x + child_x) / 2 + 0.02, (y + child_y) / 2, 'False',
               fontsize=7, color='red', fontweight='bold')

        draw_tree_diagram(ax, node.right, child_x, child_y, dx * 0.6, dy, depth + 1, max_depth)


# 1. Simple Tree with Splits Visualization
print("\n1. Generating simple tree with splits...")
np.random.seed(42)

# Create simple 2D dataset
X_simple = np.random.randn(100, 2)
y_simple = ((X_simple[:, 0] > 0) & (X_simple[:, 1] > 0)).astype(int)

# Train shallow tree
tree_simple = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_simple.fit(X_simple, y_simple)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot data and splits
ax1.scatter(X_simple[y_simple == 0, 0], X_simple[y_simple == 0, 1],
           c='red', s=50, alpha=0.6, edgecolors='black', label='Class 0')
ax1.scatter(X_simple[y_simple == 1, 0], X_simple[y_simple == 1, 1],
           c='blue', s=50, alpha=0.6, edgecolors='black', label='Class 1')

# Draw splits
plot_tree_splits_2d(ax1, tree_simple, X_simple, y_simple)

ax1.set_xlabel('Feature 1', fontsize=12)
ax1.set_ylabel('Feature 2', fontsize=12)
ax1.set_title('Decision Tree Splits (max_depth=2)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Draw tree structure
ax2.axis('off')
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
draw_tree_diagram(ax2, tree_simple.tree_, max_depth=3)
ax2.set_title('Tree Structure', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('images/tree_structure_simple.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/tree_structure_simple.png")

# 2. Deep Tree Evolution
print("\n2. Generating deep tree evolution...")
np.random.seed(42)

X_moons, y_moons = make_moons(n_samples=200, noise=0.25, random_state=42)

depths = [1, 2, 3, 5]
fig, axes = plt.subplots(2, 4, figsize=(18, 10))

for idx, depth in enumerate(depths):
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_moons, y_moons)

    # Top row: splits
    ax = axes[0, idx]
    ax.scatter(X_moons[y_moons == 0, 0], X_moons[y_moons == 0, 1],
              c='red', s=30, alpha=0.6, edgecolors='black')
    ax.scatter(X_moons[y_moons == 1, 0], X_moons[y_moons == 1, 1],
              c='blue', s=30, alpha=0.6, edgecolors='black')

    # Draw decision boundary
    h = 0.02
    x_min, x_max = X_moons[:, 0].min() - 0.5, X_moons[:, 0].max() + 0.5
    y_min, y_max = X_moons[:, 1].min() - 0.5, X_moons[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')

    ax.set_xlabel('Feature 1', fontsize=10)
    ax.set_ylabel('Feature 2', fontsize=10)
    ax.set_title(f'Depth={depth}, Acc={tree.score(X_moons, y_moons):.3f}',
                fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Bottom row: tree structure
    ax = axes[1, idx]
    ax.axis('off')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    draw_tree_diagram(ax, tree.tree_, max_depth=depth)

plt.tight_layout()
plt.savefig('images/tree_evolution_depth.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/tree_evolution_depth.png")

# 3. Detailed Tree Structure (Iris dataset)
print("\n3. Generating detailed tree structure on Iris...")
np.random.seed(42)

iris = load_iris()
X_iris = iris.data[:, [0, 2]]  # Use sepal length and petal length
y_iris = iris.target

# Train tree with moderate depth
tree_iris = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_iris.fit(X_iris, y_iris)

fig = plt.figure(figsize=(16, 10))

# Create grid for tree diagram
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Top: Tree structure diagram
ax_tree = fig.add_subplot(gs[0:2, :])
ax_tree.axis('off')
ax_tree.set_xlim([0, 1])
ax_tree.set_ylim([0, 1])
draw_tree_diagram(ax_tree, tree_iris.tree_, x=0.5, y=0.95, dx=0.2, dy=0.2)
ax_tree.set_title('Decision Tree Structure (Iris Dataset)\nDepth=3, Features: Sepal Length & Petal Length',
                 fontsize=14, fontweight='bold', pad=20)

# Bottom left: Decision regions
ax_regions = fig.add_subplot(gs[2, 0])

h = 0.02
x_min, x_max = X_iris[:, 0].min() - 0.5, X_iris[:, 0].max() + 0.5
y_min, y_max = X_iris[:, 1].min() - 0.5, X_iris[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = tree_iris.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

colors = ['#FFB6C1', '#ADD8E6', '#90EE90']
ax_regions.contourf(xx, yy, Z, alpha=0.4, colors=colors)

# Plot points
for i, color in enumerate(['red', 'blue', 'green']):
    idx = y_iris == i
    ax_regions.scatter(X_iris[idx, 0], X_iris[idx, 1],
                      c=color, label=iris.target_names[i],
                      s=50, alpha=0.8, edgecolors='black')

ax_regions.set_xlabel('Sepal Length (cm)', fontsize=11)
ax_regions.set_ylabel('Petal Length (cm)', fontsize=11)
ax_regions.set_title('Decision Regions', fontsize=12, fontweight='bold')
ax_regions.legend(fontsize=10)
ax_regions.grid(True, alpha=0.3)

# Bottom right: Node statistics
ax_stats = fig.add_subplot(gs[2, 1])
ax_stats.axis('off')

def count_nodes(node, counts={'internal': 0, 'leaf': 0, 'max_depth': 0}, depth=0):
    if node is None:
        return counts

    counts['max_depth'] = max(counts['max_depth'], depth)

    if node.value is not None:
        counts['leaf'] += 1
    else:
        counts['internal'] += 1
        count_nodes(node.left, counts, depth + 1)
        count_nodes(node.right, counts, depth + 1)

    return counts

stats = count_nodes(tree_iris.tree_, {'internal': 0, 'leaf': 0, 'max_depth': 0})

stats_text = f"""
Tree Statistics:
━━━━━━━━━━━━━━━━━━
• Total Nodes: {stats['internal'] + stats['leaf']}
• Internal Nodes: {stats['internal']}
• Leaf Nodes: {stats['leaf']}
• Max Depth: {stats['max_depth']}
• Training Accuracy: {tree_iris.score(X_iris, y_iris):.3f}

Split Information:
━━━━━━━━━━━━━━━━━━
Each internal node tests:
  X[feature] <= threshold

Left branch: condition True
Right branch: condition False

Leaf nodes contain:
  • Final class prediction
  • Number of samples
"""

ax_stats.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig('images/tree_structure_detailed.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/tree_structure_detailed.png")

# 4. Split Progression Animation Frames
print("\n4. Generating split progression frames...")
np.random.seed(42)

X_prog = np.random.randn(80, 2)
y_prog = ((X_prog[:, 0] > 0) & (X_prog[:, 1] > 0)).astype(int)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, depth in enumerate([0, 1, 2, 3, 4, 5]):
    ax = axes[idx]

    if depth == 0:
        # Show initial data
        ax.scatter(X_prog[y_prog == 0, 0], X_prog[y_prog == 0, 1],
                  c='red', s=50, alpha=0.6, edgecolors='black', label='Class 0')
        ax.scatter(X_prog[y_prog == 1, 0], X_prog[y_prog == 1, 1],
                  c='blue', s=50, alpha=0.6, edgecolors='black', label='Class 1')
        ax.set_title('Initial Data\n(No splits)', fontsize=12, fontweight='bold')
        ax.legend()
    else:
        # Train tree to this depth
        tree_prog = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree_prog.fit(X_prog, y_prog)

        # Plot regions
        h = 0.05
        x_min, x_max = X_prog[:, 0].min() - 0.5, X_prog[:, 0].max() + 0.5
        y_min, y_max = X_prog[:, 1].min() - 0.5, X_prog[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

        Z = tree_prog.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, colors=['#FFB6C1', '#ADD8E6'])
        ax.scatter(X_prog[y_prog == 0, 0], X_prog[y_prog == 0, 1],
                  c='red', s=50, alpha=0.8, edgecolors='black')
        ax.scatter(X_prog[y_prog == 1, 0], X_prog[y_prog == 1, 1],
                  c='blue', s=50, alpha=0.8, edgecolors='black')

        # Count nodes
        stats = count_nodes(tree_prog.tree_, {'internal': 0, 'leaf': 0, 'max_depth': 0})

        ax.set_title(f'Depth={depth}\n{stats["leaf"]} regions, Acc={tree_prog.score(X_prog, y_prog):.3f}',
                    fontsize=12, fontweight='bold')

    ax.set_xlabel('Feature 1', fontsize=10)
    ax.set_ylabel('Feature 2', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('Progressive Tree Growth: How Splits Create Decision Regions',
            fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('images/tree_split_progression.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/tree_split_progression.png")

print("\n" + "=" * 70)
print("All Decision Tree structure visualizations generated successfully!")
print("Images saved in: 09_DECISION_TREE/images/")
print("=" * 70)
