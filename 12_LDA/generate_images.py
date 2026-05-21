"""
Generate Comprehensive Visualizations for Linear Discriminant Analysis

This script creates visualizations demonstrating:
1. LDA vs PCA comparison
2. Decision boundaries
3. Dimensionality reduction on Iris
4. Scatter matrices visualization
5. Effect of regularization (shrinkage)
6. Discriminant directions
7. Class separation
8. Multi-class classification
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from lda import LinearDiscriminantAnalysis

# Create images directory
os.makedirs('images', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

print("Generating LDA visualizations...")

# ============================================================================
# 1. LDA vs PCA Comparison
# ============================================================================
print("1. Generating LDA vs PCA comparison...")

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

X_iris, y_iris = load_iris(return_X_y=True)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_iris)

# Apply LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_iris, y_iris)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = ['red', 'green', 'blue']
class_names = ['Setosa', 'Versicolor', 'Virginica']

# PCA plot
ax = axes[0]
for i, color, name in zip([0, 1, 2], colors, class_names):
    ax.scatter(X_pca[y_iris == i, 0], X_pca[y_iris == i, 1],
               label=name, color=color, alpha=0.6, edgecolors='k', s=50)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=11)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=11)
ax.set_title('PCA: Unsupervised (Maximizes Variance)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# LDA plot
ax = axes[1]
for i, color, name in zip([0, 1, 2], colors, class_names):
    ax.scatter(X_lda[y_iris == i, 0], X_lda[y_iris == i, 1],
               label=name, color=color, alpha=0.6, edgecolors='k', s=50)
ax.set_xlabel(f'LD1 ({lda.explained_variance_ratio_[0]:.1%} separation)', fontsize=11)
ax.set_ylabel(f'LD2 ({lda.explained_variance_ratio_[1]:.1%} separation)', fontsize=11)
ax.set_title('LDA: Supervised (Maximizes Class Separation)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/lda_vs_pca.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 2. Decision Boundaries (2D)
# ============================================================================
print("2. Generating decision boundaries...")

# Generate 2D data
np.random.seed(42)
X1 = np.random.randn(80, 2) + np.array([2, 2])
X2 = np.random.randn(80, 2) + np.array([-2, -2])
X3 = np.random.randn(80, 2) + np.array([2, -2])
X_2d = np.vstack([X1, X2, X3])
y_2d = np.array([0] * 80 + [1] * 80 + [2] * 80)

# Train LDA
lda_2d = LinearDiscriminantAnalysis()
lda_2d.fit(X_2d, y_2d)

# Create meshgrid for decision boundary
h = 0.1
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict on meshgrid
Z = lda_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

# Decision boundary
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

# Scatter plot
colors = ['red', 'green', 'blue']
for i, color in enumerate(colors):
    ax.scatter(X_2d[y_2d == i, 0], X_2d[y_2d == i, 1],
               label=f'Class {i}', color=color, alpha=0.7, edgecolors='k', s=60)

# Class means
for i, color in enumerate(colors):
    mean = lda_2d.means_[i]
    ax.scatter(mean[0], mean[1], marker='X', s=300,
               edgecolors='black', linewidths=2, color=color, label=f'Mean {i}')

ax.set_xlabel('Feature 1', fontsize=12)
ax.set_ylabel('Feature 2', fontsize=12)
ax.set_title('LDA Decision Boundaries (Linear Separators)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig('images/decision_boundaries.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 3. Iris Dimensionality Reduction (4D → 2D)
# ============================================================================
print("3. Generating Iris dimensionality reduction...")

fig, ax = plt.subplots(figsize=(10, 8))

colors = ['red', 'green', 'blue']
class_names = ['Setosa', 'Versicolor', 'Virginica']

for i, color, name in zip([0, 1, 2], colors, class_names):
    ax.scatter(X_lda[y_iris == i, 0], X_lda[y_iris == i, 1],
               label=name, color=color, alpha=0.6, edgecolors='k', s=80)

ax.set_xlabel(f'LD1 (Discriminant 1) - {lda.explained_variance_ratio_[0]:.1%} separation',
              fontsize=12)
ax.set_ylabel(f'LD2 (Discriminant 2) - {lda.explained_variance_ratio_[1]:.1%} separation',
              fontsize=12)
ax.set_title('Iris Dataset: 4D → 2D via LDA', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Add text box with info
textstr = f'Original: 4 features\nReduced: 2 discriminants\nTotal separation: {lda.explained_variance_ratio_.sum():.1%}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.savefig('images/iris_projection.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 4. Scatter Matrices Visualization
# ============================================================================
print("4. Generating scatter matrices visualization...")

# Simple 2D example for visualization
np.random.seed(42)
X_scatter = np.vstack([
    np.random.randn(30, 2) + [3, 3],
    np.random.randn(30, 2) + [-3, -3]
])
y_scatter = np.array([0] * 30 + [1] * 30)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Overall data
ax = axes[0]
ax.scatter(X_scatter[y_scatter == 0, 0], X_scatter[y_scatter == 0, 1],
           label='Class 0', color='red', alpha=0.6, edgecolors='k', s=60)
ax.scatter(X_scatter[y_scatter == 1, 0], X_scatter[y_scatter == 1, 1],
           label='Class 1', color='blue', alpha=0.6, edgecolors='k', s=60)
overall_mean = X_scatter.mean(axis=0)
ax.scatter(overall_mean[0], overall_mean[1], marker='*', s=500,
           color='yellow', edgecolors='black', linewidths=2, label='Overall Mean')
ax.set_title('Original Data', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')

# Within-class scatter
ax = axes[1]
mean_0 = X_scatter[y_scatter == 0].mean(axis=0)
mean_1 = X_scatter[y_scatter == 1].mean(axis=0)

ax.scatter(X_scatter[y_scatter == 0, 0], X_scatter[y_scatter == 0, 1],
           label='Class 0', color='red', alpha=0.6, edgecolors='k', s=60)
ax.scatter(X_scatter[y_scatter == 1, 0], X_scatter[y_scatter == 1, 1],
           label='Class 1', color='blue', alpha=0.6, edgecolors='k', s=60)
ax.scatter(mean_0[0], mean_0[1], marker='X', s=300,
           color='red', edgecolors='black', linewidths=2, label='Mean 0')
ax.scatter(mean_1[0], mean_1[1], marker='X', s=300,
           color='blue', edgecolors='black', linewidths=2, label='Mean 1')

# Draw ellipses for within-class variance
from matplotlib.patches import Ellipse

for class_idx, color, mean in zip([0, 1], ['red', 'blue'], [mean_0, mean_1]):
    class_data = X_scatter[y_scatter == class_idx]
    cov = np.cov(class_data.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * np.sqrt(eigenvalues)
    ellipse = Ellipse(mean, width, height, angle=angle, alpha=0.2, color=color)
    ax.add_patch(ellipse)

ax.set_title('Within-Class Scatter $S_W$', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')

# Between-class scatter
ax = axes[2]
ax.scatter(mean_0[0], mean_0[1], marker='X', s=400,
           color='red', edgecolors='black', linewidths=2, label='Mean 0')
ax.scatter(mean_1[0], mean_1[1], marker='X', s=400,
           color='blue', edgecolors='black', linewidths=2, label='Mean 1')
ax.scatter(overall_mean[0], overall_mean[1], marker='*', s=500,
           color='yellow', edgecolors='black', linewidths=2, label='Overall Mean')

# Draw arrow between means
ax.annotate('', xy=mean_1, xytext=mean_0,
            arrowprops=dict(arrowstyle='<->', lw=3, color='purple'))

# Draw distances to overall mean
ax.plot([mean_0[0], overall_mean[0]], [mean_0[1], overall_mean[1]],
        'r--', linewidth=2, alpha=0.7, label='Distance to overall mean')
ax.plot([mean_1[0], overall_mean[0]], [mean_1[1], overall_mean[1]],
        'b--', linewidth=2, alpha=0.7)

ax.set_title('Between-Class Scatter $S_B$', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.savefig('images/scatter_matrices.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 5. Effect of Shrinkage (Regularization)
# ============================================================================
print("5. Generating shrinkage effect...")

# High-dimensional data with small sample
np.random.seed(42)
n_samples, n_features = 60, 50
X_high = np.random.randn(n_samples, n_features)
y_high = np.random.randint(0, 3, size=n_samples)

# Add some structure
for i in range(3):
    X_high[y_high == i] += np.random.randn(n_features) * (i + 1)

shrinkage_values = [0.0, 0.3, 0.6, 0.9]
accuracies = []

for shrink in shrinkage_values:
    lda_shrink = LinearDiscriminantAnalysis(shrinkage=shrink, solver='eigen')
    lda_shrink.fit(X_high, y_high)
    acc = lda_shrink.score(X_high, y_high)
    accuracies.append(acc)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(shrinkage_values, accuracies, marker='o', markersize=10,
        linewidth=2, color='purple')
ax.set_xlabel('Shrinkage Parameter', fontsize=12)
ax.set_ylabel('Training Accuracy', fontsize=12)
ax.set_title('Effect of Shrinkage Regularization\n(High dimensions, small sample)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xticks(shrinkage_values)

# Add text annotations
for shrink, acc in zip(shrinkage_values, accuracies):
    ax.text(shrink, acc + 0.01, f'{acc:.3f}', ha='center', fontsize=10)

textstr = f'n_samples = {n_samples}\nn_features = {n_features}\nClasses = 3'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.savefig('images/shrinkage_effect.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 6. Discriminant Directions
# ============================================================================
print("6. Generating discriminant directions...")

# 2D data for visualization
np.random.seed(42)
X_dir = np.vstack([
    np.random.randn(50, 2) * [1.5, 0.5] + [2, 2],
    np.random.randn(50, 2) * [1.5, 0.5] + [-2, -2]
])
y_dir = np.array([0] * 50 + [1] * 50)

lda_dir = LinearDiscriminantAnalysis(n_components=1)
lda_dir.fit(X_dir, y_dir)

fig, ax = plt.subplots(figsize=(10, 8))

# Scatter plot
ax.scatter(X_dir[y_dir == 0, 0], X_dir[y_dir == 0, 1],
           label='Class 0', color='red', alpha=0.6, edgecolors='k', s=60)
ax.scatter(X_dir[y_dir == 1, 0], X_dir[y_dir == 1, 1],
           label='Class 1', color='blue', alpha=0.6, edgecolors='k', s=60)

# Discriminant direction (scaling vector)
discriminant = lda_dir.scalings_[:, 0]
origin = X_dir.mean(axis=0)

# Draw discriminant direction
scale = 5
ax.arrow(origin[0], origin[1],
         discriminant[0] * scale, discriminant[1] * scale,
         head_width=0.4, head_length=0.3, fc='green', ec='green',
         linewidth=3, label='Discriminant Direction', alpha=0.8)

# Draw perpendicular (decision boundary)
perpendicular = np.array([-discriminant[1], discriminant[0]])
boundary_points = np.array([origin - perpendicular * 5, origin + perpendicular * 5])
ax.plot(boundary_points[:, 0], boundary_points[:, 1], 'k--',
        linewidth=2, label='Decision Boundary', alpha=0.7)

ax.set_xlabel('Feature 1', fontsize=12)
ax.set_ylabel('Feature 2', fontsize=12)
ax.set_title('LDA Discriminant Direction\n(Maximizes Class Separation)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axis('equal')

plt.savefig('images/discriminant_directions.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 7. Class Separation Metrics
# ============================================================================
print("7. Generating class separation metrics...")

X_iris, y_iris = load_iris(return_X_y=True)

# Train LDA
lda_metrics = LinearDiscriminantAnalysis(n_components=2)
X_lda_metrics = lda_metrics.fit_transform(X_iris, y_iris)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Explained variance ratio
ax = axes[0, 0]
components = np.arange(1, len(lda_metrics.explained_variance_ratio_) + 1)
ax.bar(components, lda_metrics.explained_variance_ratio_, color='steelblue', edgecolor='black')
ax.set_xlabel('Linear Discriminant', fontsize=11)
ax.set_ylabel('Explained Variance Ratio', fontsize=11)
ax.set_title('Explained Variance by Each Discriminant', fontsize=13, fontweight='bold')
ax.set_xticks(components)
ax.grid(True, alpha=0.3, axis='y')

# Cumulative explained variance
ax = axes[0, 1]
cumsum = np.cumsum(lda_metrics.explained_variance_ratio_)
ax.plot(components, cumsum, marker='o', markersize=10, linewidth=2, color='purple')
ax.set_xlabel('Number of Discriminants', fontsize=11)
ax.set_ylabel('Cumulative Explained Variance', fontsize=11)
ax.set_title('Cumulative Class Separation', fontsize=13, fontweight='bold')
ax.set_xticks(components)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

# Class means in original space
ax = axes[1, 0]
class_means = lda_metrics.means_
feature_idx = [0, 1]  # First two features
for i, color, name in zip([0, 1, 2], ['red', 'green', 'blue'],
                          ['Setosa', 'Versicolor', 'Virginica']):
    ax.scatter(class_means[i, feature_idx[0]], class_means[i, feature_idx[1]],
               s=300, marker='X', edgecolors='black', linewidths=2,
               color=color, label=name)
ax.set_xlabel(f'Feature {feature_idx[0]} (sepal length)', fontsize=11)
ax.set_ylabel(f'Feature {feature_idx[1]} (sepal width)', fontsize=11)
ax.set_title('Class Means in Original Space', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Class means in LDA space
ax = axes[1, 1]
X_means_lda = lda_metrics.transform(class_means)
for i, color, name in zip([0, 1, 2], ['red', 'green', 'blue'],
                          ['Setosa', 'Versicolor', 'Virginica']):
    ax.scatter(X_means_lda[i, 0], X_means_lda[i, 1],
               s=300, marker='X', edgecolors='black', linewidths=2,
               color=color, label=name)
ax.set_xlabel('LD1', fontsize=11)
ax.set_ylabel('LD2', fontsize=11)
ax.set_title('Class Means in LDA Space (Maximally Separated)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/class_separation.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 8. Number of Components Effect
# ============================================================================
print("8. Generating n_components effect...")

from sklearn.model_selection import train_test_split

X_iris, y_iris = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris,
                                                      test_size=0.3, random_state=42)

n_components_list = [1, 2]
train_accs = []
test_accs = []

for n_comp in n_components_list:
    lda_comp = LinearDiscriminantAnalysis(n_components=n_comp)
    lda_comp.fit(X_train, y_train)
    train_acc = lda_comp.score(X_train, y_train)
    test_acc = lda_comp.score(X_test, y_test)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

fig, ax = plt.subplots(figsize=(10, 6))

x_pos = np.arange(len(n_components_list))
width = 0.35

bars1 = ax.bar(x_pos - width/2, train_accs, width, label='Train Accuracy',
               color='steelblue', edgecolor='black')
bars2 = ax.bar(x_pos + width/2, test_accs, width, label='Test Accuracy',
               color='orange', edgecolor='black')

ax.set_xlabel('Number of Components', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Effect of Number of LDA Components (Iris Dataset)',
             fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(n_components_list)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0.8, 1.05])

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

textstr = 'Max components = K - 1 = 2\nfor K=3 classes'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.7, 0.15, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.savefig('images/n_components_effect.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll visualizations generated successfully!")
print("Images saved in: images/")
print("\nGenerated files:")
print("  1. lda_vs_pca.png - Comparison with PCA")
print("  2. decision_boundaries.png - Linear decision boundaries")
print("  3. iris_projection.png - Dimensionality reduction on Iris")
print("  4. scatter_matrices.png - Within-class and between-class scatter")
print("  5. shrinkage_effect.png - Regularization effect")
print("  6. discriminant_directions.png - Optimal projection direction")
print("  7. class_separation.png - Separation metrics and explained variance")
print("  8. n_components_effect.png - Effect of number of components")
