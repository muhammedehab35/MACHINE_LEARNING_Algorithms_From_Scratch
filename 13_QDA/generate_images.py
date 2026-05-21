"""
Generate Comprehensive Visualizations for Quadratic Discriminant Analysis

This script creates visualizations demonstrating:
1. QDA vs LDA decision boundaries
2. Quadratic boundaries with different covariances
3. Effect of regularization
4. Covariance ellipses
5. Decision boundaries evolution
6. Multi-class classification
7. QDA vs Naive Bayes comparison
8. Probability contours
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
import os
from qda import QuadraticDiscriminantAnalysis, RegularizedQDA

# Create images directory
os.makedirs('images', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

print("Generating QDA visualizations...")

# ============================================================================
# 1. QDA vs LDA Decision Boundaries
# ============================================================================
print("1. Generating QDA vs LDA comparison...")

# Import LDA
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '12_LDA'))
from lda import LinearDiscriminantAnalysis

# Generate data with different covariances
np.random.seed(42)
X1 = np.random.randn(100, 2) * [2.5, 0.5] + [2, 2]  # Horizontal ellipse
X2 = np.random.randn(100, 2) * [0.5, 2.5] + [-2, -2]  # Vertical ellipse

X = np.vstack([X1, X2])
y = np.array([0] * 100 + [1] * 100)

# Create meshgrid
h = 0.1
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# LDA
ax = axes[0]
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
Z_lda = lda.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
ax.contourf(xx, yy, Z_lda, alpha=0.3, cmap=cmap_light)
ax.contour(xx, yy, Z_lda, colors='k', linewidths=2, levels=[0.5])

ax.scatter(X[y == 0, 0], X[y == 0, 1], c='red', alpha=0.6, edgecolors='k', s=50, label='Class 0')
ax.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', alpha=0.6, edgecolors='k', s=50, label='Class 1')
ax.set_xlabel('Feature 1', fontsize=12)
ax.set_ylabel('Feature 2', fontsize=12)
ax.set_title('LDA: Linear Decision Boundary\n(Assumes Equal Covariances)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# QDA
ax = axes[1]
qda = QuadraticDiscriminantAnalysis()
qda.fit(X, y)
Z_qda = qda.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax.contourf(xx, yy, Z_qda, alpha=0.3, cmap=cmap_light)
ax.contour(xx, yy, Z_qda, colors='k', linewidths=2, levels=[0.5])

ax.scatter(X[y == 0, 0], X[y == 0, 1], c='red', alpha=0.6, edgecolors='k', s=50, label='Class 0')
ax.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', alpha=0.6, edgecolors='k', s=50, label='Class 1')
ax.set_xlabel('Feature 1', fontsize=12)
ax.set_ylabel('Feature 2', fontsize=12)
ax.set_title('QDA: Quadratic Decision Boundary\n(Allows Different Covariances)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/qda_vs_lda.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 2. Covariance Ellipses Visualization
# ============================================================================
print("2. Generating covariance ellipses...")

fig, ax = plt.subplots(figsize=(10, 8))

# Plot data
ax.scatter(X[y == 0, 0], X[y == 0, 1], c='red', alpha=0.5, edgecolors='k', s=60, label='Class 0')
ax.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', alpha=0.5, edgecolors='k', s=60, label='Class 1')

# Plot class means
for i in range(2):
    mean = qda.means_[i]
    color = ['red', 'blue'][i]
    ax.scatter(mean[0], mean[1], marker='X', s=400,
               edgecolors='black', linewidths=2, color=color, label=f'Mean {i}', zorder=10)

# Draw covariance ellipses
for i, color in enumerate(['red', 'blue']):
    mean = qda.means_[i]
    cov = qda.covariances_[i]

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # Draw ellipses at 1, 2, 3 standard deviations
    for n_std in [1, 2, 3]:
        width, height = 2 * n_std * np.sqrt(eigenvalues)
        ellipse = Ellipse(mean, width, height, angle=angle,
                         alpha=0.2 - 0.05*n_std, color=color, linewidth=2)
        ax.add_patch(ellipse)

# Decision boundary
Z_qda = qda.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contour(xx, yy, Z_qda, colors='black', linewidths=3, levels=[0.5], linestyles='--')

ax.set_xlabel('Feature 1', fontsize=12)
ax.set_ylabel('Feature 2', fontsize=12)
ax.set_title('QDA Covariance Ellipses and Decision Boundary\n(1σ, 2σ, 3σ contours)',
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plt.savefig('images/covariance_ellipses.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 3. Different Types of Quadratic Boundaries
# ============================================================================
print("3. Generating different quadratic boundary types...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

scenarios = [
    {
        'title': 'Elliptical Boundary (Different Scales)',
        'X1_scale': [2.0, 0.5],
        'X2_scale': [0.5, 2.0],
        'X1_center': [2, 2],
        'X2_center': [-2, -2]
    },
    {
        'title': 'Parabolic-like Boundary',
        'X1_scale': [1.5, 1.5],
        'X2_scale': [3.0, 0.5],
        'X1_center': [0, 2],
        'X2_center': [0, -2]
    },
    {
        'title': 'Hyperbolic-like Boundary',
        'X1_scale': [2.5, 0.3],
        'X2_scale': [0.3, 2.5],
        'X1_center': [3, 0],
        'X2_center': [-3, 0]
    },
    {
        'title': 'Nested Classes (One Inside Other)',
        'X1_scale': [0.5, 0.5],
        'X2_scale': [2.5, 2.5],
        'X1_center': [0, 0],
        'X2_center': [0, 0]
    }
]

for idx, scenario in enumerate(scenarios):
    ax = axes[idx // 2, idx % 2]

    # Generate data
    np.random.seed(42 + idx)
    X1 = np.random.randn(80, 2) * scenario['X1_scale'] + scenario['X1_center']
    X2 = np.random.randn(80, 2) * scenario['X2_scale'] + scenario['X2_center']

    X_temp = np.vstack([X1, X2])
    y_temp = np.array([0] * 80 + [1] * 80)

    # Train QDA
    qda_temp = QuadraticDiscriminantAnalysis()
    qda_temp.fit(X_temp, y_temp)

    # Create meshgrid
    x_min_temp = X_temp[:, 0].min() - 1
    x_max_temp = X_temp[:, 0].max() + 1
    y_min_temp = X_temp[:, 1].min() - 1
    y_max_temp = X_temp[:, 1].max() + 1
    xx_temp, yy_temp = np.meshgrid(np.arange(x_min_temp, x_max_temp, 0.1),
                                     np.arange(y_min_temp, y_max_temp, 0.1))

    # Predict
    Z_temp = qda_temp.predict(np.c_[xx_temp.ravel(), yy_temp.ravel()]).reshape(xx_temp.shape)

    # Plot
    ax.contourf(xx_temp, yy_temp, Z_temp, alpha=0.3, cmap=cmap_light)
    ax.contour(xx_temp, yy_temp, Z_temp, colors='k', linewidths=2, levels=[0.5])

    ax.scatter(X_temp[y_temp == 0, 0], X_temp[y_temp == 0, 1],
               c='red', alpha=0.6, edgecolors='k', s=40)
    ax.scatter(X_temp[y_temp == 1, 0], X_temp[y_temp == 1, 1],
               c='blue', alpha=0.6, edgecolors='k', s=40)

    ax.set_title(scenario['title'], fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.savefig('images/quadratic_boundary_types.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 4. Effect of Regularization
# ============================================================================
print("4. Generating regularization effects...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

reg_configs = [
    {'shrinkage': 0.0, 'diagonal_shrinkage': 0.0, 'title': 'No Regularization (Pure QDA)'},
    {'shrinkage': 0.7, 'diagonal_shrinkage': 0.0, 'title': 'Shrinkage towards LDA (α=0.7)'},
    {'shrinkage': 0.0, 'diagonal_shrinkage': 0.7, 'title': 'Diagonal Shrinkage (β=0.7)'},
    {'shrinkage': 0.5, 'diagonal_shrinkage': 0.5, 'title': 'Combined Regularization (α=β=0.5)'}
]

for idx, config in enumerate(reg_configs):
    ax = axes[idx // 2, idx % 2]

    # Train model
    qda_reg = RegularizedQDA(
        shrinkage=config['shrinkage'],
        diagonal_shrinkage=config['diagonal_shrinkage']
    )
    qda_reg.fit(X, y)

    # Predict
    Z_reg = qda_reg.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Plot
    ax.contourf(xx, yy, Z_reg, alpha=0.3, cmap=cmap_light)
    ax.contour(xx, yy, Z_reg, colors='k', linewidths=2, levels=[0.5])

    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='red', alpha=0.6, edgecolors='k', s=50)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', alpha=0.6, edgecolors='k', s=50)

    acc = qda_reg.score(X, y)
    ax.set_title(f"{config['title']}\nAccuracy: {acc:.3f}", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.savefig('images/regularization_effects.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 5. Multi-class QDA (Iris Dataset)
# ============================================================================
print("5. Generating multi-class visualization...")

from sklearn.datasets import load_iris

X_iris, y_iris = load_iris(return_X_y=True)

# Use only first 2 features for visualization
X_iris_2d = X_iris[:, :2]

# Train QDA
qda_iris = QuadraticDiscriminantAnalysis()
qda_iris.fit(X_iris_2d, y_iris)

# Create meshgrid
x_min_iris = X_iris_2d[:, 0].min() - 0.5
x_max_iris = X_iris_2d[:, 0].max() + 0.5
y_min_iris = X_iris_2d[:, 1].min() - 0.5
y_max_iris = X_iris_2d[:, 1].max() + 0.5
xx_iris, yy_iris = np.meshgrid(np.arange(x_min_iris, x_max_iris, 0.02),
                                 np.arange(y_min_iris, y_max_iris, 0.02))

# Predict
Z_iris = qda_iris.predict(np.c_[xx_iris.ravel(), yy_iris.ravel()]).reshape(xx_iris.shape)

fig, ax = plt.subplots(figsize=(10, 8))

cmap_light_3 = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
ax.contourf(xx_iris, yy_iris, Z_iris, alpha=0.3, cmap=cmap_light_3)
ax.contour(xx_iris, yy_iris, Z_iris, colors='k', linewidths=2)

colors = ['red', 'green', 'blue']
class_names = ['Setosa', 'Versicolor', 'Virginica']

for i, color, name in zip([0, 1, 2], colors, class_names):
    ax.scatter(X_iris_2d[y_iris == i, 0], X_iris_2d[y_iris == i, 1],
               c=color, alpha=0.6, edgecolors='k', s=80, label=name)

ax.set_xlabel('Sepal Length (cm)', fontsize=12)
ax.set_ylabel('Sepal Width (cm)', fontsize=12)
ax.set_title('QDA Multi-class Classification on Iris Dataset\n(Using first 2 features)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.savefig('images/multiclass_iris.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 6. Probability Contours
# ============================================================================
print("6. Generating probability contours...")

# Use the original 2-class data
qda_prob = QuadraticDiscriminantAnalysis()
qda_prob.fit(X, y)

# Get probabilities
Z_proba = qda_prob.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10, 8))

# Probability contours
contour = ax.contourf(xx, yy, Z_proba, levels=20, cmap='RdBu_r', alpha=0.7)
ax.contour(xx, yy, Z_proba, levels=[0.1, 0.3, 0.5, 0.7, 0.9],
           colors='black', linewidths=1.5, linestyles='--')

# Decision boundary (P=0.5)
ax.contour(xx, yy, Z_proba, levels=[0.5], colors='black', linewidths=3)

# Data points
ax.scatter(X[y == 0, 0], X[y == 0, 1], c='red', alpha=0.7, edgecolors='k', s=80, label='Class 0')
ax.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', alpha=0.7, edgecolors='k', s=80, label='Class 1')

# Colorbar
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('P(Class 1 | x)', fontsize=12)

ax.set_xlabel('Feature 1', fontsize=12)
ax.set_ylabel('Feature 2', fontsize=12)
ax.set_title('QDA Probability Contours\n(Dashed lines: 0.1, 0.3, 0.5, 0.7, 0.9)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.savefig('images/probability_contours.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 7. Accuracy Comparison (Sample Size Effect)
# ============================================================================
print("7. Generating sample size effect...")

sample_sizes = [20, 40, 60, 100, 150, 200]
qda_accs = []
lda_accs = []

np.random.seed(42)
# Generate large dataset
X_large1 = np.random.randn(200, 2) * [2.0, 0.5] + [2, 2]
X_large2 = np.random.randn(200, 2) * [0.5, 2.0] + [-2, -2]
X_large = np.vstack([X_large1, X_large2])
y_large = np.array([0] * 200 + [1] * 200)

for n in sample_sizes:
    # Sample n points per class
    indices = np.concatenate([
        np.random.choice(np.where(y_large == 0)[0], n, replace=False),
        np.random.choice(np.where(y_large == 1)[0], n, replace=False)
    ])
    X_sample = X_large[indices]
    y_sample = y_large[indices]

    # QDA
    qda_temp = QuadraticDiscriminantAnalysis()
    qda_temp.fit(X_sample, y_sample)
    qda_accs.append(qda_temp.score(X_sample, y_sample))

    # LDA
    lda_temp = LinearDiscriminantAnalysis()
    lda_temp.fit(X_sample, y_sample)
    lda_accs.append(lda_temp.score(X_sample, y_sample))

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot([n*2 for n in sample_sizes], qda_accs, marker='o', markersize=10,
        linewidth=2, color='purple', label='QDA')
ax.plot([n*2 for n in sample_sizes], lda_accs, marker='s', markersize=10,
        linewidth=2, color='orange', label='LDA')

ax.set_xlabel('Total Sample Size', fontsize=12)
ax.set_ylabel('Training Accuracy', fontsize=12)
ax.set_title('QDA vs LDA: Effect of Sample Size\n(Data with different covariances)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.5, 1.05])

# Add annotation
ax.text(0.5, 0.15, 'QDA needs more data but achieves\nhigher accuracy with enough samples',
        transform=ax.transAxes, fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig('images/sample_size_effect.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 8. Mahalanobis Distance Visualization
# ============================================================================
print("8. Generating Mahalanobis distance visualization...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot class 0
ax = axes[0]
mean_0 = qda.means_[0]
cov_0 = qda.covariances_[0]
cov_inv_0 = np.linalg.inv(cov_0)

# Compute Mahalanobis distances
mahal_0 = np.zeros(xx.shape)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        point = np.array([xx[i, j], yy[i, j]])
        diff = point - mean_0
        mahal_0[i, j] = np.sqrt(diff @ cov_inv_0 @ diff)

contour_0 = ax.contourf(xx, yy, mahal_0, levels=20, cmap='Reds', alpha=0.6)
ax.contour(xx, yy, mahal_0, levels=[1, 2, 3], colors='darkred', linewidths=2)

ax.scatter(X[y == 0, 0], X[y == 0, 1], c='red', alpha=0.7, edgecolors='k', s=60)
ax.scatter(mean_0[0], mean_0[1], marker='X', s=400, c='darkred',
           edgecolors='black', linewidths=2, label='Mean')

cbar_0 = plt.colorbar(contour_0, ax=ax)
cbar_0.set_label('Mahalanobis Distance', fontsize=11)

ax.set_xlabel('Feature 1', fontsize=12)
ax.set_ylabel('Feature 2', fontsize=12)
ax.set_title('Mahalanobis Distance to Class 0 Mean\n(Contours at 1σ, 2σ, 3σ)',
             fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot class 1
ax = axes[1]
mean_1 = qda.means_[1]
cov_1 = qda.covariances_[1]
cov_inv_1 = np.linalg.inv(cov_1)

mahal_1 = np.zeros(xx.shape)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        point = np.array([xx[i, j], yy[i, j]])
        diff = point - mean_1
        mahal_1[i, j] = np.sqrt(diff @ cov_inv_1 @ diff)

contour_1 = ax.contourf(xx, yy, mahal_1, levels=20, cmap='Blues', alpha=0.6)
ax.contour(xx, yy, mahal_1, levels=[1, 2, 3], colors='darkblue', linewidths=2)

ax.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', alpha=0.7, edgecolors='k', s=60)
ax.scatter(mean_1[0], mean_1[1], marker='X', s=400, c='darkblue',
           edgecolors='black', linewidths=2, label='Mean')

cbar_1 = plt.colorbar(contour_1, ax=ax)
cbar_1.set_label('Mahalanobis Distance', fontsize=11)

ax.set_xlabel('Feature 1', fontsize=12)
ax.set_ylabel('Feature 2', fontsize=12)
ax.set_title('Mahalanobis Distance to Class 1 Mean\n(Contours at 1σ, 2σ, 3σ)',
             fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/mahalanobis_distance.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll visualizations generated successfully!")
print("Images saved in: images/")
print("\nGenerated files:")
print("  1. qda_vs_lda.png - QDA vs LDA decision boundaries")
print("  2. covariance_ellipses.png - Covariance structure visualization")
print("  3. quadratic_boundary_types.png - Different quadratic boundary shapes")
print("  4. regularization_effects.png - Effect of regularization parameters")
print("  5. multiclass_iris.png - Multi-class classification on Iris")
print("  6. probability_contours.png - Posterior probability contours")
print("  7. sample_size_effect.png - Impact of training data size")
print("  8. mahalanobis_distance.png - Mahalanobis distance visualization")
