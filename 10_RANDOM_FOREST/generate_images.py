"""
Generate Comprehensive Visualizations for Random Forest

Creates 6 detailed visualizations demonstrating:
1. Feature importance analysis
2. Number of trees effect
3. Max features comparison
4. OOB score vs test score
5. Classification decision boundaries
6. Regression performance
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_moons, make_regression
from sklearn.model_selection import train_test_split
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from random_forest import RandomForestClassifier, RandomForestRegressor

# Create images directory
os.makedirs('images', exist_ok=True)

print("Generating Random Forest visualizations...")
print("=" * 70)

# 1. Feature Importance Analysis
print("\n1. Generating feature importance analysis...")
np.random.seed(42)

iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importances)))
ax1.bar(range(len(importances)), importances[indices], color=colors)
ax1.set_xlabel('Feature Index', fontsize=12)
ax1.set_ylabel('Importance', fontsize=12)
ax1.set_title('Feature Importance', fontsize=14, fontweight='bold')
ax1.set_xticks(range(len(importances)))
ax1.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
ax1.grid(True, alpha=0.3, axis='y')

# Add values on bars
for i, v in enumerate(importances[indices]):
    ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

# Cumulative importance
cumsum = np.cumsum(importances[indices])
ax2.plot(range(1, len(importances) + 1), cumsum, 'o-', linewidth=2.5,
         markersize=8, color='#2E86AB')
ax2.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% threshold')
ax2.fill_between(range(1, len(importances) + 1), cumsum, alpha=0.3, color='#2E86AB')
ax2.set_xlabel('Number of Features', fontsize=12)
ax2.set_ylabel('Cumulative Importance', fontsize=12)
ax2.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('images/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/feature_importance.png")

# 2. Number of Trees Effect
print("\n2. Generating number of trees effect...")
np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

n_estimators_range = range(1, 201, 5)
train_scores = []
test_scores = []
oob_scores = []

for n in n_estimators_range:
    rf = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)

    train_scores.append(rf.score(X_train, y_train))
    test_scores.append(rf.score(X_test, y_test))
    oob_scores.append(rf.oob_score_)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy vs number of trees
ax1.plot(n_estimators_range, train_scores, label='Training', linewidth=2.5, color='#A23B72')
ax1.plot(n_estimators_range, test_scores, label='Test', linewidth=2.5, color='#2E86AB')
ax1.plot(n_estimators_range, oob_scores, label='OOB', linewidth=2.5,
         color='#F18F01', linestyle='--')
ax1.set_xlabel('Number of Trees', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Accuracy vs Number of Trees', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.8, 1.02])

# Convergence analysis (standard deviation across last 20 estimators)
window = 20
convergence = []
for i in range(window, len(test_scores) + 1):
    convergence.append(np.std(test_scores[i-window:i]))

ax2.plot(list(n_estimators_range)[window-1:], convergence, linewidth=2.5,
         color='#2E86AB', marker='o', markersize=4)
ax2.fill_between(list(n_estimators_range)[window-1:], convergence, alpha=0.3,
                 color='#2E86AB')
ax2.set_xlabel('Number of Trees', fontsize=12)
ax2.set_ylabel(f'Std Dev (window={window})', fontsize=12)
ax2.set_title('Test Score Convergence', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Add annotation for diminishing returns
if len(convergence) > 0:
    ax2.axhline(y=0.01, color='red', linestyle='--', linewidth=2,
                label='Convergence threshold')
    ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('images/n_estimators_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/n_estimators_effect.png")

# 3. Max Features Comparison
print("\n3. Generating max features comparison...")
np.random.seed(42)

max_features_options = ['sqrt', 'log2', None, 0.5]
max_features_labels = ['sqrt(p)', 'log2(p)', 'all (p)', '0.5p']

results = {
    'train': [],
    'test': [],
    'oob': [],
    'time': []
}

import time

for mf in max_features_options:
    start = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_features=mf, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)
    elapsed = time.time() - start

    results['train'].append(rf.score(X_train, y_train))
    results['test'].append(rf.score(X_test, y_test))
    results['oob'].append(rf.oob_score_)
    results['time'].append(elapsed)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Accuracy comparison
x = np.arange(len(max_features_labels))
width = 0.25

bars1 = ax1.bar(x - width, results['train'], width, label='Train', color='#A23B72', alpha=0.8)
bars2 = ax1.bar(x, results['test'], width, label='Test', color='#2E86AB', alpha=0.8)
bars3 = ax1.bar(x + width, results['oob'], width, label='OOB', color='#F18F01', alpha=0.8)

ax1.set_xlabel('max_features', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Accuracy vs max_features', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(max_features_labels)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim([0.85, 1.02])

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Training time
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(max_features_labels)))
bars = ax2.bar(max_features_labels, results['time'], color=colors, alpha=0.8)
ax2.set_xlabel('max_features', fontsize=12)
ax2.set_ylabel('Training Time (seconds)', fontsize=12)
ax2.set_title('Training Time vs max_features', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

for i, (bar, val) in enumerate(zip(bars, results['time'])):
    ax2.text(i, val + 0.01, f'{val:.3f}s', ha='center', fontsize=10, fontweight='bold')

# Bias-variance analysis (simulated)
# Lower max_features = higher bias, lower variance
bias = np.array([0.08, 0.06, 0.03, 0.05])  # Simulated
variance = np.array([0.02, 0.03, 0.06, 0.04])  # Simulated
total_error = bias + variance

x = np.arange(len(max_features_labels))
ax3.plot(x, bias, 'o-', label='Bias²', linewidth=2.5, markersize=8, color='#A23B72')
ax3.plot(x, variance, 's-', label='Variance', linewidth=2.5, markersize=8, color='#2E86AB')
ax3.plot(x, total_error, '^-', label='Total Error', linewidth=2.5, markersize=8, color='#F18F01')
ax3.set_xlabel('max_features', fontsize=12)
ax3.set_ylabel('Error Component', fontsize=12)
ax3.set_title('Bias-Variance Tradeoff', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(max_features_labels)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# Performance summary table
cell_text = []
for i, label in enumerate(max_features_labels):
    cell_text.append([
        label,
        f"{results['test'][i]:.3f}",
        f"{results['oob'][i]:.3f}",
        f"{results['time'][i]:.3f}s"
    ])

table = ax4.table(cellText=cell_text,
                  colLabels=['max_features', 'Test Acc', 'OOB Acc', 'Time'],
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color header
for i in range(4):
    table[(0, i)].set_facecolor('#2E86AB')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color rows
colors_alt = ['#E8F4F8', 'white']
for i in range(1, len(cell_text) + 1):
    for j in range(4):
        table[(i, j)].set_facecolor(colors_alt[(i-1) % 2])

ax4.axis('off')
ax4.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('images/max_features_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/max_features_comparison.png")

# 4. OOB Score vs Test Score
print("\n4. Generating OOB vs test score...")
np.random.seed(42)

# Generate multiple train/test splits
n_splits = 30
oob_scores = []
test_scores = []

for i in range(n_splits):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

    rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)

    oob_scores.append(rf.oob_score_)
    test_scores.append(rf.score(X_test, y_test))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot
ax1.scatter(oob_scores, test_scores, alpha=0.6, s=100, color='#2E86AB', edgecolors='black')

# Add diagonal line (perfect correlation)
min_val = min(min(oob_scores), min(test_scores))
max_val = max(max(oob_scores), max(test_scores))
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
         label='Perfect correlation')

# Add regression line
z = np.polyfit(oob_scores, test_scores, 1)
p = np.poly1d(z)
ax1.plot(oob_scores, p(oob_scores), 'g-', linewidth=2.5,
         label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')

ax1.set_xlabel('OOB Score', fontsize=12)
ax1.set_ylabel('Test Score', fontsize=12)
ax1.set_title('OOB Score vs Test Score', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Calculate correlation
correlation = np.corrcoef(oob_scores, test_scores)[0, 1]
ax1.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
         transform=ax1.transAxes, fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Distribution comparison
ax2.hist(oob_scores, bins=15, alpha=0.6, label='OOB Scores', color='#F18F01',
         edgecolor='black', density=True)
ax2.hist(test_scores, bins=15, alpha=0.6, label='Test Scores', color='#2E86AB',
         edgecolor='black', density=True)

ax2.axvline(np.mean(oob_scores), color='#F18F01', linestyle='--', linewidth=2,
            label=f'OOB Mean: {np.mean(oob_scores):.3f}')
ax2.axvline(np.mean(test_scores), color='#2E86AB', linestyle='--', linewidth=2,
            label=f'Test Mean: {np.mean(test_scores):.3f}')

ax2.set_xlabel('Score', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title('Score Distributions', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('images/oob_vs_test_score.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/oob_vs_test_score.png")

# 5. Classification Decision Boundaries
print("\n5. Generating classification decision boundaries...")
np.random.seed(42)

# Create 2D dataset
X_2d, y_2d = make_moons(n_samples=300, noise=0.3, random_state=42)
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y_2d, test_size=0.3, random_state=42
)

# Train Random Forest and single tree
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train_2d, y_train_2d)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '09_DECISION_TREE'))
from decision_tree import DecisionTreeClassifier
single_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
single_tree.fit(X_train_2d, y_train_2d)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

# Helper function for decision boundary
def plot_decision_boundary(ax, clf, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
                        edgecolors='black', s=50, alpha=0.8)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    return scatter

# Single tree
plot_decision_boundary(ax1, single_tree, X_test_2d, y_test_2d,
                      f'Single Decision Tree (Acc: {single_tree.score(X_test_2d, y_test_2d):.3f})')

# Random Forest
plot_decision_boundary(ax2, rf, X_test_2d, y_test_2d,
                      f'Random Forest (Acc: {rf.score(X_test_2d, y_test_2d):.3f})')

# Probability heatmap for Random Forest
h = 0.02
x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z_proba = rf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z_proba = Z_proba.reshape(xx.shape)

im = ax3.contourf(xx, yy, Z_proba, levels=20, cmap='RdYlBu', alpha=0.8)
ax3.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test_2d, cmap='RdYlBu',
           edgecolors='black', s=50, alpha=0.9)
ax3.set_xlabel('Feature 1', fontsize=12)
ax3.set_ylabel('Feature 2', fontsize=12)
ax3.set_title('Class Probability Heatmap', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
cbar = plt.colorbar(im, ax=ax3)
cbar.set_label('P(Class 1)', fontsize=11)

# Prediction confidence
confidence = np.max(rf.predict_proba(X_test_2d), axis=1)
scatter = ax4.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=confidence,
                     cmap='viridis', s=80, edgecolors='black', alpha=0.8)
ax4.set_xlabel('Feature 1', fontsize=12)
ax4.set_ylabel('Feature 2', fontsize=12)
ax4.set_title('Prediction Confidence', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Confidence', fontsize=11)

plt.tight_layout()
plt.savefig('images/classification_boundaries.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/classification_boundaries.png")

# 6. Regression Performance
print("\n6. Generating regression performance...")
np.random.seed(42)

# Generate regression data
X_reg, y_reg = make_regression(n_samples=200, n_features=1, noise=15, random_state=42)
X_reg = X_reg.reshape(-1)

# Sort for plotting
sort_idx = np.argsort(X_reg)
X_reg_sorted = X_reg[sort_idx]
y_reg_sorted = y_reg[sort_idx]

# Split data
split = 140
X_train_reg = X_reg_sorted[:split].reshape(-1, 1)
y_train_reg = y_reg_sorted[:split]
X_test_reg = X_reg_sorted[split:].reshape(-1, 1)
y_test_reg = y_reg_sorted[split:]

# Train models
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_reg, y_train_reg)

# Get predictions
X_plot = np.linspace(X_reg.min(), X_reg.max(), 300).reshape(-1, 1)
y_rf = rf_reg.predict(X_plot)

# Get individual tree predictions
tree_preds = [tree.predict(X_plot) for tree in rf_reg.trees_[:10]]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

# Main prediction plot
ax1.scatter(X_train_reg, y_train_reg, alpha=0.6, s=50, color='#2E86AB',
           label='Training', edgecolors='black')
ax1.scatter(X_test_reg, y_test_reg, alpha=0.6, s=50, color='#F18F01',
           label='Test', edgecolors='black')
ax1.plot(X_plot, y_rf, 'r-', linewidth=3, label='Random Forest', alpha=0.8)
ax1.set_xlabel('Feature', fontsize=12)
ax1.set_ylabel('Target', fontsize=12)
ax1.set_title(f'Random Forest Regression (R²={rf_reg.score(X_test_reg, y_test_reg):.3f})',
             fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Individual trees + ensemble
for i, tree_pred in enumerate(tree_preds):
    ax2.plot(X_plot, tree_pred, alpha=0.2, color='gray', linewidth=1)

ax2.plot(X_plot, y_rf, 'r-', linewidth=3, label='Ensemble (avg)', alpha=0.9)
ax2.scatter(X_train_reg, y_train_reg, alpha=0.4, s=30, color='#2E86AB')
ax2.set_xlabel('Feature', fontsize=12)
ax2.set_ylabel('Target', fontsize=12)
ax2.set_title('Individual Trees vs Ensemble', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Residual plot
y_pred_test = rf_reg.predict(X_test_reg)
residuals = y_test_reg - y_pred_test

ax3.scatter(y_pred_test, residuals, alpha=0.6, s=80, color='#2E86AB',
           edgecolors='black')
ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Predicted Values', fontsize=12)
ax3.set_ylabel('Residuals', fontsize=12)
ax3.set_title('Residual Plot', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Add RMSE
rmse = np.sqrt(np.mean(residuals**2))
ax3.text(0.05, 0.95, f'RMSE: {rmse:.2f}', transform=ax3.transAxes,
        fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Prediction vs actual
ax4.scatter(y_test_reg, y_pred_test, alpha=0.6, s=80, color='#2E86AB',
           edgecolors='black')

# Add diagonal line
min_val = min(y_test_reg.min(), y_pred_test.min())
max_val = max(y_test_reg.max(), y_pred_test.max())
ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
        label='Perfect prediction')

ax4.set_xlabel('True Values', fontsize=12)
ax4.set_ylabel('Predicted Values', fontsize=12)
ax4.set_title('Predicted vs Actual', fontsize=13, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

# Add R²
r2 = rf_reg.score(X_test_reg, y_test_reg)
ax4.text(0.05, 0.95, f'R²: {r2:.3f}', transform=ax4.transAxes,
        fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('images/regression_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/regression_performance.png")

print("\n" + "=" * 70)
print("All Random Forest visualizations generated successfully!")
print("Images saved in: 10_RANDOM_FOREST/images/")
print("=" * 70)
