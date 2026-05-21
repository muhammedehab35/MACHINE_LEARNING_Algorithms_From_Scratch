"""
Generate Comprehensive Visualizations for Gradient Boosting

Creates visualizations demonstrating:
- Sequential learning process
- Learning rate effects
- Residual fitting
- Ensemble evolution
- Loss function comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_regression, load_iris
from sklearn.model_selection import train_test_split
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gradient_boosting import GradientBoostingClassifier, GradientBoostingRegressor

# Create images directory
os.makedirs('images', exist_ok=True)

print("Generating Gradient Boosting visualizations...")
print("=" * 70)


# 1. Sequential Learning Process
print("\n1. Generating sequential learning process...")
np.random.seed(42)

# Simple regression problem
X_seq = np.linspace(0, 10, 100).reshape(-1, 1)
y_seq = np.sin(X_seq).ravel() + np.random.normal(0, 0.3, 100)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

iterations = [1, 5, 10, 20, 50, 100]

for idx, n_iter in enumerate(iterations):
    ax = axes[idx]

    gb = GradientBoostingRegressor(n_estimators=n_iter, learning_rate=0.1,
                                    max_depth=3, random_state=42)
    gb.fit(X_seq, y_seq)

    y_pred = gb.predict(X_seq)

    # Plot data
    ax.scatter(X_seq, y_seq, alpha=0.5, s=20, label='Data', color='blue')
    ax.plot(X_seq, y_pred, 'r-', linewidth=2, label=f'GB ({n_iter} trees)')
    ax.plot(X_seq, np.sin(X_seq), 'g--', linewidth=1, alpha=0.5, label='True function')

    # Calculate MSE
    mse = np.mean((y_seq - y_pred) ** 2)

    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.set_title(f'Iteration {n_iter}\nMSE: {mse:.4f}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.suptitle('Sequential Learning in Gradient Boosting\nEach tree improves the fit progressively',
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('images/sequential_learning.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/sequential_learning.png")


# 2. Learning Rate Effect
print("\n2. Generating learning rate effect...")
np.random.seed(42)

X_lr = np.linspace(0, 10, 100).reshape(-1, 1)
y_lr = np.sin(X_lr).ravel() + np.random.normal(0, 0.3, 100)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

learning_rates = [0.01, 0.1, 0.5, 1.0]

for idx, lr in enumerate(learning_rates):
    ax = axes[idx]

    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr,
                                    max_depth=3, random_state=42)
    gb.fit(X_lr, y_lr)

    y_pred = gb.predict(X_lr)

    ax.scatter(X_lr, y_lr, alpha=0.5, s=20, color='blue', label='Data')
    ax.plot(X_lr, y_pred, 'r-', linewidth=2, label=f'LR={lr}')
    ax.plot(X_lr, np.sin(X_lr), 'g--', linewidth=1, alpha=0.5, label='True')

    mse = np.mean((y_lr - y_pred) ** 2)

    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.set_title(f'Learning Rate = {lr}\nMSE: {mse:.4f}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Effect of Learning Rate on Gradient Boosting\nLower learning rate = smoother fit, needs more trees',
            fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('images/learning_rate_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/learning_rate_effect.png")


# 3. Residual Fitting Demonstration
print("\n3. Generating residual fitting demonstration...")
np.random.seed(42)

X_res = np.linspace(0, 10, 100).reshape(-1, 1)
y_res = np.sin(X_res).ravel() + np.random.normal(0, 0.3, 100)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# First row: predictions
# Second row: residuals

stages = [0, 1, 5, 20]

for idx, stage in enumerate(stages):
    ax_pred = axes[0, idx]
    ax_resid = axes[1, idx]

    if stage == 0:
        # Initial prediction (mean)
        y_pred = np.full_like(y_res, np.mean(y_res))
        title = 'Initial (F₀ = mean)'
    else:
        gb = GradientBoostingRegressor(n_estimators=stage, learning_rate=0.1,
                                        max_depth=3, random_state=42)
        gb.fit(X_res, y_res)
        y_pred = gb.predict(X_res)
        title = f'After {stage} tree{"s" if stage > 1 else ""}'

    # Plot predictions
    ax_pred.scatter(X_res, y_res, alpha=0.5, s=15, color='blue', label='True')
    ax_pred.plot(X_res, y_pred, 'r-', linewidth=2, label='Prediction')
    ax_pred.set_ylabel('y', fontsize=10)
    ax_pred.set_title(title, fontsize=11, fontweight='bold')
    ax_pred.legend(fontsize=8)
    ax_pred.grid(True, alpha=0.3)

    # Plot residuals
    residuals = y_res - y_pred
    ax_resid.scatter(X_res, residuals, alpha=0.5, s=15, color='green')
    ax_resid.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax_resid.set_xlabel('X', fontsize=10)
    ax_resid.set_ylabel('Residuals', fontsize=10)
    ax_resid.set_title(f'Residuals (next target)', fontsize=10, style='italic')
    ax_resid.grid(True, alpha=0.3)

plt.suptitle('Residual Fitting in Gradient Boosting\nTop: Predictions improve | Bottom: Residuals decrease',
            fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('images/residual_fitting.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/residual_fitting.png")


# 4. Classification Decision Boundaries
print("\n4. Generating classification decision boundaries...")
np.random.seed(42)

X_moons, y_moons = make_moons(n_samples=200, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_moons, y_moons, test_size=0.3, random_state=42)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

n_estimators_list = [1, 5, 10, 20, 50, 100]

for idx, n_est in enumerate(n_estimators_list):
    ax = axes[idx]

    gb = GradientBoostingClassifier(n_estimators=n_est, learning_rate=0.1,
                                     max_depth=3, random_state=42)
    gb.fit(X_train, y_train)

    # Create mesh
    h = 0.02
    x_min, x_max = X_moons[:, 0].min() - 0.5, X_moons[:, 0].max() + 0.5
    y_min, y_max = X_moons[:, 1].min() - 0.5, X_moons[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = gb.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu', levels=[-0.5, 0.5, 1.5])
    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
              c='red', s=30, edgecolors='black', linewidths=1, alpha=0.7)
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
              c='blue', s=30, edgecolors='black', linewidths=1, alpha=0.7)

    acc = gb.score(X_test, y_test)
    ax.set_title(f'{n_est} Trees\nTest Acc: {acc:.3f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature 1', fontsize=9)
    ax.set_ylabel('Feature 2', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Decision Boundaries Evolution in Gradient Boosting\nBoundaries become smoother and more accurate',
            fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('images/decision_boundaries.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/decision_boundaries.png")


# 5. Training Progress (Staged Predictions)
print("\n5. Generating training progress...")
np.random.seed(42)

X_prog, y_prog = make_moons(n_samples=300, noise=0.25, random_state=42)
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_prog, y_prog, test_size=0.3, random_state=42
)

gb_prog = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                     max_depth=3, random_state=42)
gb_prog.fit(X_train_p, y_train_p)

# Get staged accuracies
train_accs = []
test_accs = []

for train_pred, test_pred in zip(gb_prog.staged_predict(X_train_p),
                                  gb_prog.staged_predict(X_test_p)):
    train_accs.append(np.mean(train_pred == y_train_p))
    test_accs.append(np.mean(test_pred == y_test_p))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy curve
ax1.plot(range(1, 101), train_accs, 'b-', linewidth=2, label='Train Accuracy')
ax1.plot(range(1, 101), test_accs, 'r-', linewidth=2, label='Test Accuracy')
ax1.set_xlabel('Number of Trees', fontsize=11, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax1.set_title('Accuracy vs Number of Trees', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Find optimal number of trees
best_idx = np.argmax(test_accs)
ax1.axvline(x=best_idx + 1, color='green', linestyle='--', linewidth=2,
           label=f'Best: {best_idx + 1} trees')
ax1.legend(fontsize=10)

# Overfitting indicator
gap = np.array(train_accs) - np.array(test_accs)
ax2.plot(range(1, 101), gap, 'purple', linewidth=2)
ax2.fill_between(range(1, 101), 0, gap, alpha=0.3, color='purple')
ax2.set_xlabel('Number of Trees', fontsize=11, fontweight='bold')
ax2.set_ylabel('Train - Test Accuracy', fontsize=11, fontweight='bold')
ax2.set_title('Overfitting Gap', fontsize=12, fontweight='bold')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.grid(True, alpha=0.3)

plt.suptitle('Training Progress in Gradient Boosting\nMonitor performance to prevent overfitting',
            fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('images/training_progress.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/training_progress.png")


# 6. Loss Function Comparison (Regression)
print("\n6. Generating loss function comparison...")
np.random.seed(42)

# Generate data with outliers
X_loss = np.linspace(0, 10, 80).reshape(-1, 1)
y_loss = 2 * X_loss.ravel() + np.random.normal(0, 1, 80)

# Add outliers
outlier_indices = [10, 30, 50, 70]
y_loss[outlier_indices] += np.array([8, -10, 12, -8])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

losses = ['squared_error', 'absolute_error', 'huber']
titles = ['Squared Error (L2)\nSensitive to outliers',
         'Absolute Error (L1)\nRobust to outliers',
         'Huber Loss\nBalanced approach']

for idx, (loss, title) in enumerate(zip(losses, titles)):
    ax = axes[idx]

    gb = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1,
                                    max_depth=3, loss=loss, random_state=42)
    gb.fit(X_loss, y_loss)

    X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
    y_pred = gb.predict(X_plot)

    # Plot data
    ax.scatter(X_loss, y_loss, alpha=0.6, s=40, color='blue', label='Data', zorder=3)

    # Highlight outliers
    ax.scatter(X_loss[outlier_indices], y_loss[outlier_indices],
              s=200, facecolors='none', edgecolors='red', linewidths=2,
              label='Outliers', zorder=4)

    # Plot prediction
    ax.plot(X_plot, y_pred, 'g-', linewidth=3, label='GB Prediction', zorder=2)

    # Plot true line (without outliers)
    ax.plot(X_plot, 2 * X_plot, 'r--', linewidth=2, alpha=0.5,
           label='True Line', zorder=1)

    mse = np.mean((y_loss - gb.predict(X_loss)) ** 2)

    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.set_title(f'{title}\nMSE: {mse:.2f}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Loss Function Comparison in Gradient Boosting\nChoose loss based on data characteristics',
            fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('images/loss_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/loss_comparison.png")


# 7. Max Depth Effect
print("\n7. Generating max depth effect...")
np.random.seed(42)

X_depth, y_depth = make_moons(n_samples=200, noise=0.25, random_state=42)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

depths = [1, 2, 3, 5]

for idx, depth in enumerate(depths):
    ax = axes[idx]

    gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1,
                                     max_depth=depth, random_state=42)
    gb.fit(X_depth, y_depth)

    h = 0.02
    x_min, x_max = X_depth[:, 0].min() - 0.5, X_depth[:, 0].max() + 0.5
    y_min, y_max = X_depth[:, 1].min() - 0.5, X_depth[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = gb.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu', levels=[-0.5, 0.5, 1.5])
    ax.scatter(X_depth[y_depth == 0, 0], X_depth[y_depth == 0, 1],
              c='red', s=40, edgecolors='black', linewidths=1, alpha=0.7)
    ax.scatter(X_depth[y_depth == 1, 0], X_depth[y_depth == 1, 1],
              c='blue', s=40, edgecolors='black', linewidths=1, alpha=0.7)

    acc = gb.score(X_depth, y_depth)

    ax.set_title(f'Max Depth = {depth}\nAccuracy: {acc:.3f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature 1', fontsize=10)
    ax.set_ylabel('Feature 2', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('Effect of Tree Depth in Gradient Boosting\nShallow trees (2-5) usually work best',
            fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('images/max_depth_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/max_depth_effect.png")


# 8. GB vs Random Forest Comparison
print("\n8. Generating GB vs RF comparison...")
np.random.seed(42)

from random_forest import RandomForestClassifier

X_comp, y_comp = make_moons(n_samples=300, noise=0.25, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_comp, y_comp, test_size=0.3, random_state=42
)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gradient Boosting
gb_comp = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1,
                                     max_depth=3, random_state=42)
gb_comp.fit(X_train_c, y_train_c)

h = 0.02
x_min, x_max = X_comp[:, 0].min() - 0.5, X_comp[:, 0].max() + 0.5
y_min, y_max = X_comp[:, 1].min() - 0.5, X_comp[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z_gb = gb_comp.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

axes[0].contourf(xx, yy, Z_gb, alpha=0.3, cmap='RdBu', levels=[-0.5, 0.5, 1.5])
axes[0].scatter(X_train_c[y_train_c == 0, 0], X_train_c[y_train_c == 0, 1],
               c='red', s=40, edgecolors='black', linewidths=1, alpha=0.7)
axes[0].scatter(X_train_c[y_train_c == 1, 0], X_train_c[y_train_c == 1, 1],
               c='blue', s=40, edgecolors='black', linewidths=1, alpha=0.7)
acc_gb = gb_comp.score(X_test_c, y_test_c)
axes[0].set_title(f'Gradient Boosting\nTest Accuracy: {acc_gb:.3f}',
                 fontsize=12, fontweight='bold')
axes[0].set_xlabel('Feature 1', fontsize=10)
axes[0].set_ylabel('Feature 2', fontsize=10)
axes[0].grid(True, alpha=0.3)

# Random Forest
rf_comp = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
rf_comp.fit(X_train_c, y_train_c)

Z_rf = rf_comp.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

axes[1].contourf(xx, yy, Z_rf, alpha=0.3, cmap='RdBu', levels=[-0.5, 0.5, 1.5])
axes[1].scatter(X_train_c[y_train_c == 0, 0], X_train_c[y_train_c == 0, 1],
               c='red', s=40, edgecolors='black', linewidths=1, alpha=0.7)
axes[1].scatter(X_train_c[y_train_c == 1, 0], X_train_c[y_train_c == 1, 1],
               c='blue', s=40, edgecolors='black', linewidths=1, alpha=0.7)
acc_rf = rf_comp.score(X_test_c, y_test_c)
axes[1].set_title(f'Random Forest\nTest Accuracy: {acc_rf:.3f}',
                 fontsize=12, fontweight='bold')
axes[1].set_xlabel('Feature 1', fontsize=10)
axes[1].set_ylabel('Feature 2', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Gradient Boosting vs Random Forest\nSequential learning vs parallel ensemble',
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/gb_vs_rf.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/gb_vs_rf.png")

print("\n" + "=" * 70)
print("All Gradient Boosting visualizations generated successfully!")
print("Images saved in: 11_GRADIENT_BOOSTING/images/")
print("=" * 70)
