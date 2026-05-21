"""
Generate Comprehensive Visualizations for XGBoost

Creates visualizations demonstrating:
1. Sequential learning / convergence over iterations
2. XGBoost vs Gradient Boosting decision boundaries
3. Regularization effects (lambda, alpha, gamma)
4. Max depth effect (overfitting analysis)
5. Learning rate vs number of trees tradeoff
6. Feature importance
7. Regression performance (linear, nonlinear, sine)
8. Subsampling effect (subsample + colsample_bytree)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xgboost import XGBoostClassifier, XGBoostRegressor

# Also import GBM for comparison
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '11_GRADIENT_BOOSTING'))
from gradient_boosting import GradientBoostingClassifier, GradientBoostingRegressor

os.makedirs('images', exist_ok=True)

np.random.seed(42)

print("Generating XGBoost visualizations...")
print("=" * 70)


# ============================================================================
# 1. Sequential Learning / Convergence Over Iterations
# ============================================================================
print("\n1. Generating sequential learning convergence...")

X_seq = np.linspace(-5, 5, 150).reshape(-1, 1)
y_seq = np.sin(X_seq[:, 0]) * 2 + X_seq[:, 0] * 0.3 + np.random.normal(0, 0.3, 150)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

iterations = [1, 5, 15, 30, 60, 100]
colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6']

for idx, n_iter in enumerate(iterations):
    ax = axes[idx]

    xgb = XGBoostRegressor(n_estimators=n_iter, learning_rate=0.15,
                           max_depth=3, random_state=42)
    xgb.fit(X_seq, y_seq)
    y_pred = xgb.predict(X_seq)

    mse = np.mean((y_seq - y_pred) ** 2)

    ax.scatter(X_seq, y_seq, alpha=0.4, s=18, color='steelblue', label='Data')
    ax.plot(X_seq, y_pred, '-', color=colors[idx], linewidth=2.5,
            label=f'XGBoost ({n_iter} trees)')
    ax.plot(X_seq, np.sin(X_seq[:, 0]) * 2 + X_seq[:, 0] * 0.3,
            'k--', linewidth=1.2, alpha=0.5, label='True function')

    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.set_title(f'n_estimators = {n_iter}\nMSE: {mse:.4f}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

plt.suptitle('XGBoost Sequential Learning — Convergence Over Iterations\n'
             'Each tree reduces residuals using second-order optimization',
             fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('images/01_sequential_learning.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/01_sequential_learning.png")


# ============================================================================
# 2. XGBoost vs Gradient Boosting — Decision Boundaries
# ============================================================================
print("\n2. Generating XGBoost vs GBM decision boundaries...")

from sklearn.datasets import make_moons, make_circles

datasets = {
    'Moons': make_moons(n_samples=200, noise=0.25, random_state=42),
    'Circles': make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=42),
}

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

cmap_light = ListedColormap(['#ffaaaa', '#aaaaff'])
cmap_bold = ['#cc0000', '#0000cc']

for row_idx, (name, (X_d, y_d)) in enumerate(datasets.items()):
    for col_idx, (clf_name, clf) in enumerate([
        ('Gradient Boosting', GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                                          learning_rate=0.1, random_state=42)),
        ('XGBoost', XGBoostClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                                       reg_lambda=1.0, gamma=0.1, random_state=42))
    ]):
        ax = axes[row_idx, col_idx]

        clf.fit(X_d, y_d)
        acc = clf.score(X_d, y_d)

        h = 0.04
        x_min, x_max = X_d[:, 0].min() - 0.5, X_d[:, 0].max() + 0.5
        y_min, y_max = X_d[:, 1].min() - 0.5, X_d[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
        ax.contour(xx, yy, Z, colors='gray', linewidths=0.8, alpha=0.5)

        for cls, color in zip([0, 1], cmap_bold):
            mask = y_d == cls
            ax.scatter(X_d[mask, 0], X_d[mask, 1], c=color, s=40,
                       edgecolors='white', linewidth=0.5, alpha=0.85)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f'{name} — {clf_name}\nAccuracy: {acc:.3f}',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Feature 1', fontsize=9)
        ax.set_ylabel('Feature 2', fontsize=9)
        ax.grid(True, alpha=0.2)

plt.suptitle('XGBoost vs Gradient Boosting — Decision Boundaries\n'
             'XGBoost uses explicit regularization for smoother boundaries',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('images/02_xgboost_vs_gbm.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/02_xgboost_vs_gbm.png")


# ============================================================================
# 3. Regularization Effects (lambda, alpha, gamma)
# ============================================================================
print("\n3. Generating regularization effects...")

np.random.seed(42)
X_d, y_d = make_moons(n_samples=150, noise=0.3, random_state=42)

configs = [
    ('No Regularization\n(λ=0, α=0, γ=0)', dict(reg_lambda=0.0, reg_alpha=0.0, gamma=0.0)),
    ('L2 Regularization\n(λ=5, α=0, γ=0)', dict(reg_lambda=5.0, reg_alpha=0.0, gamma=0.0)),
    ('L1 Regularization\n(λ=1, α=2, γ=0)', dict(reg_lambda=1.0, reg_alpha=2.0, gamma=0.0)),
    ('Gamma Pruning\n(λ=1, α=0, γ=1)', dict(reg_lambda=1.0, reg_alpha=0.0, gamma=1.0)),
]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

h = 0.04
x_min, x_max = X_d[:, 0].min() - 0.5, X_d[:, 0].max() + 0.5
y_min, y_max = X_d[:, 1].min() - 0.5, X_d[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

for ax, (title, params) in zip(axes, configs):
    xgb = XGBoostClassifier(n_estimators=60, max_depth=4, learning_rate=0.1,
                             random_state=42, **params)
    xgb.fit(X_d, y_d)
    acc = xgb.score(X_d, y_d)

    Z = xgb.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.25, cmap=ListedColormap(['#ffcccc', '#ccccff']))
    ax.contour(xx, yy, Z, colors='gray', linewidths=1.0, alpha=0.6)

    for cls, (color, marker) in zip([0, 1], [('#cc0000', 'o'), ('#0000cc', 's')]):
        mask = y_d == cls
        ax.scatter(X_d[mask, 0], X_d[mask, 1], c=color, marker=marker,
                   s=35, edgecolors='white', linewidth=0.5, alpha=0.85)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f'{title}\nAcc: {acc:.3f}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Feature 1', fontsize=9)
    ax.set_ylabel('Feature 2', fontsize=9)
    ax.grid(True, alpha=0.2)

plt.suptitle('XGBoost Regularization — L1 (α), L2 (λ), and Gamma (γ) Effects\n'
             'Regularization controls boundary complexity and prevents overfitting',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('images/03_regularization_effects.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/03_regularization_effects.png")


# ============================================================================
# 4. Max Depth Effect — Overfitting Analysis
# ============================================================================
print("\n4. Generating max depth overfitting analysis...")

np.random.seed(42)
X_full, y_full = make_moons(n_samples=300, noise=0.25, random_state=42)
split = 200
X_train_d, X_test_d = X_full[:split], X_full[split:]
y_train_d, y_test_d = y_full[:split], y_full[split:]

depths = [1, 2, 3, 5, 7, 10]
train_accs = []
test_accs = []

for depth in depths:
    xgb = XGBoostClassifier(n_estimators=50, max_depth=depth,
                             learning_rate=0.1, random_state=42)
    xgb.fit(X_train_d, y_train_d)
    train_accs.append(xgb.score(X_train_d, y_train_d))
    test_accs.append(xgb.score(X_test_d, y_test_d))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: accuracy curves
ax = axes[0]
ax.plot(depths, train_accs, 'b-o', linewidth=2, markersize=8, label='Train Accuracy')
ax.plot(depths, test_accs, 'r-s', linewidth=2, markersize=8, label='Test Accuracy')
ax.axvline(x=depths[np.argmax(test_accs)], color='green', linestyle='--',
           alpha=0.7, label=f'Best depth = {depths[np.argmax(test_accs)]}')
ax.fill_between(depths,
                [t - tr for t, tr in zip(train_accs, test_accs)],
                [0] * len(depths),
                alpha=0.0)
ax.set_xlabel('max_depth', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Train vs Test Accuracy\nvs Max Depth', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.5, 1.05)
ax.set_xticks(depths)

# Right: decision boundaries for 3 depths
ax2 = axes[1]
ax2.axis('off')

sub_fig, sub_axes = plt.subplots(1, 3, figsize=(15, 5))
for i, depth in enumerate([1, 3, 10]):
    ax_sub = sub_axes[i]
    xgb = XGBoostClassifier(n_estimators=50, max_depth=depth,
                             learning_rate=0.1, random_state=42)
    xgb.fit(X_train_d, y_train_d)

    h = 0.05
    x_min, x_max = X_full[:, 0].min() - 0.5, X_full[:, 0].max() + 0.5
    y_min_b, y_max_b = X_full[:, 1].min() - 0.5, X_full[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min_b, y_max_b, h))

    Z = xgb.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax_sub.contourf(xx, yy, Z, alpha=0.25, cmap=ListedColormap(['#ffcccc', '#ccccff']))
    ax_sub.contour(xx, yy, Z, colors='gray', linewidths=0.8)

    for cls, color in zip([0, 1], ['#cc0000', '#0000cc']):
        mask = y_train_d == cls
        ax_sub.scatter(X_train_d[mask, 0], X_train_d[mask, 1], c=color,
                       s=30, alpha=0.7, edgecolors='white', linewidth=0.4)

    train_a = xgb.score(X_train_d, y_train_d)
    test_a = xgb.score(X_test_d, y_test_d)
    label = 'Underfitting' if depth == 1 else ('Overfitting' if depth == 10 else 'Good Fit')
    ax_sub.set_title(f'max_depth={depth} — {label}\nTrain:{train_a:.2f} Test:{test_a:.2f}',
                     fontsize=10, fontweight='bold')
    ax_sub.set_xlabel('Feature 1', fontsize=9)
    ax_sub.set_ylabel('Feature 2', fontsize=9)
    ax_sub.grid(True, alpha=0.2)

sub_fig.suptitle('XGBoost Max Depth — Boundary Complexity', fontsize=12, fontweight='bold')
sub_fig.tight_layout()
sub_fig.savefig('images/04b_depth_boundaries.png', dpi=150, bbox_inches='tight')
plt.close(sub_fig)

axes[1].text(0.5, 0.5, 'See 04b_depth_boundaries.png\nfor boundary visualizations',
             ha='center', va='center', fontsize=12, transform=axes[1].transAxes)

fig.suptitle('XGBoost Max Depth — Overfitting Analysis', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('images/04_depth_overfitting.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("   [OK] Saved: images/04_depth_overfitting.png")
print("   [OK] Saved: images/04b_depth_boundaries.png")


# ============================================================================
# 5. Learning Rate vs Number of Trees Tradeoff
# ============================================================================
print("\n5. Generating learning rate vs n_estimators tradeoff...")

from sklearn.datasets import load_iris
X_iris, y_iris_full = load_iris(return_X_y=True)
y_iris = (y_iris_full != 0).astype(int)

split_iris = 100
X_tr_i, X_te_i = X_iris[:split_iris], X_iris[split_iris:]
y_tr_i, y_te_i = y_iris[:split_iris], y_iris[split_iris:]

learning_rates = [0.01, 0.05, 0.1, 0.3, 0.5]
max_trees = 150
colors_lr = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6']

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax_train = axes[0]
ax_test = axes[1]

for lr, color in zip(learning_rates, colors_lr):
    train_scores = []
    test_scores = []
    tree_counts = list(range(5, max_trees + 1, 5))

    for n in tree_counts:
        xgb = XGBoostClassifier(n_estimators=n, learning_rate=lr,
                                 max_depth=3, random_state=42)
        xgb.fit(X_tr_i, y_tr_i)
        train_scores.append(xgb.score(X_tr_i, y_tr_i))
        test_scores.append(xgb.score(X_te_i, y_te_i))

    ax_train.plot(tree_counts, train_scores, '-', color=color,
                  linewidth=2, label=f'lr={lr}')
    ax_test.plot(tree_counts, test_scores, '-', color=color,
                 linewidth=2, label=f'lr={lr}')

for ax, title in zip([ax_train, ax_test], ['Training Accuracy', 'Test Accuracy']):
    ax.set_xlabel('Number of Trees', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title(f'{title} vs n_estimators', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)

plt.suptitle('Learning Rate vs Number of Trees Tradeoff\n'
             'Low LR needs more trees; High LR risks overshooting',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('images/05_learning_rate_tradeoff.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/05_learning_rate_tradeoff.png")


# ============================================================================
# 6. Feature Importance
# ============================================================================
print("\n6. Generating feature importance visualizations...")

from sklearn.datasets import load_iris, fetch_california_housing

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left: Iris (Classification) ---
X_iris_fi, y_iris_fi = load_iris(return_X_y=True)
y_iris_bin = (y_iris_fi != 0).astype(int)
feature_names_iris = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

xgb_iris = XGBoostClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                               reg_lambda=1.0, random_state=42)
xgb_iris.fit(X_iris_fi, y_iris_bin)
imp_iris = xgb_iris.feature_importances_

ax = axes[0]
sorted_idx = np.argsort(imp_iris)[::-1]
bars = ax.bar(range(len(imp_iris)),
              imp_iris[sorted_idx],
              color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'],
              edgecolor='white', linewidth=0.5, alpha=0.85)

for bar, val in zip(bars, imp_iris[sorted_idx]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xticks(range(len(imp_iris)))
ax.set_xticklabels([feature_names_iris[i] for i in sorted_idx], rotation=20, ha='right', fontsize=10)
ax.set_ylabel('Importance (Gain)', fontsize=11)
ax.set_title('Feature Importance — Iris (Classification)\nGain-based: total gain from splits',
             fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(imp_iris) * 1.2)

# --- Right: California Housing (Regression) ---
try:
    X_cal, y_cal = fetch_california_housing(return_X_y=True)
    X_cal, y_cal = X_cal[:600], y_cal[:600]
    feature_names_cal = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                         'Population', 'AveOccup', 'Latitude', 'Longitude']

    xgb_cal = XGBoostRegressor(n_estimators=80, max_depth=4, learning_rate=0.1,
                                reg_lambda=1.0, random_state=42)
    xgb_cal.fit(X_cal, y_cal)
    imp_cal = xgb_cal.feature_importances_

    ax2 = axes[1]
    sorted_idx_cal = np.argsort(imp_cal)[::-1]
    palette = plt.cm.viridis(np.linspace(0.2, 0.85, len(imp_cal)))
    bars2 = ax2.bar(range(len(imp_cal)), imp_cal[sorted_idx_cal],
                    color=palette, edgecolor='white', linewidth=0.5, alpha=0.85)

    for bar, val in zip(bars2, imp_cal[sorted_idx_cal]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_xticks(range(len(imp_cal)))
    ax2.set_xticklabels([feature_names_cal[i] for i in sorted_idx_cal],
                        rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('Importance (Gain)', fontsize=11)
    ax2.set_title('Feature Importance — California Housing (Regression)',
                  fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(imp_cal) * 1.2)
except Exception:
    axes[1].text(0.5, 0.5, 'California Housing\nnot available',
                 ha='center', va='center', fontsize=13, transform=axes[1].transAxes)

plt.suptitle('XGBoost Feature Importance (Gain-based)\n'
             'Higher = feature contributes more split gain across all trees',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('images/06_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/06_feature_importance.png")


# ============================================================================
# 7. Regression Performance — Linear, Polynomial, Sine
# ============================================================================
print("\n7. Generating regression performance visualizations...")

np.random.seed(42)
n = 200
X_lin = np.random.uniform(-5, 5, n).reshape(-1, 1)
y_lin = 2.5 * X_lin[:, 0] - 1.0 + np.random.normal(0, 0.8, n)

X_poly = np.random.uniform(-4, 4, n).reshape(-1, 1)
y_poly = 0.5 * X_poly[:, 0] ** 3 - 2 * X_poly[:, 0] + np.random.normal(0, 1.5, n)

X_sine = np.random.uniform(-np.pi * 2, np.pi * 2, n).reshape(-1, 1)
y_sine = np.sin(X_sine[:, 0]) + 0.5 * np.cos(2 * X_sine[:, 0]) + np.random.normal(0, 0.25, n)

datasets_reg = [
    ('Linear: y = 2.5x − 1', X_lin, y_lin),
    ('Polynomial: y = 0.5x³ − 2x', X_poly, y_poly),
    ('Sinusoidal: y = sin(x) + 0.5cos(2x)', X_sine, y_sine),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (title, X_r, y_r) in zip(axes, datasets_reg):
    split_r = int(0.7 * len(X_r))
    sort_idx = np.argsort(X_r[:, 0])
    X_r, y_r = X_r[sort_idx], y_r[sort_idx]

    X_tr_r, X_te_r = X_r[:split_r], X_r[split_r:]
    y_tr_r, y_te_r = y_r[:split_r], y_r[split_r:]

    xgb_r = XGBoostRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                              reg_lambda=1.0, random_state=42)
    xgb_r.fit(X_tr_r, y_tr_r)

    y_pred_tr = xgb_r.predict(X_tr_r)
    y_pred_te = xgb_r.predict(X_te_r)

    r2_tr = xgb_r.score(X_tr_r, y_tr_r)
    r2_te = xgb_r.score(X_te_r, y_te_r)

    ax.scatter(X_tr_r, y_tr_r, alpha=0.4, s=18, color='steelblue', label='Train data')
    ax.scatter(X_te_r, y_te_r, alpha=0.5, s=18, color='salmon', label='Test data')
    ax.plot(X_tr_r, y_pred_tr, 'b-', linewidth=2.5, label=f'Train R²={r2_tr:.3f}')
    ax.plot(X_te_r, y_pred_te, 'r-', linewidth=2.5, label=f'Test R²={r2_te:.3f}')

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('XGBoost Regression Performance\nLinear, Polynomial, and Sinusoidal Functions',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('images/07_regression_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/07_regression_performance.png")


# ============================================================================
# 8. Subsampling Effect — subsample × colsample_bytree Grid
# ============================================================================
print("\n8. Generating subsampling effect...")

np.random.seed(42)
X_sub, y_sub = make_moons(n_samples=300, noise=0.3, random_state=42)
split_sub = 200
X_tr_s, X_te_s = X_sub[:split_sub], X_sub[split_sub:]
y_tr_s, y_te_s = y_sub[:split_sub], y_sub[split_sub:]

subsample_vals = [0.5, 0.7, 1.0]
colsample_vals = [0.5, 0.7, 1.0]

fig, axes = plt.subplots(3, 3, figsize=(15, 14))

h = 0.05
x_min_s, x_max_s = X_sub[:, 0].min() - 0.4, X_sub[:, 0].max() + 0.4
y_min_s, y_max_s = X_sub[:, 1].min() - 0.4, X_sub[:, 1].max() + 0.4
xx_s, yy_s = np.meshgrid(np.arange(x_min_s, x_max_s, h),
                          np.arange(y_min_s, y_max_s, h))

for r, sub in enumerate(subsample_vals):
    for c, col in enumerate(colsample_vals):
        ax = axes[r, c]

        xgb_s = XGBoostClassifier(n_estimators=60, max_depth=3, learning_rate=0.1,
                                   subsample=sub, colsample_bytree=col,
                                   reg_lambda=1.0, random_state=42)
        xgb_s.fit(X_tr_s, y_tr_s)

        train_a = xgb_s.score(X_tr_s, y_tr_s)
        test_a = xgb_s.score(X_te_s, y_te_s)

        Z = xgb_s.predict(np.c_[xx_s.ravel(), yy_s.ravel()]).reshape(xx_s.shape)
        ax.contourf(xx_s, yy_s, Z, alpha=0.25, cmap=ListedColormap(['#ffcccc', '#ccccff']))
        ax.contour(xx_s, yy_s, Z, colors='gray', linewidths=0.7)

        for cls, color in zip([0, 1], ['#cc0000', '#0000cc']):
            mask = y_tr_s == cls
            ax.scatter(X_tr_s[mask, 0], X_tr_s[mask, 1], c=color,
                       s=25, alpha=0.65, edgecolors='white', linewidth=0.3)

        ax.set_xlim(x_min_s, x_max_s)
        ax.set_ylim(y_min_s, y_max_s)
        ax.set_title(f'sub={sub}, col={col}\nTr:{train_a:.2f} Te:{test_a:.2f}',
                     fontsize=9, fontweight='bold')

        if r == 2:
            ax.set_xlabel(f'colsample={col}', fontsize=9, color='navy')
        if c == 0:
            ax.set_ylabel(f'subsample={sub}', fontsize=9, color='darkgreen')
        ax.grid(True, alpha=0.15)

plt.suptitle('XGBoost Subsampling Effect — subsample × colsample_bytree Grid\n'
             'Lower values add stochasticity → less overfitting, more variance reduction',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('images/08_subsampling_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/08_subsampling_effect.png")


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("All XGBoost visualizations generated successfully!")
print("=" * 70)
print("\nImages saved to images/:")
images = [
    "01_sequential_learning.png    - Convergence over iterations",
    "02_xgboost_vs_gbm.png         - XGBoost vs GBM decision boundaries",
    "03_regularization_effects.png - L1, L2, gamma regularization comparison",
    "04_depth_overfitting.png      - Max depth overfitting analysis",
    "04b_depth_boundaries.png      - Decision boundaries per depth",
    "05_learning_rate_tradeoff.png - LR vs n_estimators tradeoff",
    "06_feature_importance.png     - Gain-based feature importances",
    "07_regression_performance.png - Linear, poly, sine regression",
    "08_subsampling_effect.png     - subsample x colsample grid",
]
for img in images:
    print("  " + img)
