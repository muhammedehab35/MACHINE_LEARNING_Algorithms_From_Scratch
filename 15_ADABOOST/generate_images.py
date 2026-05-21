"""
Generate Comprehensive Visualizations for AdaBoost

Creates visualizations demonstrating:
1. Boosting process — sequential weight updates
2. Decision boundaries evolution over rounds
3. AdaBoost vs single stump decision boundary
4. Effect of n_estimators (convergence)
5. Learning rate tradeoff
6. Feature importance
7. Training error per round
8. Probability contours
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaboost import AdaBoostClassifier, DecisionStump

os.makedirs('images', exist_ok=True)
np.random.seed(42)

print("Generating AdaBoost visualizations...")
print("=" * 70)


# ============================================================================
# 1. Boosting process — weight evolution over rounds
# ============================================================================
print("\n1. Generating weight evolution...")

from sklearn.datasets import make_moons
X_m, y_m = make_moons(n_samples=120, noise=0.25, random_state=42)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
rounds = [1, 3, 5, 10, 20, 50]

for ax, n in zip(axes, rounds):
    clf = AdaBoostClassifier(n_estimators=n, random_state=42)
    clf.fit(X_m, y_m)
    acc = clf.score(X_m, y_m)

    h = 0.04
    x_min, x_max = X_m[:, 0].min() - 0.5, X_m[:, 0].max() + 0.5
    y_min, y_max = X_m[:, 1].min() - 0.5, X_m[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.25, cmap=ListedColormap(['#ffcccc', '#ccccff']))
    ax.contour(xx, yy, Z, colors='gray', linewidths=0.8)
    for cls, col in zip([0, 1], ['#cc0000', '#0000cc']):
        mask = y_m == cls
        ax.scatter(X_m[mask, 0], X_m[mask, 1], c=col, s=30,
                   edgecolors='white', linewidth=0.4, alpha=0.85)

    ax.set_title(f'n_estimators = {n}\nAcc: {acc:.3f}', fontsize=11, fontweight='bold')
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.2)

plt.suptitle('AdaBoost — Decision Boundary Evolution Over Boosting Rounds\n'
             'More stumps → complex boundary that fits the data better',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('images/01_boundary_evolution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/01_boundary_evolution.png")


# ============================================================================
# 2. AdaBoost vs Single Stump — on 3 datasets
# ============================================================================
print("\n2. Generating AdaBoost vs single stump comparison...")

from sklearn.datasets import make_circles

datasets = {
    'Moons':   make_moons(n_samples=200, noise=0.25, random_state=42),
    'Circles': make_circles(n_samples=200, noise=0.15, factor=0.5, random_state=42),
}

fig, axes = plt.subplots(2, 2, figsize=(13, 11))

for row, (name, (X_d, y_d)) in enumerate(datasets.items()):
    for col, (label, n_est) in enumerate([('Single Stump', 1), ('AdaBoost (50)', 50)]):
        ax = axes[row, col]
        clf = AdaBoostClassifier(n_estimators=n_est, random_state=42)
        clf.fit(X_d, y_d)
        acc = clf.score(X_d, y_d)

        h = 0.04
        x_min, x_max = X_d[:, 0].min() - 0.5, X_d[:, 0].max() + 0.5
        y_min, y_max = X_d[:, 1].min() - 0.5, X_d[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.25, cmap=ListedColormap(['#ffcccc', '#ccccff']))
        ax.contour(xx, yy, Z, colors='gray', linewidths=1.0)
        for cls, col_c in zip([0, 1], ['#cc0000', '#0000cc']):
            mask = y_d == cls
            ax.scatter(X_d[mask, 0], X_d[mask, 1], c=col_c, s=35,
                       edgecolors='white', linewidth=0.4, alpha=0.85)

        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_title(f'{name} — {label}\nAccuracy: {acc:.3f}',
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)

plt.suptitle('AdaBoost vs Single Decision Stump\n'
             'Ensemble of stumps overcomes the linear limit of a single stump',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('images/02_adaboost_vs_stump.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/02_adaboost_vs_stump.png")


# ============================================================================
# 3. Sample weight evolution (first 3 rounds illustrated)
# ============================================================================
print("\n3. Generating sample weight evolution...")

np.random.seed(42)
X_w = np.random.randn(40, 2)
y_w = (X_w[:, 0] + X_w[:, 1] > 0).astype(int)

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
titles = ['Initial weights\n(uniform)', 'After round 1', 'After round 2', 'After round 5']
n_rounds = [0, 1, 2, 5]

for ax, n, title in zip(axes, n_rounds, titles):
    if n == 0:
        w = np.full(len(X_w), 1.0 / len(X_w))
        w_norm = w / w.max()
    else:
        # manually simulate weight update
        w = np.full(len(X_w), 1.0 / len(X_w))
        y_enc = np.where(y_w == 1, 1, -1).astype(float)
        for _ in range(n):
            clf_tmp = AdaBoostClassifier(n_estimators=1, random_state=0)
            # find stump
            stump = DecisionStump()
            min_err = float('inf')
            for fi in range(2):
                col = X_w[:, fi]
                for thresh in np.unique(col):
                    for pol in (1, -1):
                        p = np.ones(len(X_w))
                        if pol == 1:
                            p[col < thresh] = -1
                        else:
                            p[col >= thresh] = -1
                        err = np.sum(w[y_enc != p])
                        if err < min_err:
                            min_err = err
                            stump.feature_index = fi
                            stump.threshold = thresh
                            stump.polarity = pol
            min_err = np.clip(min_err, 1e-10, 1 - 1e-10)
            alpha = 0.5 * np.log((1 - min_err) / min_err)
            preds = stump.predict(X_w)
            w *= np.exp(-alpha * y_enc * preds)
            w /= w.sum()
        w_norm = w / w.max()

    sizes = 40 + w_norm * 250
    for cls, col in zip([0, 1], ['#cc0000', '#0000cc']):
        mask = y_w == cls
        ax.scatter(X_w[mask, 0], X_w[mask, 1], c=col,
                   s=sizes[mask], edgecolors='black', linewidth=0.5, alpha=0.75)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature 1', fontsize=9)
    ax.set_ylabel('Feature 2', fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.text(0.02, 0.97, 'Size = sample weight',
            transform=ax.transAxes, fontsize=8, va='top', color='gray')

plt.suptitle('AdaBoost — Sample Weight Evolution\n'
             'Misclassified samples grow larger (receive higher weights)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('images/03_weight_evolution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/03_weight_evolution.png")


# ============================================================================
# 4. Convergence — train/test accuracy vs n_estimators
# ============================================================================
print("\n4. Generating convergence curves...")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X_iris, y_iris = load_iris(return_X_y=True)
y_bin = (y_iris != 0).astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X_iris, y_bin, test_size=0.3, random_state=42)

n_vals = list(range(1, 101))
train_accs, test_accs = [], []
for n in n_vals:
    clf = AdaBoostClassifier(n_estimators=n, random_state=42)
    clf.fit(X_tr, y_tr)
    train_accs.append(clf.score(X_tr, y_tr))
    test_accs.append(clf.score(X_te, y_te))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(n_vals, train_accs, 'b-', linewidth=2, label='Train accuracy')
ax.plot(n_vals, test_accs, 'r-', linewidth=2, label='Test accuracy')
ax.axvline(x=n_vals[np.argmax(test_accs)], color='green', linestyle='--',
           alpha=0.7, label=f'Best n={n_vals[np.argmax(test_accs)]}')
ax.fill_between(n_vals, train_accs, test_accs, alpha=0.1, color='orange',
                label='Overfitting gap')
ax.set_xlabel('Number of Estimators', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('AdaBoost Convergence — Train vs Test Accuracy\n(Iris binary, Setosa vs Others)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.5, 1.05)
plt.tight_layout()
plt.savefig('images/04_convergence.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/04_convergence.png")


# ============================================================================
# 5. Learning rate effect
# ============================================================================
print("\n5. Generating learning rate effect...")

lrs = [0.1, 0.5, 1.0, 2.0]
colors_lr = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for lr, c in zip(lrs, colors_lr):
    tr_s, te_s = [], []
    for n in n_vals:
        clf = AdaBoostClassifier(n_estimators=n, learning_rate=lr, random_state=42)
        clf.fit(X_tr, y_tr)
        tr_s.append(clf.score(X_tr, y_tr))
        te_s.append(clf.score(X_te, y_te))
    axes[0].plot(n_vals, tr_s, '-', color=c, linewidth=1.8, label=f'lr={lr}')
    axes[1].plot(n_vals, te_s, '-', color=c, linewidth=1.8, label=f'lr={lr}')

for ax, title in zip(axes, ['Training Accuracy', 'Test Accuracy']):
    ax.set_xlabel('Number of Estimators', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title(f'{title} vs n_estimators', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)

plt.suptitle('AdaBoost — Learning Rate vs Number of Estimators\n'
             'Low LR needs more rounds; high LR may overshoot',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('images/05_learning_rate.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/05_learning_rate.png")


# ============================================================================
# 6. Feature importance
# ============================================================================
print("\n6. Generating feature importance...")

feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
clf_fi = AdaBoostClassifier(n_estimators=100, random_state=42)
clf_fi.fit(X_iris, y_bin)
fi = clf_fi.feature_importances_

fig, ax = plt.subplots(figsize=(9, 6))
sorted_idx = np.argsort(fi)[::-1]
bars = ax.bar(range(4), fi[sorted_idx],
              color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'],
              edgecolor='white', linewidth=0.5, alpha=0.88)

for bar, val in zip(bars, fi[sorted_idx]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xticks(range(4))
ax.set_xticklabels([feature_names[i] for i in sorted_idx], fontsize=11)
ax.set_ylabel('Importance (sum of |alpha|)', fontsize=11)
ax.set_title('AdaBoost Feature Importance — Iris Dataset\n'
             'Petal features dominate for Setosa vs Others classification',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, fi.max() * 1.25)
plt.tight_layout()
plt.savefig('images/06_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/06_feature_importance.png")


# ============================================================================
# 7. Training error per round
# ============================================================================
print("\n7. Generating training error per round...")

clf_err = AdaBoostClassifier(n_estimators=80, random_state=42)
clf_err.fit(X_tr, y_tr)

alphas = clf_err.alphas_
errors = clf_err.training_errors_

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax1 = axes[0]
ax1.plot(range(1, len(errors) + 1), errors, 'r-o', markersize=3,
         linewidth=1.5, label='Weighted error')
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.6, label='Random (0.5)')
ax1.set_xlabel('Boosting Round', fontsize=11)
ax1.set_ylabel('Weighted Error', fontsize=11)
ax1.set_title('Weighted Training Error per Round', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 0.6)

ax2 = axes[1]
ax2.plot(range(1, len(alphas) + 1), alphas, 'b-o', markersize=3,
         linewidth=1.5, label='Alpha (stump weight)')
ax2.set_xlabel('Boosting Round', fontsize=11)
ax2.set_ylabel('Alpha', fontsize=11)
ax2.set_title('Stump Weight (Alpha) per Round\nalpha = 0.5 * log((1-err)/err)',
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.suptitle('AdaBoost — Error and Alpha Evolution\n'
             'Higher alpha = more reliable stump (lower error)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('images/07_error_alpha.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/07_error_alpha.png")


# ============================================================================
# 8. Probability contours
# ============================================================================
print("\n8. Generating probability contours...")

X_p, y_p = make_moons(n_samples=200, noise=0.2, random_state=42)
clf_prob = AdaBoostClassifier(n_estimators=50, random_state=42)
clf_prob.fit(X_p, y_p)

h = 0.04
x_min, x_max = X_p[:, 0].min() - 0.5, X_p[:, 0].max() + 0.5
y_min, y_max = X_p[:, 1].min() - 0.5, X_p[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z_prob = clf_prob.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z_prob = Z_prob.reshape(xx.shape)
Z_pred = clf_prob.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Probability heatmap
cf = axes[0].contourf(xx, yy, Z_prob, levels=20, cmap='RdBu_r', alpha=0.85)
plt.colorbar(cf, ax=axes[0], label='P(class=1)')
for cls, col in zip([0, 1], ['#cc0000', '#0000cc']):
    mask = y_p == cls
    axes[0].scatter(X_p[mask, 0], X_p[mask, 1], c=col, s=30,
                    edgecolors='white', linewidth=0.4, alpha=0.9)
axes[0].set_title('Probability Contours P(y=1)\nSigmoid of decision function',
                  fontsize=11, fontweight='bold')
axes[0].grid(True, alpha=0.2)

# Hard decision boundary
axes[1].contourf(xx, yy, Z_pred, alpha=0.25, cmap=ListedColormap(['#ffcccc', '#ccccff']))
axes[1].contour(xx, yy, Z_pred, colors='black', linewidths=1.5)
for cls, col in zip([0, 1], ['#cc0000', '#0000cc']):
    mask = y_p == cls
    axes[1].scatter(X_p[mask, 0], X_p[mask, 1], c=col, s=30,
                    edgecolors='white', linewidth=0.4, alpha=0.9)
axes[1].set_title(f'Hard Decision Boundary\nAccuracy: {clf_prob.score(X_p, y_p):.3f}',
                  fontsize=11, fontweight='bold')
axes[1].grid(True, alpha=0.2)

plt.suptitle('AdaBoost — Probability Contours vs Decision Boundary',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('images/08_probability_contours.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/08_probability_contours.png")


print("\n" + "=" * 70)
print("All AdaBoost visualizations generated successfully!")
print("=" * 70)
print("\nImages saved to images/:")
for img in [
    "01_boundary_evolution.png  - Decision boundary over boosting rounds",
    "02_adaboost_vs_stump.png   - AdaBoost vs single stump comparison",
    "03_weight_evolution.png    - Sample weight evolution per round",
    "04_convergence.png         - Train/test accuracy convergence",
    "05_learning_rate.png       - Learning rate vs n_estimators",
    "06_feature_importance.png  - Feature importances",
    "07_error_alpha.png         - Training error and alpha per round",
    "08_probability_contours.png- Probability contours",
]:
    print("  " + img)
