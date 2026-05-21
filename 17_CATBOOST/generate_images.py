"""
Generate visualizations for CatBoost From-Scratch Implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from catboost_scratch import (
    CatBoostClassifier, CatBoostRegressor,
    ObliviousTree, ordered_target_statistics
)

os.makedirs("images", exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Oblivious Tree Structure — symmetric vs asymmetric
# ---------------------------------------------------------------------------

def img_oblivious_structure():
    print("Generating: 01_oblivious_tree_structure.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("Symmetric (Oblivious) Tree vs Standard Decision Tree", fontsize=13, fontweight='bold')

    def draw_tree(ax, symmetric):
        ax.set_xlim(0, 10); ax.set_ylim(0, 8)
        ax.axis('off')

        if symmetric:
            ax.set_title("Oblivious Tree (CatBoost)\nSame split at every node of each depth", fontsize=10)
            nodes = [
                (5, 7, '#2196F3', 'd=0: f1 > 3.2'),
                (2.5, 5.2, '#4CAF50', 'd=1: f3 > 1.5'),
                (7.5, 5.2, '#4CAF50', 'd=1: f3 > 1.5'),
                (1.2, 3.3, '#FF9800', 'Leaf'), (3.8, 3.3, '#FF9800', 'Leaf'),
                (6.2, 3.3, '#FF9800', 'Leaf'), (8.8, 3.3, '#FF9800', 'Leaf'),
            ]
        else:
            ax.set_title("Standard Tree (XGBoost/LightGBM)\nEach node can use any split", fontsize=10)
            nodes = [
                (5, 7, '#2196F3', 'f1 > 3.2'),
                (2.5, 5.2, '#9C27B0', 'f2 > 0.8'),
                (7.5, 5.2, '#F44336', 'f4 > 2.1'),
                (1.2, 3.3, '#FF9800', 'Leaf'), (3.8, 3.3, '#607D8B', 'f3 > 1.0'),
                (6.2, 3.3, '#FF9800', 'Leaf'), (8.8, 3.3, '#FF9800', 'Leaf'),
            ]

        edges = [(0,1),(0,2),(1,3),(1,4),(2,5),(2,6)]
        for i, j in edges:
            ax.plot([nodes[i][0], nodes[j][0]], [nodes[i][1], nodes[j][1]],
                    'k-', linewidth=1.5, zorder=1)

        for (x, y, c, lbl) in nodes:
            fc = c if 'Leaf' not in lbl else '#FFEB3B'
            circle = plt.Circle((x, y), 0.55, color=fc, zorder=3, ec='black', lw=1.5)
            ax.add_patch(circle)
            ax.text(x, y, lbl, ha='center', va='center', fontsize=7, fontweight='bold', zorder=4)

        if symmetric:
            ax.annotate('', xy=(2.5, 5.75), xytext=(7.5, 5.75),
                        arrowprops=dict(arrowstyle='<->', color='green', lw=2))
            ax.text(5, 6.0, 'Same split!', ha='center', fontsize=9, color='green', fontweight='bold')

    draw_tree(axes[0], symmetric=True)
    draw_tree(axes[1], symmetric=False)

    plt.tight_layout()
    plt.savefig("images/01_oblivious_tree_structure.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 2. Ordered Target Statistics — plain vs ordered encoding
# ---------------------------------------------------------------------------

def img_ordered_target_statistics():
    print("Generating: 02_ordered_target_statistics.png")
    np.random.seed(42)
    n = 60
    categories = np.array([0] * 20 + [1] * 20 + [2] * 20)
    true_means = {0: 1.0, 1: 3.0, 2: 5.0}
    targets = np.array([true_means[c] + np.random.randn() * 0.5 for c in categories])

    # Plain mean encoding
    cat_means = {c: targets[categories == c].mean() for c in [0, 1, 2]}
    plain_enc = np.array([cat_means[c] for c in categories])

    # Ordered encoding (shuffled)
    perm = np.random.permutation(n)
    cats_perm = categories[perm]
    tgts_perm = targets[perm]
    ordered_enc_perm = ordered_target_statistics(cats_perm, tgts_perm, prior=2.0)
    ordered_enc = np.zeros(n)
    ordered_enc[perm] = ordered_enc_perm

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Ordered Target Statistics vs Plain Mean Encoding", fontsize=13, fontweight='bold')

    colors = ['#2196F3', '#4CAF50', '#F44336']
    for ax, enc, title in zip(axes, [plain_enc, ordered_enc],
                               ['Plain Mean Encoding\n(target leakage!)', 'Ordered Target Statistics\n(leak-free)']):
        for cat in [0, 1, 2]:
            mask = categories == cat
            ax.scatter(np.where(mask)[0], enc[mask], c=colors[cat], s=30,
                       label=f'Category {cat} (true mean={true_means[cat]})', alpha=0.8)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Encoded value")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.axhline(1.0, color='#2196F3', linestyle='--', alpha=0.4)
        ax.axhline(3.0, color='#4CAF50', linestyle='--', alpha=0.4)
        ax.axhline(5.0, color='#F44336', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig("images/02_ordered_target_statistics.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 3. Prediction shift — ordered vs plain boosting
# ---------------------------------------------------------------------------

def img_ordered_boosting():
    print("Generating: 03_ordered_boosting.png")
    np.random.seed(42)
    X = np.sort(np.random.uniform(-3, 3, 200)).reshape(-1, 1)
    y = np.sin(X[:, 0]) + np.random.randn(200) * 0.15
    X_train, y_train = X[:160], y[:160]
    X_test, y_test = X[160:], y[160:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Plain Boosting vs Ordered Boosting (Reduced Prediction Shift)", fontsize=13)

    x_plot = np.linspace(-3, 3, 300).reshape(-1, 1)

    for ax, mode, title in zip(axes, [False, True], ['Plain Boosting', 'Ordered Boosting']):
        reg = CatBoostRegressor(
            iterations=100, learning_rate=0.05, depth=4,
            use_ordered_boosting=mode, n_ordered_folds=4,
            min_samples_leaf=3, random_state=42
        )
        reg.fit(X_train, y_train)
        y_plot = reg.predict(x_plot)
        r2_train = reg.score(X_train, y_train)
        r2_test = reg.score(X_test, y_test)

        ax.scatter(X_train, y_train, s=10, alpha=0.4, color='gray', label='Train')
        ax.scatter(X_test, y_test, s=20, alpha=0.7, color='orange', label='Test')
        ax.plot(x_plot, y_plot, 'b-', linewidth=2, label='Model')
        ax.set_title(f"{title}\nTrain R2={r2_train:.3f}  Test R2={r2_test:.3f}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("images/03_ordered_boosting.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 4. Decision boundary evolution
# ---------------------------------------------------------------------------

def img_decision_boundary():
    print("Generating: 04_decision_boundary.png")
    np.random.seed(42)
    X1 = np.random.randn(100, 2) + [2, 2]
    X2 = np.random.randn(100, 2) + [-2, -2]
    X = np.vstack([X1, X2])
    y = np.array([1] * 100 + [0] * 100)

    xx, yy = np.meshgrid(np.linspace(-6, 6, 80), np.linspace(-6, 6, 80))
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("CatBoost Decision Boundary Evolution (Oblivious Trees)", fontsize=12, fontweight='bold')

    for ax, n_iter in zip(axes, [1, 5, 20, 100]):
        clf = CatBoostClassifier(
            iterations=n_iter, learning_rate=0.1, depth=4,
            min_samples_leaf=3, random_state=42
        )
        clf.fit(X, y)
        proba = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

        ax.contourf(xx, yy, proba, levels=50, cmap='RdYlBu', alpha=0.7, vmin=0, vmax=1)
        ax.contour(xx, yy, proba, levels=[0.5], colors='black', linewidths=1.5)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', s=8, alpha=0.5)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=8, alpha=0.5)
        acc = clf.score(X, y)
        ax.set_title(f"iterations={n_iter}\nacc={acc:.3f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("images/04_decision_boundary.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 5. Regression performance
# ---------------------------------------------------------------------------

def img_regression():
    print("Generating: 05_regression.png")
    np.random.seed(42)
    X = np.sort(np.random.uniform(-3, 3, 300)).reshape(-1, 1)
    y = np.sin(X[:, 0] * 1.5) * np.cos(X[:, 0]) + np.random.randn(300) * 0.15
    X_train, y_train = X[:240], y[:240]
    X_test, y_test = X[240:], y[240:]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("CatBoost Regression: Nonlinear Function Approximation", fontsize=13)
    x_plot = np.linspace(-3, 3, 300).reshape(-1, 1)

    for ax, n_iter in zip(axes, [10, 50, 200]):
        reg = CatBoostRegressor(
            iterations=n_iter, learning_rate=0.1, depth=5,
            min_samples_leaf=3, random_state=42
        )
        reg.fit(X_train, y_train)
        r2 = reg.score(X_test, y_test)
        ax.scatter(X_train, y_train, s=8, alpha=0.3, color='gray', label='Train')
        ax.scatter(X_test, y_test, s=20, alpha=0.6, color='orange', label='Test')
        ax.plot(x_plot, reg.predict(x_plot), 'b-', linewidth=2, label='Prediction')
        ax.set_title(f"iterations={n_iter}  R2={r2:.3f}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        if ax == axes[0]:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("images/05_regression.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 6. Feature importances
# ---------------------------------------------------------------------------

def img_feature_importances():
    print("Generating: 06_feature_importances.png")
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Feature Importances (Gain-Based) — CatBoost Oblivious Trees", fontsize=12)

    for ax, depth in zip(axes, [3, 6]):
        clf = CatBoostClassifier(
            iterations=100, depth=depth, min_samples_leaf=3, random_state=42
        )
        clf.fit(X, y_bin)
        fi = clf.feature_importances_
        bars = ax.barh(feature_names, fi, color='steelblue', edgecolor='black', alpha=0.8)
        for bar, val in zip(bars, fi):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=9)
        ax.set_xlabel("Normalized Importance")
        ax.set_title(f"depth={depth}")
        ax.set_xlim(0, max(fi) * 1.25 + 0.05)

    plt.tight_layout()
    plt.savefig("images/06_feature_importances.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 7. Depth effect on model complexity
# ---------------------------------------------------------------------------

def img_depth_effect():
    print("Generating: 07_depth_effect.png")
    np.random.seed(42)
    X = np.sort(np.random.uniform(-3, 3, 150)).reshape(-1, 1)
    y = np.sin(X[:, 0] * 2) + np.random.randn(150) * 0.2
    x_plot = np.linspace(-3.5, 3.5, 300).reshape(-1, 1)

    depths = [1, 2, 4, 6]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("Depth Effect on Oblivious Trees (each tree has 2^depth leaves)", fontsize=12)

    for ax, d in zip(axes, depths):
        reg = CatBoostRegressor(
            iterations=50, learning_rate=0.1, depth=d,
            min_samples_leaf=3, random_state=42
        )
        reg.fit(X, y)
        y_plot = reg.predict(x_plot)
        r2 = reg.score(X, y)
        ax.scatter(X, y, s=12, alpha=0.5, color='gray')
        ax.plot(x_plot, y_plot, 'b-', linewidth=2)
        ax.set_title(f"depth={d} ({2**d} leaves/tree)\nTrain R2={r2:.3f}")
        ax.set_xlabel("x")
        if ax == axes[0]:
            ax.set_ylabel("y")

    plt.tight_layout()
    plt.savefig("images/07_depth_effect.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 8. Regularization effect
# ---------------------------------------------------------------------------

def img_regularization():
    print("Generating: 08_regularization.png")
    np.random.seed(42)
    X = np.sort(np.random.uniform(-3, 3, 80)).reshape(-1, 1)
    y = np.sin(X[:, 0] * 2) + np.random.randn(80) * 0.2
    X_test = np.sort(np.random.uniform(-3, 3, 100)).reshape(-1, 1)
    y_test = np.sin(X_test[:, 0] * 2) + np.random.randn(100) * 0.2
    x_plot = np.linspace(-3.5, 3.5, 200).reshape(-1, 1)

    lambdas = [0.01, 1.0, 10.0, 100.0]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("L2 Regularization Effect (reg_lambda)", fontsize=13)

    for ax, lam in zip(axes, lambdas):
        reg = CatBoostRegressor(
            iterations=50, learning_rate=0.1, depth=4,
            reg_lambda=lam, min_samples_leaf=3, random_state=42
        )
        reg.fit(X, y)
        r2 = reg.score(X_test, y_test)
        ax.scatter(X, y, s=12, alpha=0.5, color='gray', label='Train')
        ax.plot(x_plot, reg.predict(x_plot), 'b-', linewidth=2)
        ax.set_title(f"lambda={lam}\nTest R2={r2:.3f}")
        ax.set_xlabel("x"); ax.set_ylim(-2.5, 2.5)
        if ax == axes[0]:
            ax.set_ylabel("y")

    plt.tight_layout()
    plt.savefig("images/08_regularization.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating CatBoost images...")
    print("=" * 60)

    img_oblivious_structure()
    img_ordered_target_statistics()
    img_ordered_boosting()
    img_decision_boundary()
    img_regression()
    img_feature_importances()
    img_depth_effect()
    img_regularization()

    print("=" * 60)
    print("All 8 images saved to images/")
