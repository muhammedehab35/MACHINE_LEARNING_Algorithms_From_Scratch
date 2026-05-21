"""
Generate visualizations for LightGBM From-Scratch Implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lightgbm_scratch import (
    LightGBMClassifier, LightGBMRegressor,
    FeatureHistogram, goss_sample
)

os.makedirs("images", exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Histogram Binning Illustration
# ---------------------------------------------------------------------------

def img_histogram_binning():
    print("Generating: 01_histogram_binning.png")
    np.random.seed(42)
    values = np.concatenate([
        np.random.normal(2, 0.5, 200),
        np.random.normal(5, 1.0, 300),
        np.random.normal(8, 0.8, 200),
    ])

    hist = FeatureHistogram(n_bins=10)
    hist.fit(values)
    bins = hist.transform(values)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Histogram-Based Feature Binning", fontsize=14, fontweight='bold')

    # Left: raw distribution + bin edges
    ax = axes[0]
    ax.hist(values, bins=50, color='steelblue', alpha=0.7, label='Raw values')
    for edge in hist.bin_edges:
        ax.axvline(edge, color='red', linewidth=1.2, alpha=0.7)
    ax.set_title("Continuous Values -> Discrete Bins")
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Count")
    ax.legend()

    # Right: binned representation
    ax = axes[1]
    bin_counts = np.bincount(bins, minlength=len(hist.bin_edges))
    ax.bar(range(len(bin_counts)), bin_counts, color='coral', edgecolor='black', alpha=0.8)
    ax.set_title("Binned Histogram (B bins)")
    ax.set_xlabel("Bin Index")
    ax.set_ylabel("Number of Samples")
    ax.text(0.05, 0.95, f"Bins: {len(hist.bin_edges)}\nSamples: {len(values)}",
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig("images/01_histogram_binning.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 2. Leaf-wise vs Level-wise Tree Growth
# ---------------------------------------------------------------------------

def img_leafwise_vs_levelwise():
    print("Generating: 02_leafwise_vs_levelwise.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Leaf-wise vs Level-wise Tree Growth", fontsize=14, fontweight='bold')

    def draw_tree_levelwise(ax):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title("Level-wise (XGBoost style)\nAll leaves at same depth grow together", fontsize=10)

        nodes_by_level = [
            [(5, 7)],
            [(2.5, 5), (7.5, 5)],
            [(1.25, 3), (3.75, 3), (6.25, 3), (8.75, 3)],
        ]
        colors = ['#2196F3', '#4CAF50', '#FF9800']
        labels = ['Root (split)', 'Level 1 (split)', 'Level 2 (split)']

        for lvl, nodes in enumerate(nodes_by_level):
            for node in nodes:
                circle = plt.Circle(node, 0.4, color=colors[lvl], zorder=3)
                ax.add_patch(circle)

        for lvl in range(len(nodes_by_level) - 1):
            parents = nodes_by_level[lvl]
            children = nodes_by_level[lvl + 1]
            for i, parent in enumerate(parents):
                for child in children[2*i:2*i+2]:
                    ax.plot([parent[0], child[0]], [parent[1], child[1]],
                            'k-', linewidth=1.5, zorder=2)

        handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
        ax.legend(handles=handles, loc='lower center', fontsize=8)

    def draw_tree_leafwise(ax):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title("Leaf-wise (LightGBM style)\nBest leaf always splits next", fontsize=10)

        nodes = [
            (5, 7, '#2196F3', 'Root'),
            (2.5, 5, '#4CAF50', 'Best leaf'),
            (7.5, 5, '#9E9E9E', 'Leaf'),
            (1.25, 3, '#FF9800', 'Best leaf'),
            (3.75, 3, '#9E9E9E', 'Leaf'),
        ]
        edges = [(0, 1), (0, 2), (1, 3), (1, 4)]

        for i, (x, y, c, lbl) in enumerate(nodes):
            circle = plt.Circle((x, y), 0.4, color=c, zorder=3)
            ax.add_patch(circle)
            ax.text(x, y - 0.75, lbl, ha='center', va='top', fontsize=7)

        for i, j in edges:
            ax.plot([nodes[i][0], nodes[j][0]], [nodes[i][1], nodes[j][1]],
                    'k-', linewidth=1.5, zorder=2)

        handles2 = [
            mpatches.Patch(color='#4CAF50', label='Selected (max gain)'),
            mpatches.Patch(color='#9E9E9E', label='Not selected'),
        ]
        ax.legend(handles=handles2, loc='lower center', fontsize=8)

    draw_tree_levelwise(axes[0])
    draw_tree_leafwise(axes[1])

    plt.tight_layout()
    plt.savefig("images/02_leafwise_vs_levelwise.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 3. GOSS Sampling Illustration
# ---------------------------------------------------------------------------

def img_goss_sampling():
    print("Generating: 03_goss_sampling.png")
    np.random.seed(42)
    n = 500
    gradients = np.concatenate([
        np.random.normal(0, 0.2, 400),   # small gradients
        np.random.normal(0, 2.0, 100),    # large gradients
    ])
    np.random.shuffle(gradients)

    top_rate = 0.2
    other_rate = 0.1
    idx, weights = goss_sample(gradients, top_rate, other_rate)

    colors = np.full(n, '#9E9E9E')   # grey for not selected
    colors[idx[:int(n * top_rate)]] = '#F44336'   # red for large
    colors[idx[int(n * top_rate):]] = '#2196F3'   # blue for small sampled

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("GOSS: Gradient-based One-Side Sampling", fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.scatter(range(n), np.sort(np.abs(gradients)), c=[colors[i] for i in np.argsort(np.abs(gradients))],
               s=10, alpha=0.7)
    ax.set_title("Samples Sorted by |Gradient|")
    ax.set_xlabel("Sample Rank")
    ax.set_ylabel("|Gradient|")
    ax.axvline(int(n * top_rate), color='black', linestyle='--',
               label=f'Top {int(top_rate*100)}% threshold')
    ax.legend(fontsize=9)
    red_patch = mpatches.Patch(color='#F44336', label='Large gradient (keep all)')
    blue_patch = mpatches.Patch(color='#2196F3', label='Small gradient (sampled)')
    grey_patch = mpatches.Patch(color='#9E9E9E', label='Not selected')
    ax.legend(handles=[red_patch, blue_patch, grey_patch], fontsize=8)

    ax = axes[1]
    categories = ['Total', 'Large gradient\n(kept)', 'Small gradient\n(sampled)', 'Discarded']
    n_large = int(n * top_rate)
    n_small = len(idx) - n_large
    n_disc = n - n_large - n_small
    values_bar = [n, n_large, n_small, n_disc]
    bar_colors = ['#607D8B', '#F44336', '#2196F3', '#9E9E9E']
    bars = ax.bar(categories, values_bar, color=bar_colors, edgecolor='black', alpha=0.8)
    for bar, val in zip(bars, values_bar):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title("Sample Counts per Group")
    ax.set_ylabel("Number of Samples")

    plt.tight_layout()
    plt.savefig("images/03_goss_sampling.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 4. Split Gain Formula Visualization
# ---------------------------------------------------------------------------

def img_split_gain():
    print("Generating: 04_split_gain.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Histogram Split Gain: O(B) per Feature", fontsize=14, fontweight='bold')

    # Left: gain as function of split position
    np.random.seed(0)
    n = 200
    x_vals = np.sort(np.random.uniform(0, 10, n))
    y = (x_vals > 5).astype(float) + np.random.randn(n) * 0.3
    gradients = np.random.randn(n) * 0.5

    hist = FeatureHistogram(n_bins=20)
    hist.fit(x_vals)
    bins = hist.transform(x_vals)

    G_bins, H_bins, cnt_bins = hist.build(bins, gradients, np.ones(n))
    reg_lambda = 1.0
    G_total = gradients.sum()
    H_total = float(n)

    gains = []
    G_left = 0.0; H_left = 0.0; cnt_left = 0
    thresholds = []
    for b in range(len(hist.bin_edges) - 1):
        G_left += G_bins[b]
        H_left += H_bins[b]
        cnt_left += cnt_bins[b]
        if cnt_left < 5 or (n - cnt_left) < 5:
            gains.append(np.nan)
        else:
            G_right = G_total - G_left
            H_right = H_total - H_left
            gain = 0.5 * (
                G_left**2 / (H_left + reg_lambda) +
                G_right**2 / (H_right + reg_lambda) -
                G_total**2 / (H_total + reg_lambda)
            )
            gains.append(gain)
        thresholds.append(hist.bin_edges[b])

    ax = axes[0]
    valid = ~np.isnan(gains)
    t_arr = np.array(thresholds)
    g_arr = np.array(gains)
    ax.plot(t_arr[valid], g_arr[valid], 'b-o', markersize=4)
    if valid.any():
        best_b = np.nanargmax(gains)
        ax.axvline(thresholds[best_b], color='red', linestyle='--',
                   label=f'Best split @ {thresholds[best_b]:.2f}')
        ax.scatter([thresholds[best_b]], [gains[best_b]], color='red', s=100, zorder=5)
    ax.set_xlabel("Split Threshold (Bin Edge)")
    ax.set_ylabel("Split Gain")
    ax.set_title("Gain at Each Bin Threshold")
    ax.legend()

    # Right: O(B) vs O(n) complexity
    ax = axes[1]
    n_range = np.logspace(3, 7, 50)
    B = 255
    cost_exact = n_range * np.log2(n_range)
    cost_hist = B * np.ones_like(n_range)

    ax.loglog(n_range, cost_exact, 'r-', linewidth=2, label='Exact: O(n log n)')
    ax.loglog(n_range, cost_hist * (n_range / n_range[0]),
              'b--', linewidth=2, label='Histogram: O(B) per leaf')
    ax.set_xlabel("Dataset Size (n)")
    ax.set_ylabel("Relative Computation")
    ax.set_title("Split-Finding Complexity")
    ax.legend()

    plt.tight_layout()
    plt.savefig("images/04_split_gain.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 5. Decision Boundary Evolution
# ---------------------------------------------------------------------------

def img_decision_boundary():
    print("Generating: 05_decision_boundary.png")

    np.random.seed(42)
    X1 = np.random.randn(100, 2) + [2, 2]
    X2 = np.random.randn(100, 2) + [-2, -2]
    X = np.vstack([X1, X2])
    y = np.array([1] * 100 + [0] * 100)

    xx, yy = np.meshgrid(np.linspace(-6, 6, 80), np.linspace(-6, 6, 80))
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("Decision Boundary Evolution (Leaf-wise Growth)", fontsize=13, fontweight='bold')

    for ax, n_est in zip(axes, [1, 5, 15, 50]):
        clf = LightGBMClassifier(
            n_estimators=n_est, learning_rate=0.2, num_leaves=15,
            min_data_in_leaf=5, random_state=42
        )
        clf.fit(X, y)
        proba = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

        ax.contourf(xx, yy, proba, levels=50, cmap='RdYlBu', alpha=0.7, vmin=0, vmax=1)
        ax.contour(xx, yy, proba, levels=[0.5], colors='black', linewidths=1.5)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', s=8, alpha=0.5)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=8, alpha=0.5)
        acc = clf.score(X, y)
        ax.set_title(f"n_estimators={n_est}\nacc={acc:.3f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("images/05_decision_boundary.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 6. Regression Performance
# ---------------------------------------------------------------------------

def img_regression():
    print("Generating: 06_regression.png")

    np.random.seed(42)
    X = np.sort(np.random.uniform(-3, 3, 300)).reshape(-1, 1)
    y = np.sin(X[:, 0] * 1.5) * np.cos(X[:, 0]) + np.random.randn(300) * 0.15

    X_train, y_train = X[:240], y[:240]
    X_test, y_test = X[240:], y[240:]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("LightGBM Regression: Nonlinear Function Approximation", fontsize=13)

    n_estimators_list = [10, 50, 200]
    for ax, n_est in zip(axes, n_estimators_list):
        reg = LightGBMRegressor(
            n_estimators=n_est, learning_rate=0.1, num_leaves=31,
            min_data_in_leaf=5, random_state=42
        )
        reg.fit(X_train, y_train)

        x_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
        y_plot = reg.predict(x_plot)
        r2_test = reg.score(X_test, y_test)

        ax.scatter(X_train, y_train, s=8, alpha=0.4, color='gray', label='Train')
        ax.scatter(X_test, y_test, s=20, alpha=0.6, color='orange', label='Test')
        ax.plot(x_plot, y_plot, 'b-', linewidth=2, label='Prediction')
        ax.set_title(f"n_estimators={n_est}\nR2={r2_test:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if ax == axes[0]:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("images/06_regression.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 7. Feature Importances
# ---------------------------------------------------------------------------

def img_feature_importances():
    print("Generating: 07_feature_importances.png")

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Feature Importances (Gain-based) vs Num Leaves", fontsize=13)

    for ax, num_leaves in zip(axes, [7, 31]):
        clf = LightGBMClassifier(
            n_estimators=100, num_leaves=num_leaves,
            min_data_in_leaf=3, random_state=42
        )
        clf.fit(X, y_bin)
        fi = clf.feature_importances_

        bars = ax.barh(feature_names, fi, color='steelblue', edgecolor='black', alpha=0.8)
        for bar, val in zip(bars, fi):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=9)
        ax.set_xlabel("Normalized Importance (Gain)")
        ax.set_title(f"num_leaves={num_leaves}")
        ax.set_xlim(0, max(fi) * 1.2)

    plt.tight_layout()
    plt.savefig("images/07_feature_importances.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 8. Regularization Effect
# ---------------------------------------------------------------------------

def img_regularization():
    print("Generating: 08_regularization.png")

    np.random.seed(42)
    X_train = np.sort(np.random.uniform(-3, 3, 80)).reshape(-1, 1)
    y_train = np.sin(X_train[:, 0] * 2) + np.random.randn(80) * 0.2

    X_test = np.sort(np.random.uniform(-3, 3, 100)).reshape(-1, 1)
    y_test = np.sin(X_test[:, 0] * 2) + np.random.randn(100) * 0.2

    lambdas = [0.01, 1.0, 10.0, 100.0]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("L2 Regularization Effect (reg_lambda)", fontsize=13)

    x_plot = np.linspace(-3.5, 3.5, 200).reshape(-1, 1)

    for ax, lam in zip(axes, lambdas):
        reg = LightGBMRegressor(
            n_estimators=50, learning_rate=0.1, num_leaves=31,
            reg_lambda=lam, min_data_in_leaf=3, random_state=42
        )
        reg.fit(X_train, y_train)
        y_plot = reg.predict(x_plot)
        r2 = reg.score(X_test, y_test)

        ax.scatter(X_train, y_train, s=12, alpha=0.5, color='gray', label='Train')
        ax.plot(x_plot, y_plot, 'b-', linewidth=2)
        ax.set_title(f"lambda={lam}\nR2={r2:.3f}")
        ax.set_xlabel("x")
        ax.set_ylim(-2.5, 2.5)
        if ax == axes[0]:
            ax.set_ylabel("y")

    plt.tight_layout()
    plt.savefig("images/08_regularization.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating LightGBM images...")
    print("=" * 60)

    img_histogram_binning()
    img_leafwise_vs_levelwise()
    img_goss_sampling()
    img_split_gain()
    img_decision_boundary()
    img_regression()
    img_feature_importances()
    img_regularization()

    print("=" * 60)
    print("All 8 images saved to images/")
