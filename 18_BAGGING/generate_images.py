"""
Generate visualizations for Bagging From-Scratch Implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bagging_scratch import BaggingClassifier, BaggingRegressor, bootstrap_sample, oob_indices

os.makedirs("images", exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Bootstrap sampling illustration
# ---------------------------------------------------------------------------

def img_bootstrap():
    print("Generating: 01_bootstrap_sampling.png")
    np.random.seed(42)
    n = 20
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Bootstrap Sampling: Draw n samples with replacement", fontsize=13, fontweight='bold')

    for ax, seed in zip(axes, [0, 1, 2]):
        np.random.seed(seed)
        boot = bootstrap_sample(n, n)
        oob = oob_indices(boot, n)
        counts = np.bincount(boot, minlength=n)

        colors = ['#F44336' if c == 0 else '#2196F3' for c in counts]
        bars = ax.bar(range(n), counts, color=colors, edgecolor='black', alpha=0.8)
        ax.axhline(1, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_title(f"Bootstrap {seed+1}\n{n - len(oob)} unique | {len(oob)} OOB ({len(oob)/n*100:.0f}%)")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Times selected")
        ax.set_xticks(range(0, n, 2))

    from matplotlib.patches import Patch
    legend = [Patch(color='#2196F3', label='In-bag'), Patch(color='#F44336', label='OOB')]
    axes[0].legend(handles=legend, fontsize=9)

    plt.tight_layout()
    plt.savefig("images/01_bootstrap_sampling.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 2. OOB rate convergence
# ---------------------------------------------------------------------------

def img_oob_rate():
    print("Generating: 02_oob_rate.png")
    ns = np.arange(2, 300)
    theoretical = (1 - 1/ns)**ns
    limit = np.exp(-1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ns, theoretical, 'b-', linewidth=2, label=r'$(1 - 1/n)^n$')
    ax.axhline(limit, color='red', linestyle='--', linewidth=1.5, label=f'$e^{{-1}}$ = {limit:.4f}')
    ax.fill_between(ns, theoretical, limit, alpha=0.1, color='blue')
    ax.set_xlabel("Training set size $n$")
    ax.set_ylabel("P(sample not selected)")
    ax.set_title("OOB Rate: Probability a Sample is Out-of-Bag\nConverges to $e^{-1} \\approx 36.8\\%$", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xlim(2, 300)
    ax.set_ylim(0.30, 0.60)

    # Annotate
    ax.annotate(f'Limit = {limit:.4f}\n(~36.8%)', xy=(200, limit),
                xytext=(150, limit + 0.08),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig("images/02_oob_rate.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 3. Variance reduction: single tree vs bagging
# ---------------------------------------------------------------------------

def img_variance_reduction():
    print("Generating: 03_variance_reduction.png")
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)

    n_trials = 30
    single_accs, bag_accs_10, bag_accs_50 = [], [], []

    for seed in range(n_trials):
        np.random.seed(seed)
        idx = np.random.permutation(len(y_bin))
        Xs, ys = X[idx], y_bin[idx]
        split = int(0.6 * len(ys))

        dt = DecisionTreeClassifier(random_state=seed)
        dt.fit(Xs[:split], ys[:split])
        single_accs.append(dt.score(Xs[split:], ys[split:]))

        for n_est, lst in [(10, bag_accs_10), (50, bag_accs_50)]:
            bg = BaggingClassifier(n_estimators=n_est, random_state=seed)
            bg.fit(Xs[:split], ys[:split])
            lst.append(bg.score(Xs[split:], ys[split:]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Variance Reduction: Single Tree vs Bagging", fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.plot(single_accs, 'r-o', markersize=4, alpha=0.7, label=f'Single DT  std={np.std(single_accs):.3f}')
    ax.plot(bag_accs_10, 'b-s', markersize=4, alpha=0.7, label=f'Bagging B=10  std={np.std(bag_accs_10):.3f}')
    ax.plot(bag_accs_50, 'g-^', markersize=4, alpha=0.7, label=f'Bagging B=50  std={np.std(bag_accs_50):.3f}')
    ax.set_xlabel("Trial")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy per Trial")
    ax.legend(fontsize=9)
    ax.set_ylim(0.5, 1.05)

    ax = axes[1]
    data = [single_accs, bag_accs_10, bag_accs_50]
    labels = ['Single DT', 'Bagging\nB=10', 'Bagging\nB=50']
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ['#F44336', '#2196F3', '#4CAF50']
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Distribution of Test Accuracy\n(lower spread = lower variance)")

    plt.tight_layout()
    plt.savefig("images/03_variance_reduction.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 4. Decision boundary: single tree vs bagging
# ---------------------------------------------------------------------------

def img_decision_boundary():
    print("Generating: 04_decision_boundary.png")
    from sklearn.tree import DecisionTreeClassifier

    np.random.seed(42)
    X1 = np.random.randn(80, 2) + [2, 2]
    X2 = np.random.randn(80, 2) + [-2, -2]
    X3 = np.random.randn(40, 2) + [2, -2]
    X = np.vstack([X1, X2, X3])
    y = np.array([1]*80 + [0]*80 + [1]*40)

    xx, yy = np.meshgrid(np.linspace(-6, 6, 80), np.linspace(-6, 6, 80))
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Decision Boundary: Single Tree vs Bagging", fontsize=13, fontweight='bold')

    configs = [
        ('Single Decision Tree', DecisionTreeClassifier(random_state=42), None),
        ('Bagging B=5', None, 5),
        ('Bagging B=50', None, 50),
    ]

    for ax, (title, estimator, n_est) in zip(axes, configs):
        if estimator is not None:
            estimator.fit(X, y)
            proba = estimator.predict_proba(grid)[:, 1].reshape(xx.shape)
        else:
            clf = BaggingClassifier(n_estimators=n_est, random_state=42)
            clf.fit(X, y)
            proba = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

        ax.contourf(xx, yy, proba, levels=50, cmap='RdYlBu', alpha=0.7, vmin=0, vmax=1)
        ax.contour(xx, yy, proba, levels=[0.5], colors='black', linewidths=1.5)
        ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', s=10, alpha=0.5)
        ax.scatter(X[y==1, 0], X[y==1, 1], c='red', s=10, alpha=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("images/04_decision_boundary.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 5. OOB score vs test score
# ---------------------------------------------------------------------------

def img_oob_vs_test():
    print("Generating: 05_oob_vs_test.png")
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.3, random_state=42
    )

    n_estimators_range = [5, 10, 20, 30, 50, 75, 100]
    oob_scores, test_scores = [], []

    for n_est in n_estimators_range:
        clf = BaggingClassifier(
            n_estimators=n_est, oob_score=True, random_state=42
        )
        clf.fit(X_train, y_train)
        oob_scores.append(clf.oob_score_)
        test_scores.append(clf.score(X_test, y_test))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(n_estimators_range, oob_scores, 'b-o', linewidth=2, label='OOB Score (free estimate)')
    ax.plot(n_estimators_range, test_scores, 'r--s', linewidth=2, label='Test Score')
    ax.set_xlabel("Number of Estimators $B$")
    ax.set_ylabel("Accuracy")
    ax.set_title("OOB Score vs Test Score as $B$ Increases\n(OOB approximates test performance for free)")
    ax.legend(fontsize=11)
    ax.set_ylim(0.8, 1.02)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("images/05_oob_vs_test.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 6. Regression: variance reduction
# ---------------------------------------------------------------------------

def img_regression():
    print("Generating: 06_regression.png")
    from sklearn.tree import DecisionTreeRegressor

    np.random.seed(42)
    X = np.sort(np.random.uniform(-3, 3, 100)).reshape(-1, 1)
    y = np.sin(X[:, 0] * 2) + np.random.randn(100) * 0.3
    x_plot = np.linspace(-3.5, 3.5, 300).reshape(-1, 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Bagging Regression: Variance Reduction", fontsize=13)

    for ax, label, model in zip(axes,
        ['Single Tree (overfit)', 'Bagging B=5', 'Bagging B=50'],
        [DecisionTreeRegressor(random_state=42),
         BaggingRegressor(n_estimators=5, random_state=42),
         BaggingRegressor(n_estimators=50, random_state=42)]):

        model.fit(X, y)
        r2 = model.score(X, y)
        ax.scatter(X, y, s=15, alpha=0.6, color='gray')
        ax.plot(x_plot, model.predict(x_plot), 'b-', linewidth=2)
        ax.set_title(f"{label}\nTrain R2={r2:.3f}")
        ax.set_xlabel("x"); ax.set_ylabel("y")

    plt.tight_layout()
    plt.savefig("images/06_regression.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 7. Feature importances
# ---------------------------------------------------------------------------

def img_feature_importances():
    print("Generating: 07_feature_importances.png")
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Feature Importances — Averaged Across Bootstrap Estimators", fontsize=12)

    for ax, n_est in zip(axes, [10, 100]):
        clf = BaggingClassifier(n_estimators=n_est, random_state=42)
        clf.fit(X, y_bin)
        fi = clf.feature_importances_
        bars = ax.barh(feature_names, fi, color='steelblue', edgecolor='black', alpha=0.8)
        for bar, val in zip(bars, fi):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)
        ax.set_xlabel("Normalized Importance")
        ax.set_title(f"B={n_est} estimators")
        ax.set_xlim(0, max(fi) * 1.3)

    plt.tight_layout()
    plt.savefig("images/07_feature_importances.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 8. Max samples / max features effect
# ---------------------------------------------------------------------------

def img_subsampling_effect():
    print("Generating: 08_subsampling_effect.png")
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.3, random_state=42
    )

    sample_fracs = [0.3, 0.5, 0.7, 0.9, 1.0]
    feat_fracs   = [0.3, 0.5, 0.7, 0.9, 1.0]

    accs_samp, accs_feat = [], []
    for sf in sample_fracs:
        clf = BaggingClassifier(n_estimators=30, max_samples=sf, random_state=42)
        clf.fit(X_train, y_train)
        accs_samp.append(clf.score(X_test, y_test))

    for ff in feat_fracs:
        clf = BaggingClassifier(n_estimators=30, max_features=ff, random_state=42)
        clf.fit(X_train, y_train)
        accs_feat.append(clf.score(X_test, y_test))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Effect of max_samples and max_features on Test Accuracy", fontsize=12)

    axes[0].plot(sample_fracs, accs_samp, 'b-o', linewidth=2, markersize=8)
    axes[0].set_xlabel("max_samples (fraction)")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("Row Subsampling Effect")
    axes[0].set_ylim(0.8, 1.02)
    axes[0].grid(alpha=0.3)

    axes[1].plot(feat_fracs, accs_feat, 'r-s', linewidth=2, markersize=8)
    axes[1].set_xlabel("max_features (fraction)")
    axes[1].set_ylabel("Test Accuracy")
    axes[1].set_title("Feature Subsampling Effect")
    axes[1].set_ylim(0.8, 1.02)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("images/08_subsampling_effect.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating Bagging images...")
    print("=" * 60)

    img_bootstrap()
    img_oob_rate()
    img_variance_reduction()
    img_decision_boundary()
    img_oob_vs_test()
    img_regression()
    img_feature_importances()
    img_subsampling_effect()

    print("=" * 60)
    print("All 8 images saved to images/")
