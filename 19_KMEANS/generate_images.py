"""
Generate visualizations for K-Means From-Scratch Implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmeans_scratch import KMeans, silhouette_score, davies_bouldin_score, elbow_scores

os.makedirs("images", exist_ok=True)

CMAP = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA',
        '#00ACC1', '#F4511E', '#3949AB']


# ---------------------------------------------------------------------------
# 1. Lloyd's algorithm: convergence steps
# ---------------------------------------------------------------------------

def img_convergence_steps():
    print("Generating: 01_convergence_steps.png")
    np.random.seed(42)
    X = np.vstack([np.random.randn(40, 2) + c for c in [[3, 3], [-3, 3], [0, -3]]])

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("Lloyd's Algorithm: Convergence Steps", fontsize=13, fontweight='bold')

    k = 3
    rng = np.random.RandomState(10)
    idx = rng.choice(len(X), size=k, replace=False)
    centroids = X[idx].copy()

    for step, ax in enumerate(axes):
        # E-step: assign
        diffs = X[:, None, :] - centroids[None, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        labels = np.argmin(dists, axis=1)

        for j in range(k):
            mask = labels == j
            ax.scatter(X[mask, 0], X[mask, 1], c=CMAP[j], s=25, alpha=0.7)
            ax.scatter(*centroids[j], c=CMAP[j], s=200, marker='*',
                       edgecolors='black', linewidths=1.2, zorder=5)

        title = f"Step {step}" if step > 0 else "Initialisation"
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        if step < 3:
            # M-step: recompute centroids
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if (labels == j).any() else centroids[j]
                for j in range(k)
            ])
            # Draw shift arrows
            for j in range(k):
                ax.annotate('', xy=new_centroids[j], xytext=centroids[j],
                            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
            centroids = new_centroids

    plt.tight_layout()
    plt.savefig("images/01_convergence_steps.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 2. Voronoi / decision boundary
# ---------------------------------------------------------------------------

def img_voronoi():
    print("Generating: 02_voronoi_decision_boundary.png")
    np.random.seed(42)
    centers_true = np.array([[4, 4], [-4, 4], [0, -4], [4, -4]])
    X = np.vstack([np.random.randn(60, 2) + c for c in centers_true])

    km = KMeans(n_clusters=4, random_state=42)
    km.fit(X)

    xx, yy = np.meshgrid(np.linspace(-8, 8, 300), np.linspace(-8, 8, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = km.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
                colors=CMAP[:4], alpha=0.25)
    ax.contour(xx, yy, Z, levels=[0.5, 1.5, 2.5], colors='black', linewidths=0.8)

    for j in range(4):
        mask = km.labels_ == j
        ax.scatter(X[mask, 0], X[mask, 1], c=CMAP[j], s=25, alpha=0.8)
        ax.scatter(*km.cluster_centers_[j], c=CMAP[j], s=250,
                   marker='*', edgecolors='black', linewidths=1.2, zorder=5)

    ax.set_title("Voronoi Regions: K-Means Decision Boundary\n(K=4)", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("images/02_voronoi_decision_boundary.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 3. KMeans++ vs random init
# ---------------------------------------------------------------------------

def img_init_comparison():
    print("Generating: 03_init_comparison.png")
    np.random.seed(7)
    X = np.vstack([np.random.randn(50, 2) + c
                   for c in [[5, 5], [-5, 5], [-5, -5], [5, -5]]])

    n_trials = 30
    inertias_pp, inertias_rnd = [], []

    for seed in range(n_trials):
        km_pp = KMeans(n_clusters=4, init='k-means++', n_init=1, random_state=seed)
        km_rnd = KMeans(n_clusters=4, init='random', n_init=1, random_state=seed)
        km_pp.fit(X)
        km_rnd.fit(X)
        inertias_pp.append(km_pp.inertia_)
        inertias_rnd.append(km_rnd.inertia_)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("K-Means++ vs Random Initialisation", fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.plot(inertias_rnd, 'r-o', markersize=5, alpha=0.7,
            label=f'Random  std={np.std(inertias_rnd):.1f}')
    ax.plot(inertias_pp, 'b-s', markersize=5, alpha=0.7,
            label=f'K-Means++  std={np.std(inertias_pp):.1f}')
    ax.set_xlabel("Trial"); ax.set_ylabel("Inertia (WCSS)")
    ax.set_title("Inertia per Trial (n_init=1)")
    ax.legend(fontsize=9)

    ax = axes[1]
    bp = ax.boxplot([inertias_rnd, inertias_pp],
                    tick_labels=['Random', 'K-Means++'], patch_artist=True)
    for patch, c in zip(bp['boxes'], ['#F44336', '#2196F3']):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel("Inertia (WCSS)")
    ax.set_title("Distribution (lower = better)")

    plt.tight_layout()
    plt.savefig("images/03_init_comparison.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 4. Elbow method
# ---------------------------------------------------------------------------

def img_elbow():
    print("Generating: 04_elbow_method.png")
    np.random.seed(42)
    true_k = 4
    X = np.vstack([np.random.randn(60, 2) + c
                   for c in [[4, 4], [-4, 4], [-4, -4], [4, -4]]])

    ks = list(range(1, 11))
    scores = elbow_scores(X, ks, random_state=42)

    # Compute second derivative to find elbow
    diffs = np.diff(scores)
    diffs2 = np.diff(diffs)
    elbow_k = ks[1:-1][np.argmax(np.abs(diffs2))]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, scores, 'b-o', linewidth=2, markersize=8)
    ax.axvline(true_k, color='red', linestyle='--', linewidth=1.5,
               label=f'True K={true_k}')
    ax.axvline(elbow_k, color='green', linestyle=':', linewidth=1.5,
               label=f'Elbow K={elbow_k}')
    ax.set_xlabel("Number of Clusters K")
    ax.set_ylabel("Inertia (WCSS)")
    ax.set_title("Elbow Method: Inertia vs K\n(choose K at the elbow)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("images/04_elbow_method.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 5. Silhouette scores vs K
# ---------------------------------------------------------------------------

def img_silhouette_vs_k():
    print("Generating: 05_silhouette_vs_k.png")
    np.random.seed(42)
    X = np.vstack([np.random.randn(40, 2) + c
                   for c in [[5, 0], [-5, 0], [0, 5]]])

    ks = list(range(2, 9))
    sils, dbs = [], []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        sils.append(silhouette_score(X, km.labels_))
        dbs.append(davies_bouldin_score(X, km.labels_, km.cluster_centers_))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Cluster Quality Metrics vs K (True K=3)", fontsize=13, fontweight='bold')

    axes[0].plot(ks, sils, 'b-o', linewidth=2, markersize=8)
    axes[0].axvline(3, color='red', linestyle='--', linewidth=1.5, label='True K=3')
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Silhouette Score")
    axes[0].set_title("Silhouette (higher = better)")
    axes[0].legend(fontsize=10); axes[0].grid(alpha=0.3)

    axes[1].plot(ks, dbs, 'r-s', linewidth=2, markersize=8)
    axes[1].axvline(3, color='red', linestyle='--', linewidth=1.5, label='True K=3')
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Davies-Bouldin Index")
    axes[1].set_title("Davies-Bouldin (lower = better)")
    axes[1].legend(fontsize=10); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("images/05_silhouette_vs_k.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 6. K-Means failure: non-convex shapes
# ---------------------------------------------------------------------------

def img_failure_cases():
    print("Generating: 06_failure_cases.png")
    from sklearn.datasets import make_circles, make_moons

    np.random.seed(42)
    X_circ, y_circ = make_circles(n_samples=200, factor=0.5, noise=0.08, random_state=42)
    X_moon, y_moon = make_moons(n_samples=200, noise=0.1, random_state=42)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("K-Means Limitations: Non-Convex Clusters", fontsize=13, fontweight='bold')

    datasets = [(X_circ, y_circ, 'Circles (true)'), (X_moon, y_moon, 'Moons (true)')]

    for col, (X, y_true, title) in enumerate(datasets):
        # True labels
        ax = axes[0, col]
        for j in range(2):
            ax.scatter(X[y_true == j, 0], X[y_true == j, 1],
                       c=CMAP[j], s=20, alpha=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        # K-Means labels
        ax = axes[1, col]
        km = KMeans(n_clusters=2, random_state=42)
        km.fit(X)
        for j in range(2):
            mask = km.labels_ == j
            ax.scatter(X[mask, 0], X[mask, 1], c=CMAP[j], s=20, alpha=0.8)
        ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                   c='black', s=200, marker='*', zorder=5)
        sil = silhouette_score(X, km.labels_)
        ax.set_title(f'K-Means (K=2)  sil={sil:.2f}\n[Fails on non-convex shapes]', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("images/06_failure_cases.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 7. Effect of K: different cluster counts
# ---------------------------------------------------------------------------

def img_effect_of_k():
    print("Generating: 07_effect_of_k.png")
    np.random.seed(42)
    X = np.vstack([np.random.randn(50, 2) * 0.7 + c
                   for c in [[3, 3], [-3, 3], [-3, -3], [3, -3]]])

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("Effect of K on Clustering", fontsize=13, fontweight='bold')

    for ax, k in zip(axes, [2, 3, 4, 6]):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        for j in range(k):
            mask = km.labels_ == j
            ax.scatter(X[mask, 0], X[mask, 1], c=CMAP[j], s=20, alpha=0.8)
        ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                   c='black', s=200, marker='*', zorder=5)
        sil = silhouette_score(X, km.labels_)
        ax.set_title(f'K={k}  inertia={km.inertia_:.0f}\nsil={sil:.3f}', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("images/07_effect_of_k.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 8. Inertia convergence across multiple runs
# ---------------------------------------------------------------------------

def img_multi_run_convergence():
    print("Generating: 08_multi_run_convergence.png")
    np.random.seed(0)
    X = np.vstack([np.random.randn(50, 2) + c
                   for c in [[4, 4], [-4, 4], [-4, -4], [4, -4]]])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Inertia Convergence: Multiple Random Restarts", fontsize=13, fontweight='bold')

    ax = axes[0]
    best_inertia = np.inf
    for run in range(8):
        km = KMeans(n_clusters=4, n_init=1, init='random', random_state=run)
        km.fit(X)
        color = '#2196F3' if km.inertia_ == best_inertia else (
                '#4CAF50' if km.inertia_ < best_inertia else '#9E9E9E')
        if km.inertia_ < best_inertia:
            best_inertia = km.inertia_
            color = '#4CAF50'
        alpha = 0.9 if km.inertia_ == best_inertia else 0.5
        ax.plot(km.inertia_history_, linewidth=1.5, alpha=0.7, label=f'run {run}')

    ax.set_xlabel("Iteration"); ax.set_ylabel("Inertia (WCSS)")
    ax.set_title("Per-run Convergence (random init)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # Compare final inertias
    ax = axes[1]
    finals_rnd = [KMeans(n_clusters=4, n_init=1, init='random', random_state=s).fit(X).inertia_
                  for s in range(20)]
    finals_pp = [KMeans(n_clusters=4, n_init=1, init='k-means++', random_state=s).fit(X).inertia_
                 for s in range(20)]

    ax.hist(finals_rnd, bins=12, alpha=0.6, color='#F44336', label=f'Random  mean={np.mean(finals_rnd):.0f}')
    ax.hist(finals_pp, bins=12, alpha=0.6, color='#2196F3', label=f'K-Means++  mean={np.mean(finals_pp):.0f}')
    ax.set_xlabel("Final Inertia")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Final Inertia\n(20 single-run experiments)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("images/08_multi_run_convergence.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating K-Means images...")
    print("=" * 60)

    img_convergence_steps()
    img_voronoi()
    img_init_comparison()
    img_elbow()
    img_silhouette_vs_k()
    img_failure_cases()
    img_effect_of_k()
    img_multi_run_convergence()

    print("=" * 60)
    print("All 8 images saved to images/")
