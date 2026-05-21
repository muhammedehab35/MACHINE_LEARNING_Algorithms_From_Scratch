"""
Generate visualizations for K-Medoids (PAM) From-Scratch Implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '19_KMEANS'))
from kmedoids_scratch import KMedoids, pairwise_distances, elbow_costs
from kmeans_scratch import KMeans

os.makedirs("images", exist_ok=True)

CMAP = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']


# ---------------------------------------------------------------------------
# 1. K-Medoids vs K-Means: medoid is an actual data point
# ---------------------------------------------------------------------------

def img_medoid_vs_centroid():
    print("Generating: 01_medoid_vs_centroid.png")
    np.random.seed(42)
    X = np.vstack([np.random.randn(60, 2) * 0.8 + c for c in [[4, 4], [-4, 4], [0, -4]]])

    km_means = KMeans(n_clusters=3, random_state=42).fit(X)
    km_meds = KMedoids(n_clusters=3, random_state=42).fit(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("K-Means Centroid vs K-Medoids Medoid", fontsize=13, fontweight='bold')

    for ax, (title, centers, is_medoid) in zip(axes, [
        ('K-Means: centroids may not be data points', km_means.cluster_centers_, False),
        ('K-Medoids: medoids ARE data points', km_meds.cluster_centers_, True),
    ]):
        labels = km_means.labels_ if not is_medoid else km_meds.labels_
        for j in range(3):
            mask = labels == j
            ax.scatter(X[mask, 0], X[mask, 1], c=CMAP[j], s=25, alpha=0.7)

        for j, c in enumerate(centers):
            marker = 'D' if not is_medoid else '*'
            size = 200 if not is_medoid else 350
            ax.scatter(*c, c=CMAP[j], s=size, marker=marker,
                       edgecolors='black', linewidths=1.5, zorder=6,
                       label=f'Centroid {j}' if not is_medoid else f'Medoid {j}')

        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("images/01_medoid_vs_centroid.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 2. PAM SWAP step illustration
# ---------------------------------------------------------------------------

def img_swap_steps():
    print("Generating: 02_swap_steps.png")
    np.random.seed(7)
    X = np.vstack([np.random.randn(30, 2) * 0.6 + c for c in [[3, 3], [-3, 3], [0, -3]]])

    km = KMedoids(n_clusters=3, init='random', n_init=1, random_state=7)
    km.fit(X)

    fig, axes = plt.subplots(1, len(km.cost_history_), figsize=(5 * len(km.cost_history_), 4.5))
    if len(km.cost_history_) == 1:
        axes = [axes]
    fig.suptitle("PAM SWAP Iterations: Cost Decreasing to Convergence", fontsize=12, fontweight='bold')

    # Re-run step by step for visualization
    from kmedoids_scratch import pairwise_distances as pd2, _init_random
    D = pd2(X)
    rng = np.random.RandomState(7)
    medoids = _init_random(len(X), 3, rng)

    for step, ax in enumerate(axes):
        labels = np.argmin(D[:, medoids], axis=1)
        cost = float(np.sum(D[np.arange(len(X)), medoids[labels]]))

        for j in range(3):
            mask = labels == j
            ax.scatter(X[mask, 0], X[mask, 1], c=CMAP[j], s=25, alpha=0.7)
            ax.scatter(*X[medoids[j]], c=CMAP[j], s=300, marker='*',
                       edgecolors='black', linewidths=1.5, zorder=5)

        ax.set_title(f'Step {step}\nCost = {cost:.1f}', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        if step < len(axes) - 1:
            # do one SWAP
            n = len(X)
            non_medoids = [i for i in range(n) if i not in set(medoids)]
            dists_to_med = D[:, medoids]
            sorted_d = np.sort(dists_to_med, axis=1)
            d1 = sorted_d[:, 0]
            d2 = sorted_d[:, 1] if len(medoids) > 1 else sorted_d[:, 0]

            best_delta = 0; best_mi = -1; best_o = -1
            for mi in range(3):
                assigned = labels == mi
                for o in non_medoids:
                    d_jo = D[:, o]
                    delta = float(np.sum(
                        np.where(assigned, np.minimum(d_jo, d2) - d1,
                                 np.minimum(d_jo - d1, 0.0))
                    ))
                    if delta < best_delta:
                        best_delta = delta; best_mi = mi; best_o = o

            if best_o != -1:
                medoids = medoids.copy(); medoids[best_mi] = best_o

    plt.tight_layout()
    plt.savefig("images/02_swap_steps.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 3. Robustness to outliers: K-Means vs K-Medoids
# ---------------------------------------------------------------------------

def img_outlier_robustness():
    print("Generating: 03_outlier_robustness.png")
    np.random.seed(42)
    X_clean = np.vstack([np.random.randn(50, 2) * 0.5 + c for c in [[4, 4], [-4, -4]]])
    outliers = np.array([[20, 20], [-20, 20], [20, -20], [0, 25], [-25, 0]])
    X = np.vstack([X_clean, outliers])

    km_means = KMeans(n_clusters=2, random_state=42).fit(X)
    km_meds = KMedoids(n_clusters=2, random_state=42).fit(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Outlier Robustness: K-Means vs K-Medoids", fontsize=13, fontweight='bold')

    for ax, (title, centers, labels_arr) in zip(axes, [
        ('K-Means (centroids pulled by outliers)', km_means.cluster_centers_, km_means.labels_),
        ('K-Medoids (medoids robust to outliers)', km_meds.cluster_centers_, km_meds.labels_),
    ]):
        for j in range(2):
            mask = labels_arr == j
            ax.scatter(X[mask, 0], X[mask, 1], c=CMAP[j], s=25, alpha=0.6)
        ax.scatter(outliers[:, 0], outliers[:, 1], c='black', s=100,
                   marker='x', linewidths=2, label='Outliers', zorder=5)
        for j, c in enumerate(centers):
            ax.scatter(*c, c=CMAP[j], s=300, marker='D' if 'Means' in title else '*',
                       edgecolors='black', linewidths=1.5, zorder=6)
        # Draw true cluster centers
        for tc in [[4, 4], [-4, -4]]:
            ax.scatter(*tc, c='gold', s=200, marker='+', linewidths=2.5, zorder=7)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    legend = [mpatches.Patch(color='black', label='Outlier'),
              mpatches.Patch(color='gold', label='True center')]
    axes[0].legend(handles=legend, fontsize=8)

    plt.tight_layout()
    plt.savefig("images/03_outlier_robustness.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 4. Effect of distance metric
# ---------------------------------------------------------------------------

def img_metrics():
    print("Generating: 04_metrics_comparison.png")
    np.random.seed(42)
    X = np.vstack([np.random.randn(50, 2) * 0.7 + c for c in [[4, 4], [-4, 4], [0, -4]]])

    metrics = ['euclidean', 'manhattan', 'cosine']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("K-Medoids with Different Distance Metrics", fontsize=12, fontweight='bold')

    for ax, metric in zip(axes, metrics):
        km = KMedoids(n_clusters=3, metric=metric, random_state=42)
        km.fit(X)
        for j in range(3):
            mask = km.labels_ == j
            ax.scatter(X[mask, 0], X[mask, 1], c=CMAP[j], s=25, alpha=0.7)
            ax.scatter(*km.cluster_centers_[j], c=CMAP[j], s=300, marker='*',
                       edgecolors='black', linewidths=1.5, zorder=5)
        ax.set_title(f'Metric: {metric}\nCost={km.inertia_:.1f}', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("images/04_metrics_comparison.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 5. Elbow method: total cost vs K
# ---------------------------------------------------------------------------

def img_elbow():
    print("Generating: 05_elbow_method.png")
    np.random.seed(42)
    X = np.vstack([np.random.randn(40, 2) * 0.7 + c
                   for c in [[5, 5], [-5, 5], [-5, -5], [5, -5]]])

    ks = list(range(1, 9))
    costs = elbow_costs(X, ks, random_state=42)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, costs, 'b-o', linewidth=2, markersize=8)
    ax.axvline(4, color='red', linestyle='--', linewidth=1.5, label='True K=4')
    ax.set_xlabel("Number of Clusters K")
    ax.set_ylabel("Total Cost (sum of distances)")
    ax.set_title("Elbow Method: Total Cost vs K\n(K-Medoids uses L1-like objective)")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("images/05_elbow_method.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 6. BUILD vs SWAP convergence
# ---------------------------------------------------------------------------

def img_build_swap():
    print("Generating: 06_build_vs_swap.png")
    np.random.seed(0)
    X = np.vstack([np.random.randn(50, 2) * 0.8 + c for c in [[4, 4], [-4, 4], [4, -4], [-4, -4]]])

    n_trials = 20
    costs_build, costs_rnd = [], []
    iters_build, iters_rnd = [], []

    for seed in range(n_trials):
        kb = KMedoids(n_clusters=4, init='build', random_state=seed).fit(X)
        kr = KMedoids(n_clusters=4, init='random', n_init=1, random_state=seed).fit(X)
        costs_build.append(kb.inertia_)
        costs_rnd.append(kr.inertia_)
        iters_build.append(kb.n_iter_)
        iters_rnd.append(kr.n_iter_)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("PAM BUILD vs Random Init", fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.plot(costs_rnd, 'r-o', markersize=5, alpha=0.7,
            label=f'Random  mean={np.mean(costs_rnd):.1f}')
    ax.plot(costs_build, 'b-s', markersize=5, alpha=0.7,
            label=f'BUILD   mean={np.mean(costs_build):.1f}')
    ax.set_xlabel("Trial"); ax.set_ylabel("Total Cost")
    ax.set_title("Final Cost per Trial")
    ax.legend(fontsize=9)

    ax = axes[1]
    bp = ax.boxplot([costs_rnd, costs_build],
                    tick_labels=['Random', 'BUILD'], patch_artist=True)
    for patch, c in zip(bp['boxes'], ['#F44336', '#2196F3']):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel("Total Cost")
    ax.set_title("Distribution of Final Cost\n(lower = better)")

    plt.tight_layout()
    plt.savefig("images/06_build_vs_swap.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 7. K-Medoids on non-spherical data (Manhattan street grid)
# ---------------------------------------------------------------------------

def img_manhattan_grid():
    print("Generating: 07_manhattan_grid.png")
    np.random.seed(42)
    # Points on a grid (city blocks)
    grid_x, grid_y = np.meshgrid(np.arange(-4, 5, 2), np.arange(-4, 5, 2))
    X_grid = np.c_[grid_x.ravel(), grid_y.ravel()].astype(float)
    X_grid += np.random.randn(*X_grid.shape) * 0.25

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("K-Medoids on Grid Data: Euclidean vs Manhattan Metric", fontsize=12, fontweight='bold')

    for ax, metric in zip(axes, ['euclidean', 'manhattan']):
        km = KMedoids(n_clusters=4, metric=metric, random_state=42).fit(X_grid)
        for j in range(4):
            mask = km.labels_ == j
            ax.scatter(X_grid[mask, 0], X_grid[mask, 1], c=CMAP[j], s=60, alpha=0.8)
            ax.scatter(*km.cluster_centers_[j], c=CMAP[j], s=350, marker='*',
                       edgecolors='black', linewidths=1.5, zorder=5)
        ax.set_title(f'{metric.capitalize()} distance\nCost={km.inertia_:.2f}', fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("images/07_manhattan_grid.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 8. Cost convergence history: multiple runs
# ---------------------------------------------------------------------------

def img_cost_convergence():
    print("Generating: 08_cost_convergence.png")
    np.random.seed(0)
    X = np.vstack([np.random.randn(50, 2) + c for c in [[4, 4], [-4, 4], [0, -4]]])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PAM Cost Convergence: Random Init Runs", fontsize=13, fontweight='bold')

    ax = axes[0]
    final_costs = []
    for seed in range(10):
        km = KMedoids(n_clusters=3, init='random', n_init=1, random_state=seed).fit(X)
        ax.plot(km.cost_history_, marker='o', markersize=4, linewidth=1.5,
                alpha=0.7, label=f'run {seed}')
        final_costs.append(km.cost_history_[-1])

    ax.set_xlabel("SWAP Iteration"); ax.set_ylabel("Total Cost")
    ax.set_title("Per-run Cost History")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    ax = axes[1]
    costs_rnd = [KMedoids(n_clusters=3, init='random', n_init=1, random_state=s).fit(X).inertia_
                 for s in range(30)]
    costs_build = [KMedoids(n_clusters=3, init='build', random_state=s).fit(X).inertia_
                   for s in range(30)]

    ax.hist(costs_rnd, bins=15, alpha=0.6, color='#F44336',
            label=f'Random mean={np.mean(costs_rnd):.1f}')
    ax.hist(costs_build, bins=15, alpha=0.6, color='#2196F3',
            label=f'BUILD  mean={np.mean(costs_build):.1f}')
    ax.set_xlabel("Final Total Cost"); ax.set_ylabel("Count")
    ax.set_title("Distribution of Final Cost (30 runs)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("images/08_cost_convergence.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating K-Medoids images...")
    print("=" * 60)

    img_medoid_vs_centroid()
    img_swap_steps()
    img_outlier_robustness()
    img_metrics()
    img_elbow()
    img_build_swap()
    img_manhattan_grid()
    img_cost_convergence()

    print("=" * 60)
    print("All 8 images saved to images/")
