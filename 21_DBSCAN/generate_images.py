"""
Generate visualizations for DBSCAN From-Scratch Implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '19_KMEANS'))
from dbscan_scratch import DBSCAN, k_dist, cluster_stats
from kmeans_scratch import KMeans

os.makedirs("images", exist_ok=True)

CLUSTER_COLORS = ['#E53935', '#1E88E5', '#43A047', '#FB8C00',
                  '#8E24AA', '#00ACC1', '#F4511E', '#3949AB',
                  '#6D4C41', '#00897B']
NOISE_COLOR = '#BDBDBD'


def _color_for(label):
    if label == -1:
        return NOISE_COLOR
    return CLUSTER_COLORS[label % len(CLUSTER_COLORS)]


# ---------------------------------------------------------------------------
# 1. Core, border, noise point types
# ---------------------------------------------------------------------------

def img_point_types():
    print("Generating: 01_point_types.png")
    np.random.seed(42)
    X_core = np.random.randn(40, 2) * 0.4
    X_border = np.array([[1.0, 0.0], [-0.9, 0.5], [0.0, -1.1],
                          [0.8, -0.7], [-1.0, -0.6]])
    X_noise = np.array([[3.5, 3.5], [-3.5, -3.5], [4.0, -3.0]])
    X = np.vstack([X_core, X_border, X_noise])

    db = DBSCAN(eps=0.7, min_samples=5).fit(X)
    types = db.point_types(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[types == 2, 0], X[types == 2, 1],
               c='#1E88E5', s=60, label='Core', edgecolors='black', linewidths=0.5, alpha=0.8)
    ax.scatter(X[types == 1, 0], X[types == 1, 1],
               c='#43A047', s=60, label='Border', edgecolors='black', linewidths=0.5, alpha=0.8)
    ax.scatter(X[types == 0, 0], X[types == 0, 1],
               c='#E53935', s=80, marker='x', linewidths=2, label='Noise')

    # Draw eps circle around one core point
    center = X[20]
    circle = plt.Circle(center, 0.7, color='#1E88E5', fill=False,
                        linestyle='--', linewidth=1.5, alpha=0.6)
    ax.add_patch(circle)
    ax.annotate(r'$\varepsilon$', xy=center + [0.7, 0], fontsize=13, color='#1E88E5')

    ax.set_title("DBSCAN Point Types: Core / Border / Noise", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("images/01_point_types.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 2. DBSCAN vs K-Means on circles and moons
# ---------------------------------------------------------------------------

def img_vs_kmeans():
    print("Generating: 02_vs_kmeans.png")
    from sklearn.datasets import make_circles, make_moons

    np.random.seed(42)
    X_circ, _ = make_circles(n_samples=250, factor=0.5, noise=0.06, random_state=42)
    X_moon, _ = make_moons(n_samples=250, noise=0.08, random_state=42)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("DBSCAN vs K-Means: Non-Convex Shapes", fontsize=13, fontweight='bold')

    datasets = [
        (X_circ, 'Circles', DBSCAN(eps=0.18, min_samples=5), KMeans(n_clusters=2, random_state=42)),
        (X_moon, 'Moons', DBSCAN(eps=0.22, min_samples=5), KMeans(n_clusters=2, random_state=42)),
    ]

    for row, (X, name, db, km) in enumerate(datasets):
        db.fit(X); km.fit(X)

        # DBSCAN
        ax = axes[row, 0]
        colors = [_color_for(l) for l in db.labels_]
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=20, alpha=0.8)
        ax.set_title(f'DBSCAN on {name}  ({db.n_clusters_} clusters)', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        # K-Means
        ax = axes[row, 1]
        km_colors = [CLUSTER_COLORS[l] for l in km.labels_]
        ax.scatter(X[:, 0], X[:, 1], c=km_colors, s=20, alpha=0.8)
        ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                   c='black', s=200, marker='*', zorder=5)
        ax.set_title(f'K-Means on {name}  (K=2)', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("images/02_vs_kmeans.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 3. Effect of eps
# ---------------------------------------------------------------------------

def img_eps_effect():
    print("Generating: 03_eps_effect.png")
    np.random.seed(42)
    X = np.vstack([np.random.randn(60, 2) * 0.5 + c for c in [[3, 3], [-3, 3], [0, -3]]])

    eps_vals = [0.1, 0.5, 1.0, 3.0]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("Effect of eps (min_samples=5 fixed)", fontsize=12, fontweight='bold')

    for ax, eps in zip(axes, eps_vals):
        db = DBSCAN(eps=eps, min_samples=5).fit(X)
        colors = [_color_for(l) for l in db.labels_]
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=25, alpha=0.8)
        n_noise = (db.labels_ == -1).sum()
        ax.set_title(f'eps={eps}\n{db.n_clusters_} clusters, {n_noise} noise', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    noise_patch = mpatches.Patch(color=NOISE_COLOR, label='Noise')
    axes[0].legend(handles=[noise_patch], fontsize=8)
    plt.tight_layout()
    plt.savefig("images/03_eps_effect.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 4. Effect of min_samples
# ---------------------------------------------------------------------------

def img_min_samples_effect():
    print("Generating: 04_min_samples_effect.png")
    np.random.seed(42)
    X = np.vstack([np.random.randn(60, 2) * 0.5 + c for c in [[3, 3], [-3, 3], [0, -3]]])

    ms_vals = [2, 5, 10, 20]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("Effect of min_samples (eps=0.8 fixed)", fontsize=12, fontweight='bold')

    for ax, ms in zip(axes, ms_vals):
        db = DBSCAN(eps=0.8, min_samples=ms).fit(X)
        colors = [_color_for(l) for l in db.labels_]
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=25, alpha=0.8)
        n_noise = (db.labels_ == -1).sum()
        n_core = len(db.core_sample_indices_)
        ax.set_title(f'min_samples={ms}\n{db.n_clusters_} clusters, {n_noise} noise\n{n_core} core pts', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    noise_patch = mpatches.Patch(color=NOISE_COLOR, label='Noise')
    axes[0].legend(handles=[noise_patch], fontsize=8)
    plt.tight_layout()
    plt.savefig("images/04_min_samples_effect.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 5. k-dist graph for eps selection
# ---------------------------------------------------------------------------

def img_k_dist():
    print("Generating: 05_k_dist_graph.png")
    np.random.seed(42)
    X = np.vstack([np.random.randn(80, 2) * 0.5 + c for c in [[4, 4], [-4, 4], [0, -4]]])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("k-dist Graph: Choosing eps", fontsize=12, fontweight='bold')

    for ax, k in zip(axes, [4, 8]):
        kd = k_dist(X, k=k)
        ax.plot(range(len(kd)), kd, 'b-', linewidth=1.5)
        # Elbow: largest gap between consecutive values
        gaps = np.abs(np.diff(kd))
        elbow_i = int(np.argmax(gaps))
        elbow_eps = float(kd[elbow_i])
        ax.axhline(elbow_eps, color='red', linestyle='--', linewidth=1.5,
                   label=f'Elbow eps={elbow_eps:.2f}')
        ax.scatter([elbow_i], [elbow_eps], color='red', s=100, zorder=5)
        ax.set_xlabel("Points (sorted by distance)")
        ax.set_ylabel(f"{k}-th nearest neighbour distance")
        ax.set_title(f"k={k}: elbow at eps={elbow_eps:.2f}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("images/05_k_dist_graph.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 6. Arbitrary-shape clusters
# ---------------------------------------------------------------------------

def img_arbitrary_shapes():
    print("Generating: 06_arbitrary_shapes.png")
    np.random.seed(42)
    # Spiral
    t1 = np.linspace(0, 3 * np.pi, 150)
    t2 = np.linspace(0, 3 * np.pi, 150)
    X1 = np.c_[t1 * np.cos(t1) / 10, t1 * np.sin(t1) / 10]
    X2 = np.c_[(t2 + np.pi) * np.cos(t2) / 10, (t2 + np.pi) * np.sin(t2) / 10]
    X1 += np.random.randn(*X1.shape) * 0.04
    X2 += np.random.randn(*X2.shape) * 0.04
    X = np.vstack([X1, X2])

    db = DBSCAN(eps=0.15, min_samples=5).fit(X)
    km = KMeans(n_clusters=2, random_state=42).fit(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Arbitrary Shape: Double Spiral", fontsize=12, fontweight='bold')

    ax = axes[0]
    colors = [_color_for(l) for l in db.labels_]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=15, alpha=0.9)
    noise_n = (db.labels_ == -1).sum()
    ax.set_title(f'DBSCAN: {db.n_clusters_} clusters, {noise_n} noise', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    km_colors = [CLUSTER_COLORS[l] for l in km.labels_]
    ax.scatter(X[:, 0], X[:, 1], c=km_colors, s=15, alpha=0.9)
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
               c='black', s=200, marker='*', zorder=5)
    ax.set_title('K-Means (K=2): fails on spiral', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("images/06_arbitrary_shapes.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 7. Density-reachability illustration
# ---------------------------------------------------------------------------

def img_density_reachability():
    print("Generating: 07_density_reachability.png")
    np.random.seed(42)
    X = np.array([
        [0.0, 0.0], [0.3, 0.1], [0.1, 0.35], [-0.2, 0.15], [0.1, -0.2],  # core cluster A
        [0.7, 0.0], [1.0, 0.1], [0.85, 0.3], [1.1, -0.1],                 # cluster B
        [0.5, 0.0],                                                          # bridge border
        [3.0, 3.0],                                                          # noise
    ])
    db = DBSCAN(eps=0.45, min_samples=4).fit(X)

    fig, ax = plt.subplots(figsize=(9, 6))
    types = db.point_types(X)

    colors_type = {2: '#1E88E5', 1: '#43A047', 0: '#E53935'}
    markers_type = {2: 'o', 1: 's', 0: 'x'}
    labels_type = {2: 'Core', 1: 'Border', 0: 'Noise'}

    for t in [2, 1, 0]:
        mask = types == t
        if mask.any():
            ax.scatter(X[mask, 0], X[mask, 1], c=colors_type[t],
                       s=120 if t > 0 else 150, marker=markers_type[t],
                       edgecolors='black' if t < 2 else 'black',
                       linewidths=1.2, label=labels_type[t], zorder=5)

    # Draw eps circles for core points
    for i in db.core_sample_indices_:
        circle = plt.Circle(X[i], db.eps, color='#1E88E5', fill=False,
                            linestyle=':', linewidth=0.8, alpha=0.4)
        ax.add_patch(circle)

    # Annotate density-reachability chain
    ax.annotate('', xy=X[5], xytext=X[0],
                arrowprops=dict(arrowstyle='->', color='purple', lw=2))
    ax.text(0.4, 0.15, 'Density-\nreachable', color='purple', fontsize=8)

    ax.set_title("Density-Reachability: Cluster Expansion via Core Points",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("images/07_density_reachability.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 8. Varying density clusters
# ---------------------------------------------------------------------------

def img_varying_density():
    print("Generating: 08_varying_density.png")
    np.random.seed(42)
    # Cluster 1: tight, high density
    C1 = np.random.randn(80, 2) * 0.2 + [3, 3]
    # Cluster 2: loose, low density
    C2 = np.random.randn(40, 2) * 1.2 + [-3, -3]
    # Noise
    noise = np.random.uniform(-6, 6, (15, 2))
    X = np.vstack([C1, C2, noise])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("DBSCAN on Varying Density Clusters", fontsize=12, fontweight='bold')

    configs = [(0.4, 5, 'Single eps works for tight cluster'),
               (1.5, 5, 'Single eps captures loose cluster\n(may merge with noise)'),
               (0.5, 3, 'Tuned eps+min_samples')]

    for ax, (eps, ms, title) in zip(axes, configs):
        db = DBSCAN(eps=eps, min_samples=ms).fit(X)
        colors = [_color_for(l) for l in db.labels_]
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=25, alpha=0.8)
        n_noise = (db.labels_ == -1).sum()
        ax.set_title(f'{title}\neps={eps}, min_s={ms}, {db.n_clusters_} clusters, {n_noise} noise',
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    noise_patch = mpatches.Patch(color=NOISE_COLOR, label='Noise')
    axes[0].legend(handles=[noise_patch], fontsize=8)
    plt.tight_layout()
    plt.savefig("images/08_varying_density.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating DBSCAN images...")
    print("=" * 60)

    img_point_types()
    img_vs_kmeans()
    img_eps_effect()
    img_min_samples_effect()
    img_k_dist()
    img_arbitrary_shapes()
    img_density_reachability()
    img_varying_density()

    print("=" * 60)
    print("All 8 images saved to images/")
