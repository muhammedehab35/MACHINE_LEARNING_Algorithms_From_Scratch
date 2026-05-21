"""
Generate visualizations for Agglomerative Hierarchical Clustering
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '19_KMEANS'))
from hierarchical_scratch import (
    AgglomerativeClustering, linkage, fcluster,
    cophenetic_correlation, pairwise_distances
)
from kmeans_scratch import KMeans

os.makedirs("images", exist_ok=True)

CMAP = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA',
        '#00ACC1', '#F4511E', '#3949AB']


# ---------------------------------------------------------------------------
# Helper: draw a simple dendrogram from linkage matrix Z
# ---------------------------------------------------------------------------

def _draw_dendrogram(ax, Z, n, title='', leaf_labels=None, color_threshold=None):
    """
    Draw a dendrogram on axes `ax` from linkage matrix Z.
    Leaf nodes are at x = 0, 1, ..., n-1.
    """
    # Assign leaf positions
    pos = {i: float(i) for i in range(n)}   # x position of each node
    heights = {i: 0.0 for i in range(n)}     # height (y) of each node

    node_colors = {}
    color_cycle = ['#E53935', '#1E88E5', '#43A047', '#FB8C00',
                   '#8E24AA', '#00ACC1', '#F4511E']

    # Determine which clusters are formed above color_threshold
    if color_threshold is not None:
        # Find root clusters at the cut
        active_clusters = {}
        col_idx = 0
        for k in range(n - 1):
            ci, cj, h = int(Z[k, 0]), int(Z[k, 1]), Z[k, 2]
            new_id = n + k
            if h > color_threshold:
                # This merge is above the cut: assign different colors to ci, cj
                if ci not in active_clusters:
                    active_clusters[ci] = color_cycle[col_idx % len(color_cycle)]
                    col_idx += 1
                if cj not in active_clusters:
                    active_clusters[cj] = color_cycle[col_idx % len(color_cycle)]
                    col_idx += 1
                active_clusters[new_id] = '#555555'
            else:
                # Below cut: inherit color from children
                ci_col = active_clusters.get(ci, '#555555')
                cj_col = active_clusters.get(cj, '#555555')
                active_clusters[new_id] = ci_col if ci_col == cj_col else '#555555'
    else:
        active_clusters = {}

    for k in range(n - 1):
        ci, cj, h, sz = int(Z[k, 0]), int(Z[k, 1]), Z[k, 2], int(Z[k, 3])
        new_id = n + k

        xi = pos[ci]
        xj = pos[cj]
        new_x = (xi + xj) / 2.0

        color = active_clusters.get(new_id, '#555555')

        # Draw the U-shape
        ax.plot([xi, xi], [heights[ci], h], color=color, linewidth=1.5)
        ax.plot([xj, xj], [heights[cj], h], color=color, linewidth=1.5)
        ax.plot([xi, xj], [h, h], color=color, linewidth=1.5)

        pos[new_id] = new_x
        heights[new_id] = h

    if color_threshold is not None:
        ax.axhline(color_threshold, color='red', linestyle='--', linewidth=1.5,
                   label=f'cut h={color_threshold:.2f}')
        ax.legend(fontsize=8)

    ax.set_title(title, fontsize=9)
    ax.set_xlim(-1, n)
    ax.set_ylabel('Distance')
    ax.set_xticks([])


# ---------------------------------------------------------------------------
# 1. Dendrogram: 4 linkage methods on blobs
# ---------------------------------------------------------------------------

def img_dendrograms():
    print("Generating: 01_dendrograms.png")
    np.random.seed(42)
    X = np.vstack([np.random.randn(10, 2) * 0.5 + c
                   for c in [[5, 5], [-5, 5], [0, -5]]])
    n = len(X)

    methods = ['single', 'complete', 'average', 'ward']
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Dendrogram: 4 Linkage Methods on 3-Blob Data", fontsize=13, fontweight='bold')

    for ax, method in zip(axes, methods):
        Z = linkage(X, method=method)
        _draw_dendrogram(ax, Z, n, title=f'{method.capitalize()} Linkage')

    plt.tight_layout()
    plt.savefig("images/01_dendrograms.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 2. Cluster assignments: 4 linkage methods
# ---------------------------------------------------------------------------

def img_linkage_comparison():
    print("Generating: 02_linkage_comparison.png")
    np.random.seed(42)
    X = np.vstack([np.random.randn(40, 2) * 0.7 + c
                   for c in [[5, 5], [-5, 5], [0, -5]]])

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Cluster Assignments: 4 Linkage Methods (K=3)", fontsize=13, fontweight='bold')

    for ax, method in zip(axes, ['single', 'complete', 'average', 'ward']):
        ac = AgglomerativeClustering(n_clusters=3, linkage_method=method)
        ac.fit(X)
        for j in range(3):
            mask = ac.labels_ == j
            ax.scatter(X[mask, 0], X[mask, 1], c=CMAP[j], s=25, alpha=0.8)
        ax.set_title(f'{method.capitalize()}', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("images/02_linkage_comparison.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 3. Dendrogram cut: different K values
# ---------------------------------------------------------------------------

def img_dendrogram_cuts():
    print("Generating: 03_dendrogram_cuts.png")
    np.random.seed(42)
    X = np.vstack([np.random.randn(12, 2) * 0.5 + c
                   for c in [[5, 5], [-5, 5], [0, -5], [5, -5]]])
    n = len(X)
    Z = linkage(X, method='ward')
    heights = Z[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Dendrogram Cuts at Different Heights (Ward)", fontsize=12, fontweight='bold')

    for ax, k in zip(axes, [2, 3, 4]):
        labels_k = fcluster(Z, k, criterion='maxclust')
        sorted_h = np.sort(heights)[::-1]
        if k >= n:
            cut_h = np.inf
        else:
            cut_h = (sorted_h[k-2] + sorted_h[k-1]) / 2.0

        _draw_dendrogram(ax, Z, n, title=f'K={k} clusters', color_threshold=cut_h)

    plt.tight_layout()
    plt.savefig("images/03_dendrogram_cuts.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 4. Single linkage chaining effect
# ---------------------------------------------------------------------------

def img_chaining():
    print("Generating: 04_chaining_effect.png")
    np.random.seed(42)
    # Two elongated clusters bridged by a few points
    X1 = np.c_[np.linspace(0, 4, 20), np.random.randn(20) * 0.15]
    X2 = np.c_[np.linspace(0, 4, 20), np.random.randn(20) * 0.15 + 2.0]
    X = np.vstack([X1, X2])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Single Linkage Chaining Effect", fontsize=13, fontweight='bold')

    for ax, method in zip(axes, ['single', 'ward']):
        ac = AgglomerativeClustering(n_clusters=2, linkage_method=method)
        ac.fit(X)
        for j in range(2):
            mask = ac.labels_ == j
            ax.scatter(X[mask, 0], X[mask, 1], c=CMAP[j], s=30, alpha=0.9)
        ax.set_title(f'{method.capitalize()} linkage (K=2)', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("images/04_chaining_effect.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 5. Cophenetic correlation vs linkage method
# ---------------------------------------------------------------------------

def img_cophenetic():
    print("Generating: 05_cophenetic_correlation.png")
    np.random.seed(42)
    X = np.vstack([np.random.randn(30, 2) * 0.5 + c
                   for c in [[5, 5], [-5, 5], [0, -5]]])

    methods = ['single', 'complete', 'average', 'ward', 'centroid', 'median']
    coph_vals = []
    for method in methods:
        Z = linkage(X, method=method)
        coph_vals.append(cophenetic_correlation(X, Z))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(methods, coph_vals,
                  color=['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA', '#00ACC1'],
                  edgecolor='black', alpha=0.8)
    for bar, val in zip(bars, coph_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Cophenetic Correlation')
    ax.set_title('Cophenetic Correlation by Linkage Method\n(higher = dendrogram better represents distances)',
                 fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.75, color='red', linestyle='--', linewidth=1.2, label='Good threshold (0.75)')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("images/05_cophenetic_correlation.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 6. Hierarchical vs K-Means on non-convex data
# ---------------------------------------------------------------------------

def img_vs_kmeans():
    print("Generating: 06_vs_kmeans.png")
    from sklearn.datasets import make_moons
    np.random.seed(42)
    X, _ = make_moons(n_samples=200, noise=0.07, random_state=42)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Hierarchical (Single Linkage) vs K-Means on Moons", fontsize=12, fontweight='bold')

    configs = [
        ('Single Linkage (K=2)', AgglomerativeClustering(n_clusters=2, linkage_method='single')),
        ('Ward Linkage (K=2)', AgglomerativeClustering(n_clusters=2, linkage_method='ward')),
        ('K-Means (K=2)', None),
    ]
    for ax, (title, model) in zip(axes, configs):
        if model is None:
            model = KMeans(n_clusters=2, random_state=42)
        model.fit(X)
        labels = model.labels_
        for j in range(2):
            mask = labels == j
            ax.scatter(X[mask, 0], X[mask, 1], c=CMAP[j], s=20, alpha=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("images/06_vs_kmeans.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 7. Ward step-by-step merge illustration
# ---------------------------------------------------------------------------

def img_ward_steps():
    print("Generating: 07_ward_merge_steps.png")
    np.random.seed(42)
    X = np.vstack([np.random.randn(8, 2) * 0.4 + c
                   for c in [[4, 4], [-4, 4], [0, -4]]])
    n = len(X)
    Z = linkage(X, method='ward')

    # Show 4 snapshots: initial, after 5, 15, 20 merges
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("Ward Linkage: Merge Progress", fontsize=12, fontweight='bold')

    for ax, n_merges in zip(axes, [0, 6, 14, n - 1]):
        if n_merges == 0:
            labels = np.arange(n)
        else:
            k = max(1, n - n_merges)
            labels = fcluster(Z, k, criterion='maxclust')

        n_clust = len(np.unique(labels))
        for j in range(n_clust):
            mask = labels == j
            ax.scatter(X[mask, 0], X[mask, 1], c=CMAP[j % len(CMAP)], s=50, alpha=0.85,
                       edgecolors='black', linewidths=0.5)

        ax.set_title(f'{n_merges} merges done\n{n_clust} clusters', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("images/07_ward_merge_steps.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 8. Inertia vs K: hierarchical (Ward) vs K-Means
# ---------------------------------------------------------------------------

def img_inertia_vs_k():
    print("Generating: 08_inertia_vs_k.png")
    np.random.seed(42)
    X = np.vstack([np.random.randn(30, 2) * 0.6 + c
                   for c in [[5, 5], [-5, 5], [-5, -5], [5, -5]]])

    def inertia(X, labels):
        return sum(float(np.sum((X[labels == j] - X[labels == j].mean(0)) ** 2))
                   for j in np.unique(labels))

    Z = linkage(X, method='ward')
    ks = list(range(1, 9))

    inertias_hc = []
    inertias_km = []
    for k in ks:
        labels_hc = fcluster(Z, k, criterion='maxclust')
        inertias_hc.append(inertia(X, labels_hc))
        km = KMeans(n_clusters=k, random_state=42).fit(X)
        inertias_km.append(inertia(X, km.labels_))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, inertias_hc, 'b-o', linewidth=2, markersize=8, label='Ward (hierarchical)')
    ax.plot(ks, inertias_km, 'r--s', linewidth=2, markersize=8, label='K-Means')
    ax.axvline(4, color='green', linestyle=':', linewidth=1.5, label='True K=4')
    ax.set_xlabel('Number of Clusters K')
    ax.set_ylabel('Inertia (WCSS)')
    ax.set_title('Inertia vs K: Ward Hierarchical vs K-Means', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("images/08_inertia_vs_k.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating Hierarchical Clustering images...")
    print("=" * 60)

    img_dendrograms()
    img_linkage_comparison()
    img_dendrogram_cuts()
    img_chaining()
    img_cophenetic()
    img_vs_kmeans()
    img_ward_steps()
    img_inertia_vs_k()

    print("=" * 60)
    print("All 8 images saved to images/")
