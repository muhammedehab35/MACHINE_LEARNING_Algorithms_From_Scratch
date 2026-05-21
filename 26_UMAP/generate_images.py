"""
Generate visualizations for UMAP
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '24_PCA'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '25_TSNE'))
from umap_scratch import UMAP, compute_fuzzy_simplicial_set, find_ab_params
from pca_scratch import PCA
from tsne_scratch import TSNE

os.makedirs("images", exist_ok=True)

CMAP10 = [plt.cm.tab10(i / 10) for i in range(10)]
CMAP = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA',
        '#00ACC1', '#F4511E', '#3949AB']


# ---------------------------------------------------------------------------
# 1. UMAP on MNIST digits
# ---------------------------------------------------------------------------

def img_umap_digits():
    print("Generating: 01_umap_digits.png")
    from sklearn.datasets import load_digits
    np.random.seed(42)
    digits = load_digits()
    idx = np.random.choice(len(digits.data), 500, replace=False)
    X = digits.data[idx].astype(float)
    y = digits.target[idx]

    umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                n_epochs=300, learning_rate=1.0, random_state=42, verbose=False)
    Z = umap.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    for digit in range(10):
        mask = y == digit
        ax.scatter(Z[mask, 0], Z[mask, 1], s=18, alpha=0.8,
                   color=CMAP10[digit], label=str(digit))
    ax.legend(title='Digit', fontsize=8, ncol=2, markerscale=2, loc='upper right')
    ax.set_title('UMAP on 500 MNIST Digits (n_neighbors=15, min_dist=0.1)', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("images/01_umap_digits.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 2. PCA vs UMAP on 4-cluster nonlinear data
# ---------------------------------------------------------------------------

def img_pca_vs_umap():
    print("Generating: 02_pca_vs_umap.png")
    np.random.seed(42)
    n_per = 60
    centers_2d = [(4, 0), (0, 4), (-4, 0), (0, -4)]
    X_list, y_list = [], []
    for k, (cx, cy) in enumerate(centers_2d):
        blob = np.random.randn(n_per, 20) * 0.4
        blob[:, 0] += cx; blob[:, 1] += cy
        X_list.append(blob); y_list.append(np.full(n_per, k))
    X = np.vstack(X_list); y = np.concatenate(y_list)

    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(X)

    umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                n_epochs=300, random_state=42)
    Z_umap = umap.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("PCA vs UMAP on 4-Cluster 20D Data", fontsize=12, fontweight='bold')
    for ax, Z, title in zip(axes, [Z_pca, Z_umap], ['PCA (linear)', 'UMAP (nonlinear)']):
        for k in range(4):
            mask = y == k
            ax.scatter(Z[mask, 0], Z[mask, 1], c=CMAP[k], s=25, alpha=0.8,
                       label=f'Cluster {k}')
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("images/02_pca_vs_umap.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 3. Effect of n_neighbors
# ---------------------------------------------------------------------------

def img_n_neighbors_effect():
    print("Generating: 03_n_neighbors_effect.png")
    np.random.seed(42)
    X = np.vstack([np.random.randn(50, 2) * 0.3 + c
                   for c in [[4, 0], [-4, 0], [0, 4]]])
    y = np.repeat([0, 1, 2], 50)
    X = np.hstack([X, np.random.randn(len(X), 8) * 0.1])

    n_neighbors_list = [3, 10, 20, 50]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("Effect of n_neighbors on UMAP Embedding (3 Blobs in 10D)",
                 fontsize=12, fontweight='bold')

    for ax, nn in zip(axes, n_neighbors_list):
        umap = UMAP(n_components=2, n_neighbors=nn, min_dist=0.1,
                    n_epochs=300, random_state=0)
        Z = umap.fit_transform(X)
        for k in range(3):
            ax.scatter(Z[y == k, 0], Z[y == k, 1], c=CMAP[k], s=20, alpha=0.85)
        ax.set_title(f'n_neighbors = {nn}', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("images/03_n_neighbors_effect.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 4. Effect of min_dist
# ---------------------------------------------------------------------------

def img_min_dist_effect():
    print("Generating: 04_min_dist_effect.png")
    np.random.seed(42)
    X = np.vstack([np.random.randn(60, 5) * 0.4 + c
                   for c in [[4, 0, 0, 0, 0], [-4, 0, 0, 0, 0], [0, 4, 0, 0, 0]]])
    y = np.repeat([0, 1, 2], 60)

    min_dists = [0.0, 0.1, 0.5, 1.0]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("Effect of min_dist on UMAP Embedding (3 Blobs in 5D)",
                 fontsize=12, fontweight='bold')

    for ax, md in zip(axes, min_dists):
        umap = UMAP(n_components=2, n_neighbors=15, min_dist=md,
                    n_epochs=300, random_state=0)
        Z = umap.fit_transform(X)
        for k in range(3):
            ax.scatter(Z[y == k, 0], Z[y == k, 1], c=CMAP[k], s=20, alpha=0.85)
        ax.set_title(f'min_dist = {md}', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("images/04_min_dist_effect.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 5. Swiss roll: PCA vs UMAP
# ---------------------------------------------------------------------------

def img_swiss_roll():
    print("Generating: 05_swiss_roll.png")
    from sklearn.datasets import make_swiss_roll
    np.random.seed(42)
    X3d, color = make_swiss_roll(n_samples=400, noise=0.1, random_state=42)

    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(X3d)

    umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                n_epochs=400, random_state=42)
    Z_umap = umap.fit_transform(X3d)

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle("Swiss Roll: 3D Original, PCA, UMAP Projections",
                 fontsize=12, fontweight='bold')

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X3d[:, 0], X3d[:, 1], X3d[:, 2], c=color,
                cmap='Spectral', s=15, alpha=0.7)
    ax1.set_title("3D Swiss Roll", fontsize=10)
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])

    ax2 = fig.add_subplot(132)
    ax2.scatter(Z_pca[:, 0], Z_pca[:, 1], c=color, cmap='Spectral', s=15, alpha=0.7)
    ax2.set_title("PCA (2D) — does NOT unroll", fontsize=10)
    ax2.set_xticks([]); ax2.set_yticks([])

    ax3 = fig.add_subplot(133)
    ax3.scatter(Z_umap[:, 0], Z_umap[:, 1], c=color, cmap='Spectral', s=15, alpha=0.7)
    ax3.set_title("UMAP (2D) — unrolls the manifold", fontsize=10)
    ax3.set_xticks([]); ax3.set_yticks([])

    plt.tight_layout()
    plt.savefig("images/05_swiss_roll.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 6. High-dimensional blobs (50D) projected to 2D
# ---------------------------------------------------------------------------

def img_high_dim_blobs():
    print("Generating: 06_high_dim_blobs.png")
    np.random.seed(42)
    n_per, p = 50, 50
    centers = [np.random.randn(p) * 5 for _ in range(5)]
    X = np.vstack([np.random.randn(n_per, p) * 0.7 + c for c in centers])
    y = np.repeat(np.arange(5), n_per)

    pca2 = PCA(n_components=2)
    Z_pca = pca2.fit_transform(X)

    umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                n_epochs=300, random_state=42)
    Z_umap = umap.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("5 Clusters in 50D: PCA vs UMAP", fontsize=12, fontweight='bold')
    for ax, Z, title in zip(axes, [Z_pca, Z_umap], ['PCA 2D', 'UMAP 2D']):
        for k in range(5):
            ax.scatter(Z[y == k, 0], Z[y == k, 1], c=CMAP[k], s=25, alpha=0.85,
                       label=f'Cluster {k}')
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("images/06_high_dim_blobs.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 7. UMAP vs t-SNE comparison
# ---------------------------------------------------------------------------

def img_umap_vs_tsne():
    print("Generating: 07_umap_vs_tsne.png")
    np.random.seed(42)
    X = np.vstack([np.random.randn(60, 10) * 0.5 + c
                   for c in [np.eye(10)[i] * 6 for i in range(4)]])
    y = np.repeat(np.arange(4), 60)

    tsne = TSNE(n_components=2, perplexity=20, learning_rate=150,
                n_iter=600, random_state=42)
    Z_tsne = tsne.fit_transform(X)

    umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                n_epochs=300, random_state=42)
    Z_umap = umap.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("t-SNE vs UMAP on 4-Cluster 10D Data", fontsize=12, fontweight='bold')
    for ax, Z, title in zip(axes, [Z_tsne, Z_umap], ['t-SNE', 'UMAP']):
        for k in range(4):
            ax.scatter(Z[y == k, 0], Z[y == k, 1], c=CMAP[k], s=25, alpha=0.85,
                       label=f'Cluster {k}')
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("images/07_umap_vs_tsne.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 8. Effect of set_op_mix_ratio (fuzzy union vs intersection)
# ---------------------------------------------------------------------------

def img_set_op_mix():
    print("Generating: 08_set_op_mix.png")
    np.random.seed(42)
    X = np.vstack([np.random.randn(50, 8) * 0.5 + c
                   for c in [[4, 0, 0, 0, 0, 0, 0, 0],
                              [-4, 0, 0, 0, 0, 0, 0, 0],
                              [0, 4, 0, 0, 0, 0, 0, 0],
                              [0, -4, 0, 0, 0, 0, 0, 0]]])
    y = np.repeat([0, 1, 2, 3], 50)

    mix_ratios = [1.0, 0.5, 0.0]
    labels = ['Fuzzy Union (mix=1.0)', 'Mixed (mix=0.5)', 'Fuzzy Intersection (mix=0.0)']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Effect of set_op_mix_ratio on UMAP (4 Clusters in 8D)",
                 fontsize=12, fontweight='bold')

    for ax, ratio, label in zip(axes, mix_ratios, labels):
        umap = UMAP(n_components=2, n_neighbors=12, min_dist=0.1,
                    n_epochs=300, set_op_mix_ratio=ratio, random_state=0)
        Z = umap.fit_transform(X)
        for k in range(4):
            ax.scatter(Z[y == k, 0], Z[y == k, 1], c=CMAP[k], s=25, alpha=0.85,
                       label=f'Cluster {k}')
        ax.set_title(label, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("images/08_set_op_mix.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating UMAP images...")
    print("=" * 60)

    img_umap_digits()
    img_pca_vs_umap()
    img_n_neighbors_effect()
    img_min_dist_effect()
    img_swiss_roll()
    img_high_dim_blobs()
    img_umap_vs_tsne()
    img_set_op_mix()

    print("=" * 60)
    print("All 8 images saved to images/")
