"""
Generate visualizations for t-SNE
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '24_PCA'))
from tsne_scratch import TSNE, compute_joint_probabilities
from pca_scratch import PCA

os.makedirs("images", exist_ok=True)

CMAP10 = [plt.cm.tab10(i / 10) for i in range(10)]
CMAP = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA',
        '#00ACC1', '#F4511E', '#3949AB']


# ---------------------------------------------------------------------------
# 1. t-SNE on MNIST digits
# ---------------------------------------------------------------------------

def img_tsne_digits():
    print("Generating: 01_tsne_digits.png")
    from sklearn.datasets import load_digits
    np.random.seed(42)
    digits = load_digits()
    # Use 500 samples for speed
    idx = np.random.choice(len(digits.data), 500, replace=False)
    X = digits.data[idx].astype(float)
    y = digits.target[idx]

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200,
                n_iter=750, random_state=42, verbose=False)
    Z = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    for digit in range(10):
        mask = y == digit
        ax.scatter(Z[mask, 0], Z[mask, 1], s=18, alpha=0.8,
                   color=CMAP10[digit], label=str(digit))
    ax.legend(title='Digit', fontsize=8, ncol=2, markerscale=2,
              loc='upper right')
    ax.set_title(f't-SNE on 500 MNIST Digits (perplexity=30, KL={tsne.kl_divergence_:.3f})',
                 fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("images/01_tsne_digits.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 2. PCA vs t-SNE on 4-cluster nonlinear data
# ---------------------------------------------------------------------------

def img_pca_vs_tsne():
    print("Generating: 02_pca_vs_tsne.png")
    np.random.seed(42)
    # 4 clusters arranged on a circle in 20D
    n_per = 60
    centers_2d = [(4, 0), (0, 4), (-4, 0), (0, -4)]
    X_list, y_list = [], []
    for k, (cx, cy) in enumerate(centers_2d):
        blob = np.random.randn(n_per, 20) * 0.4
        blob[:, 0] += cx
        blob[:, 1] += cy
        X_list.append(blob)
        y_list.append(np.full(n_per, k))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(X)

    tsne = TSNE(n_components=2, perplexity=20, learning_rate=150,
                n_iter=600, random_state=42)
    Z_tsne = tsne.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("PCA vs t-SNE on 4-Cluster 20D Data", fontsize=12, fontweight='bold')

    for ax, Z, title in zip(axes, [Z_pca, Z_tsne], ['PCA (linear)', 't-SNE (nonlinear)']):
        for k in range(4):
            mask = y == k
            ax.scatter(Z[mask, 0], Z[mask, 1], c=CMAP[k], s=25, alpha=0.8,
                       label=f'Cluster {k}')
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("images/02_pca_vs_tsne.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 3. Effect of perplexity
# ---------------------------------------------------------------------------

def img_perplexity_effect():
    print("Generating: 03_perplexity_effect.png")
    np.random.seed(42)
    # 3 tight blobs
    X = np.vstack([np.random.randn(50, 2) * 0.3 + c
                   for c in [[4, 0], [-4, 0], [0, 4]]])
    y = np.repeat([0, 1, 2], 50)
    # Lift to 10D
    X = np.hstack([X, np.random.randn(len(X), 8) * 0.1])

    perplexities = [5, 15, 30, 50]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("Effect of Perplexity on t-SNE Embedding (3 Blobs in 10D)",
                 fontsize=12, fontweight='bold')

    for ax, perp in zip(axes, perplexities):
        tsne = TSNE(n_components=2, perplexity=perp, learning_rate=150,
                    n_iter=500, random_state=0)
        Z = tsne.fit_transform(X)
        for k in range(3):
            mask = y == k
            ax.scatter(Z[mask, 0], Z[mask, 1], c=CMAP[k], s=20, alpha=0.85)
        ax.set_title(f'Perplexity = {perp}\nKL = {tsne.kl_divergence_:.3f}',
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("images/03_perplexity_effect.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 4. KL divergence convergence
# ---------------------------------------------------------------------------

def img_kl_convergence():
    print("Generating: 04_kl_convergence.png")
    np.random.seed(42)

    datasets = {
        '3 Blobs (60D)': (
            np.vstack([np.random.randn(80, 60) * 0.5 + np.random.randn(60) * 3
                       for _ in range(3)]),
            '#E53935'
        ),
        '5 Blobs (30D)': (
            np.vstack([np.random.randn(50, 30) * 0.5 + np.random.randn(30) * 4
                       for _ in range(5)]),
            '#1E88E5'
        ),
        'Random (20D)': (
            np.random.randn(150, 20),
            '#43A047'
        ),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, (X, color) in datasets.items():
        tsne = TSNE(n_components=2, perplexity=20, learning_rate=200,
                    n_iter=800, n_iter_early_exag=200, random_state=0)
        tsne.fit(X)
        iters = [h[0] for h in tsne.kl_divergence_history_]
        kls = [h[1] for h in tsne.kl_divergence_history_]
        ax.plot(iters, kls, lw=2, color=color, label=label)
        ax.axvline(200, color='gray', linestyle=':', lw=1)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('KL Divergence KL(P || Q)')
    ax.set_title('t-SNE KL Divergence over Iterations', fontsize=11)
    ax.text(205, ax.get_ylim()[1] * 0.95, 'End of early\nexaggeration',
            fontsize=8, color='gray', va='top')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("images/04_kl_convergence.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 5. Swiss roll: PCA vs t-SNE
# ---------------------------------------------------------------------------

def img_swiss_roll():
    print("Generating: 05_swiss_roll.png")
    from sklearn.datasets import make_swiss_roll
    np.random.seed(42)
    X3d, color = make_swiss_roll(n_samples=400, noise=0.1, random_state=42)

    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(X3d)

    tsne = TSNE(n_components=2, perplexity=25, learning_rate=200,
                n_iter=750, random_state=42)
    Z_tsne = tsne.fit_transform(X3d)

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle("Swiss Roll: 3D Original, PCA, t-SNE Projections",
                 fontsize=12, fontweight='bold')

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X3d[:, 0], X3d[:, 1], X3d[:, 2], c=color,
                cmap='Spectral', s=15, alpha=0.7)
    ax1.set_title("3D Swiss Roll", fontsize=10)
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])

    ax2 = fig.add_subplot(132)
    ax2.scatter(Z_pca[:, 0], Z_pca[:, 1], c=color, cmap='Spectral', s=15, alpha=0.7)
    ax2.set_title(f"PCA (2D) — does NOT unroll", fontsize=10)
    ax2.set_xticks([]); ax2.set_yticks([])

    ax3 = fig.add_subplot(133)
    ax3.scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=color, cmap='Spectral', s=15, alpha=0.7)
    ax3.set_title(f"t-SNE (2D) — unrolls the manifold", fontsize=10)
    ax3.set_xticks([]); ax3.set_yticks([])

    plt.tight_layout()
    plt.savefig("images/05_swiss_roll.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 6. Embedding progression: snapshots at different iterations
# ---------------------------------------------------------------------------

def img_iterations():
    print("Generating: 06_iterations.png")
    np.random.seed(42)
    X = np.vstack([np.random.randn(60, 5) * 0.4 + c
                   for c in [[5, 0, 0, 0, 0],
                              [-5, 0, 0, 0, 0],
                              [0, 5, 0, 0, 0]]])
    y = np.repeat([0, 1, 2], 60)

    checkpoints = {0: None, 50: None, 150: None, 300: None, 700: None}
    tsne = TSNE(n_components=2, perplexity=20, learning_rate=150,
                n_iter=750, n_iter_early_exag=150, random_state=42)
    tsne.fit(X, _checkpoints=checkpoints)
    checkpoints[750] = tsne.embedding_

    iters_show = [0, 50, 150, 300, 750]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("t-SNE Embedding Evolution (3 Blobs in 5D)",
                 fontsize=12, fontweight='bold')

    for ax, it in zip(axes, iters_show):
        Z = checkpoints.get(it)
        if Z is None:
            ax.set_visible(False)
            continue
        for k in range(3):
            mask = y == k
            ax.scatter(Z[mask, 0], Z[mask, 1], c=CMAP[k], s=20, alpha=0.85)
        tag = 'Early exag.' if it <= 150 else 'Normal'
        ax.set_title(f'Iter {it}\n({tag})', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("images/06_iterations.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 7. High-dimensional blobs (50D) projected to 2D
# ---------------------------------------------------------------------------

def img_high_dim_blobs():
    print("Generating: 07_high_dim_blobs.png")
    np.random.seed(42)
    n_per, p = 50, 50
    # 5 well-separated clusters in 50D
    centers = [np.random.randn(p) * 5 for _ in range(5)]
    X = np.vstack([np.random.randn(n_per, p) * 0.7 + c for c in centers])
    y = np.repeat(np.arange(5), n_per)

    pca2 = PCA(n_components=2)
    Z_pca = pca2.fit_transform(X)

    tsne = TSNE(n_components=2, perplexity=15, learning_rate=150,
                n_iter=600, random_state=42)
    Z_tsne = tsne.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("5 Clusters in 50D: PCA vs t-SNE",
                 fontsize=12, fontweight='bold')

    for ax, Z, title in zip(axes, [Z_pca, Z_tsne],
                             ['PCA 2D', 't-SNE 2D']):
        for k in range(5):
            mask = y == k
            ax.scatter(Z[mask, 0], Z[mask, 1], c=CMAP[k], s=25, alpha=0.85,
                       label=f'Cluster {k}')
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("images/07_high_dim_blobs.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 8. Early exaggeration: with vs without
# ---------------------------------------------------------------------------

def img_early_exaggeration():
    print("Generating: 08_early_exaggeration.png")
    np.random.seed(42)
    X = np.vstack([np.random.randn(50, 8) * 0.5 + c
                   for c in [[4, 0, 0, 0, 0, 0, 0, 0],
                              [-4, 0, 0, 0, 0, 0, 0, 0],
                              [0, 4, 0, 0, 0, 0, 0, 0],
                              [0, -4, 0, 0, 0, 0, 0, 0]]])
    y = np.repeat([0, 1, 2, 3], 50)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Early Exaggeration: Effect on Final Embedding (4 Clusters in 8D)",
                 fontsize=12, fontweight='bold')

    configs = [
        (12.0, 250, "With early exaggeration\n(factor=12, 250 iters)"),
        (1.0,  0,   "Without early exaggeration"),
    ]
    for ax, (exag, exag_iter, title) in zip(axes, configs):
        tsne = TSNE(n_components=2, perplexity=15, learning_rate=150,
                    n_iter=600, early_exaggeration=exag,
                    n_iter_early_exag=exag_iter, random_state=0)
        Z = tsne.fit_transform(X)
        for k in range(4):
            mask = y == k
            ax.scatter(Z[mask, 0], Z[mask, 1], c=CMAP[k], s=25, alpha=0.85,
                       label=f'Cluster {k}')
        ax.set_title(f"{title}\nKL = {tsne.kl_divergence_:.3f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("images/08_early_exaggeration.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating t-SNE images...")
    print("=" * 60)

    img_tsne_digits()
    img_pca_vs_tsne()
    img_perplexity_effect()
    img_kl_convergence()
    img_swiss_roll()
    img_iterations()
    img_high_dim_blobs()
    img_early_exaggeration()

    print("=" * 60)
    print("All 8 images saved to images/")
