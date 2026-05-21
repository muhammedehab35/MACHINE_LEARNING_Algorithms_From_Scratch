"""
Generate visualizations for Principal Component Analysis (PCA)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '19_KMEANS'))
from pca_scratch import PCA
from kmeans_scratch import KMeans

os.makedirs("images", exist_ok=True)

CMAP = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA',
        '#00ACC1', '#F4511E', '#3949AB']


# ---------------------------------------------------------------------------
# 1. PCA directions on 2D data
# ---------------------------------------------------------------------------

def img_pca_directions():
    print("Generating: 01_pca_directions.png")
    np.random.seed(42)
    # Correlated 2D data
    cov = np.array([[3.0, 2.2], [2.2, 2.0]])
    X = np.random.multivariate_normal([0, 0], cov, 200)

    pca = PCA(n_components=2).fit(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("PCA: Principal Directions and 1D Projection", fontsize=12, fontweight='bold')

    # Left: scatter + PC arrows
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], alpha=0.4, s=20, color='#90CAF9', edgecolors='none')
    origin = pca.mean_
    scale = [np.sqrt(pca.explained_variance_[k]) * 2.5 for k in range(2)]
    colors_pc = ['#E53935', '#1E88E5']
    labels_pc = [f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                 f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)']
    for k in range(2):
        v = pca.components_[k] * scale[k]
        ax.annotate('', xy=origin + v, xytext=origin,
                    arrowprops=dict(arrowstyle='->', color=colors_pc[k], lw=2.5))
        ax.annotate('', xy=origin - v, xytext=origin,
                    arrowprops=dict(arrowstyle='->', color=colors_pc[k], lw=2.5))
    for k in range(2):
        v = pca.components_[k] * scale[k]
        ax.plot([], [], color=colors_pc[k], lw=2, label=labels_pc[k])
    ax.set_title("Original Data + PC Directions", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)

    # Right: projection onto PC1 (1D)
    ax2 = axes[1]
    Z = pca.transform(X)
    proj_1d = Z[:, 0:1] @ pca.components_[0:1]  # project back to 2D on PC1 axis
    ax2.scatter(X[:, 0], X[:, 1], alpha=0.3, s=15, color='#90CAF9', label='Original')
    ax2.scatter(proj_1d[:, 0], proj_1d[:, 1], alpha=0.5, s=15,
                color='#E53935', label='Projected onto PC1')
    for i in range(0, len(X), 5):
        ax2.plot([X[i, 0], proj_1d[i, 0]], [X[i, 1], proj_1d[i, 1]],
                 'k-', alpha=0.1, lw=0.5)
    ax2.set_title("Projection onto PC1", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.set_aspect('equal')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("images/01_pca_directions.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 2. Scree plot + cumulative explained variance
# ---------------------------------------------------------------------------

def img_scree_plot():
    print("Generating: 02_scree_plot.png")
    np.random.seed(42)
    # Low-rank + noise
    rank = 4
    n, p = 200, 15
    A = np.random.randn(n, rank)
    B = np.random.randn(rank, p)
    X = A @ B + np.random.randn(n, p) * 0.5

    pca = PCA().fit(X)
    ratios = pca.explained_variance_ratio_
    cumsum = np.cumsum(ratios)
    ks = np.arange(1, len(ratios) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    bars = ax1.bar(ks, ratios * 100, color='#1E88E5', alpha=0.75,
                   edgecolor='white', label='Individual variance (%)')
    ax2.plot(ks, cumsum * 100, 'ro-', linewidth=2, markersize=7,
             label='Cumulative variance (%)')
    ax2.axhline(95, color='green', linestyle='--', linewidth=1.5, label='95% threshold')

    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance (%)', color='#1E88E5')
    ax2.set_ylabel('Cumulative Variance (%)', color='red')
    ax1.set_title('Scree Plot: Individual and Cumulative Explained Variance', fontsize=11)
    ax1.set_xticks(ks)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='center right')

    plt.tight_layout()
    plt.savefig("images/02_scree_plot.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 3. Image compression: digits reconstructed at different K
# ---------------------------------------------------------------------------

def img_reconstruction():
    print("Generating: 03_reconstruction.png")
    from sklearn.datasets import load_digits
    digits = load_digits()
    X = digits.data.astype(float)       # (1797, 64)  — 8×8 digit images

    np.random.seed(42)
    sample_idx = [0, 1, 2, 3, 4]       # show 5 example digits
    X_samples = X[sample_idx]

    ks = [2, 5, 15, 32, 64]
    n_show = len(sample_idx)
    n_k = len(ks)

    fig, axes = plt.subplots(n_show, n_k + 1, figsize=(14, 8))
    fig.suptitle("PCA Image Compression: Digits Reconstructed at Different K",
                 fontsize=12, fontweight='bold')

    pca_full = PCA().fit(X)

    for row, idx in enumerate(sample_idx):
        # Original
        axes[row, 0].imshow(X[idx].reshape(8, 8), cmap='gray_r')
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
        if row == 0:
            axes[row, 0].set_title('Original', fontsize=9)

        for col, k in enumerate(ks):
            pca_k = PCA(n_components=k).fit(X)
            X_rec = pca_k.inverse_transform(pca_k.transform(X_samples))
            axes[row, col + 1].imshow(X_rec[row].reshape(8, 8), cmap='gray_r')
            axes[row, col + 1].set_xticks([]); axes[row, col + 1].set_yticks([])
            if row == 0:
                ratio = pca_k.explained_variance_ratio_.sum()
                axes[row, col + 1].set_title(f'K={k}\n({ratio*100:.0f}%)', fontsize=8)

    plt.tight_layout()
    plt.savefig("images/03_reconstruction.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 4. 3D data projected to 2D
# ---------------------------------------------------------------------------

def img_3d_projection():
    print("Generating: 04_3d_projection.png")
    np.random.seed(42)
    # 3 clusters in 3D
    centers = [[3, 3, 3], [-3, -3, 3], [0, -3, -3]]
    X3 = np.vstack([np.random.randn(80, 3) * 0.8 + c for c in centers])
    true = np.repeat([0, 1, 2], 80)

    pca3 = PCA(n_components=2).fit(X3)
    Z2 = pca3.transform(X3)

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle("PCA: 3D Data Projected to 2D Principal Subspace",
                 fontsize=12, fontweight='bold')

    # 3D view
    ax1 = fig.add_subplot(121, projection='3d')
    for j in range(3):
        mask = true == j
        ax1.scatter(X3[mask, 0], X3[mask, 1], X3[mask, 2],
                    c=CMAP[j], s=20, alpha=0.7, label=f'Cluster {j}')
    ax1.set_title("Original 3D Data", fontsize=10)
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])
    ax1.legend(fontsize=8)

    # 2D projection
    ax2 = fig.add_subplot(122)
    for j in range(3):
        mask = true == j
        ax2.scatter(Z2[mask, 0], Z2[mask, 1], c=CMAP[j], s=25, alpha=0.8,
                    label=f'Cluster {j}')
    ax2.set_xlabel(f'PC1 ({pca3.explained_variance_ratio_[0]*100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({pca3.explained_variance_ratio_[1]*100:.1f}%)')
    ax2.set_title("2D PCA Projection", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("images/04_3d_projection.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 5. Biplot
# ---------------------------------------------------------------------------

def img_biplot():
    print("Generating: 05_biplot.png")
    np.random.seed(42)
    # Synthetic 6-feature dataset: 3 clusters
    centers = [[2, 0, 1, -1, 0, 1],
               [-1, 2, -1, 0, 1, -1],
               [0, -2, 0, 1, -1, 0]]
    feature_names = [f'F{i+1}' for i in range(6)]
    X = np.vstack([np.random.randn(50, 6) * 0.6 + c for c in centers])
    true = np.repeat([0, 1, 2], 50)

    pca = PCA(n_components=2).fit(X)
    Z = pca.transform(X)
    loadings = pca.components_.T   # (6, 2) — feature contributions to each PC

    fig, ax = plt.subplots(figsize=(9, 7))
    for j in range(3):
        mask = true == j
        ax.scatter(Z[mask, 0], Z[mask, 1], c=CMAP[j], s=30, alpha=0.7,
                   label=f'Cluster {j}')

    # Scale loadings for visibility
    scale = max(abs(Z[:, 0]).max(), abs(Z[:, 1]).max()) * 0.4
    for i, name in enumerate(feature_names):
        ax.annotate('', xy=(loadings[i, 0] * scale, loadings[i, 1] * scale),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        offset = 0.15 * scale
        ax.text(loadings[i, 0] * scale + offset * np.sign(loadings[i, 0]),
                loadings[i, 1] * scale + offset * np.sign(loadings[i, 1]),
                name, fontsize=9, color='black', fontweight='bold')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
    ax.set_title('PCA Biplot: Scores and Feature Loadings', fontsize=11)
    ax.legend(fontsize=9)
    ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.5)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig("images/05_biplot.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 6. Whitening comparison
# ---------------------------------------------------------------------------

def img_whitening():
    print("Generating: 06_whitening.png")
    np.random.seed(42)
    cov = np.array([[6.0, 4.5], [4.5, 4.0]])
    X = np.random.multivariate_normal([0, 0], cov, 300)

    pca_plain = PCA(n_components=2, whiten=False).fit(X)
    pca_white = PCA(n_components=2, whiten=True).fit(X)

    Z_plain = pca_plain.transform(X)
    Z_white = pca_white.transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("PCA Whitening: Removing Correlations and Scaling Variance",
                 fontsize=12, fontweight='bold')

    configs = [
        (X, "Original Data\n(correlated, anisotropic)"),
        (Z_plain, f"PCA Scores (no whitening)\nVar(PC1)={pca_plain.explained_variance_[0]:.2f}, "
                  f"Var(PC2)={pca_plain.explained_variance_[1]:.2f}"),
        (Z_white, "Whitened PCA Scores\nVar(PC1)≈1, Var(PC2)≈1"),
    ]
    for ax, (data, title) in zip(axes, configs):
        ax.scatter(data[:, 0], data[:, 1], alpha=0.4, s=15, color='#1E88E5')
        var_x = data[:, 0].var()
        var_y = data[:, 1].var()
        ax.set_title(title, fontsize=9)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='gray', lw=0.5)

    plt.tight_layout()
    plt.savefig("images/06_whitening.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 7. PCA + K-Means clustering in PC space
# ---------------------------------------------------------------------------

def img_pca_clustering():
    print("Generating: 07_pca_clustering.png")
    from sklearn.datasets import load_digits
    np.random.seed(42)
    digits = load_digits()
    X = digits.data.astype(float)
    y_true = digits.target

    # PCA to 2D for visualization
    pca2 = PCA(n_components=2).fit(X)
    Z2 = pca2.transform(X)

    # PCA to 20D for clustering
    pca20 = PCA(n_components=20).fit(X)
    Z20 = pca20.transform(X)
    km = KMeans(n_clusters=10, n_init=3, random_state=42).fit(Z20)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("PCA + K-Means: Digit Clustering in PC Space (10 clusters)",
                 fontsize=12, fontweight='bold')

    # True labels
    for digit in range(10):
        mask = y_true == digit
        axes[0].scatter(Z2[mask, 0], Z2[mask, 1], s=8, alpha=0.5,
                        color=plt.cm.tab10(digit / 10), label=str(digit))
    axes[0].set_title(f"True Labels\nPC1={pca2.explained_variance_ratio_[0]*100:.1f}%,"
                      f" PC2={pca2.explained_variance_ratio_[1]*100:.1f}%", fontsize=9)
    axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')
    axes[0].legend(title='Digit', fontsize=7, ncol=2, markerscale=2)

    # K-Means labels
    for k in range(10):
        mask = km.labels_ == k
        axes[1].scatter(Z2[mask, 0], Z2[mask, 1], s=8, alpha=0.5,
                        color=plt.cm.tab10(k / 10))
    axes[1].set_title("K-Means Clusters (K=10) on 20-PC Space", fontsize=9)
    axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')

    plt.tight_layout()
    plt.savefig("images/07_pca_clustering.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 8. Noise filtering via PCA
# ---------------------------------------------------------------------------

def img_noise_filtering():
    print("Generating: 08_noise_filtering.png")
    np.random.seed(42)
    # Structured signal: rank-3 data in 20D
    n, p, rank = 200, 20, 3
    signal = np.random.randn(n, rank) @ np.random.randn(rank, p) * 2.0
    noise = np.random.randn(n, p) * 1.5
    X_noisy = signal + noise

    ks = [1, 3, 5, 10, 20]
    mse_signal, mse_noisy = [], []
    for k in ks:
        pca = PCA(n_components=k).fit(X_noisy)
        X_rec = pca.inverse_transform(pca.transform(X_noisy))
        mse_signal.append(float(np.mean((X_rec - signal) ** 2)))
        mse_noisy.append(float(np.mean((X_rec - X_noisy) ** 2)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PCA as a Noise Filter: Reconstruction Error vs K",
                 fontsize=12, fontweight='bold')

    axes[0].plot(ks, mse_signal, 'b-o', lw=2, ms=8, label='MSE vs clean signal')
    axes[0].plot(ks, mse_noisy, 'r--s', lw=2, ms=8, label='MSE vs noisy data')
    axes[0].axvline(rank, color='green', linestyle=':', lw=1.8, label=f'True rank={rank}')
    axes[0].set_xlabel('Number of PCs (K)')
    axes[0].set_ylabel('Mean Squared Error')
    axes[0].set_title('MSE vs K', fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Visualise one row: noisy vs denoised vs signal
    pca_best = PCA(n_components=rank).fit(X_noisy)
    denoised = pca_best.inverse_transform(pca_best.transform(X_noisy))
    idx = 0
    ax = axes[1]
    ax.plot(signal[idx], 'g-', lw=2, label='Clean signal', alpha=0.9)
    ax.plot(X_noisy[idx], 'r:', lw=1, label='Noisy input', alpha=0.7)
    ax.plot(denoised[idx], 'b--', lw=2, label=f'PCA denoised (K={rank})', alpha=0.9)
    ax.set_xlabel('Feature index')
    ax.set_ylabel('Value')
    ax.set_title(f'One sample: signal vs noisy vs denoised', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("images/08_noise_filtering.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating PCA images...")
    print("=" * 60)

    img_pca_directions()
    img_scree_plot()
    img_reconstruction()
    img_3d_projection()
    img_biplot()
    img_whitening()
    img_pca_clustering()
    img_noise_filtering()

    print("=" * 60)
    print("All 8 images saved to images/")
