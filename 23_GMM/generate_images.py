"""
Generate visualizations for Gaussian Mixture Models (GMM)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '19_KMEANS'))
from gmm_scratch import GaussianMixture
from kmeans_scratch import KMeans

os.makedirs("images", exist_ok=True)

CMAP = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA',
        '#00ACC1', '#F4511E', '#3949AB']


# ---------------------------------------------------------------------------
# Helper: draw 2D Gaussian confidence ellipse
# ---------------------------------------------------------------------------

def _draw_ellipse(ax, mu, Sigma, n_std=2.0, **kwargs):
    """Draw a 2-sigma confidence ellipse for a 2D Gaussian."""
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    # Largest eigenvalue -> major axis
    angle = np.degrees(np.arctan2(eigenvectors[1, -1], eigenvectors[0, -1]))
    width  = 2 * n_std * np.sqrt(max(eigenvalues[-1], 0))
    height = 2 * n_std * np.sqrt(max(eigenvalues[0],  0))
    ell = Ellipse(xy=mu, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ell)


# ---------------------------------------------------------------------------
# 1. GMM contours on 3-blob data
# ---------------------------------------------------------------------------

def img_gmm_contours():
    print("Generating: 01_gmm_contours.png")
    np.random.seed(42)
    centers = [[5, 5], [-5, 5], [0, -5]]
    X = np.vstack([np.random.randn(60, 2) * 0.8 + c for c in centers])
    true_labels = np.repeat([0, 1, 2], 60)

    gmm = GaussianMixture(n_components=3, n_init=5, random_state=0).fit(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("GMM on 3-Blob Data: True Labels vs Fitted Components",
                 fontsize=12, fontweight='bold')

    titles = ["True Labels", "GMM Hard Assignment + 2σ Ellipses"]
    label_sets = [true_labels, gmm.labels_]

    for ax, title, labels in zip(axes, titles, label_sets):
        for j in range(3):
            mask = labels == j
            ax.scatter(X[mask, 0], X[mask, 1], c=CMAP[j], s=25, alpha=0.8,
                       label=f"Cluster {j}")

        if ax == axes[1]:
            for k in range(3):
                _draw_ellipse(ax, gmm.means_[k], gmm.covariances_[k],
                              n_std=2.0, edgecolor=CMAP[k], facecolor='none',
                              linewidth=2.5, linestyle='--')
                ax.scatter(*gmm.means_[k], c=CMAP[k], s=150, marker='*',
                           edgecolors='black', linewidths=0.8, zorder=5)

        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("images/01_gmm_contours.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 2. EM convergence: log-likelihood vs iterations for K = 2, 3, 4
# ---------------------------------------------------------------------------

def img_em_convergence():
    print("Generating: 02_em_convergence.png")
    np.random.seed(42)
    X = np.vstack([np.random.randn(50, 2) * 0.7 + c
                   for c in [[5, 5], [-5, 5], [0, -5]]])

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00']

    for k, col in zip([2, 3, 4, 5], colors):
        gmm = GaussianMixture(n_components=k, max_iter=80, random_state=0).fit(X)
        h = gmm.log_likelihoods_
        ax.plot(range(1, len(h) + 1), h, '-o', color=col, markersize=4,
                linewidth=2, label=f'K={k} ({len(h)} iters)')

    ax.set_xlabel("EM Iteration")
    ax.set_ylabel("Mean Log-Likelihood")
    ax.set_title("EM Convergence: Log-Likelihood vs Iteration", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("images/02_em_convergence.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 3. Covariance types comparison
# ---------------------------------------------------------------------------

def img_covariance_types():
    print("Generating: 03_covariance_types.png")
    np.random.seed(42)
    # Anisotropic blobs to highlight differences
    np.random.seed(42)
    X1 = np.random.randn(80, 2) @ np.array([[2.5, 0.8], [0.8, 0.5]]) + [4, 0]
    X2 = np.random.randn(80, 2) @ np.array([[0.4, 0.0], [0.0, 2.0]]) + [-4, 0]
    X = np.vstack([X1, X2])

    cov_types = ['full', 'diag', 'spherical', 'tied']
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("GMM Covariance Types (K=2, 2σ ellipses)", fontsize=12, fontweight='bold')

    for ax, ct in zip(axes, cov_types):
        gmm = GaussianMixture(n_components=2, covariance_type=ct,
                              n_init=5, random_state=0).fit(X)
        for j in range(2):
            mask = gmm.labels_ == j
            ax.scatter(X[mask, 0], X[mask, 1], c=CMAP[j], s=18, alpha=0.7)

        for k in range(2):
            if ct == 'full':
                sigma = gmm.covariances_[k]
            elif ct == 'diag':
                sigma = np.diag(gmm.covariances_[k])
            elif ct == 'spherical':
                sigma = np.eye(2) * gmm.covariances_[k]
            elif ct == 'tied':
                sigma = gmm.covariances_
            _draw_ellipse(ax, gmm.means_[k], sigma, n_std=2.0,
                          edgecolor=CMAP[k], facecolor='none',
                          linewidth=2.5, linestyle='--')

        score = gmm.score(X)
        ax.set_title(f"{ct.capitalize()}\nLL = {score:.2f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("images/03_covariance_types.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 4. Soft assignments: responsibility heatmap
# ---------------------------------------------------------------------------

def img_soft_assignments():
    print("Generating: 04_soft_assignments.png")
    np.random.seed(42)
    X = np.vstack([np.random.randn(80, 2) * 0.9 + c
                   for c in [[4, 4], [-4, 4], [0, -4]]])

    gmm = GaussianMixture(n_components=3, n_init=5, random_state=0).fit(X)
    proba = gmm.predict_proba(X)          # (n, 3)
    max_prob = proba.max(axis=1)          # confidence of assignment
    # RGB color: blend component colors by probability
    R = np.array([0xE5, 0x1E, 0x43]) / 255.0
    G = np.array([0x39, 0x88, 0xA0]) / 255.0
    B = np.array([0x35, 0xE5, 0x47]) / 255.0
    colors_rgb = (proba[:, 0:1] * R + proba[:, 1:2] * G + proba[:, 2:3] * B)
    colors_rgb = np.clip(colors_rgb, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("GMM Soft Assignments: Hard vs Soft Cluster Membership",
                 fontsize=12, fontweight='bold')

    # Hard assignment
    for j in range(3):
        mask = gmm.labels_ == j
        axes[0].scatter(X[mask, 0], X[mask, 1], c=CMAP[j], s=25, alpha=0.9)
    axes[0].set_title("Hard Assignment (argmax)", fontsize=10)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # Soft assignment: color by max probability
    sc = axes[1].scatter(X[:, 0], X[:, 1], c=max_prob, cmap='RdYlGn',
                         vmin=0.3, vmax=1.0, s=30, alpha=0.9)
    plt.colorbar(sc, ax=axes[1], label='max r_ik (confidence)')
    axes[1].set_title("Soft Assignment (max responsibility)", fontsize=10)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    plt.tight_layout()
    plt.savefig("images/04_soft_assignments.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 5. BIC / AIC model selection
# ---------------------------------------------------------------------------

def img_bic_aic():
    print("Generating: 05_bic_aic.png")
    np.random.seed(42)
    X = np.vstack([np.random.randn(80, 2) * 0.5 + c
                   for c in [[6, 6], [-6, 6], [0, -6]]])

    ks = list(range(1, 9))
    bics, aics = [], []
    for k in ks:
        gmm = GaussianMixture(n_components=k, n_init=3, random_state=0).fit(X)
        bics.append(gmm.bic(X))
        aics.append(gmm.aic(X))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, bics, 'b-o', linewidth=2, markersize=8, label='BIC')
    ax.plot(ks, aics, 'r--s', linewidth=2, markersize=8, label='AIC')
    ax.axvline(3, color='green', linestyle=':', linewidth=1.8, label='True K=3')
    ax.set_xlabel('Number of Components K')
    ax.set_ylabel('Information Criterion (lower = better)')
    ax.set_title('BIC / AIC Model Selection for GMM', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("images/05_bic_aic.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 6. 1D density estimation
# ---------------------------------------------------------------------------

def img_1d_density():
    print("Generating: 06_1d_density.png")
    np.random.seed(42)
    # 1D mixture: two Gaussians
    X1 = np.random.randn(100) * 0.8 + 3.0
    X2 = np.random.randn(70) * 1.2 - 2.0
    X3 = np.random.randn(50) * 0.4 + 7.0
    X_1d = np.concatenate([X1, X2, X3]).reshape(-1, 1)

    gmm = GaussianMixture(n_components=3, covariance_type='full',
                          n_init=5, random_state=0).fit(X_1d)

    grid = np.linspace(-6, 11, 400).reshape(-1, 1)
    log_dens = gmm.score_samples(grid)
    density = np.exp(log_dens)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(X_1d.ravel(), bins=40, density=True, alpha=0.4, color='gray',
            label='Data histogram')
    ax.plot(grid.ravel(), density, 'k-', linewidth=2.5, label='Fitted GMM density')

    # Individual component densities
    for k in range(3):
        mu_k = gmm.means_[k, 0]
        sigma_k = np.sqrt(gmm.covariances_[k, 0, 0])
        comp_density = (gmm.weights_[k]
                        * np.exp(-0.5 * ((grid.ravel() - mu_k) / sigma_k) ** 2)
                        / (sigma_k * np.sqrt(2 * np.pi)))
        ax.fill_between(grid.ravel(), comp_density, alpha=0.25, color=CMAP[k],
                        label=f'Component {k} (π={gmm.weights_[k]:.2f})')

    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.set_title('1D GMM Density Estimation (K=3)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("images/06_1d_density.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 7. GMM vs K-Means on anisotropic data
# ---------------------------------------------------------------------------

def img_vs_kmeans():
    print("Generating: 07_vs_kmeans.png")
    np.random.seed(42)
    # Anisotropic blobs: GMM (full) can adapt, K-Means cannot
    X1 = np.random.randn(100, 2) @ np.array([[3.0, 1.0], [1.0, 0.3]]) + [5, 0]
    X2 = np.random.randn(100, 2) @ np.array([[0.3, 0.0], [0.0, 3.0]]) + [-5, 0]
    X = np.vstack([X1, X2])
    true = np.repeat([0, 1], 100)

    gmm = GaussianMixture(n_components=2, covariance_type='full',
                          n_init=5, random_state=0).fit(X)
    km = KMeans(n_clusters=2, random_state=0).fit(X)

    # Align K-Means labels to true labels by majority vote
    km_labels = km.labels_.copy()
    if np.sum(km_labels == 0) > 0 and np.mean(true[km_labels == 0]) > 0.5:
        km_labels = 1 - km_labels

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("GMM vs K-Means on Anisotropic Data", fontsize=12, fontweight='bold')

    configs = [
        ("True Labels", true, None),
        ("GMM (Full Cov)", gmm.labels_, gmm),
        ("K-Means", km_labels, None),
    ]

    for ax, (title, labels, model) in zip(axes, configs):
        for j in range(2):
            mask = labels == j
            ax.scatter(X[mask, 0], X[mask, 1], c=CMAP[j], s=20, alpha=0.8)
        if model is not None and hasattr(model, 'means_'):
            for k in range(2):
                _draw_ellipse(ax, model.means_[k], model.covariances_[k],
                              n_std=2.0, edgecolor=CMAP[k], facecolor='none',
                              linewidth=2.5, linestyle='--')
        acc = np.mean(labels == true)
        ax.set_title(f"{title}\nAccuracy = {acc:.0%}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("images/07_vs_kmeans.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 8. Sampling from a fitted GMM
# ---------------------------------------------------------------------------

def img_sampling():
    print("Generating: 08_sampling.png")
    np.random.seed(42)
    X_train = np.vstack([
        np.random.randn(60, 2) * 0.6 + [4, 4],
        np.random.randn(60, 2) @ np.array([[2, 0.5], [0.5, 0.3]]) + [-4, 2],
        np.random.randn(60, 2) * 0.5 + [0, -4],
    ])

    gmm = GaussianMixture(n_components=3, n_init=5, random_state=0).fit(X_train)
    X_samp, comp_idx = gmm.sample(400, random_state=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("GMM: Training Data vs Generated Samples", fontsize=12, fontweight='bold')

    # Training data
    for j in range(3):
        mask = gmm.labels_ == j
        axes[0].scatter(X_train[mask, 0], X_train[mask, 1],
                        c=CMAP[j], s=30, alpha=0.8)
    for k in range(3):
        _draw_ellipse(axes[0], gmm.means_[k], gmm.covariances_[k],
                      n_std=2.0, edgecolor=CMAP[k], facecolor='none',
                      linewidth=2, linestyle='--')
    axes[0].set_title("Training Data + 2σ Ellipses", fontsize=10)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # Samples
    for j in range(3):
        mask = comp_idx == j
        axes[1].scatter(X_samp[mask, 0], X_samp[mask, 1],
                        c=CMAP[j], s=15, alpha=0.6)
    axes[1].set_title("400 Samples Generated from Fitted GMM", fontsize=10)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    plt.tight_layout()
    plt.savefig("images/08_sampling.png", dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating Gaussian Mixture Models images...")
    print("=" * 60)

    img_gmm_contours()
    img_em_convergence()
    img_covariance_types()
    img_soft_assignments()
    img_bic_aic()
    img_1d_density()
    img_vs_kmeans()
    img_sampling()

    print("=" * 60)
    print("All 8 images saved to images/")
