"""
UMAP (Uniform Manifold Approximation and Projection) — From Scratch

McInnes, L., Healy, J., & Melville, J. (2018). "UMAP: Uniform Manifold
Approximation and Projection for Dimension Reduction." arXiv:1802.03426.

Core idea:
  1. Build a fuzzy topological graph over the high-dim data: for each point,
     assign adaptive Gaussian weights to its k nearest neighbours (fuzzy
     simplicial sets), then symmetrise via fuzzy union.
  2. Fit a smooth low-dim kernel q_ij = (1 + a*d^{2b})^{-1} whose shape
     is controlled by (min_dist, spread).
  3. Minimise the cross-entropy between the high-dim weights and the
     low-dim kernel values via SGD with negative sampling.
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Distance utilities
# ---------------------------------------------------------------------------

def _pairwise_sq_distances(X: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance matrix (n x n)."""
    sq = np.sum(X ** 2, axis=1)
    return np.maximum(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T), 0.0)


def _knn_distances(X: np.ndarray, n_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    k-nearest-neighbour distances and indices (self excluded).

    Returns
    -------
    dists : (n, k)  sorted ascending
    idx   : (n, k)  neighbour indices
    """
    D2 = _pairwise_sq_distances(X)
    np.fill_diagonal(D2, np.inf)
    idx = np.argsort(D2, axis=1)[:, :n_neighbors]
    dists = np.sqrt(np.take_along_axis(D2, idx, axis=1))
    return dists, idx


# ---------------------------------------------------------------------------
# Fuzzy simplicial set
# ---------------------------------------------------------------------------

def _smooth_knn_dist(
    knn_dists: np.ndarray,
    n_neighbors: int,
    n_iter: int = 64,
    bandwidth: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-point rho_i and sigma_i via vectorised binary search.

    rho_i = distance to the nearest neighbour of i.
    sigma_i solves:
        sum_j exp(-(d_ij - rho_i) / sigma_i) = log2(k) * bandwidth

    Returns
    -------
    rho   : (n,)  nearest-neighbour distances
    sigma : (n,)  bandwidth values (all > 0)
    """
    n = knn_dists.shape[0]
    target = np.log2(n_neighbors) * bandwidth

    rho = knn_dists[:, 0].copy()
    shifted = np.maximum(knn_dists - rho[:, None], 0.0)  # (n, k)

    # Start small so sum starts below target; binary search upward
    sigma = np.ones(n)
    lo = np.zeros(n)
    hi = np.full(n, np.inf)

    for _ in range(n_iter):
        vals = np.sum(np.exp(-shifted / sigma[:, None]), axis=1)
        diff = vals - target
        converged = np.abs(diff) < 1e-5

        # Larger sigma → larger sum; smaller sigma → smaller sum
        too_large = diff > 0   # sum > target → sigma too large → decrease
        too_small = diff < 0   # sum < target → sigma too small → increase

        hi_new = np.where(too_large, sigma, hi)
        lo_new = np.where(too_small, sigma, lo)

        sigma_dec = (lo + sigma) / 2.0                                      # decrease
        sigma_inc = np.where(np.isinf(hi), sigma * 2.0, (sigma + hi) / 2.0)  # increase
        sigma_new = np.where(too_large, sigma_dec,
                             np.where(too_small, sigma_inc, sigma))

        lo = np.where(converged, lo, lo_new)
        hi = np.where(converged, hi, hi_new)
        sigma = np.where(converged, sigma, sigma_new)

    return rho, np.maximum(sigma, 1e-8)


def compute_fuzzy_simplicial_set(
    X: np.ndarray,
    n_neighbors: int,
    set_op_mix_ratio: float = 1.0,
) -> np.ndarray:
    """
    Build the n x n fuzzy weight matrix W.

    Directed memberships for j in N_k(i):
        v_ij = exp(-(d_ij - rho_i) / sigma_i)

    Symmetrised via fuzzy union (set_op_mix_ratio=1):
        w_ij = v_ij + v_ji - v_ij * v_ji

    Returns
    -------
    W : (n, n) symmetric weight matrix with values in [0, 1]
    """
    n = len(X)
    knn_dists, knn_idx = _knn_distances(X, n_neighbors)
    rho, sigma = _smooth_knn_dist(knn_dists, n_neighbors)

    shifted = np.maximum(knn_dists - rho[:, None], 0.0)
    v_vals = np.exp(-shifted / sigma[:, None])

    V = np.zeros((n, n))
    i_idx = np.repeat(np.arange(n), n_neighbors)
    j_idx = knn_idx.ravel()
    V[i_idx, j_idx] = v_vals.ravel()

    W = V + V.T - V * V.T

    if set_op_mix_ratio < 1.0:
        W = set_op_mix_ratio * W + (1.0 - set_op_mix_ratio) * np.minimum(V, V.T)

    return W


# ---------------------------------------------------------------------------
# Low-dimensional kernel parameters
# ---------------------------------------------------------------------------

def find_ab_params(spread: float = 1.0, min_dist: float = 0.1) -> Tuple[float, float]:
    """
    Fit (a, b) so that q(d) = (1 + a*d^{2b})^{-1} matches the piecewise target:
        f(d) = 1                            if d < min_dist
               exp(-(d - min_dist)/spread)  otherwise

    Returns a, b > 0.
    """
    xv = np.linspace(0.0, spread * 3.0, 300)
    yv = np.where(xv < min_dist, 1.0, np.exp(-(xv - min_dist) / spread))

    def _q(d, a, b):
        return 1.0 / (1.0 + a * np.maximum(d, 1e-10) ** (2.0 * b))

    try:
        (a, b), _ = curve_fit(_q, xv, yv, p0=[1.0, 1.0], maxfev=10000,
                              bounds=([0.0, 0.0], [np.inf, np.inf]))
        return float(a), float(b)
    except RuntimeError:
        return 1.929, 0.791


# ---------------------------------------------------------------------------
# Spectral initialisation
# ---------------------------------------------------------------------------

def _spectral_init(
    W: np.ndarray,
    n_components: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Laplacian eigenmaps initialisation from the fuzzy weight matrix."""
    n = len(W)
    Ws = (W + W.T) / 2.0
    D = np.maximum(Ws.sum(axis=1), 1e-10)
    D_inv_sqrt = 1.0 / np.sqrt(D)
    L = np.eye(n) - D_inv_sqrt[:, None] * Ws * D_inv_sqrt[None, :]

    try:
        vals, vecs = np.linalg.eigh(L)
        order = np.argsort(vals)
        Y = vecs[:, order[1:n_components + 1]]
        std = Y.std(axis=0)
        std = np.where(std > 1e-10, std, 1.0)
        return (Y / std) * 0.1
    except np.linalg.LinAlgError:
        return rng.standard_normal((n, n_components)) * 0.01


# ---------------------------------------------------------------------------
# Layout optimisation
# ---------------------------------------------------------------------------

def umap_optimize_layout(
    W: np.ndarray,
    Y: np.ndarray,
    a: float,
    b: float,
    n_epochs: int = 200,
    learning_rate: float = 1.0,
    negative_sample_rate: int = 5,
    repulsion_strength: float = 1.0,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Mini-batch SGD layout optimisation via cross-entropy minimisation.

    Processes edges in mini-batches of size n (one update per point per step),
    matching the per-edge sequential behaviour of the original UMAP algorithm.

    Attractive: Delta_y_i = -lr * 2ab(d^2)^{b-1}/(1+a(d^2)^b) * w_ij * (y_i - y_j)
    Repulsive:  Delta_y_i = +lr * 2b*gamma/((eps+d^2)(1+a(d^2)^b)) * (y_i - y_k)
    """
    rng = np.random.default_rng(random_state)
    n = len(Y)
    Y = Y.copy()

    rows, cols = np.where(W > 0)
    weights = W[rows, cols]
    n_edges = len(rows)
    eps = 1e-6
    mini_batch = max(1, n)  # ~one update per point per step

    for epoch in range(n_epochs):
        lr = max(learning_rate * (1.0 - epoch / n_epochs), 1e-4)
        perm = rng.permutation(n_edges)

        for start in range(0, n_edges, mini_batch):
            end = min(start + mini_batch, n_edges)
            b_idx = perm[start:end]
            r_b = rows[b_idx]
            c_b = cols[b_idx]
            w_b = weights[b_idx]
            m = end - start

            # Attractive forces
            diff_a = Y[r_b] - Y[c_b]
            d2_a = np.maximum(np.sum(diff_a ** 2, axis=1, keepdims=True), eps)
            ad2b_a = a * d2_a ** b
            g_a = np.clip(-2.0 * a * b * d2_a ** (b - 1.0) / (1.0 + ad2b_a), -4.0, 0.0)
            upd_a = lr * g_a * diff_a * w_b[:, None]
            np.add.at(Y, r_b, upd_a)
            np.add.at(Y, c_b, -upd_a)

            # Repulsive forces via negative sampling
            neg_k = rng.integers(0, n, size=(m, negative_sample_rate))
            rep_i = np.repeat(r_b, negative_sample_rate)
            rep_k = neg_k.ravel()

            diff_r = Y[rep_i] - Y[rep_k]
            d2_r = np.maximum(np.sum(diff_r ** 2, axis=1, keepdims=True), eps)
            ad2b_r = a * d2_r ** b
            g_r = np.clip(
                2.0 * b * repulsion_strength / ((1e-3 + d2_r) * (1.0 + ad2b_r)),
                0.0, 4.0,
            )
            np.add.at(Y, rep_i, lr * g_r * diff_r)

    return Y


# ---------------------------------------------------------------------------
# UMAP class
# ---------------------------------------------------------------------------

class UMAP:
    """
    UMAP: Uniform Manifold Approximation and Projection.

    Parameters
    ----------
    n_components         : target dimensionality
    n_neighbors          : k, number of nearest neighbours for the fuzzy graph
    min_dist             : minimum distance between points in embedding
    spread               : effective scale of embedded points
    n_epochs             : SGD iterations
    learning_rate        : initial learning rate (decays linearly to 1e-4)
    negative_sample_rate : negative samples per positive edge per epoch
    repulsion_strength   : gamma — weight of repulsive cross-entropy term
    set_op_mix_ratio     : 1.0 = fuzzy union, 0.0 = fuzzy intersection
    init                 : 'spectral' | 'pca' | 'random'
    random_state         : int or None
    verbose              : print progress

    Attributes (after fit)
    ----------------------
    embedding_ : (n, n_components) final embedding
    graph_     : (n, n) symmetrised fuzzy weight matrix W
    a_, b_     : fitted low-dim kernel parameters
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        spread: float = 1.0,
        n_epochs: int = 200,
        learning_rate: float = 1.0,
        negative_sample_rate: int = 5,
        repulsion_strength: float = 1.0,
        set_op_mix_ratio: float = 1.0,
        init: str = 'spectral',
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.spread = spread
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.negative_sample_rate = negative_sample_rate
        self.repulsion_strength = repulsion_strength
        self.set_op_mix_ratio = set_op_mix_ratio
        self.init = init
        self.random_state = random_state
        self.verbose = verbose

        self.embedding_: Optional[np.ndarray] = None
        self.graph_: Optional[np.ndarray] = None
        self.a_: Optional[float] = None
        self.b_: Optional[float] = None

    def fit(self, X: np.ndarray) -> 'UMAP':
        X = np.asarray(X, dtype=float)
        n = len(X)
        rng = np.random.default_rng(self.random_state)
        k = min(self.n_neighbors, n - 1)

        if self.verbose:
            print(f"  UMAP fit: n={n}, k={k}, init={self.init!r}")

        W = compute_fuzzy_simplicial_set(X, k, self.set_op_mix_ratio)
        self.graph_ = W

        self.a_, self.b_ = find_ab_params(self.spread, self.min_dist)

        if self.init == 'spectral':
            Y = _spectral_init(W, self.n_components, rng)
        elif self.init == 'pca':
            import os, sys as _sys
            _sys.path.insert(0, os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', '24_PCA'))
            from pca_scratch import PCA as _PCA
            pca = _PCA(n_components=self.n_components)
            Z = pca.fit_transform(X)
            std = Z.std(axis=0)
            std = np.where(std > 1e-10, std, 1.0)
            Y = (Z / std) * 0.1
        else:
            Y = rng.standard_normal((n, self.n_components)) * 0.01

        if self.verbose:
            print(f"  Optimizing layout ({self.n_epochs} epochs)...")

        self.embedding_ = umap_optimize_layout(
            W, Y, self.a_, self.b_,
            n_epochs=self.n_epochs,
            learning_rate=self.learning_rate,
            negative_sample_rate=self.negative_sample_rate,
            repulsion_strength=self.repulsion_strength,
            random_state=self.random_state,
        )
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and return the embedding."""
        return self.fit(X).embedding_
