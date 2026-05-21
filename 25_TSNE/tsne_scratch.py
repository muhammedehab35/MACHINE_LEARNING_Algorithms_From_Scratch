"""
t-SNE (t-distributed Stochastic Neighbor Embedding) — From Scratch

van der Maaten, L. & Hinton, G. (2008). "Visualizing data using t-SNE."
Journal of Machine Learning Research, 9, 2579-2605.

Core idea:
  1. Build a probability distribution P over high-dimensional pairs using
     Gaussian kernels whose bandwidth sigma_i is set by binary search to
     match a target perplexity.
  2. Build a probability distribution Q over low-dimensional pairs using
     the heavy-tailed Student-t(1) kernel (Cauchy distribution).
  3. Minimise KL(P || Q) via gradient descent with momentum and early
     exaggeration so that clusters form and repel each other clearly.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict


# ---------------------------------------------------------------------------
# High-dimensional affinity matrix
# ---------------------------------------------------------------------------

def _pairwise_sq_distances(X: np.ndarray) -> np.ndarray:
    """Compute squared Euclidean distance matrix in O(n^2 p) via BLAS."""
    sq = np.sum(X ** 2, axis=1)
    D = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    return np.maximum(D, 0.0)


def _cond_probs_and_entropy(dist2_i: np.ndarray, beta: float) -> Tuple[float, np.ndarray]:
    """
    Compute P_i (conditional probabilities) and Shannon entropy H_i
    for a single point given precision beta = 1 / (2 * sigma_i^2).

    dist2_i[i] must be 0.0 (self-distance already excluded by the caller).
    """
    d = -dist2_i * beta
    d -= d.max()                         # numerical stability (log-sum-exp shift)
    p = np.exp(d)
    Z = p.sum()
    if Z < 1e-300:
        Z = 1e-300
    p /= Z
    H = -np.sum(p * np.log(np.maximum(p, 1e-300)))   # nats
    return H, p


def compute_joint_probabilities(
    X: np.ndarray,
    perplexity: float = 30.0,
    tol: float = 1e-5,
    max_binary_search: int = 50,
) -> np.ndarray:
    """
    Compute the symmetric joint probability matrix P (n x n).

    For each point i, a binary search over beta_i = 1/(2*sigma_i^2) finds
    the bandwidth that matches the target perplexity:
        H(P_i) = log(perplexity)   (entropy in nats)

    Then the symmetric version is:
        p_ij = (p_{j|i} + p_{i|j}) / (2n)

    Parameters
    ----------
    X          : (n, p) data matrix (should be mean-centred)
    perplexity : target perplexity (effective number of neighbors)
    tol        : binary search convergence tolerance on entropy
    max_binary_search : max iterations per binary search

    Returns
    -------
    P : (n, n) symmetric probability matrix with zero diagonal
    """
    n = len(X)
    target_H = np.log(perplexity)        # target entropy in nats
    D2 = _pairwise_sq_distances(X)       # (n, n)
    P = np.zeros((n, n))

    for i in range(n):
        d_i = D2[i].copy()
        d_i[i] = 0.0                     # exclude self (will get p=0 after shift)

        beta_lo, beta_hi = -np.inf, np.inf
        beta = 1.0

        for _ in range(max_binary_search):
            # Mask self by setting self-distance contribution to zero in exp
            d_tmp = d_i.copy()
            d_tmp[i] = np.inf            # maps to exp(-inf) = 0 after shift
            H, p = _cond_probs_and_entropy(d_tmp, beta)
            p[i] = 0.0                   # ensure exactly zero

            diff = H - target_H
            if abs(diff) < tol:
                break
            if diff > 0:                 # H too high → beta too small → increase
                beta_lo = beta
                beta = 2 * beta if beta_hi == np.inf else (beta + beta_hi) / 2
            else:                        # H too low → beta too large → decrease
                beta_hi = beta
                beta = beta / 2 if beta_lo == -np.inf else (beta + beta_lo) / 2

        P[i] = p

    # Symmetrize and normalize
    P = (P + P.T) / (2.0 * n)
    P = np.maximum(P, 1e-12)
    return P


# ---------------------------------------------------------------------------
# Low-dimensional affinity (Student-t kernel)
# ---------------------------------------------------------------------------

def student_t_affinities(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Q matrix and unnormalised kernel values from current embedding Y.

    q_ij ∝ (1 + ||y_i - y_j||^2)^{-1}   (Student-t with 1 degree of freedom)

    Returns
    -------
    Q   : (n, n) joint probability matrix (zero diagonal)
    num : (n, n) unnormalised kernel values (1 / (1 + dist^2)), zero diagonal
    """
    D2 = _pairwise_sq_distances(Y)
    num = 1.0 / (1.0 + D2)
    np.fill_diagonal(num, 0.0)
    Q = num / num.sum()
    Q = np.maximum(Q, 1e-12)
    return Q, num


# ---------------------------------------------------------------------------
# Gradient of KL(P || Q) w.r.t. Y
# ---------------------------------------------------------------------------

def tsne_gradient(P: np.ndarray, Q: np.ndarray,
                  num: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of KL(P || Q) w.r.t. Y:

        dC/dy_i = 4 * sum_j (p_ij - q_ij)(y_i - y_j)(1 + ||y_i - y_j||^2)^{-1}

    In matrix form with A_ij = (P_ij - Q_ij) * num_ij:

        dC/dY = 4 * (diag(A @ 1) - A) @ Y

    Parameters
    ----------
    P, Q, num : (n, n)
    Y         : (n, d)

    Returns
    -------
    grad : (n, d)
    """
    A = (P - Q) * num                                        # (n, n)
    grad = 4.0 * (A.sum(axis=1, keepdims=True) * Y - A @ Y) # (n, d)
    return grad


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------

def kl_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    """KL(P || Q) = sum_ij p_ij * log(p_ij / q_ij)."""
    mask = P > 1e-12
    return float(np.sum(P[mask] * np.log(P[mask] / Q[mask])))


# ---------------------------------------------------------------------------
# TSNE class
# ---------------------------------------------------------------------------

class TSNE:
    """
    t-SNE: t-distributed Stochastic Neighbor Embedding.

    Parameters
    ----------
    n_components       : target dimensionality (almost always 2)
    perplexity         : effective number of neighbors (typically 5-50)
    learning_rate      : gradient descent step size (100-1000)
    n_iter             : total gradient descent iterations
    early_exaggeration : scale P by this factor for first n_iter_early_exag
                         iterations to form tight, well-separated clusters
    n_iter_early_exag  : number of early-exaggeration iterations
    momentum_init      : momentum for t < n_iter_early_exag
    momentum_final     : momentum for t >= n_iter_early_exag
    init               : 'random' | 'pca'
    random_state       : int or None
    verbose            : if True, print KL divergence every 100 iterations

    Attributes (after fit)
    ----------------------
    embedding_             : (n, n_components) final low-dim embedding
    kl_divergence_         : float  final KL(P || Q)
    kl_divergence_history_ : list[(iter, kl)] sampled every 50 iterations
    n_iter_                : int
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        early_exaggeration: float = 12.0,
        n_iter_early_exag: int = 250,
        momentum_init: float = 0.5,
        momentum_final: float = 0.8,
        init: str = 'random',
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.n_iter_early_exag = n_iter_early_exag
        self.momentum_init = momentum_init
        self.momentum_final = momentum_final
        self.init = init
        self.random_state = random_state
        self.verbose = verbose

        self.embedding_: Optional[np.ndarray] = None
        self.kl_divergence_: float = np.inf
        self.kl_divergence_history_: List[Tuple[int, float]] = []
        self.n_iter_: int = 0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray,
            _checkpoints: Optional[Dict[int, np.ndarray]] = None) -> 'TSNE':
        """
        Fit t-SNE to data X.

        Parameters
        ----------
        X            : (n, p) input data
        _checkpoints : optional dict; will be filled with embedding snapshots
                       at keys that are iteration numbers (used by generate_images)
        """
        X = np.asarray(X, dtype=float)
        n = len(X)
        rng = np.random.default_rng(self.random_state)

        # Centre X (good practice; doesn't affect P)
        X = X - X.mean(axis=0)

        # 1. Compute joint probability matrix P
        if self.verbose:
            print("  Computing P matrix (binary search for sigma)...")
        P = compute_joint_probabilities(X, self.perplexity)
        P_exag = P * self.early_exaggeration

        # 2. Initialise embedding Y
        if self.init == 'pca':
            import os, sys
            sys.path.insert(0, os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', '24_PCA'))
            from pca_scratch import PCA as _PCA
            pca = _PCA(n_components=self.n_components)
            Y = pca.fit_transform(X) * 1e-4
        else:
            Y = rng.standard_normal((n, self.n_components)) * 1e-4

        # 3. Gradient descent with momentum
        velocity = np.zeros_like(Y)
        history: List[Tuple[int, float]] = []

        for t in range(self.n_iter):
            # Choose P (with or without early exaggeration) and momentum
            P_t = P_exag if t < self.n_iter_early_exag else P
            mom = self.momentum_init if t < self.n_iter_early_exag else self.momentum_final

            # E-step: low-dim affinities
            Q, num = student_t_affinities(Y)

            # Gradient
            grad = tsne_gradient(P_t, Q, num, Y)

            # Update
            velocity = mom * velocity - self.learning_rate * grad
            Y = Y + velocity

            # Re-centre (prevents drifting to infinity)
            Y -= Y.mean(axis=0)

            # Log KL divergence (using true P, not exaggerated)
            if t % 50 == 0 or t == self.n_iter - 1:
                kl = kl_divergence(P, Q)
                history.append((t, kl))
                if self.verbose:
                    print(f"  Iter {t:4d}: KL = {kl:.5f}")

            # Store checkpoint if requested
            if _checkpoints is not None and t in _checkpoints:
                _checkpoints[t] = Y.copy()

        self.embedding_ = Y
        self.kl_divergence_history_ = history
        self.kl_divergence_ = history[-1][1] if history else np.inf
        self.n_iter_ = self.n_iter
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and return the embedding."""
        return self.fit(X).embedding_
