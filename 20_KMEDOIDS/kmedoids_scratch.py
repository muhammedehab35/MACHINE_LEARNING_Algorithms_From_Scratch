"""
K-Medoids Clustering (PAM) — From Scratch Implementation

Kaufman, L. & Rousseeuw, P.J. (1987). "Clustering by means of Medoids."
Statistical Data Analysis Based on the L1-Norm, 405-416.

Schubert, E. & Rousseeuw, P.J. (2021). "Fast and eager k-medoids clustering:
O(k) runtime improvement of the PAM, CLARA, and CLARANS algorithms."
Information Systems, 101, 101804.

Core idea: unlike K-Means, medoids are actual data points from the dataset.
The objective minimises total distance (not squared distance) to medoids,
making K-Medoids robust to outliers and compatible with any distance metric.
"""

import numpy as np
from typing import Optional, Union, Callable


# ---------------------------------------------------------------------------
# Distance utilities
# ---------------------------------------------------------------------------

def pairwise_distances(
    X: np.ndarray,
    metric: Union[str, Callable] = 'euclidean',
) -> np.ndarray:
    """
    Compute the (n x n) pairwise distance matrix for data matrix X.

    Supported metrics: 'euclidean', 'manhattan', 'cosine'.
    A callable metric(u, v) -> float is also accepted.
    """
    n = len(X)
    if callable(metric):
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = metric(X[i], X[j])
        return D

    if metric == 'euclidean':
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        sq = np.sum(X ** 2, axis=1)
        D_sq = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        return np.sqrt(np.maximum(D_sq, 0.0))

    if metric == 'manhattan':
        D = np.zeros((n, n))
        for i in range(n):
            D[i] = np.sum(np.abs(X[i] - X), axis=1)
        return D

    if metric == 'cosine':
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        X_norm = X / norms
        sim = X_norm @ X_norm.T
        return np.maximum(1.0 - sim, 0.0)

    raise ValueError(f"Unknown metric: {metric!r}. Use 'euclidean', 'manhattan', 'cosine', or a callable.")


def _total_cost(D: np.ndarray, labels: np.ndarray, medoid_indices: np.ndarray) -> float:
    """Total distance from every point to its assigned medoid."""
    return float(np.sum(D[np.arange(len(labels)), medoid_indices[labels]]))


# ---------------------------------------------------------------------------
# Initialisation strategies
# ---------------------------------------------------------------------------

def _init_random(n: int, k: int, rng: np.random.RandomState) -> np.ndarray:
    return rng.choice(n, size=k, replace=False)


def _init_build(D: np.ndarray, k: int) -> np.ndarray:
    """
    PAM BUILD phase (greedy initialisation).

    1. Select the single point minimising total distance to all others as m_1.
    2. For each subsequent medoid, choose the point that most reduces the
       total cost given existing medoids.
    """
    n = len(D)
    # First medoid: minimise sum of all distances
    first = int(np.argmin(D.sum(axis=1)))
    medoids = [first]

    for _ in range(k - 1):
        # Current cost for each point: distance to nearest existing medoid
        curr_cost = D[:, medoids].min(axis=1)   # (n,)
        best_gain = -np.inf
        best_m = -1

        for o in range(n):
            if o in medoids:
                continue
            # If we add o: each point gains if d(j,o) < curr_cost[j]
            gain = np.sum(np.maximum(curr_cost - D[:, o], 0.0))
            if gain > best_gain:
                best_gain = gain
                best_m = o

        medoids.append(best_m)

    return np.array(medoids)


# ---------------------------------------------------------------------------
# KMedoids (PAM)
# ---------------------------------------------------------------------------

class KMedoids:
    """
    K-Medoids clustering via PAM (Partitioning Around Medoids).

    Medoids are actual data points; the objective minimises total (not
    squared) distance to medoids. Works with any distance metric and is
    robust to outliers.

    Parameters
    ----------
    n_clusters  : number of clusters K
    metric      : 'euclidean' | 'manhattan' | 'cosine' | callable(u,v)->float
    init        : 'build' (PAM BUILD, greedy) | 'random'
    max_iter    : maximum SWAP iterations
    n_init      : number of independent runs (best cost kept)
    random_state: reproducibility seed

    Attributes
    ----------
    medoid_indices_ : indices into X of the K medoids, shape (K,)
    cluster_centers_: medoid vectors, shape (K, p)
    labels_         : cluster assignments, shape (n,)
    inertia_        : total distance (cost) of best solution
    n_iter_         : SWAP iterations of best run
    cost_history_   : total cost per SWAP iteration (best run)
    """

    def __init__(
        self,
        n_clusters: int = 3,
        metric: Union[str, Callable] = 'euclidean',
        init: str = 'build',
        max_iter: int = 300,
        n_init: int = 5,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.metric = metric
        self.init = init
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

        self.medoid_indices_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: Optional[int] = None
        self.cost_history_: Optional[list] = None

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _assign(self, D: np.ndarray, medoids: np.ndarray) -> np.ndarray:
        """Assign each point to the nearest medoid (E-step)."""
        return np.argmin(D[:, medoids], axis=1)   # (n,) cluster indices 0..K-1

    def _swap_cost(self, D: np.ndarray, medoids: np.ndarray, labels: np.ndarray) -> tuple:
        """
        PAM SWAP: find the best (medoid_idx, non_medoid) exchange.

        For each current medoid m_i and each non-medoid candidate o,
        compute the change in total cost delta_T if m_i is replaced by o.

        For each sample j:
          - d1j = distance to its current medoid (nearest)
          - d2j = distance to its second nearest medoid
          - if j is assigned to m_i:
              contribution = min(d(j,o), d2j) - d1j
          - else:
              contribution = min(d(j,o), d1j) - d1j  = min(0, d(j,o)-d1j)

        Returns (best_delta, swap_medoid_pos, swap_non_medoid_idx).
        """
        n = len(D)
        k = len(medoids)
        non_medoids = np.array([i for i in range(n) if i not in set(medoids)])

        # Pre-compute d1 and d2 for all points
        dists_to_medoids = D[:, medoids]       # (n, K)
        sorted_dists = np.sort(dists_to_medoids, axis=1)
        d1 = sorted_dists[:, 0]               # nearest medoid distance
        d2 = sorted_dists[:, 1] if k > 1 else sorted_dists[:, 0]  # second nearest

        best_delta = 0.0
        best_mi = -1
        best_o = -1

        for mi in range(k):
            m = medoids[mi]
            is_assigned_to_mi = (labels == mi)

            for o in non_medoids:
                d_jo = D[:, o]   # distance from each point to candidate o

                # Contribution from points currently assigned to m_i
                delta_assigned = np.where(
                    is_assigned_to_mi,
                    np.minimum(d_jo, d2) - d1,
                    0.0
                )
                # Contribution from points NOT assigned to m_i
                delta_other = np.where(
                    ~is_assigned_to_mi,
                    np.minimum(d_jo - d1, 0.0),
                    0.0
                )
                delta = float(np.sum(delta_assigned + delta_other))

                if delta < best_delta:
                    best_delta = delta
                    best_mi = mi
                    best_o = o

        return best_delta, best_mi, best_o

    def _run_once(self, X: np.ndarray, D: np.ndarray, rng: np.random.RandomState):
        """One full PAM run from a fresh initialisation."""
        if self.init == 'build':
            medoids = _init_build(D, self.n_clusters)
        else:
            medoids = _init_random(len(X), self.n_clusters, rng)

        labels = self._assign(D, medoids)
        cost = _total_cost(D, labels, medoids)
        history = [cost]

        for iteration in range(1, self.max_iter + 1):
            delta, mi, o = self._swap_cost(D, medoids, labels)

            if delta >= -1e-10 or o == -1:
                break   # no improving swap found

            medoids = medoids.copy()
            medoids[mi] = o
            labels = self._assign(D, medoids)
            cost += delta
            history.append(float(_total_cost(D, labels, medoids)))

        return medoids, labels, float(_total_cost(D, labels, medoids)), iteration, history

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> 'KMedoids':
        X = np.array(X, dtype=float)
        D = pairwise_distances(X, self.metric)
        rng = np.random.RandomState(self.random_state)

        best_cost = np.inf
        best_medoids = best_labels = best_history = None
        best_iter = 0

        n_runs = 1 if self.init == 'build' else self.n_init
        for _ in range(n_runs):
            medoids, labels, cost, n_iter, history = self._run_once(X, D, rng)
            if cost < best_cost:
                best_cost = cost
                best_medoids = medoids
                best_labels = labels
                best_iter = n_iter
                best_history = history

        self.medoid_indices_ = best_medoids
        self.cluster_centers_ = X[best_medoids]
        self.labels_ = best_labels
        self.inertia_ = best_cost
        self.n_iter_ = best_iter
        self.cost_history_ = best_history
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new points to the nearest medoid (by Euclidean distance to stored medoid vectors)."""
        X = np.array(X, dtype=float)
        D_new = pairwise_distances(
            np.vstack([X, self.cluster_centers_]),
            metric='euclidean' if not callable(self.metric) else self.metric
        )
        n_new = len(X)
        # distances from new points to each medoid
        D_sub = D_new[:n_new, n_new:]
        return np.argmin(D_sub, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return distances from each sample to each medoid, shape (n, K)."""
        X = np.array(X, dtype=float)
        D_new = pairwise_distances(
            np.vstack([X, self.cluster_centers_]),
            metric='euclidean' if not callable(self.metric) else self.metric
        )
        n_new = len(X)
        return D_new[:n_new, n_new:]

    def score(self, X: np.ndarray) -> float:
        """Negative total cost (higher = better, sklearn convention)."""
        X = np.array(X, dtype=float)
        D = pairwise_distances(X, self.metric)
        labels = np.argmin(D[:, self.medoid_indices_], axis=1) if self.medoid_indices_ is not None else self.labels_
        return -_total_cost(D, labels, self.medoid_indices_)


# ---------------------------------------------------------------------------
# Utility: cost curve over K for elbow method
# ---------------------------------------------------------------------------

def elbow_costs(X: np.ndarray, k_range, metric='euclidean', random_state=None) -> list:
    """Fit KMedoids for each K in k_range and return list of total costs."""
    results = []
    for k in k_range:
        km = KMedoids(n_clusters=k, metric=metric, random_state=random_state)
        km.fit(X)
        results.append(km.inertia_)
    return results
