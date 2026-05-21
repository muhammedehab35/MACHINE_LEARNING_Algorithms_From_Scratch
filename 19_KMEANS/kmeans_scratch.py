"""
K-Means Clustering — From Scratch Implementation

Lloyd (1957/1982). "Least Squares Quantization in PCM." IEEE Transactions
on Information Theory, 28(2), 129-137.

Arthur & Vassilvitskii (2007). "k-means++: The advantages of careful
seeding." SODA 2007.

Core idea: iteratively alternate between assigning each point to its
nearest centroid (E-step) and recomputing centroids as cluster means
(M-step), minimising the within-cluster sum of squares (WCSS / inertia).
"""

import numpy as np
from typing import Optional, Literal


# ---------------------------------------------------------------------------
# Distance utilities
# ---------------------------------------------------------------------------

def euclidean_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Compute pairwise squared Euclidean distances between rows of X and centroids.

    Returns shape (n_samples, n_clusters).
    Uses ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b for efficiency.
    """
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)          # (n, 1)
    C_sq = np.sum(centroids ** 2, axis=1, keepdims=True)   # (k, 1)
    cross = X @ centroids.T                                 # (n, k)
    dists_sq = X_sq + C_sq.T - 2 * cross
    return np.maximum(dists_sq, 0.0)                        # numerical safety


# ---------------------------------------------------------------------------
# Initialisation strategies
# ---------------------------------------------------------------------------

def _init_random(X: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    """Pick k random data points as initial centroids."""
    idx = rng.choice(len(X), size=k, replace=False)
    return X[idx].copy()


def _init_kmeans_plus_plus(X: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    """
    K-Means++ seeding (Arthur & Vassilvitskii, 2007).

    Each successive centroid is chosen with probability proportional to
    its squared distance from the nearest already-chosen centroid:

        P(x_i) = D(x_i)^2 / sum_j D(x_j)^2

    This gives O(log k) approximation guarantee on the initial inertia.
    """
    n = len(X)
    first = rng.randint(0, n)
    centroids = [X[first].copy()]

    for _ in range(k - 1):
        C = np.array(centroids)
        dists_sq = euclidean_distances(X, C).min(axis=1)   # (n,)
        probs = dists_sq / dists_sq.sum()
        cumprobs = np.cumsum(probs)
        r = rng.rand()
        idx = np.searchsorted(cumprobs, r)
        centroids.append(X[min(idx, n - 1)].copy())

    return np.array(centroids)


# ---------------------------------------------------------------------------
# KMeans
# ---------------------------------------------------------------------------

class KMeans:
    """
    K-Means clustering via Lloyd's algorithm with K-Means++ seeding.

    Parameters
    ----------
    n_clusters      : number of clusters K
    init            : 'k-means++' (default) or 'random'
    max_iter        : maximum Lloyd iterations per run
    tol             : convergence tolerance (centroid shift L2 norm)
    n_init          : number of independent runs; best inertia is kept
    random_state    : reproducibility seed

    Attributes
    ----------
    cluster_centers_    : final centroid positions, shape (K, p)
    labels_             : cluster assignments for training data, shape (n,)
    inertia_            : WCSS of the best run
    n_iter_             : iterations taken in the best run
    inertia_history_    : WCSS per iteration (best run)
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: Literal['k-means++', 'random'] = 'k-means++',
        max_iter: int = 300,
        tol: float = 1e-4,
        n_init: int = 10,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: Optional[int] = None
        self.inertia_history_: Optional[list] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_centroids(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        if self.init == 'k-means++':
            return _init_kmeans_plus_plus(X, self.n_clusters, rng)
        return _init_random(X, self.n_clusters, rng)

    @staticmethod
    def _assign(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """E-step: assign each point to the nearest centroid."""
        return np.argmin(euclidean_distances(X, centroids), axis=1)

    @staticmethod
    def _update(X: np.ndarray, labels: np.ndarray, k: int, old_centroids: np.ndarray) -> np.ndarray:
        """M-step: recompute each centroid as the mean of its assigned points."""
        centroids = np.empty_like(old_centroids)
        for j in range(k):
            mask = labels == j
            if mask.any():
                centroids[j] = X[mask].mean(axis=0)
            else:
                centroids[j] = old_centroids[j]   # keep old if cluster is empty
        return centroids

    @staticmethod
    def _inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """Within-cluster sum of squared distances."""
        diffs = X - centroids[labels]
        return float(np.sum(diffs ** 2))

    def _run_once(self, X: np.ndarray, rng: np.random.RandomState):
        """One full Lloyd run from a fresh initialisation."""
        centroids = self._init_centroids(X, rng)
        history = []

        for iteration in range(1, self.max_iter + 1):
            labels = self._assign(X, centroids)
            new_centroids = self._update(X, labels, self.n_clusters, centroids)

            wcss = self._inertia(X, labels, new_centroids)
            history.append(wcss)

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < self.tol:
                break

        return labels, centroids, self._inertia(X, labels, centroids), iteration, history

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> 'KMeans':
        X = np.array(X, dtype=float)
        rng = np.random.RandomState(self.random_state)

        best_inertia = np.inf
        best_labels = best_centers = best_history = None
        best_iter = 0

        for _ in range(self.n_init):
            labels, centers, inertia, n_iter, history = self._run_once(X, rng)
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centers = centers
                best_iter = n_iter
                best_history = history

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_iter
        self.inertia_history_ = best_history
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        return self._assign(X, self.cluster_centers_)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return distances from each sample to each centroid, shape (n, K)."""
        X = np.array(X, dtype=float)
        return np.sqrt(np.maximum(euclidean_distances(X, self.cluster_centers_), 0.0))

    def score(self, X: np.ndarray) -> float:
        """Return negative inertia (higher is better, sklearn convention)."""
        X = np.array(X, dtype=float)
        labels = self.predict(X)
        return -self._inertia(X, labels, self.cluster_centers_)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Within-cluster sum of squares (WCSS)."""
    return float(np.sum((X - centroids[labels]) ** 2))


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Mean silhouette coefficient over all samples.

    For sample i:
        a(i) = mean intra-cluster distance
        b(i) = mean distance to nearest other cluster
        s(i) = (b(i) - a(i)) / max(a(i), b(i))

    Range: [-1, 1]. Higher is better; > 0.5 indicates good separation.
    """
    X = np.array(X, dtype=float)
    labels = np.array(labels)
    n = len(X)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return 0.0

    s_vals = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        if same.any():
            a = np.mean(np.sqrt(np.sum((X[same] - X[i]) ** 2, axis=1)))
        else:
            a = 0.0

        b = np.inf
        for lbl in unique_labels:
            if lbl == labels[i]:
                continue
            other = labels == lbl
            d = np.mean(np.sqrt(np.sum((X[other] - X[i]) ** 2, axis=1)))
            if d < b:
                b = d

        denom = max(a, b)
        s_vals[i] = (b - a) / denom if denom > 0 else 0.0

    return float(np.mean(s_vals))


def davies_bouldin_score(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """
    Davies-Bouldin index (lower is better).

    DB = (1/K) * sum_i max_{j != i} [ (s_i + s_j) / d(c_i, c_j) ]

    where s_k = mean intra-cluster distance for cluster k,
          d(c_i, c_j) = Euclidean distance between centroids i and j.
    """
    k = len(centroids)
    s = np.zeros(k)
    for j in range(k):
        mask = labels == j
        if mask.any():
            s[j] = np.mean(np.sqrt(np.sum((X[mask] - centroids[j]) ** 2, axis=1)))

    db = 0.0
    for i in range(k):
        worst = -np.inf
        for j in range(k):
            if i == j:
                continue
            d_ij = np.linalg.norm(centroids[i] - centroids[j])
            if d_ij > 0:
                ratio = (s[i] + s[j]) / d_ij
                if ratio > worst:
                    worst = ratio
        db += worst

    return float(db / k)


def elbow_scores(X: np.ndarray, k_range, random_state=None) -> list:
    """Fit KMeans for each K in k_range and return list of inertias."""
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state)
        km.fit(X)
        results.append(km.inertia_)
    return results
