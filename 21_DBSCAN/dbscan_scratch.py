"""
DBSCAN — Density-Based Spatial Clustering of Applications with Noise
From Scratch Implementation

Ester, M., Kriegel, H.P., Sander, J. & Xu, X. (1996).
"A density-based algorithm for discovering clusters in large spatial databases
with noise." KDD-96, 226-231.

Core idea: clusters are dense regions of points separated by sparser regions.
Points are classified as core, border, or noise based on how many neighbours
they have within radius eps. Clusters grow by chaining density-reachable points.
"""

import numpy as np
from collections import deque
from typing import Optional, Union, Callable


# ---------------------------------------------------------------------------
# Distance utilities
# ---------------------------------------------------------------------------

def pairwise_distances(X: np.ndarray, metric: Union[str, Callable] = 'euclidean') -> np.ndarray:
    """Compute (n x n) pairwise distance matrix."""
    n = len(X)

    if callable(metric):
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = metric(X[i], X[j])
        return D

    if metric == 'euclidean':
        sq = np.sum(X ** 2, axis=1)
        D_sq = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        return np.sqrt(np.maximum(D_sq, 0.0))

    if metric == 'manhattan':
        D = np.zeros((n, n))
        for i in range(n):
            D[i] = np.sum(np.abs(X[i] - X), axis=1)
        return D

    raise ValueError(f"Unknown metric: {metric!r}. Use 'euclidean', 'manhattan', or a callable.")


def radius_neighbors(D: np.ndarray, idx: int, eps: float) -> np.ndarray:
    """Return indices of all points within eps of point idx (inclusive of idx)."""
    return np.where(D[idx] <= eps)[0]


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------

NOISE = -1
UNVISITED = -2


class DBSCAN:
    """
    DBSCAN clustering.

    Classifies each point as:
    - Core point  : has >= min_samples neighbours within eps (including itself)
    - Border point: within eps of a core point but not itself a core point
    - Noise point : neither core nor border (label = -1)

    Parameters
    ----------
    eps         : neighbourhood radius
    min_samples : minimum neighbours (inclusive of self) to be a core point
    metric      : 'euclidean' | 'manhattan' | callable(u,v)->float

    Attributes
    ----------
    labels_              : cluster label per point (-1 = noise), shape (n,)
    core_sample_indices_ : indices of core points
    n_clusters_          : number of clusters found (excluding noise)
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: Union[str, Callable] = 'euclidean',
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

        self.labels_: Optional[np.ndarray] = None
        self.core_sample_indices_: Optional[np.ndarray] = None
        self.n_clusters_: Optional[int] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> 'DBSCAN':
        X = np.array(X, dtype=float)
        n = len(X)

        D = pairwise_distances(X, self.metric)

        labels = np.full(n, UNVISITED, dtype=int)
        cluster_id = 0

        for i in range(n):
            if labels[i] != UNVISITED:
                continue

            nbrs = radius_neighbors(D, i, self.eps)

            if len(nbrs) < self.min_samples:
                labels[i] = NOISE
                continue

            # i is a core point — start a new cluster via BFS
            labels[i] = cluster_id
            queue = deque(nbrs[nbrs != i])   # seed with neighbours (excluding self)

            while queue:
                j = queue.popleft()

                if labels[j] == NOISE:
                    labels[j] = cluster_id   # promote noise to border
                    continue

                if labels[j] != UNVISITED:
                    continue                  # already processed

                labels[j] = cluster_id
                j_nbrs = radius_neighbors(D, j, self.eps)

                if len(j_nbrs) >= self.min_samples:
                    # j is also a core point — add its unvisited neighbours
                    for q in j_nbrs:
                        if labels[q] == UNVISITED or labels[q] == NOISE:
                            queue.append(q)

            cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.array([
            i for i in range(n)
            if len(radius_neighbors(D, i, self.eps)) >= self.min_samples
        ])
        self.n_clusters_ = cluster_id
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def point_types(self, X: np.ndarray) -> np.ndarray:
        """
        Return per-point type array after fit:
          2 = core, 1 = border, 0 = noise
        """
        X = np.array(X, dtype=float)
        D = pairwise_distances(X, self.metric)
        n = len(X)
        types = np.zeros(n, dtype=int)
        core_set = set(self.core_sample_indices_)

        for i in range(n):
            if i in core_set:
                types[i] = 2
            elif self.labels_[i] != NOISE:
                types[i] = 1   # border
            else:
                types[i] = 0   # noise
        return types


# ---------------------------------------------------------------------------
# Parameter selection helpers
# ---------------------------------------------------------------------------

def k_dist(X: np.ndarray, k: int = 4, metric: str = 'euclidean') -> np.ndarray:
    """
    Compute the k-th nearest-neighbour distance for each point (sorted ascending).

    Plotting the sorted k-dist array produces the k-dist graph; the elbow
    suggests a good eps value (Ester et al., 1996).
    """
    D = pairwise_distances(X, metric)
    np.fill_diagonal(D, np.inf)
    k_distances = np.sort(D, axis=1)[:, k - 1]   # (k-1)-th column = k-th nearest
    return np.sort(k_distances)[::-1]             # descending for elbow plot


def cluster_stats(labels: np.ndarray) -> dict:
    """Return summary statistics about a DBSCAN labelling."""
    n = len(labels)
    noise_mask = labels == NOISE
    n_noise = int(noise_mask.sum())
    unique_clusters = np.unique(labels[~noise_mask])
    cluster_sizes = [int((labels == c).sum()) for c in unique_clusters]
    return {
        'n_clusters': len(unique_clusters),
        'n_noise': n_noise,
        'noise_fraction': n_noise / n,
        'cluster_sizes': cluster_sizes,
        'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
        'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
    }
