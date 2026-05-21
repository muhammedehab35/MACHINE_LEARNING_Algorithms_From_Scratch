"""
Agglomerative Hierarchical Clustering — From Scratch Implementation

Ward, J.H. (1963). "Hierarchical grouping to optimize an objective function."
Journal of the American Statistical Association, 58(301), 236-244.

Lance, G.N. & Williams, W.T. (1967). "A general theory of classificatory
sorting strategies." Computer Journal, 9(4), 373-380.

Core idea: start with n singleton clusters and iteratively merge the two
closest clusters until a single cluster remains. The merge history forms a
binary tree (dendrogram). Different linkage criteria define "distance between
clusters" differently, leading to different cluster shapes.
"""

import numpy as np
from typing import Optional, Union, Callable


# ---------------------------------------------------------------------------
# Distance utilities
# ---------------------------------------------------------------------------

def pairwise_distances(X: np.ndarray, metric: Union[str, Callable] = 'euclidean') -> np.ndarray:
    """Compute symmetric (n x n) pairwise distance matrix."""
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
    raise ValueError(f"Unknown metric: {metric!r}")


# ---------------------------------------------------------------------------
# Lance-Williams linkage update coefficients
# ---------------------------------------------------------------------------

# Each linkage has coefficients (alpha_i, alpha_j, beta, gamma) in the
# Lance-Williams formula:
#   d(C_k, C_{ij}) = alpha_i * d(C_k, C_i)
#                  + alpha_j * d(C_k, C_j)
#                  + beta    * d(C_i, C_j)
#                  + gamma   * |d(C_k, C_i) - d(C_k, C_j)|

LINKAGE_METHODS = ('single', 'complete', 'average', 'ward', 'centroid', 'median')


def _lance_williams(method: str, ni: int, nj: int, nk: int,
                    d_ki: float, d_kj: float, d_ij: float) -> float:
    """
    Apply the Lance-Williams recurrence to compute d(C_k, C_{ij}).

    Parameters
    ----------
    ni, nj, nk : cluster sizes of C_i, C_j, C_k
    d_ki, d_kj : distances d(C_k, C_i) and d(C_k, C_j)
    d_ij       : distance d(C_i, C_j)
    """
    if method == 'single':
        # alpha_i = alpha_j = 0.5, gamma = -0.5
        return 0.5 * d_ki + 0.5 * d_kj - 0.5 * abs(d_ki - d_kj)

    if method == 'complete':
        return 0.5 * d_ki + 0.5 * d_kj + 0.5 * abs(d_ki - d_kj)

    if method == 'average':
        n_ij = ni + nj
        return (ni / n_ij) * d_ki + (nj / n_ij) * d_kj

    if method == 'ward':
        n_ij = ni + nj
        ai = (nk + ni) / (nk + n_ij)
        aj = (nk + nj) / (nk + n_ij)
        b  = -nk / (nk + n_ij)
        return ai * d_ki + aj * d_kj + b * d_ij

    if method == 'centroid':
        n_ij = ni + nj
        ai = ni / n_ij
        aj = nj / n_ij
        b  = -(ni * nj) / (n_ij ** 2)
        return ai * d_ki + aj * d_kj + b * d_ij

    if method == 'median':
        return 0.5 * d_ki + 0.5 * d_kj - 0.25 * d_ij

    raise ValueError(f"Unknown linkage: {method!r}. Choose from {LINKAGE_METHODS}")


# ---------------------------------------------------------------------------
# Linkage matrix builder (produces scipy-compatible Z)
# ---------------------------------------------------------------------------

def linkage(X: np.ndarray,
            method: str = 'ward',
            metric: Union[str, Callable] = 'euclidean') -> np.ndarray:
    """
    Perform agglomerative hierarchical clustering and return a linkage matrix.

    The linkage matrix Z has shape (n-1, 4):
        Z[k] = [cluster_i, cluster_j, merge_distance, new_cluster_size]

    Cluster indices 0..n-1 are leaf nodes; n..2n-2 are internal nodes.
    This format matches scipy.cluster.hierarchy.linkage.

    Parameters
    ----------
    X      : data matrix, shape (n, p)
    method : linkage criterion ('single', 'complete', 'average', 'ward',
             'centroid', 'median')
    metric : distance metric

    Returns
    -------
    Z : linkage matrix, shape (n-1, 4)
    """
    X = np.array(X, dtype=float)
    n = len(X)

    # Initial pairwise distance matrix; Ward uses squared distances internally
    D_orig = pairwise_distances(X, metric)
    if method == 'ward':
        D = D_orig ** 2          # Ward's formula works on squared distances
    else:
        D = D_orig.copy()

    # Active cluster set and sizes
    active = list(range(n))
    sizes = {i: 1 for i in range(n)}

    # Map: original index -> row/col in D (we extend D as merges happen)
    # Instead, use a dictionary-based distance store for flexibility.
    # dist[i][j] = current distance between clusters i and j (i < j)
    dist = {}
    for i in range(n):
        for j in range(i + 1, n):
            dist[(i, j)] = float(D[i, j])

    Z = np.zeros((n - 1, 4))
    next_id = n

    for step in range(n - 1):
        # Find the pair with minimum distance
        min_d = np.inf
        ci, cj = -1, -1
        for idx_a in range(len(active)):
            for idx_b in range(idx_a + 1, len(active)):
                i, j = active[idx_a], active[idx_b]
                key = (min(i, j), max(i, j))
                d = dist[key]
                if d < min_d:
                    min_d = d
                    ci, cj = i, j

        ni, nj = sizes[ci], sizes[cj]
        d_ij = min_d

        # Record merge: store original (non-squared) distance for Ward
        Z[step] = [ci, cj, np.sqrt(d_ij) if method == 'ward' else d_ij,
                   ni + nj]

        # Remove ci and cj from active
        active.remove(ci)
        active.remove(cj)

        # Compute distances from new cluster to all remaining active clusters
        new_id = next_id
        sizes[new_id] = ni + nj

        for ck in active:
            nk = sizes[ck]
            key_ki = (min(ck, ci), max(ck, ci))
            key_kj = (min(ck, cj), max(ck, cj))
            d_ki = dist[key_ki]
            d_kj = dist[key_kj]

            new_d = _lance_williams(method, ni, nj, nk, d_ki, d_kj, d_ij)
            dist[(min(ck, new_id), max(ck, new_id))] = new_d

        active.append(new_id)
        next_id += 1

    return Z


# ---------------------------------------------------------------------------
# Cut the dendrogram
# ---------------------------------------------------------------------------

def fcluster(Z: np.ndarray, t: float, criterion: str = 'maxclust') -> np.ndarray:
    """
    Form flat clusters from a linkage matrix.

    Parameters
    ----------
    Z         : linkage matrix from linkage()
    t         : threshold value
    criterion : 'maxclust' — cut to get exactly t clusters
                'distance'  — cut at height t

    Returns
    -------
    labels : cluster labels 0..K-1, shape (n,)
    """
    n = Z.shape[0] + 1

    if criterion == 'maxclust':
        k = int(t)
        # Find the height at which the dendrogram has exactly k clusters
        # Sort merge distances; the k-th cut from the top gives k clusters
        merge_heights = Z[:, 2]
        sorted_heights = np.sort(merge_heights)[::-1]
        if k >= n or k <= 1:
            cut_height = np.inf
        else:
            cut_height = (sorted_heights[k - 2] + sorted_heights[k - 1]) / 2.0
        return fcluster(Z, cut_height, criterion='distance')

    # criterion == 'distance'
    cut_height = float(t)

    # Build the merge tree and find connected components below cut_height
    parent = list(range(2 * n - 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y, new_id):
        parent[x] = new_id
        parent[y] = new_id
        parent[new_id] = new_id

    for k in range(n - 1):
        ci, cj, h, _ = int(Z[k, 0]), int(Z[k, 1]), Z[k, 2], Z[k, 3]
        if h <= cut_height:
            new_id = n + k
            union(find(ci), find(cj), new_id)

    # Assign cluster labels to leaf nodes
    roots = [find(i) for i in range(n)]
    unique_roots = sorted(set(roots))
    root_to_label = {r: idx for idx, r in enumerate(unique_roots)}
    return np.array([root_to_label[r] for r in roots])


# ---------------------------------------------------------------------------
# Cophenetic correlation coefficient
# ---------------------------------------------------------------------------

def cophenetic_correlation(X: np.ndarray, Z: np.ndarray,
                           metric: Union[str, Callable] = 'euclidean') -> float:
    """
    Compute the cophenetic correlation coefficient.

    For each pair (i,j), the cophenetic distance c(i,j) is the height at which
    they are first joined in the dendrogram. The cophenetic correlation is the
    Pearson correlation between c(i,j) and d(i,j) for all pairs.

    Range: [-1, 1]. Higher values indicate the dendrogram faithfully
    represents the pairwise distances. Values > 0.75 are considered good.
    """
    n = Z.shape[0] + 1
    D = pairwise_distances(X, metric)

    # Build cophenetic distance matrix
    coph = np.zeros((n, n))
    # Track which original leaves each node contains
    members = {i: {i} for i in range(n)}

    for k in range(n - 1):
        ci, cj, h = int(Z[k, 0]), int(Z[k, 1]), Z[k, 2]
        new_id = n + k
        for i in members[ci]:
            for j in members[cj]:
                coph[i, j] = coph[j, i] = h
        members[new_id] = members[ci] | members[cj]

    # Extract upper triangle
    idx = np.triu_indices(n, k=1)
    d_vals = D[idx]
    c_vals = coph[idx]

    # Pearson correlation
    if d_vals.std() < 1e-12 or c_vals.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(d_vals, c_vals)[0, 1])


# ---------------------------------------------------------------------------
# AgglomerativeClustering convenience class
# ---------------------------------------------------------------------------

class AgglomerativeClustering:
    """
    Agglomerative hierarchical clustering.

    Parameters
    ----------
    n_clusters  : number of clusters for the flat cut (or None to keep Z only)
    linkage     : 'single' | 'complete' | 'average' | 'ward' | 'centroid' | 'median'
    metric      : 'euclidean' | 'manhattan' | callable(u,v)->float

    Attributes
    ----------
    labels_      : flat cluster labels, shape (n,)
    Z_           : linkage matrix, shape (n-1, 4)
    n_clusters_  : actual number of clusters
    """

    def __init__(
        self,
        n_clusters: int = 2,
        linkage_method: str = 'ward',
        metric: Union[str, Callable] = 'euclidean',
    ):
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.metric = metric

        self.labels_: Optional[np.ndarray] = None
        self.Z_: Optional[np.ndarray] = None
        self.n_clusters_: Optional[int] = None

    def fit(self, X: np.ndarray) -> 'AgglomerativeClustering':
        X = np.array(X, dtype=float)
        self.Z_ = linkage(X, method=self.linkage_method, metric=self.metric)
        self.labels_ = fcluster(self.Z_, self.n_clusters, criterion='maxclust')
        self.n_clusters_ = len(np.unique(self.labels_))
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_
