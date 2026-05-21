"""
Principal Component Analysis (PCA) — From Scratch Implementation

Pearson, K. (1901). "On lines and planes of closest fit to systems of points
in space." Philosophical Magazine, 2(11), 559-572.

Hotelling, H. (1933). "Analysis of a complex of statistical variables into
principal components." Journal of Educational Psychology, 24(6), 417-441.

Core idea: find an orthonormal basis that maximises the variance of projections
(equivalently, minimises the mean squared reconstruction error). Computed via
the economy SVD of the mean-centred data matrix.
"""

import numpy as np
from typing import Optional, Union


class PCA:
    """
    Principal Component Analysis via truncated SVD.

    Parameters
    ----------
    n_components : int | float | None
        Number of components to keep.
        - None : keep all min(n, p) components
        - int  : exact number
        - float in (0, 1) : keep the minimum number that explains this
                            fraction of total variance
    whiten      : if True, divide scores by sqrt(explained_variance) so
                  that the projected data has unit variance along each PC
    copy        : operate on a copy of X (safe default)

    Attributes (after fit)
    ----------------------
    components_              : (n_components_, p)  — principal axes (rows = PCs)
    explained_variance_      : (n_components_,)    — variance along each PC
    explained_variance_ratio_: (n_components_,)    — fraction of total variance
    singular_values_         : (n_components_,)    — singular values of X_c
    mean_                    : (p,)                — per-feature mean
    n_components_            : int
    n_samples_               : int
    n_features_              : int
    noise_variance_          : float — PPCA noise estimate (avg discarded eigenvalue)
    """

    def __init__(
        self,
        n_components: Optional[Union[int, float]] = None,
        whiten: bool = False,
        copy: bool = True,
    ):
        self.n_components = n_components
        self.whiten = whiten
        self.copy = copy

        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.n_components_: Optional[int] = None
        self.n_samples_: Optional[int] = None
        self.n_features_: Optional[int] = None
        self.noise_variance_: float = 0.0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> 'PCA':
        X = np.array(X, dtype=float)
        if self.copy:
            X = X.copy()

        n, p = X.shape
        self.n_samples_ = n
        self.n_features_ = p

        # 1. Centre data
        self.mean_ = X.mean(axis=0)
        X -= self.mean_

        # 2. Economy SVD: X_c = U S Vt  (U: n×r, s: r, Vt: r×p, r=min(n,p))
        U, s, Vt = np.linalg.svd(X, full_matrices=False)

        # 3. Deterministic sign: ensure the element with largest absolute
        #    value in each PC is positive (matches sklearn convention)
        max_abs_cols = np.argmax(np.abs(Vt), axis=1)
        signs = np.sign(Vt[np.arange(len(Vt)), max_abs_cols])
        Vt *= signs[:, None]
        U *= signs[None, :]

        # 4. Eigenvalues of sample covariance C = X_c^T X_c / (n-1)
        explained_variance = s ** 2 / (n - 1)
        total_var = explained_variance.sum()
        expl_ratio = explained_variance / total_var

        # 5. Determine n_components_
        n_comp = self.n_components
        if n_comp is None:
            n_comp = min(n, p)
        elif isinstance(n_comp, float):
            cumvar = np.cumsum(expl_ratio)
            n_comp = int(np.searchsorted(cumvar, n_comp) + 1)
            n_comp = min(n_comp, min(n, p))
        else:
            n_comp = int(n_comp)
        n_comp = max(1, min(n_comp, min(n, p)))

        self.n_components_ = n_comp
        self.components_ = Vt[:n_comp]                    # (n_comp, p)
        self.singular_values_ = s[:n_comp]
        self.explained_variance_ = explained_variance[:n_comp]
        self.explained_variance_ratio_ = expl_ratio[:n_comp]

        # PPCA noise estimate: mean of discarded eigenvalues
        if n_comp < len(explained_variance):
            self.noise_variance_ = float(explained_variance[n_comp:].mean())
        else:
            self.noise_variance_ = 0.0

        return self

    # ------------------------------------------------------------------
    # Transform / inverse transform
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X onto the principal subspace. Returns (n, n_components_)."""
        X = np.array(X, dtype=float) - self.mean_
        Z = X @ self.components_.T                        # (n, n_comp)
        if self.whiten:
            Z /= np.sqrt(self.explained_variance_)
        return Z

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruct data from PC scores. Returns (n, p)."""
        Z = np.array(Z, dtype=float)
        if self.whiten:
            Z = Z * np.sqrt(self.explained_variance_)
        return Z @ self.components_ + self.mean_

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, X: np.ndarray) -> float:
        """Mean negative reconstruction MSE — higher is better."""
        return float(np.mean(self.score_samples(X)))

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Per-sample negative reconstruction MSE. Shape: (n,)."""
        X = np.array(X, dtype=float)
        X_rec = self.inverse_transform(self.transform(X))
        return -np.mean((X - X_rec) ** 2, axis=1)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_covariance(self) -> np.ndarray:
        """Reconstruct the feature covariance matrix from PCA parameters."""
        return (self.components_.T * self.explained_variance_) @ self.components_ \
               + np.eye(self.n_features_) * self.noise_variance_


# ---------------------------------------------------------------------------
# Standalone helper
# ---------------------------------------------------------------------------

def pca_svd(X: np.ndarray, n_components: int) -> dict:
    """
    Convenience wrapper: fit PCA and return a results dict.

    Returns
    -------
    dict with keys: scores, components, explained_variance,
                    explained_variance_ratio, mean, singular_values
    """
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    return {
        'scores': scores,
        'components': pca.components_,
        'explained_variance': pca.explained_variance_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'singular_values': pca.singular_values_,
        'mean': pca.mean_,
    }
