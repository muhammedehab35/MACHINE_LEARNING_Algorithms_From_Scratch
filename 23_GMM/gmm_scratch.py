"""
Gaussian Mixture Models (GMM) — From Scratch Implementation

Dempster, A.P., Laird, N.M. & Rubin, D.B. (1977). "Maximum likelihood from
incomplete data via the EM algorithm." Journal of the Royal Statistical
Society, Series B, 39(1), 1-38.

Core idea: model data as a weighted sum of K Gaussian components. Parameters
(means, covariances, mixing weights) are estimated via Expectation-Maximization.
"""

import numpy as np
from typing import Optional, List, Tuple

_LOG2PI = np.log(2.0 * np.pi)


# ---------------------------------------------------------------------------
# Numerical utilities
# ---------------------------------------------------------------------------

def _log_sum_exp(a: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable log-sum-exp along a given axis."""
    a_max = np.max(a, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis))
    return out + np.squeeze(a_max, axis=axis)


# ---------------------------------------------------------------------------
# Log-density helpers for each covariance type
# ---------------------------------------------------------------------------

def _log_pdf_full(X: np.ndarray, mu: np.ndarray,
                  Sigma: np.ndarray, reg: float) -> np.ndarray:
    """log N(X | mu, Sigma) for full covariance. Shape: (n,)"""
    p = X.shape[1]
    diff = X - mu
    try:
        L = np.linalg.cholesky(Sigma + np.eye(p) * reg)
    except np.linalg.LinAlgError:
        var = max(np.trace(Sigma) / p, reg)
        L = np.eye(p) * np.sqrt(var)
    log_det = 2.0 * np.sum(np.log(np.maximum(np.diag(L), 1e-300)))
    y = np.linalg.solve(L, diff.T)         # (p, n)
    maha = np.sum(y ** 2, axis=0)          # (n,)
    return -0.5 * (p * _LOG2PI + log_det + maha)


def _log_pdf_diag(X: np.ndarray, mu: np.ndarray,
                  var: np.ndarray, reg: float) -> np.ndarray:
    """log N(X | mu, diag(var)). Shape: (n,)"""
    p = X.shape[1]
    v = np.maximum(var, reg)
    diff = X - mu
    log_det = np.sum(np.log(v))
    maha = np.sum(diff ** 2 / v, axis=1)
    return -0.5 * (p * _LOG2PI + log_det + maha)


def _log_pdf_spherical(X: np.ndarray, mu: np.ndarray,
                       var: float, reg: float) -> np.ndarray:
    """log N(X | mu, var * I). Shape: (n,)"""
    p = X.shape[1]
    v = max(float(var), reg)
    diff = X - mu
    maha = np.sum(diff ** 2, axis=1) / v
    return -0.5 * (p * _LOG2PI + p * np.log(v) + maha)


# ---------------------------------------------------------------------------
# GaussianMixture
# ---------------------------------------------------------------------------

class GaussianMixture:
    """
    Gaussian Mixture Model fitted by Expectation-Maximization (EM).

    Parameters
    ----------
    n_components    : K — number of mixture components
    covariance_type : 'full' | 'diag' | 'spherical' | 'tied'
    max_iter        : maximum EM iterations per run
    tol             : convergence tolerance on mean log-likelihood change
    n_init          : number of random restarts (best is kept)
    init_params     : 'kmeans' | 'random'
    reg_covar       : regularisation added to covariance diagonals
    random_state    : int seed or None

    Attributes (after fit)
    ----------------------
    weights_         : (K,)       mixing weights pi_k
    means_           : (K, p)     component means
    covariances_     : shape depends on covariance_type
                       'full'     -> (K, p, p)
                       'diag'     -> (K, p)
                       'spherical'-> (K,)
                       'tied'     -> (p, p)
    converged_       : bool
    n_iter_          : int
    lower_bound_     : float      mean log-likelihood at convergence
    log_likelihoods_ : list[float] per-iteration log-likelihoods (best run)
    labels_          : (n,) most likely component index per training sample
    """

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: str = 'full',
        max_iter: int = 100,
        tol: float = 1e-4,
        n_init: int = 1,
        init_params: str = 'kmeans',
        reg_covar: float = 1e-6,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.init_params = init_params
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.verbose = verbose

        self.weights_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.covariances_ = None
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.lower_bound_: float = -np.inf
        self.log_likelihoods_: List[float] = []
        self.labels_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _kmeans_init(self, X: np.ndarray, rng) -> np.ndarray:
        """K-Means clustering; returns hard responsibility matrix (n, K)."""
        n, p = X.shape
        K = self.n_components
        idx = rng.choice(n, K, replace=False)
        centers = X[idx].copy()
        for _ in range(50):
            dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            assign = np.argmin(dists, axis=1)
            new_centers = np.array([
                X[assign == k].mean(axis=0) if (assign == k).any()
                else X[rng.integers(0, n)]
                for k in range(K)
            ])
            if np.allclose(centers, new_centers, atol=1e-8):
                break
            centers = new_centers
        dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        assign = np.argmin(dists, axis=1)
        resp = np.zeros((n, K))
        resp[np.arange(n), assign] = 1.0
        return resp

    def _initialize(self, X: np.ndarray, rng):
        n, p = X.shape
        K = self.n_components
        if self.init_params == 'kmeans':
            resp = self._kmeans_init(X, rng)
        else:
            raw = rng.random((n, K))
            resp = raw / raw.sum(axis=1, keepdims=True)
        resp = np.clip(resp, 1e-300, 1.0)
        self._m_step(X, np.log(resp))

    # ------------------------------------------------------------------
    # E-step and M-step
    # ------------------------------------------------------------------

    def _log_joint(self, X: np.ndarray) -> np.ndarray:
        """log(pi_k * N(x | mu_k, Sigma_k)) for all k. Shape: (n, K)."""
        K = self.n_components
        n = len(X)
        lj = np.empty((n, K))
        ct = self.covariance_type
        reg = self.reg_covar
        for k in range(K):
            log_pi = np.log(max(self.weights_[k], 1e-300))
            mu = self.means_[k]
            if ct == 'full':
                lj[:, k] = log_pi + _log_pdf_full(X, mu, self.covariances_[k], reg)
            elif ct == 'diag':
                lj[:, k] = log_pi + _log_pdf_diag(X, mu, self.covariances_[k], reg)
            elif ct == 'spherical':
                lj[:, k] = log_pi + _log_pdf_spherical(X, mu, self.covariances_[k], reg)
            elif ct == 'tied':
                lj[:, k] = log_pi + _log_pdf_full(X, mu, self.covariances_, reg)
            else:
                raise ValueError(f"Unknown covariance_type: {ct!r}")
        return lj

    def _e_step(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return log responsibilities (n, K) and mean log-likelihood."""
        lj = self._log_joint(X)                         # (n, K)
        log_prob = _log_sum_exp(lj, axis=1)             # (n,)
        log_resp = lj - log_prob[:, None]               # (n, K)
        return log_resp, float(np.mean(log_prob))

    def _m_step(self, X: np.ndarray, log_resp: np.ndarray):
        """Update all parameters from log responsibilities."""
        n, p = X.shape
        K = self.n_components
        resp = np.exp(log_resp)                         # (n, K)
        N_k = np.maximum(resp.sum(axis=0), 1e-10)      # (K,)

        self.weights_ = N_k / n
        self.means_ = (resp.T @ X) / N_k[:, None]      # (K, p)

        ct = self.covariance_type
        reg = self.reg_covar

        if ct == 'full':
            covs = np.empty((K, p, p))
            for k in range(K):
                d = X - self.means_[k]
                w = resp[:, k]
                covs[k] = (w[:, None] * d).T @ d / N_k[k] + np.eye(p) * reg
            self.covariances_ = covs

        elif ct == 'diag':
            covs = np.empty((K, p))
            for k in range(K):
                d = X - self.means_[k]
                w = resp[:, k]
                covs[k] = np.maximum(
                    (w[:, None] * d ** 2).sum(axis=0) / N_k[k], reg)
            self.covariances_ = covs

        elif ct == 'spherical':
            covs = np.empty(K)
            for k in range(K):
                d = X - self.means_[k]
                w = resp[:, k]
                covs[k] = max(
                    np.sum(w * np.sum(d ** 2, axis=1)) / (N_k[k] * p), reg)
            self.covariances_ = covs

        elif ct == 'tied':
            cov = np.zeros((p, p))
            for k in range(K):
                d = X - self.means_[k]
                w = resp[:, k]
                cov += (w[:, None] * d).T @ d
            cov = cov / n + np.eye(p) * reg
            self.covariances_ = cov

    # ------------------------------------------------------------------
    # Single EM run
    # ------------------------------------------------------------------

    def _fit_single(self, X: np.ndarray, rng) -> Tuple[float, bool, int, List[float]]:
        self._initialize(X, rng)
        lower_bound = -np.inf
        history: List[float] = []
        for iteration in range(self.max_iter):
            prev = lower_bound
            log_resp, lower_bound = self._e_step(X)
            self._m_step(X, log_resp)
            history.append(lower_bound)
            if self.verbose:
                print(f"  iter {iteration+1:3d}: log-likelihood = {lower_bound:.6f}")
            if abs(lower_bound - prev) < self.tol:
                return lower_bound, True, iteration + 1, history
        return lower_bound, False, self.max_iter, history

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> 'GaussianMixture':
        X = np.array(X, dtype=float)
        base_rng = np.random.default_rng(self.random_state)
        best_lb = -np.inf
        best_state = None

        for _ in range(self.n_init):
            seed = int(base_rng.integers(0, 2 ** 31))
            lb, converged, n_iter, history = self._fit_single(
                X, np.random.default_rng(seed))
            if lb > best_lb:
                best_lb = lb
                best_state = {
                    'weights_': self.weights_.copy(),
                    'means_': self.means_.copy(),
                    'covariances_': (self.covariances_.copy()
                                     if isinstance(self.covariances_, np.ndarray)
                                     else self.covariances_),
                    'converged_': converged,
                    'n_iter_': n_iter,
                    'log_likelihoods_': history,
                }

        for attr, val in best_state.items():
            setattr(self, attr, val)
        self.lower_bound_ = best_lb
        self.labels_ = self.predict(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Most likely component index for each sample."""
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Posterior component probabilities r_ik. Shape: (n, K)."""
        X = np.array(X, dtype=float)
        lj = self._log_joint(X)
        lp = _log_sum_exp(lj, axis=1)[:, None]
        return np.exp(lj - lp)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).predict(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Per-sample log-likelihood log p(x). Shape: (n,)."""
        X = np.array(X, dtype=float)
        return _log_sum_exp(self._log_joint(X), axis=1)

    def score(self, X: np.ndarray) -> float:
        """Mean log-likelihood (higher is better)."""
        return float(np.mean(self.score_samples(X)))

    def sample(self, n_samples: int = 1,
               random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random samples from the fitted mixture.

        Returns
        -------
        X          : (n_samples, p) samples
        components : (n_samples,)   component index of each sample
        """
        rng = np.random.default_rng(
            random_state if random_state is not None else self.random_state)
        K = self.n_components
        p = self.means_.shape[1]
        components = rng.choice(K, size=n_samples, p=self.weights_)
        X = np.zeros((n_samples, p))
        ct = self.covariance_type
        for k in range(K):
            mask = components == k
            n_k = int(mask.sum())
            if n_k == 0:
                continue
            mu = self.means_[k]
            if ct == 'full':
                X[mask] = rng.multivariate_normal(mu, self.covariances_[k], size=n_k)
            elif ct == 'diag':
                X[mask] = mu + rng.standard_normal((n_k, p)) * np.sqrt(self.covariances_[k])
            elif ct == 'spherical':
                X[mask] = mu + rng.standard_normal((n_k, p)) * np.sqrt(self.covariances_[k])
            elif ct == 'tied':
                X[mask] = rng.multivariate_normal(mu, self.covariances_, size=n_k)
        return X, components

    def _n_parameters(self, p: int) -> int:
        """Free parameter count for BIC/AIC."""
        K = self.n_components
        ct = self.covariance_type
        n_weights = K - 1
        n_means = K * p
        if ct == 'full':
            n_cov = K * p * (p + 1) // 2
        elif ct == 'diag':
            n_cov = K * p
        elif ct == 'spherical':
            n_cov = K
        elif ct == 'tied':
            n_cov = p * (p + 1) // 2
        else:
            n_cov = 0
        return n_weights + n_means + n_cov

    def bic(self, X: np.ndarray) -> float:
        """Bayesian Information Criterion: -2*ll + d*log(n)."""
        X = np.array(X, dtype=float)
        n, p = X.shape
        ll = self.score(X) * n
        d = self._n_parameters(p)
        return float(-2 * ll + d * np.log(n))

    def aic(self, X: np.ndarray) -> float:
        """Akaike Information Criterion: -2*ll + 2*d."""
        X = np.array(X, dtype=float)
        n, p = X.shape
        ll = self.score(X) * n
        d = self._n_parameters(p)
        return float(-2 * ll + 2 * d)
