"""
Gaussian Process Regression

Complete implementation of Gaussian Processes for regression from scratch.

Key Features:
- Multiple kernel functions (RBF, Matern, Periodic, etc.)
- Mean and covariance predictions
- Uncertainty quantification
- Hyperparameter optimization
- Log marginal likelihood

Author: ML Algorithms from Scratch
"""

import numpy as np
from typing import Optional, Tuple
from kernels import Kernel, RBFKernel
import warnings


class GaussianProcessRegressor:
    """
    Gaussian Process Regressor.

    Parameters
    ----------
    kernel : Kernel, default=None
        Covariance function. If None, uses RBF kernel.
    alpha : float, default=1e-10
        Noise level (observation noise variance).
    normalize_y : bool, default=False
        Whether to normalize target values.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    X_train_ : np.ndarray
        Training features.
    y_train_ : np.ndarray
        Training targets.
    K_ : np.ndarray
        Kernel matrix on training data.
    K_inv_ : np.ndarray
        Inverse of (K + αI).
    alpha_ : np.ndarray
        Weights in representer theorem: α = (K + αI)^(-1) y

    Examples
    --------
    >>> from gp import GaussianProcessRegressor
    >>> from kernels import RBFKernel
    >>> X_train = np.array([[1], [3], [5], [6], [8]])
    >>> y_train = np.sin(X_train).ravel()
    >>> kernel = RBFKernel(length_scale=1.0)
    >>> gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
    >>> gp.fit(X_train, y_train)
    >>> X_test = np.array([[2], [4], [7]])
    >>> y_mean, y_std = gp.predict(X_test, return_std=True)
    """

    def __init__(
        self,
        kernel: Optional[Kernel] = None,
        alpha: float = 1e-10,
        normalize_y: bool = False,
        random_state: Optional[int] = None
    ):
        self.kernel = kernel if kernel is not None else RBFKernel()
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.random_state = random_state

        # Will be set during fit
        self.X_train_ = None
        self.y_train_ = None
        self.K_ = None
        self.K_inv_ = None
        self.alpha_ = None
        self.y_mean_ = None
        self.y_std_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianProcessRegressor':
        """
        Fit Gaussian Process model.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training features.
        y : np.ndarray, shape (n_samples,)
            Training targets.

        Returns
        -------
        self : GaussianProcessRegressor
            Fitted model.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Store training data
        self.X_train_ = np.array(X)
        self.y_train_ = np.array(y).ravel()

        if len(self.y_train_) != len(self.X_train_):
            raise ValueError("X and y must have the same number of samples")

        # Normalize y if requested
        if self.normalize_y:
            self.y_mean_ = np.mean(self.y_train_)
            self.y_std_ = np.std(self.y_train_)
            if self.y_std_ == 0:
                self.y_std_ = 1.0
            y_normalized = (self.y_train_ - self.y_mean_) / self.y_std_
        else:
            self.y_mean_ = 0.0
            self.y_std_ = 1.0
            y_normalized = self.y_train_

        # Compute kernel matrix
        self.K_ = self.kernel(self.X_train_, self.X_train_)

        # Add noise to diagonal (for numerical stability and observation noise)
        K_with_noise = self.K_ + self.alpha * np.eye(len(self.X_train_))

        # Compute inverse using Cholesky decomposition (more stable)
        try:
            L = np.linalg.cholesky(K_with_noise)
            self.L_ = L
            # Solve L * L.T * alpha = y
            self.alpha_ = np.linalg.solve(L.T, np.linalg.solve(L, y_normalized))
        except np.linalg.LinAlgError:
            # Fallback to regular inverse if Cholesky fails
            warnings.warn("Cholesky decomposition failed, using standard inverse")
            self.K_inv_ = np.linalg.inv(K_with_noise)
            self.alpha_ = np.dot(self.K_inv_, y_normalized)
            self.L_ = None

        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
        return_cov: bool = False
    ) -> Tuple[np.ndarray, ...]:
        """
        Predict using Gaussian Process.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Test features.
        return_std : bool, default=False
            If True, return standard deviation.
        return_cov : bool, default=False
            If True, return full covariance matrix.

        Returns
        -------
        y_mean : np.ndarray, shape (n_samples,)
            Predicted mean.
        y_std : np.ndarray, shape (n_samples,), optional
            Standard deviation (if return_std=True).
        y_cov : np.ndarray, shape (n_samples, n_samples), optional
            Covariance matrix (if return_cov=True).
        """
        if self.X_train_ is None:
            raise ValueError("Model must be fitted before prediction")

        X_test = np.array(X)

        # Compute kernel between test and training points
        K_test_train = self.kernel(X_test, self.X_train_)

        # Predictive mean: K(X*, X) @ alpha
        y_mean = np.dot(K_test_train, self.alpha_)

        # Denormalize
        y_mean = self.y_std_ * y_mean + self.y_mean_

        if not (return_std or return_cov):
            return y_mean

        # Compute predictive variance/covariance
        K_test_test = self.kernel(X_test, X_test)

        if self.L_ is not None:
            # Use Cholesky factor
            v = np.linalg.solve(self.L_, K_test_train.T)
            y_cov = K_test_test - np.dot(v.T, v)
        else:
            # Use inverse
            y_cov = K_test_test - np.dot(K_test_train, np.dot(self.K_inv_, K_test_train.T))

        # Scale variance by y_std_
        y_cov = self.y_std_**2 * y_cov

        # Ensure numerical stability (variance should be non-negative)
        y_var = np.diag(y_cov)
        y_var = np.maximum(y_var, 0)

        if return_std and return_cov:
            return y_mean, np.sqrt(y_var), y_cov
        elif return_std:
            return y_mean, np.sqrt(y_var)
        else:  # return_cov
            return y_mean, y_cov

    def sample_y(self, X: np.ndarray, n_samples: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """
        Sample from the posterior distribution.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Test points.
        n_samples : int, default=1
            Number of samples to draw.
        random_state : int, optional
            Random seed.

        Returns
        -------
        y_samples : np.ndarray, shape (n_samples, n_test_points)
            Samples from posterior.
        """
        if random_state is not None:
            np.random.seed(random_state)

        y_mean, y_cov = self.predict(X, return_cov=True)

        # Add small diagonal for numerical stability
        y_cov = y_cov + 1e-10 * np.eye(y_cov.shape[0])

        # Sample from multivariate normal
        samples = np.random.multivariate_normal(y_mean, y_cov, size=n_samples)

        return samples

    def log_marginal_likelihood(self) -> float:
        """
        Compute log marginal likelihood of the training data.

        log p(y | X) = -0.5 * y^T * (K + αI)^(-1) * y
                       - 0.5 * log|K + αI|
                       - 0.5 * n * log(2π)

        Returns
        -------
        lml : float
            Log marginal likelihood.
        """
        if self.X_train_ is None:
            raise ValueError("Model must be fitted first")

        # Normalize y
        if self.normalize_y:
            y = (self.y_train_ - self.y_mean_) / self.y_std_
        else:
            y = self.y_train_

        n = len(y)

        if self.L_ is not None:
            # Using Cholesky decomposition
            # log|K + αI| = 2 * sum(log(diag(L)))
            log_det = 2 * np.sum(np.log(np.diag(self.L_)))

            # y^T * (K + αI)^(-1) * y = alpha^T * y
            y_K_inv_y = np.dot(y, self.alpha_)
        else:
            # Using direct inverse
            K_with_noise = self.K_ + self.alpha * np.eye(n)
            log_det = np.linalg.slogdet(K_with_noise)[1]
            y_K_inv_y = np.dot(y, np.dot(self.K_inv_, y))

        lml = -0.5 * y_K_inv_y - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)

        return lml

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return R² score.

        Parameters
        ----------
        X : np.ndarray
            Test features.
        y : np.ndarray
            True targets.

        Returns
        -------
        score : float
            R² score.
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return 1 - (ss_res / ss_tot)


if __name__ == "__main__":
    # Example usage
    print("\n" + "=" * 70)
    print(" " * 20 + "GAUSSIAN PROCESS EXAMPLES")
    print("=" * 70)

    np.random.seed(42)

    # Generate synthetic data
    X_train = np.array([1, 3, 5, 6, 8]).reshape(-1, 1)
    y_train = np.sin(X_train).ravel() + 0.1 * np.random.randn(5)

    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    y_true = np.sin(X_test).ravel()

    # Fit GP
    print("\n1. Fitting Gaussian Process with RBF kernel...")
    from kernels import RBFKernel

    kernel = RBFKernel(length_scale=1.0, variance=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.01)
    gp.fit(X_train, y_train)

    # Predict
    y_mean, y_std = gp.predict(X_test, return_std=True)

    print(f"   Training R² score: {gp.score(X_train, y_train):.4f}")
    print(f"   Log marginal likelihood: {gp.log_marginal_likelihood():.4f}")

    # Sample from posterior
    print("\n2. Sampling from posterior distribution...")
    samples = gp.sample_y(X_test, n_samples=3, random_state=42)
    print(f"   Generated {samples.shape[0]} samples of shape {samples.shape[1]}")

    print("\n" + "=" * 70)
