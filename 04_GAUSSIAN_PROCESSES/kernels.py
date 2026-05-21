"""
Kernel Functions for Gaussian Processes

Implements various kernel functions (covariance functions) used in GP regression.

Author: ML Algorithms from Scratch
"""

import numpy as np
from typing import Optional


class Kernel:
    """Base class for kernel functions."""

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute kernel matrix between X1 and X2.

        Parameters
        ----------
        X1 : np.ndarray, shape (n1, d)
            First set of points.
        X2 : np.ndarray, shape (n2, d)
            Second set of points.

        Returns
        -------
        K : np.ndarray, shape (n1, n2)
            Kernel matrix.
        """
        raise NotImplementedError("Subclasses must implement __call__")

    def diag(self, X: np.ndarray) -> np.ndarray:
        """
        Compute diagonal of kernel matrix (variance at each point).

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Points.

        Returns
        -------
        diag_K : np.ndarray, shape (n,)
            Diagonal elements.
        """
        return np.diag(self(X, X))


class RBFKernel(Kernel):
    """
    Radial Basis Function (RBF) / Squared Exponential Kernel.

    Also known as Gaussian kernel:
    k(x, x') = σ² * exp(-||x - x'||² / (2 * ℓ²))

    Parameters
    ----------
    length_scale : float, default=1.0
        Length scale parameter ℓ. Controls smoothness.
    variance : float, default=1.0
        Signal variance σ². Controls vertical scale.
    """

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        self.length_scale = length_scale
        self.variance = variance

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute RBF kernel matrix."""
        # Efficient computation of squared distances
        sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                   np.sum(X2**2, axis=1) - \
                   2 * np.dot(X1, X2.T)

        return self.variance * np.exp(-sq_dists / (2 * self.length_scale**2))

    def __repr__(self):
        return f"RBFKernel(length_scale={self.length_scale}, variance={self.variance})"


class MaternKernel(Kernel):
    """
    Matern Kernel.

    More general than RBF, allows different smoothness levels.

    k(x, x') = σ² * (1 + √(2*ν) * r / ℓ) * exp(-√(2*ν) * r / ℓ)

    where r = ||x - x'||

    Parameters
    ----------
    length_scale : float, default=1.0
        Length scale parameter ℓ.
    variance : float, default=1.0
        Signal variance σ².
    nu : float, default=1.5
        Smoothness parameter. Common values: 0.5, 1.5, 2.5, ∞ (→ RBF).
    """

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0, nu: float = 1.5):
        self.length_scale = length_scale
        self.variance = variance
        self.nu = nu

        if nu not in [0.5, 1.5, 2.5]:
            raise ValueError("Only nu in {0.5, 1.5, 2.5} are supported")

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute Matern kernel matrix."""
        dists = np.sqrt(np.sum(X1**2, axis=1).reshape(-1, 1) +
                       np.sum(X2**2, axis=1) -
                       2 * np.dot(X1, X2.T) + 1e-10)

        if self.nu == 0.5:
            # Exponential kernel
            K = self.variance * np.exp(-dists / self.length_scale)

        elif self.nu == 1.5:
            scaled_dists = np.sqrt(3) * dists / self.length_scale
            K = self.variance * (1 + scaled_dists) * np.exp(-scaled_dists)

        elif self.nu == 2.5:
            scaled_dists = np.sqrt(5) * dists / self.length_scale
            K = self.variance * (1 + scaled_dists + scaled_dists**2 / 3) * np.exp(-scaled_dists)

        return K

    def __repr__(self):
        return f"MaternKernel(length_scale={self.length_scale}, variance={self.variance}, nu={self.nu})"


class RationalQuadraticKernel(Kernel):
    """
    Rational Quadratic Kernel.

    Scale mixture of RBF kernels with different length scales.

    k(x, x') = σ² * (1 + ||x - x'||² / (2 * α * ℓ²))^(-α)

    Parameters
    ----------
    length_scale : float, default=1.0
        Length scale parameter ℓ.
    variance : float, default=1.0
        Signal variance σ².
    alpha : float, default=1.0
        Scale mixture parameter. As α → ∞, becomes RBF.
    """

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0, alpha: float = 1.0):
        self.length_scale = length_scale
        self.variance = variance
        self.alpha = alpha

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute Rational Quadratic kernel matrix."""
        sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                   np.sum(X2**2, axis=1) - \
                   2 * np.dot(X1, X2.T)

        return self.variance * (1 + sq_dists / (2 * self.alpha * self.length_scale**2)) ** (-self.alpha)

    def __repr__(self):
        return f"RationalQuadraticKernel(length_scale={self.length_scale}, variance={self.variance}, alpha={self.alpha})"


class PeriodicKernel(Kernel):
    """
    Periodic Kernel.

    For periodic functions with period p.

    k(x, x') = σ² * exp(-2 * sin²(π * ||x - x'|| / p) / ℓ²)

    Parameters
    ----------
    length_scale : float, default=1.0
        Length scale parameter ℓ.
    variance : float, default=1.0
        Signal variance σ².
    period : float, default=1.0
        Period p of the function.
    """

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0, period: float = 1.0):
        self.length_scale = length_scale
        self.variance = variance
        self.period = period

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute Periodic kernel matrix."""
        dists = np.sqrt(np.sum(X1**2, axis=1).reshape(-1, 1) +
                       np.sum(X2**2, axis=1) -
                       2 * np.dot(X1, X2.T) + 1e-10)

        sin_term = np.sin(np.pi * dists / self.period)
        return self.variance * np.exp(-2 * sin_term**2 / self.length_scale**2)

    def __repr__(self):
        return f"PeriodicKernel(length_scale={self.length_scale}, variance={self.variance}, period={self.period})"


class LinearKernel(Kernel):
    """
    Linear Kernel.

    k(x, x') = σ² * (x - c)ᵀ (x' - c)

    Parameters
    ----------
    variance : float, default=1.0
        Signal variance σ².
    offset : float, default=0.0
        Offset c.
    """

    def __init__(self, variance: float = 1.0, offset: float = 0.0):
        self.variance = variance
        self.offset = offset

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute Linear kernel matrix."""
        X1_centered = X1 - self.offset
        X2_centered = X2 - self.offset
        return self.variance * np.dot(X1_centered, X2_centered.T)

    def __repr__(self):
        return f"LinearKernel(variance={self.variance}, offset={self.offset})"


class WhiteNoiseKernel(Kernel):
    """
    White Noise Kernel.

    k(x, x') = σ² * δ(x, x')

    where δ is the Kronecker delta (1 if x == x', 0 otherwise).

    Parameters
    ----------
    noise_level : float, default=1.0
        Noise variance σ².
    """

    def __init__(self, noise_level: float = 1.0):
        self.noise_level = noise_level

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute White Noise kernel matrix."""
        if X1.shape != X2.shape or not np.allclose(X1, X2):
            # Different inputs → zero covariance
            return np.zeros((X1.shape[0], X2.shape[0]))
        return self.noise_level * np.eye(X1.shape[0])

    def __repr__(self):
        return f"WhiteNoiseKernel(noise_level={self.noise_level})"


class KernelSum(Kernel):
    """
    Sum of two kernels: k(x, x') = k1(x, x') + k2(x, x')

    Parameters
    ----------
    kernel1 : Kernel
        First kernel.
    kernel2 : Kernel
        Second kernel.
    """

    def __init__(self, kernel1: Kernel, kernel2: Kernel):
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute sum of kernels."""
        return self.kernel1(X1, X2) + self.kernel2(X1, X2)

    def __repr__(self):
        return f"KernelSum({self.kernel1}, {self.kernel2})"


class KernelProduct(Kernel):
    """
    Product of two kernels: k(x, x') = k1(x, x') * k2(x, x')

    Parameters
    ----------
    kernel1 : Kernel
        First kernel.
    kernel2 : Kernel
        Second kernel.
    """

    def __init__(self, kernel1: Kernel, kernel2: Kernel):
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute product of kernels."""
        return self.kernel1(X1, X2) * self.kernel2(X1, X2)

    def __repr__(self):
        return f"KernelProduct({self.kernel1}, {self.kernel2})"


if __name__ == "__main__":
    # Example usage
    print("\n" + "=" * 70)
    print(" " * 25 + "KERNEL EXAMPLES")
    print("=" * 70)

    np.random.seed(42)
    X1 = np.random.randn(5, 2)
    X2 = np.random.randn(3, 2)

    # RBF Kernel
    print("\n1. RBF Kernel:")
    rbf = RBFKernel(length_scale=1.0, variance=1.0)
    K_rbf = rbf(X1, X2)
    print(f"   Shape: {K_rbf.shape}")
    print(f"   Sample values:\n{K_rbf[:2, :2]}")

    # Matern Kernel
    print("\n2. Matern Kernel (nu=1.5):")
    matern = MaternKernel(length_scale=1.0, nu=1.5)
    K_matern = matern(X1, X2)
    print(f"   Shape: {K_matern.shape}")
    print(f"   Sample values:\n{K_matern[:2, :2]}")

    # Periodic Kernel
    print("\n3. Periodic Kernel:")
    periodic = PeriodicKernel(period=2.0)
    K_periodic = periodic(X1, X2)
    print(f"   Shape: {K_periodic.shape}")

    # Kernel composition
    print("\n4. Kernel Sum (RBF + Linear):")
    linear = LinearKernel(variance=0.5)
    combined = KernelSum(rbf, linear)
    K_combined = combined(X1, X2)
    print(f"   Shape: {K_combined.shape}")

    print("\n" + "=" * 70)
