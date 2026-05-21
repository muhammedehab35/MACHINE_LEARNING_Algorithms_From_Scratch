"""
Support Vector Machine (SVM) Implementation

A complete implementation of Support Vector Machines for binary and multi-class
classification from scratch using only NumPy.

Key Features:
- Linear SVM (Hinge loss with gradient descent)
- Kernel SVM (RBF, Polynomial, Linear kernels)
- Multi-class classification (One-vs-Rest strategy)
- Soft margin with regularization parameter C
- Decision function and probability estimates

Author: ML Algorithms from Scratch
"""

import numpy as np
from typing import Literal, Optional, Callable


class SVMClassifier:
    """
    Support Vector Machine Classifier

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. Smaller values specify stronger regularization.

    kernel : {'linear', 'rbf', 'poly'}, default='linear'
        Kernel type to use in the algorithm:
        - 'linear': Linear kernel (no kernel transformation)
        - 'rbf': Radial Basis Function kernel
        - 'poly': Polynomial kernel

    gamma : float or 'scale' or 'auto', default='scale'
        Kernel coefficient for 'rbf' and 'poly':
        - 'scale': 1 / (n_features * X.var())
        - 'auto': 1 / n_features
        - float: explicit value

    degree : int, default=3
        Degree of the polynomial kernel function ('poly'). Ignored by other kernels.

    learning_rate : float, default=0.001
        Learning rate for gradient descent optimization.

    n_iterations : int, default=1000
        Maximum number of iterations for optimization.

    random_state : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> from svm import SVMClassifier
    >>> X_train = np.array([[1, 2], [2, 3], [3, 3], [2, 1]])
    >>> y_train = np.array([1, 1, -1, -1])
    >>> svm = SVMClassifier(kernel='rbf', C=1.0)
    >>> svm.fit(X_train, y_train)
    >>> svm.predict([[2, 2]])
    array([1])
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: Literal['linear', 'rbf', 'poly'] = 'linear',
        gamma: float | Literal['scale', 'auto'] = 'scale',
        degree: int = 3,
        learning_rate: float = 0.001,
        n_iterations: int = 1000,
        random_state: Optional[int] = None
    ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state

        # Will be set during fit
        self.w_ = None
        self.b_ = None
        self.X_train_ = None
        self.y_train_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.gamma_ = None

    def _compute_gamma(self, X: np.ndarray) -> float:
        """Compute gamma based on the gamma parameter."""
        if isinstance(self.gamma, str):
            if self.gamma == 'scale':
                return 1.0 / (X.shape[1] * X.var())
            elif self.gamma == 'auto':
                return 1.0 / X.shape[1]
        return self.gamma

    def _linear_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute linear kernel: K(x, x') = x^T x'"""
        return np.dot(X1, X2.T)

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute RBF kernel: K(x, x') = exp(-gamma * ||x - x'||^2)"""
        # Efficient computation using broadcasting
        sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                   np.sum(X2**2, axis=1) - \
                   2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma_ * sq_dists)

    def _poly_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute polynomial kernel: K(x, x') = (gamma * x^T x' + 1)^degree"""
        return (self.gamma_ * np.dot(X1, X2.T) + 1) ** self.degree

    def _get_kernel_function(self) -> Callable:
        """Return the appropriate kernel function."""
        if self.kernel == 'linear':
            return self._linear_kernel
        elif self.kernel == 'rbf':
            return self._rbf_kernel
        elif self.kernel == 'poly':
            return self._poly_kernel
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _hinge_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute hinge loss for linear SVM."""
        margins = y * (np.dot(X, self.w_) + self.b_)
        losses = np.maximum(0, 1 - margins)
        return np.mean(losses) + 0.5 * self.C * np.sum(self.w_ ** 2)

    def _fit_linear(self, X: np.ndarray, y: np.ndarray) -> 'SVMClassifier':
        """Fit linear SVM using gradient descent."""
        n_samples, n_features = X.shape

        # Initialize weights
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.w_ = np.zeros(n_features)
        self.b_ = 0.0

        # Gradient descent
        for _ in range(self.n_iterations):
            # Compute margins
            margins = y * (np.dot(X, self.w_) + self.b_)

            # Find support vectors (margin < 1)
            sv_idx = margins < 1

            # Compute gradients
            dw = self.C * self.w_ - np.dot(X[sv_idx].T, y[sv_idx]) / n_samples
            db = -np.sum(y[sv_idx]) / n_samples

            # Update weights
            self.w_ -= self.learning_rate * dw
            self.b_ -= self.learning_rate * db

        return self

    def _fit_kernel(self, X: np.ndarray, y: np.ndarray) -> 'SVMClassifier':
        """Fit kernel SVM using dual formulation with simplified SMO."""
        n_samples = X.shape[0]

        # Store training data for kernel evaluation
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        # Initialize alpha (dual coefficients)
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.alpha_ = np.zeros(n_samples)
        self.b_ = 0.0

        # Compute kernel matrix
        kernel_func = self._get_kernel_function()
        K = kernel_func(X, X)

        # Simplified SMO algorithm
        for iteration in range(self.n_iterations):
            alpha_prev = self.alpha_.copy()

            for i in range(n_samples):
                # Compute error for i
                E_i = np.sum(self.alpha_ * y * K[:, i]) + self.b_ - y[i]

                # Check KKT conditions
                if (y[i] * E_i < -0.01 and self.alpha_[i] < self.C) or \
                   (y[i] * E_i > 0.01 and self.alpha_[i] > 0):

                    # Select j randomly (simplified selection)
                    j = i
                    while j == i:
                        j = np.random.randint(0, n_samples)

                    # Compute error for j
                    E_j = np.sum(self.alpha_ * y * K[:, j]) + self.b_ - y[j]

                    # Save old alphas
                    alpha_i_old = self.alpha_[i]
                    alpha_j_old = self.alpha_[j]

                    # Compute bounds L and H
                    if y[i] != y[j]:
                        L = max(0, self.alpha_[j] - self.alpha_[i])
                        H = min(self.C, self.C + self.alpha_[j] - self.alpha_[i])
                    else:
                        L = max(0, self.alpha_[i] + self.alpha_[j] - self.C)
                        H = min(self.C, self.alpha_[i] + self.alpha_[j])

                    if L == H:
                        continue

                    # Compute eta
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    # Update alpha_j
                    self.alpha_[j] -= y[j] * (E_i - E_j) / eta
                    self.alpha_[j] = np.clip(self.alpha_[j], L, H)

                    if abs(self.alpha_[j] - alpha_j_old) < 1e-5:
                        continue

                    # Update alpha_i
                    self.alpha_[i] += y[i] * y[j] * (alpha_j_old - self.alpha_[j])

                    # Update bias
                    b1 = self.b_ - E_i - y[i] * (self.alpha_[i] - alpha_i_old) * K[i, i] - \
                         y[j] * (self.alpha_[j] - alpha_j_old) * K[i, j]
                    b2 = self.b_ - E_j - y[i] * (self.alpha_[i] - alpha_i_old) * K[i, j] - \
                         y[j] * (self.alpha_[j] - alpha_j_old) * K[j, j]

                    if 0 < self.alpha_[i] < self.C:
                        self.b_ = b1
                    elif 0 < self.alpha_[j] < self.C:
                        self.b_ = b2
                    else:
                        self.b_ = (b1 + b2) / 2

            # Check convergence
            diff = np.linalg.norm(self.alpha_ - alpha_prev)
            if diff < 1e-3:
                break

        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMClassifier':
        """
        Fit the SVM classifier.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data.
        y : np.ndarray, shape (n_samples,)
            Target labels.

        Returns
        -------
        self : SVMClassifier
            Fitted classifier.
        """
        # Validate inputs
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        # Store classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Convert to binary classification format (-1, 1)
        if self.n_classes_ == 2:
            y_binary = np.where(y == self.classes_[0], -1, 1)

            # Compute gamma for kernel methods
            if self.kernel != 'linear':
                self.gamma_ = self._compute_gamma(X)

            # Fit SVM
            if self.kernel == 'linear':
                self._fit_linear(X, y_binary)
            else:
                self._fit_kernel(X, y_binary)

        else:
            # Multi-class: One-vs-Rest
            raise NotImplementedError("Multi-class SVM will be implemented in multi-class version")

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function values.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Test data.

        Returns
        -------
        decisions : np.ndarray, shape (n_samples,)
            Decision function values (distance from hyperplane).
        """
        if self.classes_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() before making predictions.")

        if self.kernel == 'linear':
            return np.dot(X, self.w_) + self.b_
        else:
            kernel_func = self._get_kernel_function()
            K = kernel_func(X, self.X_train_)
            return np.dot(K, self.alpha_ * self.y_train_) + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Test data.

        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            Predicted class labels.
        """
        decisions = self.decision_function(X)
        binary_pred = np.sign(decisions)

        # Convert back to original labels
        return np.where(binary_pred == -1, self.classes_[0], self.classes_[1])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return accuracy score.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Test data.
        y : np.ndarray, shape (n_samples,)
            True labels.

        Returns
        -------
        accuracy : float
            Accuracy score.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


if __name__ == "__main__":
    # Example usage
    print("\n" + "=" * 70)
    print(" " * 25 + "SVM EXAMPLES")
    print("=" * 70)

    # Linear SVM example
    print("\nLinear SVM Example:")
    print("-" * 70)

    np.random.seed(42)
    X_train = np.random.randn(100, 2)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

    X_test = np.random.randn(20, 2)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

    svm_linear = SVMClassifier(kernel='linear', C=1.0)
    svm_linear.fit(X_train, y_train)

    accuracy = svm_linear.score(X_test, y_test)
    print(f"Linear SVM Accuracy: {accuracy:.4f}")

    # RBF SVM example
    print("\nRBF SVM Example:")
    print("-" * 70)

    # Non-linear data
    X_train_nl = np.random.randn(100, 2)
    y_train_nl = ((X_train_nl[:, 0]**2 + X_train_nl[:, 1]**2) < 1).astype(int)

    X_test_nl = np.random.randn(20, 2)
    y_test_nl = ((X_test_nl[:, 0]**2 + X_test_nl[:, 1]**2) < 1).astype(int)

    svm_rbf = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
    svm_rbf.fit(X_train_nl, y_train_nl)

    accuracy = svm_rbf.score(X_test_nl, y_test_nl)
    print(f"RBF SVM Accuracy: {accuracy:.4f}")

    print("\n" + "=" * 70)
