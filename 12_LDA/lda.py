"""
Linear Discriminant Analysis (LDA) - From Scratch Implementation

A linear classification and dimensionality reduction technique that finds
the linear combination of features that best separates classes.

Key Concepts:
- Supervised dimensionality reduction
- Maximizes between-class variance while minimizing within-class variance
- Fisher's Linear Discriminant
- Optimal for multi-class classification
- Projects data to a lower-dimensional space
"""

import numpy as np
from typing import Optional


class LinearDiscriminantAnalysis:
    """
    Linear Discriminant Analysis (LDA)

    LDA is a supervised technique that finds the projection that maximizes
    class separation. It assumes that features are normally distributed
    with equal covariance matrices across classes.

    Mathematical Foundation:
    - Maximize: J(W) = |W^T S_B W| / |W^T S_W W|
    - S_B: Between-class scatter matrix
    - S_W: Within-class scatter matrix
    - W: Projection matrix (discriminant vectors)

    Parameters
    ----------
    n_components : int, default=None
        Number of components to keep. If None, min(n_classes - 1, n_features)

    solver : {'svd', 'eigen'}, default='svd'
        Solver to use:
        - 'svd': Singular Value Decomposition (more stable)
        - 'eigen': Eigenvalue decomposition (faster for small datasets)

    shrinkage : float, default=None
        Shrinkage parameter for regularization (0 to 1)
        If None, no shrinkage is applied

    store_covariance : bool, default=False
        Whether to store the within-class covariance matrix

    tol : float, default=1e-4
        Threshold for eigenvalue significance
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        solver: str = 'svd',
        shrinkage: Optional[float] = None,
        store_covariance: bool = False,
        tol: float = 1e-4
    ):
        self.n_components = n_components
        self.solver = solver
        self.shrinkage = shrinkage
        self.store_covariance = store_covariance
        self.tol = tol

        # Fitted attributes
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.covariance_ = None
        self.scalings_ = None  # Projection matrix (discriminant vectors)
        self.explained_variance_ratio_ = None
        self.n_features_in_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Linear Discriminant Analysis model

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data

        y : array-like of shape (n_samples,)
            Target values (class labels)

        Returns
        -------
        self : object
            Fitted estimator
        """
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Get unique classes and their counts
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("LDA requires at least 2 classes")

        # Determine number of components
        if self.n_components is None:
            self.n_components = min(n_classes - 1, n_features)
        else:
            self.n_components = min(self.n_components, n_classes - 1, n_features)

        # Compute class priors (proportion of each class)
        self.priors_ = np.array([np.sum(y == c) / n_samples for c in self.classes_])

        # Compute class means
        self.means_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])

        # Compute overall mean
        overall_mean = X.mean(axis=0)

        # Compute within-class scatter matrix S_W
        S_W = np.zeros((n_features, n_features))
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            # Center the class samples
            X_c_centered = X_c - self.means_[idx]
            # Add to within-class scatter
            S_W += X_c_centered.T @ X_c_centered

        # Apply shrinkage if specified (regularization)
        if self.shrinkage is not None:
            # Ledoit-Wolf shrinkage
            shrinkage = np.clip(self.shrinkage, 0, 1)
            S_W = (1 - shrinkage) * S_W + shrinkage * np.trace(S_W) / n_features * np.eye(n_features)

        # Store covariance if requested
        if self.store_covariance:
            self.covariance_ = S_W / (n_samples - n_classes)

        # Compute between-class scatter matrix S_B
        S_B = np.zeros((n_features, n_features))
        for idx, c in enumerate(self.classes_):
            n_c = np.sum(y == c)
            mean_diff = (self.means_[idx] - overall_mean).reshape(-1, 1)
            S_B += n_c * (mean_diff @ mean_diff.T)

        # Solve the generalized eigenvalue problem: S_B w = λ S_W w
        # This is equivalent to: S_W^(-1) S_B w = λ w

        if self.solver == 'svd':
            # SVD approach (more numerically stable)
            # S_W = U Σ V^T
            U, s, Vt = np.linalg.svd(S_W)

            # Compute S_W^(-1/2) for whitening
            # Only use significant singular values
            rank = np.sum(s > self.tol)
            s_inv_sqrt = np.zeros_like(s)
            s_inv_sqrt[:rank] = 1.0 / np.sqrt(s[:rank])

            # S_W^(-1/2) = U Σ^(-1/2) U^T
            S_W_inv_sqrt = U @ np.diag(s_inv_sqrt) @ U.T

            # Transform S_B to whitened space
            S_B_white = S_W_inv_sqrt @ S_B @ S_W_inv_sqrt

            # Eigendecomposition of whitened between-class scatter
            eigenvalues, eigenvectors = np.linalg.eigh(S_B_white)

            # Transform eigenvectors back to original space
            eigenvectors = S_W_inv_sqrt @ eigenvectors

        else:  # solver == 'eigen'
            # Eigenvalue decomposition approach
            try:
                # Add small regularization for numerical stability
                S_W_reg = S_W + 1e-6 * np.eye(n_features)
                S_W_inv = np.linalg.inv(S_W_reg)

                # Solve eigenvalue problem
                eigenvalues, eigenvectors = np.linalg.eig(S_W_inv @ S_B)
                eigenvalues = eigenvalues.real
                eigenvectors = eigenvectors.real

            except np.linalg.LinAlgError:
                raise ValueError(
                    "Singular within-class scatter matrix. "
                    "Try solver='svd' or add shrinkage."
                )

        # Sort eigenvectors by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Keep only the top n_components discriminant vectors
        self.scalings_ = eigenvectors[:, :self.n_components]

        # Compute explained variance ratio
        eigenvalues_sum = np.sum(eigenvalues[eigenvalues > 0])
        if eigenvalues_sum > 0:
            self.explained_variance_ratio_ = eigenvalues[:self.n_components] / eigenvalues_sum
        else:
            self.explained_variance_ratio_ = np.zeros(self.n_components)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data to the discriminant space

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform

        Returns
        -------
        X_new : array of shape (n_samples, n_components)
            Transformed data in discriminant space
        """
        if self.scalings_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.array(X)

        # Project data: X_new = X @ W
        return X @ self.scalings_

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit the model and transform the data

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data

        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        X_new : array of shape (n_samples, n_components)
            Transformed data
        """
        return self.fit(X, y).transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples

        Uses Bayes' rule with multivariate Gaussian assumption:
        P(y=k|x) ∝ P(x|y=k) * P(y=k)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        if self.means_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.array(X)

        # Compute discriminant scores for each class
        # Using simplified form assuming equal covariances
        scores = np.zeros((len(X), len(self.classes_)))

        for idx, c in enumerate(self.classes_):
            # Distance to class mean (Mahalanobis-like)
            # d^2 = (x - μ_k)^T Σ^(-1) (x - μ_k)
            diff = X - self.means_[idx]

            # Simplified: use Euclidean distance + prior
            # (Full implementation would use shared covariance matrix)
            scores[:, idx] = -0.5 * np.sum(diff ** 2, axis=1) + np.log(self.priors_[idx])

        # Predict class with highest score
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate probability of each class

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples

        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities
        """
        if self.means_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.array(X)

        # Compute scores (log-likelihoods)
        scores = np.zeros((len(X), len(self.classes_)))

        for idx, c in enumerate(self.classes_):
            diff = X - self.means_[idx]
            scores[:, idx] = -0.5 * np.sum(diff ** 2, axis=1) + np.log(self.priors_[idx])

        # Convert to probabilities using softmax
        # Subtract max for numerical stability
        scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        proba = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)

        return proba

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples

        y : array-like of shape (n_samples,)
            True labels

        Returns
        -------
        score : float
            Accuracy score
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function (discriminant scores)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples

        Returns
        -------
        scores : array of shape (n_samples, n_classes)
            Decision function values
        """
        if self.means_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.array(X)
        scores = np.zeros((len(X), len(self.classes_)))

        for idx, c in enumerate(self.classes_):
            diff = X - self.means_[idx]
            scores[:, idx] = -0.5 * np.sum(diff ** 2, axis=1) + np.log(self.priors_[idx])

        return scores
