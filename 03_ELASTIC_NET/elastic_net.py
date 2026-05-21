"""
Elastic Net Regression Implementation from Scratch

This module implements Elastic Net regression, which combines L1 (Lasso) and L2 (Ridge)
regularization to leverage the benefits of both approaches.

Mathematical formulation:
    Loss = MSE + λ₁||w||₁ + λ₂||w||₂²
    where λ₁ controls L1 penalty and λ₂ controls L2 penalty

Author: ML Algorithms from Scratch
"""

import numpy as np
from typing import Optional, Literal, Tuple


class ElasticNet:
    """
    Elastic Net Regression with combined L1 and L2 regularization.

    Elastic Net combines the penalties of Lasso (L1) and Ridge (L2) regression:
    - L1 penalty: Feature selection (sparsity)
    - L2 penalty: Coefficient shrinkage (stability)

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for gradient descent optimization.

    n_iterations : int, default=1000
        Maximum number of iterations for gradient descent.

    l1_ratio : float, default=0.5
        The mixing parameter between L1 and L2 regularization.
        - l1_ratio = 0: Pure Ridge regression (L2 only)
        - l1_ratio = 1: Pure Lasso regression (L1 only)
        - 0 < l1_ratio < 1: Elastic Net (combination)

    alpha : float, default=1.0
        Overall regularization strength.
        The penalties are computed as:
        - L1 penalty = alpha * l1_ratio * ||w||₁
        - L2 penalty = alpha * (1 - l1_ratio) * ||w||₂²

    batch_size : int or None, default=None
        Batch size for mini-batch gradient descent.
        - None: Batch gradient descent (use all samples)
        - int: Mini-batch gradient descent
        - 1: Stochastic gradient descent (SGD)

    early_stopping : bool, default=True
        Whether to stop training when validation loss stops improving.

    validation_fraction : float, default=0.1
        Fraction of training data to use for validation (early stopping).

    patience : int, default=10
        Number of iterations with no improvement to wait before stopping.

    tol : float, default=1e-4
        Tolerance for improvement in validation loss.

    decay_rate : float or None, default=None
        Learning rate decay factor. If not None, learning rate is multiplied
        by decay_rate every 100 iterations.

    random_state : int or None, default=None
        Random seed for reproducibility.

    verbose : bool, default=False
        Whether to print progress during training.

    Attributes
    ----------
    weights_ : np.ndarray of shape (n_features,)
        Coefficients of the model.

    bias_ : float
        Intercept term.

    loss_history_ : list
        Training loss at each iteration.

    val_loss_history_ : list
        Validation loss at each iteration (if early_stopping=True).

    n_iter_ : int
        Actual number of iterations performed.

    l1_penalty_ : float
        L1 regularization strength (alpha * l1_ratio).

    l2_penalty_ : float
        L2 regularization strength (alpha * (1 - l1_ratio)).
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        l1_ratio: float = 0.5,
        alpha: float = 1.0,
        batch_size: Optional[int] = None,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        patience: int = 10,
        tol: float = 1e-4,
        decay_rate: Optional[float] = None,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        # Validate parameters
        if not 0 <= l1_ratio <= 1:
            raise ValueError(f"l1_ratio must be between 0 and 1, got {l1_ratio}")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if not 0 < validation_fraction < 1:
            raise ValueError(f"validation_fraction must be between 0 and 1, got {validation_fraction}")

        # Hyperparameters
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.tol = tol
        self.decay_rate = decay_rate
        self.random_state = random_state
        self.verbose = verbose

        # Computed regularization strengths
        self.l1_penalty_ = alpha * l1_ratio
        self.l2_penalty_ = alpha * (1 - l1_ratio)

        # Model parameters (learned)
        self.weights_ = None
        self.bias_ = None

        # Training history
        self.loss_history_ = []
        self.val_loss_history_ = []
        self.n_iter_ = 0

        # Random state
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ElasticNet':
        """
        Fit Elastic Net model using gradient descent with proximal operator for L1.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : ElasticNet
            Fitted estimator.
        """
        # Input validation
        X, y = self._validate_input(X, y)
        n_samples, n_features = X.shape

        # Split data for early stopping
        if self.early_stopping:
            X_train, X_val, y_train, y_val = self._train_val_split(X, y)
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        # Initialize parameters
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0

        # Training loop
        current_lr = self.learning_rate
        best_val_loss = np.inf
        patience_counter = 0

        for iteration in range(self.n_iterations):
            # Learning rate decay
            if self.decay_rate is not None and iteration > 0 and iteration % 100 == 0:
                current_lr *= self.decay_rate

            # Get batch
            X_batch, y_batch = self._get_batch(X_train, y_train)

            # Forward pass
            y_pred = self._compute_predictions(X_batch)

            # Compute gradients (only from MSE and L2 terms)
            dw, db = self._compute_gradients(X_batch, y_batch, y_pred)

            # Update parameters with L2 regularization
            self.weights_ = (1 - current_lr * self.l2_penalty_) * self.weights_ - current_lr * dw
            self.bias_ -= current_lr * db

            # Apply soft-thresholding (proximal operator) for L1 regularization
            self.weights_ = self._soft_threshold(self.weights_, current_lr * self.l1_penalty_)

            # Clip weights to prevent numerical overflow
            self.weights_ = np.clip(self.weights_, -1e8, 1e8)
            self.bias_ = np.clip(self.bias_, -1e8, 1e8)

            # Compute losses
            train_loss = self._compute_loss(X_train, y_train)
            self.loss_history_.append(train_loss)

            if self.early_stopping and X_val is not None:
                val_loss = self._compute_loss(X_val, y_val)
                self.val_loss_history_.append(val_loss)

                # Check for improvement
                if val_loss < best_val_loss - self.tol:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at iteration {iteration + 1}")
                    break

            # Verbose output
            if self.verbose and (iteration + 1) % 100 == 0:
                if self.early_stopping and X_val is not None:
                    print(f"Iteration {iteration + 1}/{self.n_iterations} - "
                          f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    print(f"Iteration {iteration + 1}/{self.n_iterations} - "
                          f"Train Loss: {train_loss:.6f}")

        self.n_iter_ = iteration + 1
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values.
        """
        if self.weights_ is None:
            raise ValueError("Model must be fitted before making predictions")

        X = self._validate_input(X, None, fit=False)
        return self._compute_predictions(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score on the given data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        y : np.ndarray of shape (n_samples,)
            True target values.

        Returns
        -------
        r2 : float
            R² score (coefficient of determination).
        """
        y_pred = self.predict(X)

        # R² = 1 - (SS_res / SS_tot)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (ss_res / ss_tot)

    def get_nonzero_features(self) -> np.ndarray:
        """
        Get indices of features with non-zero coefficients (selected by L1).

        Returns
        -------
        indices : np.ndarray
            Indices of selected features.
        """
        if self.weights_ is None:
            raise ValueError("Model must be fitted first")

        return np.where(np.abs(self.weights_) > 1e-10)[0]

    def get_sparsity(self) -> float:
        """
        Compute the sparsity level (fraction of zero coefficients).

        Returns
        -------
        sparsity : float
            Fraction of coefficients that are zero.
        """
        if self.weights_ is None:
            raise ValueError("Model must be fitted first")

        n_zero = np.sum(np.abs(self.weights_) <= 1e-10)
        return n_zero / len(self.weights_)

    # Private methods

    def _validate_input(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        fit: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Validate and preprocess input data."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if fit and y is not None:
            if not isinstance(y, np.ndarray):
                y = np.array(y)

            if y.ndim != 1:
                raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")

            if X.shape[0] != y.shape[0]:
                raise ValueError(f"X and y must have same number of samples: "
                               f"{X.shape[0]} vs {y.shape[0]}")

        return (X, y) if fit else X

    def _train_val_split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and validation sets."""
        n_samples = X.shape[0]
        n_val = int(n_samples * self.validation_fraction)

        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        return X[train_indices], X[val_indices], y[train_indices], y[val_indices]

    def _get_batch(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get a batch of data for mini-batch gradient descent."""
        if self.batch_size is None:
            return X, y

        n_samples = X.shape[0]
        batch_size = min(self.batch_size, n_samples)

        indices = np.random.choice(n_samples, batch_size, replace=False)
        return X[indices], y[indices]

    def _compute_predictions(self, X: np.ndarray) -> np.ndarray:
        """Compute predictions."""
        predictions = np.dot(X, self.weights_) + self.bias_
        return np.clip(predictions, -1e10, 1e10)

    def _compute_gradients(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute gradients for MSE + L2 term.
        L1 term is handled separately via soft-thresholding.
        """
        n_samples = X.shape[0]

        # Error
        error = np.clip(y_pred - y_true, -1e10, 1e10)

        # Gradients from MSE only (L2 is in update rule, L1 is in soft-threshold)
        dw = (1 / n_samples) * np.dot(X.T, error)
        db = (1 / n_samples) * np.sum(error)

        # Clip gradients
        dw = np.clip(dw, -1e5, 1e5)
        db = np.clip(db, -1e5, 1e5)

        return dw, db

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute total loss (MSE + L1 + L2 penalties)."""
        y_pred = self._compute_predictions(X)

        # MSE term
        mse = np.mean((y - y_pred) ** 2)

        # L1 penalty
        l1_term = self.l1_penalty_ * np.sum(np.abs(self.weights_))

        # L2 penalty
        l2_term = 0.5 * self.l2_penalty_ * np.sum(self.weights_ ** 2)

        return mse + l1_term + l2_term

    def _soft_threshold(self, w: np.ndarray, threshold: float) -> np.ndarray:
        """
        Soft-thresholding operator (proximal operator for L1 norm).

        This induces sparsity by setting small coefficients to exactly zero.

        Formula: sign(w) * max(|w| - threshold, 0)
        """
        return np.sign(w) * np.maximum(np.abs(w) - threshold, 0)

    def __repr__(self) -> str:
        return (f"ElasticNet(learning_rate={self.learning_rate}, "
                f"n_iterations={self.n_iterations}, "
                f"l1_ratio={self.l1_ratio}, alpha={self.alpha})")


if __name__ == "__main__":
    # Quick test
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Generate data with some irrelevant features
    X, y = make_regression(
        n_samples=200,
        n_features=20,
        n_informative=10,  # Only 10 features are useful
        noise=10,
        random_state=42
    )

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Elastic Net
    model = ElasticNet(
        learning_rate=0.01,
        n_iterations=2000,
        l1_ratio=0.5,
        alpha=1.0,
        verbose=True,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)

    print(f"\nTrain R² Score: {train_score:.4f}")
    print(f"Test R² Score: {test_score:.4f}")
    print(f"Sparsity: {model.get_sparsity():.2%}")
    print(f"Non-zero features: {len(model.get_nonzero_features())}/{len(model.weights_)}")
