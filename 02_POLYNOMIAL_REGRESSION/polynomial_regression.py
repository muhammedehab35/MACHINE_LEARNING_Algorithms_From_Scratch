"""Polynomial Regression with Gradient Descent optimization."""

import numpy as np
from typing import Optional, Literal


class PolynomialRegression:
    """Polynomial Regression using feature expansion and Gradient Descent."""

    def __init__(
        self,
        degree: int = 2,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        method: Literal['batch', 'mini-batch', 'sgd'] = 'batch',
        batch_size: int = 32,
        regularization: float = 0.0,
        learning_rate_decay: float = 0.0,
        early_stopping: bool = False,
        patience: int = 10,
        tolerance: float = 1e-4,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        """Initialize Polynomial Regression model."""
        self.degree = degree
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.batch_size = batch_size
        self.regularization = regularization
        self.learning_rate_decay = learning_rate_decay
        self.early_stopping = early_stopping
        self.patience = patience
        self.tolerance = tolerance
        self.random_state = random_state
        self.verbose = verbose

        self.weights_ = None
        self.bias_ = 0.0
        self.losses_ = []
        self.n_iterations_used_ = 0
        self.is_fitted_ = False

        if random_state is not None:
            np.random.seed(random_state)

    def _polynomial_features(self, X: np.ndarray) -> np.ndarray:
        """Generate polynomial features up to specified degree."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        if n_features == 1:
            # For single feature: [x, x^2, x^3, ..., x^degree]
            X_poly = np.zeros((n_samples, self.degree))
            for i in range(self.degree):
                # Use clip to prevent overflow
                X_poly[:, i] = np.clip(X[:, 0] ** (i + 1), -1e10, 1e10)
        else:
            # For multiple features: include all polynomial combinations
            from itertools import combinations_with_replacement

            features = []
            for d in range(1, self.degree + 1):
                for combo in combinations_with_replacement(range(n_features), d):
                    feature = np.ones(n_samples)
                    for idx in combo:
                        feature *= X[:, idx]
                    # Clip to prevent overflow
                    features.append(np.clip(feature, -1e10, 1e10))

            X_poly = np.column_stack(features)

        return X_poly

    def _initialize_parameters(self, n_features: int) -> None:
        """Initialize weights and bias."""
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        self.losses_ = []
        self.n_iterations_used_ = 0

    def _compute_predictions(self, X: np.ndarray) -> np.ndarray:
        """Compute predictions."""
        predictions = np.dot(X, self.weights_) + self.bias_
        # Clip predictions to prevent overflow
        return np.clip(predictions, -1e10, 1e10)

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute MSE loss with optional L2 regularization."""
        error = np.clip(y_pred - y_true, -1e10, 1e10)
        mse = np.mean(error ** 2)
        if self.regularization > 0:
            l2_penalty = (self.regularization / 2) * np.sum(np.clip(self.weights_ ** 2, 0, 1e10))
            return np.clip(mse + l2_penalty, 0, 1e15)
        return np.clip(mse, 0, 1e15)

    def _compute_gradients(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
        """Compute gradients for weights and bias."""
        n_samples = X.shape[0]
        error = np.clip(y_pred - y_true, -1e10, 1e10)
        dw = (1 / n_samples) * np.dot(X.T, error)
        db = (1 / n_samples) * np.sum(error)
        # Clip gradients to prevent explosion
        dw = np.clip(dw, -1e5, 1e5)
        db = np.clip(db, -1e5, 1e5)

        if self.regularization > 0:
            dw += self.regularization * np.clip(self.weights_, -1e5, 1e5)
            dw = np.clip(dw, -1e5, 1e5)

        return dw, db

    def _update_parameters(self, dw: np.ndarray, db: float) -> None:
        """Update weights and bias."""
        self.weights_ -= self.learning_rate * dw
        self.bias_ -= self.learning_rate * db
        # Clip weights to prevent explosion
        self.weights_ = np.clip(self.weights_, -1e8, 1e8)
        self.bias_ = np.clip(self.bias_, -1e8, 1e8)

    def _update_learning_rate(self, iteration: int) -> None:
        """Apply learning rate decay."""
        if self.learning_rate_decay > 0:
            self.learning_rate = self.initial_learning_rate / (1 + self.learning_rate_decay * iteration)

    def _batch_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """Batch gradient descent optimization."""
        best_loss = float('inf')
        patience_counter = 0

        for iteration in range(self.n_iterations):
            y_pred = self._compute_predictions(X)
            loss = self._compute_loss(y, y_pred)
            self.losses_.append(loss)

            dw, db = self._compute_gradients(X, y, y_pred)
            self._update_parameters(dw, db)
            self._update_learning_rate(iteration)

            if self.early_stopping:
                if loss < best_loss - self.tolerance:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.verbose:
                            print(f"Early stopping at iteration {iteration + 1}")
                        break

            if self.verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iterations}, Loss: {loss:.6f}, LR: {self.learning_rate:.6f}")

            self.n_iterations_used_ = iteration + 1

    def _mini_batch_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """Mini-batch gradient descent optimization."""
        n_samples = X.shape[0]
        best_loss = float('inf')
        patience_counter = 0

        for iteration in range(self.n_iterations):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                y_pred = self._compute_predictions(X_batch)
                dw, db = self._compute_gradients(X_batch, y_batch, y_pred)
                self._update_parameters(dw, db)

            y_pred_full = self._compute_predictions(X)
            loss = self._compute_loss(y, y_pred_full)
            self.losses_.append(loss)
            self._update_learning_rate(iteration)

            if self.early_stopping:
                if loss < best_loss - self.tolerance:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.verbose:
                            print(f"Early stopping at iteration {iteration + 1}")
                        break

            if self.verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iterations}, Loss: {loss:.6f}, LR: {self.learning_rate:.6f}")

            self.n_iterations_used_ = iteration + 1

    def _stochastic_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """Stochastic gradient descent optimization."""
        n_samples = X.shape[0]
        best_loss = float('inf')
        patience_counter = 0

        for iteration in range(self.n_iterations):
            indices = np.random.permutation(n_samples)

            for idx in indices:
                X_sample = X[idx:idx+1]
                y_sample = y[idx:idx+1]

                y_pred = self._compute_predictions(X_sample)
                dw, db = self._compute_gradients(X_sample, y_sample, y_pred)
                self._update_parameters(dw, db)

            y_pred_full = self._compute_predictions(X)
            loss = self._compute_loss(y, y_pred_full)
            self.losses_.append(loss)
            self._update_learning_rate(iteration)

            if self.early_stopping:
                if loss < best_loss - self.tolerance:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.verbose:
                            print(f"Early stopping at iteration {iteration + 1}")
                        break

            if self.verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iterations}, Loss: {loss:.6f}, LR: {self.learning_rate:.6f}")

            self.n_iterations_used_ = iteration + 1

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PolynomialRegression':
        """Fit polynomial regression model."""
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D array, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples. Got X: {X.shape[0]}, y: {y.shape[0]}")

        # Transform to polynomial features
        X_poly = self._polynomial_features(X)

        # Store for later use in predict
        self.n_features_in_ = X.shape[1]

        # Initialize parameters
        n_samples, n_poly_features = X_poly.shape
        self._initialize_parameters(n_poly_features)

        # Select optimization method
        if self.method == 'batch':
            self._batch_gradient_descent(X_poly, y)
        elif self.method == 'mini-batch':
            self._mini_batch_gradient_descent(X_poly, y)
        elif self.method == 'sgd':
            self._stochastic_gradient_descent(X_poly, y)
        else:
            raise ValueError(f"Unknown method: {self.method}. Choose from: 'batch', 'mini-batch', 'sgd'")

        self.is_fitted_ = True

        if self.verbose:
            print(f"\nTraining completed!")
            print(f"Final loss: {self.losses_[-1]:.6f}")
            print(f"Iterations used: {self.n_iterations_used_}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but model was trained with {self.n_features_in_} features")

        # Transform to polynomial features
        X_poly = self._polynomial_features(X)

        return self._compute_predictions(X_poly)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

    def get_params(self) -> dict:
        """Get model parameters."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted first.")
        return {
            'degree': self.degree,
            'weights': self.weights_.copy(),
            'bias': self.bias_,
            'n_poly_features': len(self.weights_)
        }

    def __repr__(self) -> str:
        return f"PolynomialRegression(degree={self.degree}, lr={self.learning_rate}, n_iter={self.n_iterations}, method='{self.method}', reg={self.regularization})"
