"""Linear Regression with Gradient Descent optimization."""

import numpy as np
from typing import Optional, Tuple, Literal


class LinearRegression:
    """Linear Regression with Gradient Descent (Batch/Mini-Batch/SGD), L2 regularization, early stopping, and LR decay."""

    def __init__(
        self,
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

        # Attributes set during fitting
        self.weights_ = None
        self.bias_ = None
        self.losses_ = []
        self.n_iterations_used_ = 0
        self.is_fitted_ = False

        if random_state is not None:
            np.random.seed(random_state)

    def _initialize_parameters(self, n_features: int) -> None:
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        self.losses_ = []
        self.n_iterations_used_ = 0

    def _compute_predictions(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights_) + self.bias_

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mse = np.mean((y_pred - y_true) ** 2)
        if self.regularization > 0:
            l2_penalty = (self.regularization / 2) * np.sum(self.weights_ ** 2)
            return mse + l2_penalty
        return mse

    def _compute_gradients(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
        n_samples = X.shape[0]
        error = y_pred - y_true
        dw = (1 / n_samples) * np.dot(X.T, error)
        db = (1 / n_samples) * np.sum(error)
        if self.regularization > 0:
            dw += self.regularization * self.weights_
        return dw, db

    def _update_parameters(self, dw: np.ndarray, db: float) -> None:
        self.weights_ -= self.learning_rate * dw
        self.bias_ -= self.learning_rate * db

    def _update_learning_rate(self, iteration: int) -> None:
        if self.learning_rate_decay > 0:
            self.learning_rate = self.initial_learning_rate / (1 + self.learning_rate_decay * iteration)

    def _batch_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D array, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples. Got X: {X.shape[0]}, y: {y.shape[0]}")

        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)

        if self.method == 'batch':
            self._batch_gradient_descent(X, y)
        elif self.method == 'mini-batch':
            self._mini_batch_gradient_descent(X, y)
        elif self.method == 'sgd':
            self._stochastic_gradient_descent(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}. Choose from: 'batch', 'mini-batch', 'sgd'")

        self.is_fitted_ = True

        if self.verbose:
            print(f"\nTraining completed!")
            print(f"Final loss: {self.losses_[-1]:.6f}")
            print(f"Iterations used: {self.n_iterations_used_}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        if X.shape[1] != self.weights_.shape[0]:
            raise ValueError(f"X has {X.shape[1]} features, but model was trained with {self.weights_.shape[0]} features")

        return self._compute_predictions(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

    def get_params(self) -> dict:
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted first.")
        return {'weights': self.weights_.copy(), 'bias': self.bias_, 'n_features': len(self.weights_)}

    def __repr__(self) -> str:
        return f"LinearRegression(lr={self.learning_rate}, n_iter={self.n_iterations}, method='{self.method}', reg={self.regularization})"
