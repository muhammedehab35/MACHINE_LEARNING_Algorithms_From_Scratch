"""
Logistic Regression Implementation from Scratch

This module implements binary logistic regression using gradient descent optimization.

Mathematical formulation:
    P(y=1|x) = σ(w^T x + b) where σ is the sigmoid function
    Loss = -1/n Σ[y log(ŷ) + (1-y) log(1-ŷ)] (Binary Cross-Entropy)

Author: ML Algorithms from Scratch
"""

import numpy as np
from typing import Optional, Tuple


class LogisticRegression:
    """
    Binary Logistic Regression using gradient descent optimization.

    Logistic Regression is a linear model for binary classification that uses
    the sigmoid function to map predictions to probabilities between 0 and 1.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for gradient descent optimization.

    n_iterations : int, default=1000
        Maximum number of iterations for gradient descent.

    regularization : {'none', 'l1', 'l2', 'elasticnet'}, default='none'
        Type of regularization to apply.
        - 'none': No regularization
        - 'l1': Lasso regularization (induces sparsity)
        - 'l2': Ridge regularization (shrinks coefficients)
        - 'elasticnet': Combination of L1 and L2

    alpha : float, default=0.0
        Regularization strength. Must be non-negative.
        Higher values specify stronger regularization.

    l1_ratio : float, default=0.5
        The mixing parameter for elasticnet regularization (0 <= l1_ratio <= 1).
        - l1_ratio = 0: Pure L2 (Ridge)
        - l1_ratio = 1: Pure L1 (Lasso)
        Only used when regularization='elasticnet'.

    batch_size : int or None, default=None
        Batch size for mini-batch gradient descent.
        - None: Batch gradient descent (all samples)
        - int: Mini-batch gradient descent
        - 1: Stochastic gradient descent (SGD)

    early_stopping : bool, default=True
        Whether to stop training when validation loss stops improving.

    validation_fraction : float, default=0.1
        Fraction of training data to use for validation.

    patience : int, default=10
        Number of iterations with no improvement before stopping.

    tol : float, default=1e-4
        Tolerance for improvement in validation loss.

    decay_rate : float or None, default=None
        Learning rate decay factor applied every 100 iterations.

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

    classes_ : np.ndarray of shape (2,)
        Class labels [0, 1].
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        regularization: str = 'none',
        alpha: float = 0.0,
        l1_ratio: float = 0.5,
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
        if regularization not in ['none', 'l1', 'l2', 'elasticnet']:
            raise ValueError(f"Invalid regularization: {regularization}")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if not 0 <= l1_ratio <= 1:
            raise ValueError(f"l1_ratio must be between 0 and 1, got {l1_ratio}")
        if not 0 < validation_fraction < 1:
            raise ValueError(f"validation_fraction must be between 0 and 1")

        # Hyperparameters
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.tol = tol
        self.decay_rate = decay_rate
        self.random_state = random_state
        self.verbose = verbose

        # Model parameters (learned)
        self.weights_ = None
        self.bias_ = None

        # Training history
        self.loss_history_ = []
        self.val_loss_history_ = []
        self.n_iter_ = 0

        # Classes
        self.classes_ = np.array([0, 1])

        # Random state
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Fit logistic regression model using gradient descent.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        y : np.ndarray of shape (n_samples,)
            Target values (binary: 0 or 1).

        Returns
        -------
        self : LogisticRegression
            Fitted estimator.
        """
        # Input validation
        X, y = self._validate_input(X, y)
        n_samples, n_features = X.shape

        # Check that y is binary
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(f"y must be binary (0 and 1), found {unique_classes}")
        if not np.array_equal(unique_classes, [0, 1]):
            raise ValueError("y must contain only 0 and 1")

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
            y_pred_proba = self._sigmoid(self._compute_linear(X_batch))

            # Compute gradients
            dw, db = self._compute_gradients(X_batch, y_batch, y_pred_proba)

            # Apply regularization to gradients
            if self.regularization == 'l2':
                dw += self.alpha * self.weights_
            elif self.regularization == 'l1':
                dw += self.alpha * np.sign(self.weights_)
            elif self.regularization == 'elasticnet':
                l1_grad = self.alpha * self.l1_ratio * np.sign(self.weights_)
                l2_grad = self.alpha * (1 - self.l1_ratio) * self.weights_
                dw += l1_grad + l2_grad

            # Update parameters
            self.weights_ -= current_lr * dw
            self.bias_ -= current_lr * db

            # Apply soft-thresholding for L1/Elasticnet
            if self.regularization == 'l1':
                self.weights_ = self._soft_threshold(
                    self.weights_, current_lr * self.alpha
                )
            elif self.regularization == 'elasticnet':
                self.weights_ = self._soft_threshold(
                    self.weights_, current_lr * self.alpha * self.l1_ratio
                )

            # Clip weights to prevent overflow
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, 2)
            Probability estimates for each class.
            Column 0: P(y=0|X), Column 1: P(y=1|X)
        """
        if self.weights_ is None:
            raise ValueError("Model must be fitted before making predictions")

        X = self._validate_input(X, None, fit=False)

        # Compute P(y=1|X)
        proba_class_1 = self._sigmoid(self._compute_linear(X))

        # Stack probabilities for both classes
        proba = np.column_stack([1 - proba_class_1, proba_class_1])

        return proba

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        threshold : float, default=0.5
            Classification threshold. Predictions >= threshold are class 1.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        y : np.ndarray of shape (n_samples,)
            True labels.

        Returns
        -------
        accuracy : float
            Fraction of correctly classified samples.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    # Private methods

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.

        σ(z) = 1 / (1 + e^(-z))

        Numerically stable implementation.
        """
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)

        # Numerically stable sigmoid
        positive = z >= 0
        negative = ~positive

        result = np.zeros_like(z)

        # For positive values: σ(z) = 1 / (1 + e^(-z))
        result[positive] = 1 / (1 + np.exp(-z[positive]))

        # For negative values: σ(z) = e^z / (1 + e^z)
        exp_z = np.exp(z[negative])
        result[negative] = exp_z / (1 + exp_z)

        return result

    def _compute_linear(self, X: np.ndarray) -> np.ndarray:
        """Compute linear combination: X @ w + b"""
        return np.dot(X, self.weights_) + self.bias_

    def _compute_gradients(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute gradients for weights and bias.

        Gradient of binary cross-entropy:
        ∂L/∂w = (1/n) X^T (ŷ - y)
        ∂L/∂b = (1/n) Σ(ŷ - y)
        """
        n_samples = X.shape[0]

        # Error
        error = y_pred_proba - y_true

        # Gradients
        dw = (1 / n_samples) * np.dot(X.T, error)
        db = (1 / n_samples) * np.sum(error)

        # Clip gradients
        dw = np.clip(dw, -1e5, 1e5)
        db = np.clip(db, -1e5, 1e5)

        return dw, db

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss with regularization.

        Loss = -1/n Σ[y log(ŷ) + (1-y) log(1-ŷ)] + regularization
        """
        n_samples = X.shape[0]

        # Predictions
        y_pred_proba = self._sigmoid(self._compute_linear(X))

        # Clip probabilities to prevent log(0)
        epsilon = 1e-15
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)

        # Binary cross-entropy
        bce = -np.mean(
            y * np.log(y_pred_proba) + (1 - y) * np.log(1 - y_pred_proba)
        )

        # Regularization term
        reg_term = 0.0
        if self.regularization == 'l2':
            reg_term = 0.5 * self.alpha * np.sum(self.weights_ ** 2)
        elif self.regularization == 'l1':
            reg_term = self.alpha * np.sum(np.abs(self.weights_))
        elif self.regularization == 'elasticnet':
            l1_term = self.alpha * self.l1_ratio * np.sum(np.abs(self.weights_))
            l2_term = 0.5 * self.alpha * (1 - self.l1_ratio) * np.sum(self.weights_ ** 2)
            reg_term = l1_term + l2_term

        return bce + reg_term

    def _soft_threshold(self, w: np.ndarray, threshold: float) -> np.ndarray:
        """Soft-thresholding operator for L1 regularization."""
        return np.sign(w) * np.maximum(np.abs(w) - threshold, 0)

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
                raise ValueError(f"X and y must have same number of samples")

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

    def __repr__(self) -> str:
        return (f"LogisticRegression(learning_rate={self.learning_rate}, "
                f"n_iterations={self.n_iterations}, "
                f"regularization='{self.regularization}')")


if __name__ == "__main__":
    # Quick test
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Generate binary classification data
    X, y = make_classification(
        n_samples=300,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(
        learning_rate=0.01,
        n_iterations=1000,
        regularization='l2',
        alpha=0.1,
        verbose=True,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)

    print(f"\nTrain Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
