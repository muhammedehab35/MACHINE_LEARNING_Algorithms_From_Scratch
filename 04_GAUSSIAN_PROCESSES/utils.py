"""
Utility Functions for Gaussian Processes

Data preprocessing and helper functions.

Author: ML Algorithms from Scratch
"""

import numpy as np
from typing import Tuple, Optional


class StandardScaler:
    """
    Standardize features by removing mean and scaling to unit variance.

    z = (x - μ) / σ

    Attributes
    ----------
    mean_ : np.ndarray
        Mean of training data.
    std_ : np.ndarray
        Standard deviation of training data.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """Compute mean and std from training data."""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply standardization."""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler must be fitted before transform")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse standardization."""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler must be fitted before inverse_transform")
        return X * self.std_ + self.mean_


class MinMaxScaler:
    """
    Scale features to [0, 1] range.

    x_scaled = (x - min) / (max - min)

    Attributes
    ----------
    min_ : np.ndarray
        Minimum of training data.
    max_ : np.ndarray
        Maximum of training data.
    """

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """Compute min and max from training data."""
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        # Avoid division by zero
        range_ = self.max_ - self.min_
        range_[range_ == 0] = 1.0
        self.max_ = self.min_ + range_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply min-max scaling."""
        if self.min_ is None or self.max_ is None:
            raise ValueError("Scaler must be fitted before transform")
        return (X - self.min_) / (self.max_ - self.min_)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse min-max scaling."""
        if self.min_ is None or self.max_ is None:
            raise ValueError("Scaler must be fitted before inverse_transform")
        return X * (self.max_ - self.min_) + self.min_


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.

    Parameters
    ----------
    X : np.ndarray
        Features.
    y : np.ndarray
        Targets.
    test_size : float
        Fraction of data for test set.
    random_state : int, optional
        Random seed.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_test = int(n_samples * test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def make_regression(
    n_samples: int = 100,
    n_features: int = 1,
    noise: float = 0.0,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of features.
    noise : float
        Standard deviation of Gaussian noise.
    random_state : int, optional
        Random seed.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
        Features.
    y : np.ndarray, shape (n_samples,)
        Targets.
    """
    if random_state is not None:
        np.random.seed(random_state)

    X = np.random.randn(n_samples, n_features)

    # Generate true function (non-linear)
    if n_features == 1:
        y = np.sin(3 * X[:, 0]) + 0.5 * X[:, 0]
    else:
        # Linear combination with non-linearity
        weights = np.random.randn(n_features)
        y = np.dot(X, weights) + np.sin(np.sum(X, axis=1))

    # Add noise
    if noise > 0:
        y += noise * np.random.randn(n_samples)

    return X, y


def k_fold_split(
    n_samples: int,
    n_folds: int = 5,
    random_state: Optional[int] = None
) -> list:
    """
    Generate k-fold cross-validation splits.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_folds : int
        Number of folds.
    random_state : int, optional
        Random seed.

    Returns
    -------
    folds : list of tuples
        List of (train_indices, val_indices) for each fold.
    """
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    fold_sizes = np.full(n_folds, n_samples // n_folds, dtype=int)
    fold_sizes[:n_samples % n_folds] += 1

    folds = []
    current = 0

    for fold_size in fold_sizes:
        val_indices = indices[current:current + fold_size]
        train_indices = np.concatenate([indices[:current], indices[current + fold_size:]])
        folds.append((train_indices, val_indices))
        current += fold_size

    return folds


def cross_val_score(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Perform k-fold cross-validation.

    Parameters
    ----------
    model : object
        Model with fit and score methods.
    X : np.ndarray
        Features.
    y : np.ndarray
        Targets.
    n_folds : int
        Number of folds.
    random_state : int, optional
        Random seed.

    Returns
    -------
    scores : np.ndarray
        Score for each fold.
    """
    folds = k_fold_split(len(X), n_folds, random_state)
    scores = []

    for train_idx, val_idx in folds:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        scores.append(score)

    return np.array(scores)


def grid_search(
    model_class,
    param_grid: dict,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: Optional[int] = None
) -> Tuple[dict, float]:
    """
    Simple grid search for hyperparameter tuning.

    Parameters
    ----------
    model_class : class
        Model class to instantiate.
    param_grid : dict
        Dictionary of parameter names to lists of values.
    X : np.ndarray
        Features.
    y : np.ndarray
        Targets.
    n_folds : int
        Number of CV folds.
    random_state : int, optional
        Random seed.

    Returns
    -------
    best_params : dict
        Best parameters found.
    best_score : float
        Best cross-validation score.
    """
    from itertools import product

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    best_score = -np.inf
    best_params = None

    for values in product(*param_values):
        params = dict(zip(param_names, values))

        # Create model with these parameters
        model = model_class(**params)

        # Cross-validate
        scores = cross_val_score(model, X, y, n_folds, random_state)
        mean_score = np.mean(scores)

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return best_params, best_score


if __name__ == "__main__":
    # Example usage
    print("\n" + "=" * 70)
    print(" " * 25 + "UTILS EXAMPLES")
    print("=" * 70)

    # StandardScaler
    print("\n1. StandardScaler:")
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"   Original:\n{X}")
    print(f"   Scaled:\n{X_scaled}")
    print(f"   Mean: {np.mean(X_scaled, axis=0)}")
    print(f"   Std: {np.std(X_scaled, axis=0)}")

    # train_test_split
    print("\n2. train_test_split:")
    X = np.arange(100).reshape(-1, 1)
    y = np.arange(100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Train size: {len(X_train)}")
    print(f"   Test size: {len(X_test)}")

    # make_regression
    print("\n3. make_regression:")
    X, y = make_regression(n_samples=50, n_features=1, noise=0.1, random_state=42)
    print(f"   Generated {len(X)} samples")
    print(f"   X shape: {X.shape}")
    print(f"   y range: [{y.min():.2f}, {y.max():.2f}]")

    print("\n" + "=" * 70)
