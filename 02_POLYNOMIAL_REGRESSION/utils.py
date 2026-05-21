"""Utility functions for Linear Regression."""

import numpy as np
from typing import Optional, Tuple


class StandardScaler:
    """Standardize features by removing mean and scaling to unit variance."""

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.std_ = None
        self.n_features_ = None
        self.is_fitted_ = False

    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """Compute mean and std for later scaling."""
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")

        self.n_features_ = X.shape[1]

        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros(self.n_features_)

        if self.with_std:
            self.std_ = np.std(X, axis=0)
            self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        else:
            self.std_ = np.ones(self.n_features_)

        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Perform standardization."""
        if not self.is_fitted_:
            raise RuntimeError("Scaler must be fitted before transform. Call fit() first.")

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but scaler was fitted with {self.n_features_} features"
            )

        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit then transform."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Scale back to original representation."""
        if not self.is_fitted_:
            raise RuntimeError("Scaler must be fitted before inverse_transform.")

        X_scaled = np.asarray(X_scaled)

        if X_scaled.ndim != 2:
            raise ValueError(f"X_scaled must be 2D array, got shape {X_scaled.shape}")

        if X_scaled.shape[1] != self.n_features_:
            raise ValueError(
                f"X_scaled has {X_scaled.shape[1]} features, "
                f"but scaler was fitted with {self.n_features_} features"
            )

        return X_scaled * self.std_ + self.mean_

    def __repr__(self) -> str:
        return f"StandardScaler(with_mean={self.with_mean}, with_std={self.with_std})"


class MinMaxScaler:
    """Transform features by scaling to a given range."""

    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        self.feature_range = feature_range

        if feature_range[0] >= feature_range[1]:
            raise ValueError(
                f"Minimum of feature_range must be smaller than maximum. "
                f"Got {feature_range}"
            )

        self.min_ = None
        self.max_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.n_features_ = None
        self.is_fitted_ = False

    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """Compute min and max for later scaling."""
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")

        self.n_features_ = X.shape[1]
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        self.data_range_ = np.where(self.data_range_ == 0, 1.0, self.data_range_)

        feature_range_min, feature_range_max = self.feature_range
        scale = (feature_range_max - feature_range_min) / self.data_range_

        self.min_ = feature_range_min - self.data_min_ * scale
        self.max_ = feature_range_max
        self.is_fitted_ = True

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features to specified range."""
        if not self.is_fitted_:
            raise RuntimeError("Scaler must be fitted before transform. Call fit() first.")

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but scaler was fitted with {self.n_features_} features"
            )

        X_std = (X - self.data_min_) / self.data_range_
        feature_range_min, feature_range_max = self.feature_range
        return X_std * (feature_range_max - feature_range_min) + feature_range_min

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit then transform."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Scale back to original representation."""
        if not self.is_fitted_:
            raise RuntimeError("Scaler must be fitted before inverse_transform.")

        X_scaled = np.asarray(X_scaled)

        if X_scaled.ndim != 2:
            raise ValueError(f"X_scaled must be 2D array, got shape {X_scaled.shape}")

        if X_scaled.shape[1] != self.n_features_:
            raise ValueError(
                f"X_scaled has {X_scaled.shape[1]} features, "
                f"but scaler was fitted with {self.n_features_} features"
            )

        feature_range_min, feature_range_max = self.feature_range
        X_std = (X_scaled - feature_range_min) / (feature_range_max - feature_range_min)
        return X_std * self.data_range_ + self.data_min_

    def __repr__(self) -> str:
        return f"MinMaxScaler(feature_range={self.feature_range})"


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into random train and test subsets."""
    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have same number of samples. Got X: {X.shape[0]}, y: {y.shape[0]}")

    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0.0 and 1.0, got {test_size}")

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)

    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test
