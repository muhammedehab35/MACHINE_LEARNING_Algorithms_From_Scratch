"""
Utility Functions for SVM

Data preprocessing and utility functions.

Author: ML Algorithms from Scratch
"""

import numpy as np


class StandardScaler:
    """
    Standardize features by removing mean and scaling to unit variance.

    z = (x - mean) / std
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """
        Compute mean and std for later scaling.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : StandardScaler
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features using computed mean and std.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to scale.

        Returns
        -------
        X_scaled : np.ndarray
            Scaled data.
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler must be fitted before transform")

        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse scaling transformation.

        Parameters
        ----------
        X : np.ndarray
            Scaled data.

        Returns
        -------
        X_original : np.ndarray
            Data in original scale.
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler must be fitted before inverse_transform")

        return X * self.std_ + self.mean_


class MinMaxScaler:
    """
    Scale features to a given range (default [0, 1]).

    X_scaled = (X - X.min) / (X.max - X.min)
    """

    def __init__(self, feature_range: tuple = (0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """Compute min and max for later scaling."""
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)

        # Compute scaling factor
        data_range = self.max_ - self.min_
        data_range[data_range == 0] = 1.0

        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / data_range

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features to feature_range."""
        if self.min_ is None or self.scale_ is None:
            raise ValueError("Scaler must be fitted before transform")

        feature_min = self.feature_range[0]
        X_scaled = (X - self.min_) * self.scale_ + feature_min

        return X_scaled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = None
) -> tuple:
    """
    Split data into training and test sets.

    Parameters
    ----------
    X : np.ndarray
        Features.
    y : np.ndarray
        Labels.
    test_size : float
        Proportion of data for test set.
    random_state : int, optional
        Random seed.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple of arrays
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)

    # Random permutation
    indices = np.random.permutation(n_samples)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


if __name__ == "__main__":
    # Example usage
    print("\nStandardScaler Example:")
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Original:\n{X}")
    print(f"\nScaled:\n{X_scaled}")
    print(f"\nMean: {np.mean(X_scaled, axis=0)}")
    print(f"Std: {np.std(X_scaled, axis=0)}")

    print("\n" + "=" * 50)
    print("\nMinMaxScaler Example:")

    scaler_mm = MinMaxScaler()
    X_scaled_mm = scaler_mm.fit_transform(X)

    print(f"Original:\n{X}")
    print(f"\nScaled to [0, 1]:\n{X_scaled_mm}")
