"""
Utility Functions for Data Preprocessing

This module provides essential preprocessing utilities:
- Feature scaling (StandardScaler, MinMaxScaler)
- Data splitting (train_test_split)
- Feature polynomial expansion

All implementations use only NumPy.

Author: ML Algorithms from Scratch
"""

import numpy as np
from typing import Tuple, Optional, Union


class StandardScaler:
    """
    Standardize features by removing mean and scaling to unit variance.

    Formula: z = (x - μ) / σ

    where:
    - μ: mean of the training samples
    - σ: standard deviation of the training samples

    Attributes
    ----------
    mean_ : np.ndarray of shape (n_features,)
        Mean value for each feature in the training set.

    std_ : np.ndarray of shape (n_features,)
        Standard deviation for each feature in the training set.

    n_features_ : int
        Number of features.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.n_features_ = None

    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """
        Compute mean and standard deviation for later scaling.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : StandardScaler
            Fitted scaler.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0

        self.n_features_ = X.shape[1]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize features using previously computed mean and std.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_scaled : np.ndarray of shape (n_samples, n_features)
            Standardized data.
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        X_scaled : np.ndarray of shape (n_samples, n_features)
            Standardized data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Reverse the standardization.

        Parameters
        ----------
        X_scaled : np.ndarray of shape (n_samples, n_features)
            Standardized data.

        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features)
            Original scale data.
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")

        return X_scaled * self.std_ + self.mean_


class MinMaxScaler:
    """
    Scale features to a given range (default [0, 1]).

    Formula: X_scaled = (X - X_min) / (X_max - X_min)

    Attributes
    ----------
    min_ : np.ndarray of shape (n_features,)
        Minimum value for each feature in the training set.

    max_ : np.ndarray of shape (n_features,)
        Maximum value for each feature in the training set.

    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    """

    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.data_range_ = None

    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """
        Compute min and max for later scaling.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : MinMaxScaler
            Fitted scaler.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.data_range_ = self.max_ - self.min_

        # Avoid division by zero
        self.data_range_[self.data_range_ == 0] = 1.0

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features to the specified range.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_scaled : np.ndarray of shape (n_samples, n_features)
            Scaled data.
        """
        if self.min_ is None or self.max_ is None:
            raise ValueError("Scaler has not been fitted yet. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Scale to [0, 1]
        X_std = (X - self.min_) / self.data_range_

        # Scale to feature_range
        min_val, max_val = self.feature_range
        X_scaled = X_std * (max_val - min_val) + min_val

        return X_scaled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        X_scaled : np.ndarray of shape (n_samples, n_features)
            Scaled data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Reverse the scaling.

        Parameters
        ----------
        X_scaled : np.ndarray of shape (n_samples, n_features)
            Scaled data.

        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features)
            Original scale data.
        """
        if self.min_ is None or self.max_ is None:
            raise ValueError("Scaler has not been fitted yet.")

        min_val, max_val = self.feature_range
        X_std = (X_scaled - min_val) / (max_val - min_val)

        return X_std * self.data_range_ + self.min_


def train_test_split(
    *arrays,
    test_size: Union[float, int] = 0.25,
    random_state: Optional[int] = None,
    shuffle: bool = True
) -> list:
    """
    Split arrays into random train and test subsets.

    Parameters
    ----------
    *arrays : sequence of arrays
        Allowed inputs are lists, numpy arrays with same length.

    test_size : float or int, default=0.25
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split.
        If int, represents the absolute number of test samples.

    random_state : int, optional
        Controls the shuffling for reproducible output.

    shuffle : bool, default=True
        Whether to shuffle the data before splitting.

    Returns
    -------
    splitting : list
        List containing train-test split of inputs.
        For example, if inputs are X and y:
        X_train, X_test, y_train, y_test
    """
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")

    # Validate all arrays have same length
    n_samples = len(arrays[0])
    for array in arrays[1:]:
        if len(array) != n_samples:
            raise ValueError("All input arrays must have the same length")

    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)

    # Create indices
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    # Compute split point
    if isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0.0 and 1.0")
        n_test = int(n_samples * test_size)
    else:
        if not 0 < test_size < n_samples:
            raise ValueError(f"test_size must be between 0 and {n_samples}")
        n_test = test_size

    # Split indices
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    # Split arrays
    result = []
    for array in arrays:
        array = np.array(array)
        result.append(array[train_indices])
        result.append(array[test_indices])

    return result


def polynomial_features(
    X: np.ndarray,
    degree: int = 2,
    include_bias: bool = False
) -> np.ndarray:
    """
    Generate polynomial features up to specified degree.

    For a single feature x, generates: [x, x², x³, ..., x^degree]
    For multiple features, generates all combinations up to degree.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input data.

    degree : int, default=2
        Degree of polynomial features.

    include_bias : bool, default=False
        If True, include a column of ones (bias term).

    Returns
    -------
    X_poly : np.ndarray of shape (n_samples, n_output_features)
        Polynomial features.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples, n_features = X.shape

    if n_features == 1:
        # Single feature case
        X_poly = np.zeros((n_samples, degree + (1 if include_bias else 0)))

        col_idx = 0
        if include_bias:
            X_poly[:, 0] = 1
            col_idx = 1

        for d in range(1, degree + 1):
            X_poly[:, col_idx] = X[:, 0] ** d
            col_idx += 1

        return X_poly
    else:
        # Multiple features: generate all combinations
        # This is more complex, simplified version for degree 2
        if degree == 2:
            features = [X]

            # Add squared terms
            features.append(X ** 2)

            # Add interaction terms
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
                    features.append(interaction)

            X_poly = np.hstack(features)

            if include_bias:
                bias = np.ones((n_samples, 1))
                X_poly = np.hstack([bias, X_poly])

            return X_poly
        else:
            # For higher degrees with multiple features, use simple power expansion
            features = []

            if include_bias:
                features.append(np.ones((n_samples, 1)))

            for d in range(1, degree + 1):
                features.append(X ** d)

            return np.hstack(features)


if __name__ == "__main__":
    # Test StandardScaler
    print("Testing StandardScaler:")
    X = np.array([[1, 2], [3, 4], [5, 6]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Original:\n{X}")
    print(f"Scaled:\n{X_scaled}")
    print(f"Mean: {scaler.mean_}")
    print(f"Std: {scaler.std_}")

    # Test train_test_split
    print("\nTesting train_test_split:")
    X = np.arange(10).reshape(-1, 1)
    y = np.arange(10)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"y_train: {y_train}")
    print(f"y_test: {y_test}")

    # Test polynomial_features
    print("\nTesting polynomial_features:")
    X = np.array([[1], [2], [3]])
    X_poly = polynomial_features(X, degree=3)
    print(f"Original:\n{X}")
    print(f"Polynomial (degree 3):\n{X_poly}")
