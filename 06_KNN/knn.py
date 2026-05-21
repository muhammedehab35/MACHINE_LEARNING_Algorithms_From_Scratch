"""
K-Nearest Neighbors (KNN) Implementation

A complete implementation of K-Nearest Neighbors for both classification
and regression from scratch using only NumPy.

Key Features:
- Multiple distance metrics (Euclidean, Manhattan, Minkowski, Cosine)
- Weighted voting (uniform and distance-based)
- Both classification and regression support
- Efficient nearest neighbor search
- Cross-validation support

Author: ML Algorithms from Scratch
"""

import numpy as np
from typing import Literal, Optional, Callable


class KNNClassifier:
    """
    K-Nearest Neighbors Classifier

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use for prediction.

    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction:
        - 'uniform': All neighbors have equal weight
        - 'distance': Weights inversely proportional to distance

    metric : {'euclidean', 'manhattan', 'minkowski', 'cosine'}, default='euclidean'
        Distance metric to use.

    p : int, default=2
        Power parameter for Minkowski metric. p=2 is Euclidean, p=1 is Manhattan.

    algorithm : {'brute'}, default='brute'
        Algorithm used to compute nearest neighbors. Currently only brute force.

    Examples
    --------
    >>> from knn import KNNClassifier
    >>> X_train = np.array([[0, 0], [1, 1], [2, 2]])
    >>> y_train = np.array([0, 0, 1])
    >>> knn = KNNClassifier(n_neighbors=2)
    >>> knn.fit(X_train, y_train)
    >>> knn.predict([[1.5, 1.5]])
    array([0])
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: Literal['uniform', 'distance'] = 'uniform',
        metric: Literal['euclidean', 'manhattan', 'minkowski', 'cosine'] = 'euclidean',
        p: int = 2,
        algorithm: Literal['brute'] = 'brute'
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        self.algorithm = algorithm

        # Will be set during fit
        self.X_train_ = None
        self.y_train_ = None
        self.classes_ = None
        self.n_classes_ = None

    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Euclidean distance between two vectors."""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _manhattan_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Manhattan (L1) distance between two vectors."""
        return np.sum(np.abs(x1 - x2))

    def _minkowski_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Minkowski distance between two vectors."""
        return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)

    def _cosine_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute cosine distance between two vectors."""
        dot_product = np.dot(x1, x2)
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)

        if norm_x1 == 0 or norm_x2 == 0:
            return 1.0

        cosine_similarity = dot_product / (norm_x1 * norm_x2)
        return 1 - cosine_similarity

    def _get_distance_function(self) -> Callable:
        """Return the appropriate distance function based on metric."""
        if self.metric == 'euclidean':
            return self._euclidean_distance
        elif self.metric == 'manhattan':
            return self._manhattan_distance
        elif self.metric == 'minkowski':
            return self._minkowski_distance
        elif self.metric == 'cosine':
            return self._cosine_distance
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _compute_distances(self, x: np.ndarray) -> np.ndarray:
        """
        Compute distances from x to all training samples.

        Parameters
        ----------
        x : np.ndarray, shape (n_features,)
            Query point.

        Returns
        -------
        distances : np.ndarray, shape (n_train_samples,)
            Distances to all training samples.
        """
        distance_func = self._get_distance_function()
        distances = np.array([distance_func(x, x_train) for x_train in self.X_train_])
        return distances

    def _get_neighbors(self, x: np.ndarray) -> tuple:
        """
        Get k nearest neighbors for a query point.

        Parameters
        ----------
        x : np.ndarray, shape (n_features,)
            Query point.

        Returns
        -------
        indices : np.ndarray, shape (n_neighbors,)
            Indices of nearest neighbors.
        distances : np.ndarray, shape (n_neighbors,)
            Distances to nearest neighbors.
        """
        distances = self._compute_distances(x)

        # Get indices of k smallest distances
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_distances = distances[k_indices]

        return k_indices, k_distances

    def _predict_single(self, x: np.ndarray) -> int:
        """
        Predict class for a single sample.

        Parameters
        ----------
        x : np.ndarray, shape (n_features,)
            Query point.

        Returns
        -------
        prediction : int
            Predicted class label.
        """
        indices, distances = self._get_neighbors(x)
        neighbor_labels = self.y_train_[indices]

        if self.weights == 'uniform':
            # Simple majority vote
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            prediction = unique[np.argmax(counts)]

        elif self.weights == 'distance':
            # Distance-weighted vote
            # Add small epsilon to avoid division by zero
            weights = 1 / (distances + 1e-10)

            # Sum weights for each class
            class_weights = np.zeros(self.n_classes_)
            for label, weight in zip(neighbor_labels, weights):
                class_idx = np.where(self.classes_ == label)[0][0]
                class_weights[class_idx] += weight

            prediction = self.classes_[np.argmax(class_weights)]

        return prediction

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNClassifier':
        """
        Fit the KNN classifier.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data.
        y : np.ndarray, shape (n_samples,)
            Target labels.

        Returns
        -------
        self : KNNClassifier
            Fitted classifier.
        """
        # Validate inputs
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        if self.n_neighbors > X.shape[0]:
            raise ValueError(f"n_neighbors ({self.n_neighbors}) cannot be greater than "
                           f"number of samples ({X.shape[0]})")

        # Store training data
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        # Get unique classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Test data.

        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            Predicted class labels.
        """
        if self.X_train_ is None:
            raise ValueError("Model must be fitted before prediction")

        predictions = np.array([self._predict_single(x) for x in X])
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Test data.

        Returns
        -------
        probabilities : np.ndarray, shape (n_samples, n_classes)
            Class probabilities for each sample.
        """
        if self.X_train_ is None:
            raise ValueError("Model must be fitted before prediction")

        n_samples = X.shape[0]
        probabilities = np.zeros((n_samples, self.n_classes_))

        for i, x in enumerate(X):
            indices, distances = self._get_neighbors(x)
            neighbor_labels = self.y_train_[indices]

            if self.weights == 'uniform':
                # Count votes for each class
                for label in neighbor_labels:
                    class_idx = np.where(self.classes_ == label)[0][0]
                    probabilities[i, class_idx] += 1

                # Normalize to get probabilities
                probabilities[i] /= self.n_neighbors

            elif self.weights == 'distance':
                # Distance-weighted probabilities
                weights = 1 / (distances + 1e-10)
                total_weight = np.sum(weights)

                for label, weight in zip(neighbor_labels, weights):
                    class_idx = np.where(self.classes_ == label)[0][0]
                    probabilities[i, class_idx] += weight

                # Normalize
                probabilities[i] /= total_weight

        return probabilities

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


class KNNRegressor:
    """
    K-Nearest Neighbors Regressor

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use for prediction.

    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction.

    metric : {'euclidean', 'manhattan', 'minkowski', 'cosine'}, default='euclidean'
        Distance metric to use.

    p : int, default=2
        Power parameter for Minkowski metric.

    algorithm : {'brute'}, default='brute'
        Algorithm used to compute nearest neighbors.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: Literal['uniform', 'distance'] = 'uniform',
        metric: Literal['euclidean', 'manhattan', 'minkowski', 'cosine'] = 'euclidean',
        p: int = 2,
        algorithm: Literal['brute'] = 'brute'
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        self.algorithm = algorithm

        self.X_train_ = None
        self.y_train_ = None

    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Euclidean distance."""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _manhattan_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Manhattan distance."""
        return np.sum(np.abs(x1 - x2))

    def _minkowski_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Minkowski distance."""
        return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)

    def _cosine_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute cosine distance."""
        dot_product = np.dot(x1, x2)
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)

        if norm_x1 == 0 or norm_x2 == 0:
            return 1.0

        return 1 - (dot_product / (norm_x1 * norm_x2))

    def _get_distance_function(self) -> Callable:
        """Return distance function."""
        if self.metric == 'euclidean':
            return self._euclidean_distance
        elif self.metric == 'manhattan':
            return self._manhattan_distance
        elif self.metric == 'minkowski':
            return self._minkowski_distance
        elif self.metric == 'cosine':
            return self._cosine_distance
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _compute_distances(self, x: np.ndarray) -> np.ndarray:
        """Compute distances to all training samples."""
        distance_func = self._get_distance_function()
        distances = np.array([distance_func(x, x_train) for x_train in self.X_train_])
        return distances

    def _get_neighbors(self, x: np.ndarray) -> tuple:
        """Get k nearest neighbors."""
        distances = self._compute_distances(x)
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_distances = distances[k_indices]
        return k_indices, k_distances

    def _predict_single(self, x: np.ndarray) -> float:
        """Predict target for a single sample."""
        indices, distances = self._get_neighbors(x)
        neighbor_values = self.y_train_[indices]

        if self.weights == 'uniform':
            # Simple average
            prediction = np.mean(neighbor_values)

        elif self.weights == 'distance':
            # Distance-weighted average
            weights = 1 / (distances + 1e-10)
            prediction = np.sum(weights * neighbor_values) / np.sum(weights)

        return prediction

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNRegressor':
        """Fit the regressor."""
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        if self.n_neighbors > X.shape[0]:
            raise ValueError(f"n_neighbors ({self.n_neighbors}) cannot be greater than "
                           f"number of samples ({X.shape[0]})")

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets for samples."""
        if self.X_train_ is None:
            raise ValueError("Model must be fitted before prediction")

        predictions = np.array([self._predict_single(x) for x in X])
        return predictions

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R² score."""
        predictions = self.predict(X)

        # R² score
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1 - (ss_res / ss_tot)


if __name__ == "__main__":
    # Example usage
    print("\n" + "=" * 70)
    print(" " * 25 + "KNN EXAMPLES")
    print("=" * 70)

    # Classification example
    print("\nClassification Example:")
    print("-" * 70)

    np.random.seed(42)
    X_train = np.random.randn(100, 2)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

    X_test = np.random.randn(20, 2)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

    knn_clf = KNNClassifier(n_neighbors=5, weights='distance')
    knn_clf.fit(X_train, y_train)

    accuracy = knn_clf.score(X_test, y_test)
    print(f"Classification Accuracy: {accuracy:.4f}")

    # Regression example
    print("\nRegression Example:")
    print("-" * 70)

    X_train_reg = np.random.rand(100, 1) * 10
    y_train_reg = 2 * X_train_reg.ravel() + 3 + np.random.randn(100)

    X_test_reg = np.random.rand(20, 1) * 10
    y_test_reg = 2 * X_test_reg.ravel() + 3 + np.random.randn(20)

    knn_reg = KNNRegressor(n_neighbors=5, weights='distance')
    knn_reg.fit(X_train_reg, y_train_reg)

    r2_score = knn_reg.score(X_test_reg, y_test_reg)
    print(f"Regression R² Score: {r2_score:.4f}")

    print("\n" + "=" * 70)
