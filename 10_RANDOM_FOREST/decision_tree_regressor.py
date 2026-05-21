"""
Simple Decision Tree Regressor for Random Forest

A simplified implementation of Decision Tree Regressor used by Random Forest.
For full implementation, see 09_DECISION_TREE module.
"""

import numpy as np
from typing import Optional


class Node:
    """A node in the decision tree"""

    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional['Node'] = None,
        right: Optional['Node'] = None,
        value: Optional[float] = None,
        impurity: float = 0.0,
        n_samples: int = 0
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.impurity = impurity
        self.n_samples = n_samples


class DecisionTreeRegressor:
    """
    Simple Decision Tree Regressor

    Uses mean squared error for splitting criterion.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.tree_ = None
        self.n_features_ = None
        self.feature_importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeRegressor':
        """Build decision tree"""
        X = np.asarray(X)
        y = np.asarray(y)

        self.n_features_ = X.shape[1]
        self._feature_importances = np.zeros(self.n_features_)

        # Set random state
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Build tree
        self.tree_ = self._build_tree(X, y, depth=0)

        # Normalize feature importances
        if self._feature_importances.sum() > 0:
            self.feature_importances_ = self._feature_importances / self._feature_importances.sum()
        else:
            self.feature_importances_ = self._feature_importances

        return self

    def _mse(self, y: np.ndarray) -> float:
        """Calculate mean squared error"""
        if len(y) == 0:
            return 0.0
        return np.var(y) * len(y)

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """Find the best split"""
        n_samples, n_features = X.shape

        if n_samples < self.min_samples_split:
            return None, None

        # Parent impurity
        parent_impurity = self._mse(y)
        if parent_impurity == 0:
            return None, None

        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        # Determine which features to consider
        if self.max_features is not None and self.max_features < n_features:
            features = np.random.choice(n_features, self.max_features, replace=False)
        else:
            features = np.arange(n_features)

        # Try each feature
        for feature in features:
            # Sort samples by feature
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                # Split
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                # Calculate gain
                left_impurity = self._mse(y[left_mask])
                right_impurity = self._mse(y[right_mask])

                gain = parent_impurity - (left_impurity + right_impurity)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        if best_feature is not None:
            # Update feature importance
            self._feature_importances[best_feature] += best_gain

        return best_feature, best_threshold

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """Recursively build the tree"""
        n_samples = len(y)
        impurity = self._mse(y)
        node_value = np.mean(y)

        # Create leaf node if stopping criteria met
        if (
            (self.max_depth is not None and depth >= self.max_depth) or
            n_samples < self.min_samples_split or
            impurity == 0
        ):
            return Node(value=node_value, impurity=impurity, n_samples=n_samples)

        # Find best split
        feature, threshold = self._best_split(X, y)

        if feature is None:
            return Node(value=node_value, impurity=impurity, n_samples=n_samples)

        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        # Build subtrees
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature=feature,
            threshold=threshold,
            left=left,
            right=right,
            impurity=impurity,
            n_samples=n_samples
        )

    def _predict_sample(self, x: np.ndarray, node: Node) -> float:
        """Predict a single sample"""
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regression targets"""
        if self.tree_ is None:
            raise ValueError("Tree not fitted. Call fit() first")

        X = np.asarray(X)
        return np.array([self._predict_sample(x, self.tree_) for x in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
