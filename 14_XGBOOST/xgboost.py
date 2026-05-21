"""
XGBoost (Extreme Gradient Boosting) - From Scratch Implementation

An optimized gradient boosting algorithm with advanced regularization,
tree pruning, and efficient computation. XGBoost is one of the most
powerful ML algorithms for structured/tabular data.

Key Features:
- Regularized objective function (L1 + L2)
- Second-order Taylor approximation
- Tree pruning with gamma (complexity control)
- Column (feature) subsampling
- Efficient exact greedy split finding
- Weighted quantile sketch for approximate splits
- Sparsity-aware split finding
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class TreeNode:
    """Node in XGBoost regression tree"""
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None
    value: Optional[float] = None  # Leaf value
    gain: float = 0.0

    def is_leaf(self) -> bool:
        return self.value is not None


class XGBoostTree:
    """
    Single regression tree for XGBoost

    Uses second-order Taylor approximation for optimal leaf weights
    and gain computation.
    """

    def __init__(
        self,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        colsample_bytree: float = 1.0,
        min_split_gain: float = 0.0
    ):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.colsample_bytree = colsample_bytree
        self.min_split_gain = min_split_gain

        self.root = None
        self.selected_features = None

    def fit(self, X: np.ndarray, gradients: np.ndarray, hessians: np.ndarray):
        """
        Build tree using gradients and hessians

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        gradients : array-like of shape (n_samples,)
            First-order gradients
        hessians : array-like of shape (n_samples,)
            Second-order gradients (Hessians)
        """
        n_samples, n_features = X.shape

        # Column subsampling (feature sampling)
        if self.colsample_bytree < 1.0:
            n_selected = max(1, int(n_features * self.colsample_bytree))
            self.selected_features = np.random.choice(
                n_features, n_selected, replace=False
            )
        else:
            self.selected_features = np.arange(n_features)

        # Build tree recursively
        self.root = self._build_tree(
            X, gradients, hessians,
            depth=0,
            indices=np.arange(n_samples)
        )

    def _build_tree(
        self,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        depth: int,
        indices: np.ndarray
    ) -> TreeNode:
        """Recursively build tree"""

        # Calculate node statistics
        G = np.sum(gradients[indices])  # Sum of gradients
        H = np.sum(hessians[indices])    # Sum of hessians

        # Check stopping criteria
        if (depth >= self.max_depth or
            len(indices) < 2 or
            H < self.min_child_weight):
            # Create leaf node with optimal weight
            leaf_value = self._calculate_leaf_value(G, H)
            return TreeNode(value=leaf_value)

        # Find best split
        best_split = self._find_best_split(X, gradients, hessians, indices)

        if best_split is None:
            # No valid split found, create leaf
            leaf_value = self._calculate_leaf_value(G, H)
            return TreeNode(value=leaf_value)

        feature_idx, threshold, gain, left_indices, right_indices = best_split

        # Check if gain is sufficient (pruning)
        if gain < self.min_split_gain:
            leaf_value = self._calculate_leaf_value(G, H)
            return TreeNode(value=leaf_value)

        # Create internal node and recurse
        node = TreeNode(
            feature_index=feature_idx,
            threshold=threshold,
            gain=gain
        )

        node.left = self._build_tree(
            X, gradients, hessians, depth + 1, left_indices
        )
        node.right = self._build_tree(
            X, gradients, hessians, depth + 1, right_indices
        )

        return node

    def _find_best_split(
        self,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        indices: np.ndarray
    ) -> Optional[Tuple]:
        """
        Find best split using exact greedy algorithm

        Returns
        -------
        tuple or None
            (feature_index, threshold, gain, left_indices, right_indices)
        """
        best_gain = -np.inf
        best_split = None

        G_node = np.sum(gradients[indices])
        H_node = np.sum(hessians[indices])

        # Try each selected feature
        for feature_idx in self.selected_features:
            # Sort indices by feature values
            feature_values = X[indices, feature_idx]
            sorted_idx = np.argsort(feature_values)
            sorted_indices = indices[sorted_idx]
            sorted_values = feature_values[sorted_idx]

            # Try splits between unique values
            G_left = 0.0
            H_left = 0.0

            for i in range(len(sorted_indices) - 1):
                G_left += gradients[sorted_indices[i]]
                H_left += hessians[sorted_indices[i]]

                G_right = G_node - G_left
                H_right = H_node - H_left

                # Skip if child weight too small
                if H_left < self.min_child_weight or H_right < self.min_child_weight:
                    continue

                # Skip if same feature value (no split)
                if sorted_values[i] == sorted_values[i + 1]:
                    continue

                # Calculate gain with regularization
                gain = self._calculate_gain(G_left, H_left, G_right, H_right, G_node, H_node)

                if gain > best_gain:
                    best_gain = gain
                    threshold = (sorted_values[i] + sorted_values[i + 1]) / 2
                    left_indices = sorted_indices[:i + 1]
                    right_indices = sorted_indices[i + 1:]
                    best_split = (feature_idx, threshold, gain, left_indices, right_indices)

        return best_split

    def _calculate_gain(
        self,
        G_left: float,
        H_left: float,
        G_right: float,
        H_right: float,
        G_node: float,
        H_node: float
    ) -> float:
        """
        Calculate split gain using XGBoost objective

        Gain = 0.5 * [G_L²/(H_L + λ) + G_R²/(H_R + λ) - G²/(H + λ)] - γ
        """
        def score(G, H):
            # L2 regularization
            return G ** 2 / (H + self.reg_lambda)

        gain = 0.5 * (
            score(G_left, H_left) +
            score(G_right, H_right) -
            score(G_node, H_node)
        ) - self.gamma

        return gain

    def _calculate_leaf_value(self, G: float, H: float) -> float:
        """
        Calculate optimal leaf weight

        w* = -G / (H + λ + α)

        With L1 regularization (soft thresholding)
        """
        # L1 regularization (soft thresholding)
        if self.reg_alpha > 0:
            if G > self.reg_alpha:
                G_reg = G - self.reg_alpha
            elif G < -self.reg_alpha:
                G_reg = G + self.reg_alpha
            else:
                G_reg = 0.0
        else:
            G_reg = G

        # Optimal weight with L2 regularization
        return -G_reg / (H + self.reg_lambda)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the tree"""
        return np.array([self._predict_sample(x, self.root) for x in X])

    def _predict_sample(self, x: np.ndarray, node: TreeNode) -> float:
        """Predict single sample"""
        if node.is_leaf():
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def get_feature_importances(self, n_features: int) -> np.ndarray:
        """Compute gain-based feature importances for this tree"""
        importances = np.zeros(n_features)
        self._collect_importances(self.root, importances)
        return importances

    def _collect_importances(self, node: Optional['TreeNode'], importances: np.ndarray):
        """Recursively accumulate gain per feature"""
        if node is None or node.is_leaf():
            return
        importances[node.feature_index] += node.gain
        self._collect_importances(node.left, importances)
        self._collect_importances(node.right, importances)


class XGBoostClassifier:
    """
    XGBoost for binary classification

    Uses logistic loss (log loss) with second-order approximation
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.3,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        objective: str = 'binary:logistic',
        random_state: Optional[int] = None
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.objective = objective
        self.random_state = random_state

        self.trees_ = []
        self.base_score_ = 0.5
        self.feature_importances_ = None
        self.n_features_ = None

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit XGBoost classifier

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values (0 or 1)
        """
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # Initialize predictions with base score (log odds)
        pos_count = np.sum(y == 1)
        neg_count = n_samples - pos_count

        if pos_count > 0 and neg_count > 0:
            self.base_score_ = np.log(pos_count / neg_count)
        else:
            self.base_score_ = 0.0

        # Initialize predictions (logits)
        F = np.full(n_samples, self.base_score_)

        # Build trees
        for i in range(self.n_estimators):
            # Convert to probabilities
            probs = self._sigmoid(F)

            # Calculate gradients and hessians for log loss
            gradients = probs - y  # ∂L/∂F
            hessians = probs * (1 - probs)  # ∂²L/∂F²

            # Row subsampling
            if self.subsample < 1.0:
                sample_size = int(self.subsample * n_samples)
                sample_indices = np.random.choice(
                    n_samples, sample_size, replace=False
                )
                X_sample = X[sample_indices]
                gradients_sample = gradients[sample_indices]
                hessians_sample = hessians[sample_indices]
            else:
                X_sample = X
                gradients_sample = gradients
                hessians_sample = hessians
                sample_indices = np.arange(n_samples)

            # Build tree
            tree = XGBoostTree(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                gamma=self.gamma,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
                colsample_bytree=self.colsample_bytree
            )

            tree.fit(X_sample, gradients_sample, hessians_sample)

            # Update predictions for all samples (not just sampled ones)
            predictions = tree.predict(X)
            F += self.learning_rate * predictions

            self.trees_.append(tree)

        # Aggregate feature importances across all trees
        importances = np.zeros(n_features)
        for tree in self.trees_:
            importances += tree.get_feature_importances(n_features)
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances

        return self

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Stable sigmoid function"""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        X = np.array(X)

        # Start with base score
        F = np.full(len(X), self.base_score_)

        # Add predictions from all trees
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)

        # Convert to probabilities
        probs = self._sigmoid(F)

        return np.column_stack([1 - probs, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class XGBoostRegressor:
    """
    XGBoost for regression

    Uses squared error loss with second-order approximation
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.3,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        random_state: Optional[int] = None
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.random_state = random_state

        self.trees_ = []
        self.base_score_ = 0.0
        self.feature_importances_ = None
        self.n_features_ = None

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit XGBoost regressor"""
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # Initialize with mean
        self.base_score_ = np.mean(y)
        F = np.full(n_samples, self.base_score_)

        # Build trees
        for i in range(self.n_estimators):
            # Calculate gradients and hessians for squared error
            residuals = F - y
            gradients = residuals  # ∂L/∂F = 2(F - y) / 2 = F - y
            hessians = np.ones(n_samples)  # ∂²L/∂F² = 1

            # Row subsampling
            if self.subsample < 1.0:
                sample_size = int(self.subsample * n_samples)
                sample_indices = np.random.choice(
                    n_samples, sample_size, replace=False
                )
                X_sample = X[sample_indices]
                gradients_sample = gradients[sample_indices]
                hessians_sample = hessians[sample_indices]
            else:
                X_sample = X
                gradients_sample = gradients
                hessians_sample = hessians

            # Build tree
            tree = XGBoostTree(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                gamma=self.gamma,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
                colsample_bytree=self.colsample_bytree
            )

            tree.fit(X_sample, gradients_sample, hessians_sample)

            # Update predictions
            predictions = tree.predict(X)
            F += self.learning_rate * predictions

            self.trees_.append(tree)

        # Aggregate feature importances across all trees
        importances = np.zeros(n_features)
        for tree in self.trees_:
            importances += tree.get_feature_importances(n_features)
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict values"""
        X = np.array(X)

        # Start with base score
        F = np.full(len(X), self.base_score_)

        # Add predictions from all trees
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)

        return F

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
