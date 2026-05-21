"""
Decision Tree Classifier Implementation from Scratch
===================================================

A complete implementation of a Decision Tree classifier using only NumPy.
Implements CART (Classification and Regression Trees) algorithm with support
for both binary and multi-class classification.

Features:
---------
- Binary and multi-class classification
- Multiple split criteria (Gini, Entropy, Misclassification Error)
- Pre-pruning (max_depth, min_samples_split, min_samples_leaf)
- Cost-complexity pruning (ccp_alpha)
- Feature importance calculation
- Tree visualization
- sklearn-compatible API

Mathematical Background:
-----------------------
1. Gini Impurity:
   Gini(D) = 1 - Σ(p_i²)
   where p_i is the proportion of class i samples

2. Entropy:
   H(D) = -Σ(p_i * log2(p_i))

3. Information Gain:
   IG = H(parent) - [n_left/n * H(left) + n_right/n * H(right)]

4. Feature Importance:
   Importance(f) = Σ (n_node/n_total) * (impurity_decrease)
   where impurity_decrease = impurity - weighted_child_impurity

Author: ML Algorithms from Scratch
Date: 2026
"""

import numpy as np
from collections import Counter


class Node:
    """
    Node in the decision tree.

    A node can be either:
    - Internal node: contains a split rule (feature_index, threshold)
    - Leaf node: contains a prediction (value)

    Attributes:
    -----------
    feature_index : int or None
        Index of feature to split on (None for leaf nodes)
    threshold : float or None
        Threshold value for the split (None for leaf nodes)
    left : Node or None
        Left child node (samples where X[:, feature_index] <= threshold)
    right : Node or None
        Right child node (samples where X[:, feature_index] > threshold)
    value : int or None
        Predicted class for leaf nodes (None for internal nodes)
    impurity : float
        Node impurity (Gini, Entropy, or Error)
    n_samples : int
        Number of samples at this node
    class_counts : dict
        Distribution of classes at this node {class: count}
    depth : int
        Depth of this node in the tree (root is 0)
    """

    def __init__(self, depth=0):
        """
        Initialize a tree node.

        Parameters:
        -----------
        depth : int, default=0
            Depth of this node in the tree
        """
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.impurity = None
        self.n_samples = None
        self.class_counts = None
        self.depth = depth

    def is_leaf(self):
        """Check if node is a leaf."""
        return self.value is not None


class DecisionTreeClassifier:
    """
    Decision Tree Classifier using CART algorithm.

    This classifier uses recursive binary splitting to build a decision tree.
    At each node, it finds the best feature and threshold that maximizes
    information gain (or minimizes impurity).

    Parameters:
    -----------
    criterion : str, default='gini'
        Function to measure quality of split.
        Options: 'gini', 'entropy', 'error'

    max_depth : int or None, default=None
        Maximum depth of the tree. None means unlimited.

    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.

    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.

    max_features : int, float, str or None, default=None
        Number of features to consider for best split:
        - int: consider max_features features
        - float: consider int(max_features * n_features) features
        - 'sqrt': consider sqrt(n_features) features
        - 'log2': consider log2(n_features) features
        - None: consider all features

    random_state : int or None, default=None
        Random seed for reproducibility.

    ccp_alpha : float, default=0.0
        Complexity parameter for cost-complexity pruning.
        Larger values mean more pruning.

    Attributes:
    -----------
    tree_ : Node
        The root node of the fitted tree.

    n_features_ : int
        Number of features in training data.

    n_classes_ : int
        Number of classes in training data.

    classes_ : ndarray of shape (n_classes,)
        Array of class labels.

    feature_importances_ : ndarray of shape (n_features,)
        Normalized feature importance scores.

    Examples:
    ---------
    >>> from decision_tree import DecisionTreeClassifier
    >>> import numpy as np
    >>> X = np.array([[0, 0], [1, 1]])
    >>> y = np.array([0, 1])
    >>> clf = DecisionTreeClassifier(max_depth=2)
    >>> clf.fit(X, y)
    >>> clf.predict([[0.5, 0.5]])
    array([0])
    """

    def __init__(
        self,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
        ccp_alpha=0.0
    ):
        """Initialize DecisionTreeClassifier with parameters."""
        # Validate criterion
        if criterion not in ['gini', 'entropy', 'error']:
            raise ValueError(f"criterion must be 'gini', 'entropy', or 'error', got {criterion}")

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = max(2, min_samples_split)  # Must be at least 2
        self.min_samples_leaf = max(1, min_samples_leaf)    # Must be at least 1
        self.max_features = max_features
        self.random_state = random_state
        self.ccp_alpha = ccp_alpha

        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)

        # Attributes set during fit
        self.tree_ = None
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None
        self.feature_importances_ = None
        self._impurity_importance = None  # For feature importance calculation

    def fit(self, X, y):
        """
        Build decision tree from training data.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Training features.

        y : ndarray of shape (n_samples,)
            Training labels.

        Returns:
        --------
        self : DecisionTreeClassifier
            Fitted classifier.
        """
        # Input validation
        X, y = self._validate_input(X, y)

        # Store dataset properties
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Initialize feature importance tracking
        self._impurity_importance = np.zeros(self.n_features_)

        # Build the tree
        self.tree_ = self._build_tree(X, y, depth=0)

        # Calculate normalized feature importances
        total_importance = np.sum(self._impurity_importance)
        if total_importance > 0:
            self.feature_importances_ = self._impurity_importance / total_importance
        else:
            self.feature_importances_ = np.zeros(self.n_features_)

        # Apply cost-complexity pruning if requested
        if self.ccp_alpha > 0:
            self._prune_tree(self.tree_)

        return self

    def _validate_input(self, X, y):
        """
        Validate and convert input data.

        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Labels

        Returns:
        --------
        X : ndarray
            Validated features
        y : ndarray
            Validated labels
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")

        if y.ndim != 1:
            raise ValueError(f"y must be 1D array, got {y.ndim}D")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples: "
                f"X has {X.shape[0]}, y has {y.shape[0]}"
            )

        if X.shape[0] == 0:
            raise ValueError("Cannot fit with 0 samples")

        return X, y

    def _build_tree(self, X, y, depth):
        """
        Recursively build decision tree.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Training features for this node
        y : ndarray of shape (n_samples,)
            Training labels for this node
        depth : int
            Current depth in the tree

        Returns:
        --------
        node : Node
            Root of the subtree
        """
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))

        # Create node
        node = Node(depth=depth)
        node.n_samples = n_samples
        node.class_counts = dict(Counter(y))

        # Calculate node impurity
        node.impurity = self._calculate_impurity(y)

        # Stopping criteria
        should_stop = (
            depth == self.max_depth or                    # Max depth reached
            n_samples < self.min_samples_split or         # Too few samples to split
            n_classes == 1 or                             # Pure node (single class)
            node.impurity == 0                            # No impurity
        )

        if should_stop:
            # Create leaf node
            node.value = self._most_common_class(y)
            return node

        # Find best split
        best_split = self._find_best_split(X, y)

        if best_split is None:
            # No valid split found, create leaf
            node.value = self._most_common_class(y)
            return node

        feature_idx, threshold, left_indices, right_indices, impurity_decrease = best_split

        # Check min_samples_leaf constraint
        if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
            # Split would violate constraint, create leaf
            node.value = self._most_common_class(y)
            return node

        # Update feature importance
        # Weight by number of samples at this node
        importance_weight = n_samples / X.shape[0] if hasattr(self, 'tree_') and self.tree_ is not None else 1.0
        self._impurity_importance[feature_idx] += importance_weight * impurity_decrease

        # Create internal node
        node.feature_index = feature_idx
        node.threshold = threshold

        # Recursively build children
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        node.left = self._build_tree(X_left, y_left, depth + 1)
        node.right = self._build_tree(X_right, y_right, depth + 1)

        return node

    def _find_best_split(self, X, y):
        """
        Find the best feature and threshold to split on.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Features
        y : ndarray of shape (n_samples,)
            Labels

        Returns:
        --------
        best_split : tuple or None
            (feature_idx, threshold, left_indices, right_indices, impurity_decrease)
            Returns None if no valid split found
        """
        n_samples, n_features = X.shape

        if n_samples < self.min_samples_split:
            return None

        # Determine which features to consider
        feature_indices = self._get_feature_indices(n_features)

        best_gain = -np.inf
        best_split = None

        parent_impurity = self._calculate_impurity(y)

        # Try each feature
        for feature_idx in feature_indices:
            # Get unique values for this feature (potential thresholds)
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)

            # Try thresholds between consecutive unique values
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2

                # Split data
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Calculate information gain
                y_left, y_right = y[left_mask], y[right_mask]
                gain, impurity_decrease = self._information_gain(
                    y, y_left, y_right, parent_impurity
                )

                # Update best split
                if gain > best_gain:
                    best_gain = gain
                    left_indices = np.where(left_mask)[0]
                    right_indices = np.where(right_mask)[0]
                    best_split = (
                        feature_idx,
                        threshold,
                        left_indices,
                        right_indices,
                        impurity_decrease
                    )

        return best_split

    def _get_feature_indices(self, n_features):
        """
        Get indices of features to consider for splitting.

        Parameters:
        -----------
        n_features : int
            Total number of features

        Returns:
        --------
        indices : ndarray
            Indices of features to consider
        """
        if self.max_features is None:
            return np.arange(n_features)

        if isinstance(self.max_features, int):
            n_features_to_use = min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            n_features_to_use = max(1, int(self.max_features * n_features))
        elif self.max_features == 'sqrt':
            n_features_to_use = max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            n_features_to_use = max(1, int(np.log2(n_features)))
        else:
            raise ValueError(f"Invalid max_features: {self.max_features}")

        # Randomly select features
        return np.random.choice(n_features, n_features_to_use, replace=False)

    def _calculate_impurity(self, y):
        """
        Calculate impurity of labels using specified criterion.

        Parameters:
        -----------
        y : ndarray
            Labels

        Returns:
        --------
        impurity : float
            Impurity value
        """
        if self.criterion == 'gini':
            return self._gini_impurity(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        elif self.criterion == 'error':
            return self._misclassification_error(y)

    def _gini_impurity(self, y):
        """
        Calculate Gini impurity.

        Gini(D) = 1 - Σ(p_i²)
        where p_i is the proportion of samples belonging to class i

        Parameters:
        -----------
        y : ndarray
            Labels

        Returns:
        --------
        gini : float
            Gini impurity in [0, 1-1/n_classes]
        """
        if len(y) == 0:
            return 0.0

        # Count occurrences of each class
        _, counts = np.unique(y, return_counts=True)

        # Calculate proportions
        proportions = counts / len(y)

        # Gini = 1 - sum of squared proportions
        gini = 1.0 - np.sum(proportions ** 2)

        return gini

    def _entropy(self, y):
        """
        Calculate entropy (information content).

        H(D) = -Σ(p_i * log2(p_i))
        where p_i is the proportion of samples belonging to class i

        Parameters:
        -----------
        y : ndarray
            Labels

        Returns:
        --------
        entropy : float
            Entropy value in [0, log2(n_classes)]
        """
        if len(y) == 0:
            return 0.0

        # Count occurrences of each class
        _, counts = np.unique(y, return_counts=True)

        # Calculate proportions
        proportions = counts / len(y)

        # Remove zero proportions to avoid log(0)
        proportions = proportions[proportions > 0]

        # Entropy = -sum of p * log2(p)
        entropy = -np.sum(proportions * np.log2(proportions))

        return entropy

    def _misclassification_error(self, y):
        """
        Calculate misclassification error.

        Error(D) = 1 - max(p_i)
        where p_i is the proportion of samples belonging to class i

        Parameters:
        -----------
        y : ndarray
            Labels

        Returns:
        --------
        error : float
            Misclassification error in [0, 1-1/n_classes]
        """
        if len(y) == 0:
            return 0.0

        # Count occurrences of each class
        _, counts = np.unique(y, return_counts=True)

        # Calculate proportions
        proportions = counts / len(y)

        # Error = 1 - max proportion
        error = 1.0 - np.max(proportions)

        return error

    def _information_gain(self, parent, left_child, right_child, parent_impurity):
        """
        Calculate information gain from a split.

        IG = H(parent) - [n_left/n * H(left) + n_right/n * H(right)]

        Parameters:
        -----------
        parent : ndarray
            Parent node labels
        left_child : ndarray
            Left child labels
        right_child : ndarray
            Right child labels
        parent_impurity : float
            Pre-calculated parent impurity (for efficiency)

        Returns:
        --------
        gain : float
            Information gain
        impurity_decrease : float
            Decrease in impurity (weighted)
        """
        n_parent = len(parent)
        n_left = len(left_child)
        n_right = len(right_child)

        if n_left == 0 or n_right == 0:
            return 0.0, 0.0

        # Calculate child impurities
        left_impurity = self._calculate_impurity(left_child)
        right_impurity = self._calculate_impurity(right_child)

        # Weighted average of child impurities
        weighted_child_impurity = (
            (n_left / n_parent) * left_impurity +
            (n_right / n_parent) * right_impurity
        )

        # Information gain
        gain = parent_impurity - weighted_child_impurity

        # Impurity decrease (for feature importance)
        impurity_decrease = n_parent * gain

        return gain, impurity_decrease

    def _most_common_class(self, y):
        """
        Find the most common class in labels.

        Parameters:
        -----------
        y : ndarray
            Labels

        Returns:
        --------
        most_common : int
            Most common class label
        """
        if len(y) == 0:
            return self.classes_[0]  # Default to first class

        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]

    def predict(self, X):
        """
        Predict class labels for samples.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict

        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels
        """
        if self.tree_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was fitted with {self.n_features_}"
            )

        predictions = np.array([self._predict_sample(x, self.tree_) for x in X])
        return predictions

    def _predict_sample(self, x, node):
        """
        Predict class for a single sample by traversing the tree.

        Parameters:
        -----------
        x : ndarray of shape (n_features,)
            Single sample
        node : Node
            Current node in traversal

        Returns:
        --------
        prediction : int
            Predicted class label
        """
        # If leaf node, return prediction
        if node.is_leaf():
            return node.value

        # Traverse to left or right child
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def predict_proba(self, X):
        """
        Predict class probabilities for samples.

        The probability is based on the class distribution in the leaf node.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict

        Returns:
        --------
        proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        if self.tree_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was fitted with {self.n_features_}"
            )

        probas = np.array([self._predict_proba_sample(x, self.tree_) for x in X])
        return probas

    def _predict_proba_sample(self, x, node):
        """
        Predict class probabilities for a single sample.

        Parameters:
        -----------
        x : ndarray of shape (n_features,)
            Single sample
        node : Node
            Current node in traversal

        Returns:
        --------
        proba : ndarray of shape (n_classes,)
            Class probabilities
        """
        # If leaf node, return class distribution
        if node.is_leaf():
            proba = np.zeros(self.n_classes_)
            for class_label, count in node.class_counts.items():
                class_idx = np.where(self.classes_ == class_label)[0][0]
                proba[class_idx] = count / node.n_samples
            return proba

        # Traverse to left or right child
        if x[node.feature_index] <= node.threshold:
            return self._predict_proba_sample(x, node.left)
        else:
            return self._predict_proba_sample(x, node.right)

    def score(self, X, y):
        """
        Return the accuracy score.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples,)
            True labels

        Returns:
        --------
        accuracy : float
            Accuracy score in [0, 1]
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def get_depth(self):
        """
        Get the maximum depth of the tree.

        Returns:
        --------
        depth : int
            Maximum depth of the tree
        """
        if self.tree_ is None:
            return 0
        return self._get_depth_recursive(self.tree_)

    def _get_depth_recursive(self, node):
        """Recursively calculate tree depth."""
        if node.is_leaf():
            return node.depth

        left_depth = self._get_depth_recursive(node.left)
        right_depth = self._get_depth_recursive(node.right)

        return max(left_depth, right_depth)

    def get_n_leaves(self):
        """
        Get the number of leaf nodes in the tree.

        Returns:
        --------
        n_leaves : int
            Number of leaf nodes
        """
        if self.tree_ is None:
            return 0
        return self._count_leaves(self.tree_)

    def _count_leaves(self, node):
        """Recursively count leaf nodes."""
        if node.is_leaf():
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def _prune_tree(self, node):
        """
        Apply cost-complexity pruning to the tree.

        This is a simplified version that prunes nodes with low impurity decrease.

        Parameters:
        -----------
        node : Node
            Node to potentially prune
        """
        if node.is_leaf():
            return

        # Recursively prune children
        self._prune_tree(node.left)
        self._prune_tree(node.right)

        # Check if both children are leaves
        if node.left.is_leaf() and node.right.is_leaf():
            # Calculate cost-complexity criterion
            # If the impurity decrease is less than ccp_alpha, prune
            total_samples = node.left.n_samples + node.right.n_samples
            weighted_impurity = (
                node.left.n_samples / total_samples * node.left.impurity +
                node.right.n_samples / total_samples * node.right.impurity
            )

            impurity_decrease = node.impurity - weighted_impurity

            if impurity_decrease < self.ccp_alpha:
                # Convert to leaf
                node.value = self._most_common_class_from_counts(node.class_counts)
                node.left = None
                node.right = None
                node.feature_index = None
                node.threshold = None

    def _most_common_class_from_counts(self, class_counts):
        """Get most common class from class_counts dict."""
        return max(class_counts.items(), key=lambda x: x[1])[0]

    def print_tree(self, node=None, depth=0, prefix="Root: "):
        """
        Print a text representation of the tree.

        Parameters:
        -----------
        node : Node or None
            Node to print (None for root)
        depth : int
            Current depth (for indentation)
        prefix : str
            Prefix for the line
        """
        if self.tree_ is None:
            print("Tree not fitted yet.")
            return

        if node is None:
            node = self.tree_

        indent = "  " * depth

        if node.is_leaf():
            # Leaf node
            class_dist = ", ".join([f"class {k}: {v}" for k, v in node.class_counts.items()])
            print(f"{indent}{prefix}Predict class {node.value} "
                  f"(samples={node.n_samples}, impurity={node.impurity:.4f})")
            print(f"{indent}     [{class_dist}]")
        else:
            # Internal node
            print(f"{indent}{prefix}X[{node.feature_index}] <= {node.threshold:.4f} "
                  f"(samples={node.n_samples}, impurity={node.impurity:.4f})")

            # Print children
            if node.left is not None:
                self.print_tree(node.left, depth + 1, "|- True:  ")
            if node.right is not None:
                self.print_tree(node.right, depth + 1, "|_ False: ")

    def get_params(self):
        """Get parameters for this estimator."""
        return {
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'ccp_alpha': self.ccp_alpha
        }

    def __repr__(self):
        """String representation of the classifier."""
        params = self.get_params()
        params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        return f"DecisionTreeClassifier({params_str})"
