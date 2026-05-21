"""
Random Forest Implementation from Scratch

A Random Forest is an ensemble learning method that constructs multiple decision trees
during training and outputs the mode (classification) or mean (regression) of the
predictions from individual trees.

Key Concepts:
1. Bootstrap Aggregating (Bagging): Each tree is trained on a random sample with replacement
2. Feature Randomness: Each split considers a random subset of features
3. Ensemble Voting: Final prediction is aggregated from all trees
4. Out-of-Bag (OOB) Score: Validation using samples not in bootstrap sample

Mathematical Foundation:
- For classification: y_pred = mode({tree_i.predict(x) for i in 1..n_trees})
- For regression: y_pred = mean({tree_i.predict(x) for i in 1..n_trees})
- OOB Error: Error on samples not used in training each tree
- Feature Importance: Weighted average of feature importance across all trees

Author: ML Algorithms from Scratch
Date: March 2026
"""

import numpy as np
from typing import Optional, Union, Literal
import sys
import os

# Add parent directory to path to import DecisionTree
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '09_DECISION_TREE'))
from decision_tree import DecisionTreeClassifier
from decision_tree_regressor import DecisionTreeRegressor


class RandomForestClassifier:
    """
    Random Forest Classifier

    An ensemble of decision trees trained on bootstrap samples with feature randomness.
    Combines predictions through majority voting.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest

    criterion : {'gini', 'entropy'}, default='gini'
        The function to measure split quality

    max_depth : int or None, default=None
        Maximum depth of the trees. None means unlimited

    min_samples_split : int, default=2
        Minimum samples required to split a node

    min_samples_leaf : int, default=1
        Minimum samples required in a leaf node

    max_features : {'sqrt', 'log2', None} or int or float, default='sqrt'
        Number of features to consider for best split:
        - 'sqrt': sqrt(n_features)
        - 'log2': log2(n_features)
        - None: n_features
        - int: exact number
        - float: fraction of n_features

    bootstrap : bool, default=True
        Whether to use bootstrap samples

    oob_score : bool, default=False
        Whether to compute out-of-bag score

    random_state : int or None, default=None
        Random seed for reproducibility

    n_jobs : int or None, default=None
        Number of parallel jobs (not implemented, for API compatibility)

    Attributes
    ----------
    trees_ : list of DecisionTreeClassifier
        The collection of fitted trees

    n_features_ : int
        Number of features

    n_classes_ : int
        Number of classes

    classes_ : ndarray
        Unique class labels

    feature_importances_ : ndarray
        Feature importances aggregated across trees

    oob_score_ : float
        Out-of-bag score (only if oob_score=True)

    oob_decision_function_ : ndarray
        OOB predictions for training samples

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    >>> rf = RandomForestClassifier(n_estimators=100, random_state=42)
    >>> rf.fit(X_train, y_train)
    >>> accuracy = rf.score(X_test, y_test)
    >>> print(f"Accuracy: {accuracy:.3f}")
    """

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: Literal['gini', 'entropy'] = 'gini',
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[Literal['sqrt', 'log2'], int, float, None] = 'sqrt',
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.trees_ = []
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None
        self.feature_importances_ = None
        self.oob_score_ = None
        self.oob_decision_function_ = None
        self._oob_samples = []  # Track which samples are OOB for each tree

    def _get_max_features(self, n_features: int) -> int:
        """Calculate the number of features to consider at each split"""
        if self.max_features is None:
            return n_features
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        else:
            raise ValueError(f"Invalid max_features: {self.max_features}")

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
        """Create a bootstrap sample of the data"""
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, size=n_samples, replace=True)

        # Track out-of-bag samples
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[indices] = False
        oob_indices = np.where(oob_mask)[0]

        return X[indices], y[indices], oob_indices

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifier':
        """
        Build a forest of trees from the training data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data

        y : ndarray of shape (n_samples,)
            Target values

        Returns
        -------
        self : RandomForestClassifier
            Fitted classifier
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if y.ndim != 1:
            raise ValueError("y must be 1-dimensional")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Initialize random number generator
        rng = np.random.default_rng(self.random_state)

        # Calculate max_features for trees
        max_features = self._get_max_features(n_features)

        # Initialize OOB tracking if needed
        if self.oob_score:
            oob_predictions = np.zeros((n_samples, self.n_classes_))
            oob_counts = np.zeros(n_samples)

        # Build trees
        self.trees_ = []
        self._oob_samples = []

        for i in range(self.n_estimators):
            # Create bootstrap sample
            if self.bootstrap:
                X_sample, y_sample, oob_indices = self._bootstrap_sample(X, y, rng)
            else:
                X_sample, y_sample = X, y
                oob_indices = np.array([])

            # Create and train tree
            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=rng.integers(0, 2**31 - 1)
            )
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)
            self._oob_samples.append(oob_indices)

            # Update OOB predictions
            if self.oob_score and len(oob_indices) > 0:
                oob_proba = tree.predict_proba(X[oob_indices])
                oob_predictions[oob_indices] += oob_proba
                oob_counts[oob_indices] += 1

        # Compute feature importances
        self._compute_feature_importances()

        # Compute OOB score
        if self.oob_score:
            # Average OOB predictions
            valid_oob = oob_counts > 0
            if not np.any(valid_oob):
                raise ValueError("No out-of-bag samples found. Increase n_estimators or set bootstrap=True")

            oob_predictions[valid_oob] /= oob_counts[valid_oob][:, np.newaxis]
            self.oob_decision_function_ = oob_predictions

            # Compute OOB score
            oob_pred_classes = self.classes_[np.argmax(oob_predictions[valid_oob], axis=1)]
            self.oob_score_ = np.mean(oob_pred_classes == y[valid_oob])

        return self

    def _compute_feature_importances(self):
        """Compute feature importances as average over all trees"""
        importances = np.zeros(self.n_features_)

        for tree in self.trees_:
            if hasattr(tree, 'feature_importances_') and tree.feature_importances_ is not None:
                importances += tree.feature_importances_

        # Normalize
        importances /= len(self.trees_)

        # Normalize to sum to 1
        if importances.sum() > 0:
            importances /= importances.sum()

        self.feature_importances_ = importances

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test data

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities
        """
        if not self.trees_:
            raise ValueError("Forest not fitted. Call fit() first")

        X = np.asarray(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_}")

        # Average predictions from all trees
        all_proba = np.array([tree.predict_proba(X) for tree in self.trees_])
        return np.mean(all_proba, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test data

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return accuracy score

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test data

        y : ndarray of shape (n_samples,)
            True labels

        Returns
        -------
        score : float
            Accuracy score
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class RandomForestRegressor:
    """
    Random Forest Regressor

    An ensemble of decision trees trained on bootstrap samples with feature randomness.
    Combines predictions through averaging.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest

    max_depth : int or None, default=None
        Maximum depth of the trees

    min_samples_split : int, default=2
        Minimum samples required to split a node

    min_samples_leaf : int, default=1
        Minimum samples required in a leaf node

    max_features : {'sqrt', 'log2', None} or int or float, default='sqrt'
        Number of features to consider for best split

    bootstrap : bool, default=True
        Whether to use bootstrap samples

    oob_score : bool, default=False
        Whether to compute out-of-bag score

    random_state : int or None, default=None
        Random seed for reproducibility

    Attributes
    ----------
    trees_ : list of DecisionTreeRegressor
        The collection of fitted trees

    n_features_ : int
        Number of features

    feature_importances_ : ndarray
        Feature importances

    oob_score_ : float
        Out-of-bag R² score

    oob_prediction_ : ndarray
        OOB predictions for training samples
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[Literal['sqrt', 'log2'], int, float, None] = 'sqrt',
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.trees_ = []
        self.n_features_ = None
        self.feature_importances_ = None
        self.oob_score_ = None
        self.oob_prediction_ = None
        self._oob_samples = []

    def _get_max_features(self, n_features: int) -> int:
        """Calculate the number of features to consider at each split"""
        if self.max_features is None:
            return n_features
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        else:
            raise ValueError(f"Invalid max_features: {self.max_features}")

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
        """Create a bootstrap sample of the data"""
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, size=n_samples, replace=True)

        # Track out-of-bag samples
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[indices] = False
        oob_indices = np.where(oob_mask)[0]

        return X[indices], y[indices], oob_indices

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestRegressor':
        """
        Build a forest of trees from the training data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data

        y : ndarray of shape (n_samples,)
            Target values

        Returns
        -------
        self : RandomForestRegressor
            Fitted regressor
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if y.ndim != 1:
            raise ValueError("y must be 1-dimensional")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # Initialize random number generator
        rng = np.random.default_rng(self.random_state)

        # Calculate max_features for trees
        max_features = self._get_max_features(n_features)

        # Initialize OOB tracking if needed
        if self.oob_score:
            oob_predictions = np.zeros(n_samples)
            oob_counts = np.zeros(n_samples)

        # Build trees
        self.trees_ = []
        self._oob_samples = []

        for i in range(self.n_estimators):
            # Create bootstrap sample
            if self.bootstrap:
                X_sample, y_sample, oob_indices = self._bootstrap_sample(X, y, rng)
            else:
                X_sample, y_sample = X, y
                oob_indices = np.array([])

            # Create and train tree
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=rng.integers(0, 2**31 - 1)
            )
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)
            self._oob_samples.append(oob_indices)

            # Update OOB predictions
            if self.oob_score and len(oob_indices) > 0:
                oob_pred = tree.predict(X[oob_indices])
                oob_predictions[oob_indices] += oob_pred
                oob_counts[oob_indices] += 1

        # Compute feature importances
        self._compute_feature_importances()

        # Compute OOB score
        if self.oob_score:
            # Average OOB predictions
            valid_oob = oob_counts > 0
            if not np.any(valid_oob):
                raise ValueError("No out-of-bag samples found")

            oob_predictions[valid_oob] /= oob_counts[valid_oob]
            self.oob_prediction_ = oob_predictions

            # Compute OOB R² score
            y_oob = y[valid_oob]
            pred_oob = oob_predictions[valid_oob]
            ss_res = np.sum((y_oob - pred_oob) ** 2)
            ss_tot = np.sum((y_oob - np.mean(y_oob)) ** 2)
            self.oob_score_ = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return self

    def _compute_feature_importances(self):
        """Compute feature importances as average over all trees"""
        importances = np.zeros(self.n_features_)

        for tree in self.trees_:
            if hasattr(tree, 'feature_importances_') and tree.feature_importances_ is not None:
                importances += tree.feature_importances_

        # Normalize
        importances /= len(self.trees_)

        # Normalize to sum to 1
        if importances.sum() > 0:
            importances /= importances.sum()

        self.feature_importances_ = importances

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regression targets

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test data

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        if not self.trees_:
            raise ValueError("Forest not fitted. Call fit() first")

        X = np.asarray(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_}")

        # Average predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees_])
        return np.mean(predictions, axis=0)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return R² score

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test data

        y : ndarray of shape (n_samples,)
            True values

        Returns
        -------
        score : float
            R² score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


if __name__ == "__main__":
    print("Testing Random Forest Implementation")
    print("=" * 60)

    # Test RandomForestClassifier
    print("\n1. Random Forest Classifier Test")
    print("-" * 60)
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, oob_score=True)
    rf_clf.fit(X_train, y_train)

    train_acc = rf_clf.score(X_train, y_train)
    test_acc = rf_clf.score(X_test, y_test)

    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    print(f"OOB score: {rf_clf.oob_score_:.3f}")
    print(f"Number of trees: {len(rf_clf.trees_)}")
    print(f"Feature importances: {rf_clf.feature_importances_}")

    # Test RandomForestRegressor
    print("\n2. Random Forest Regressor Test")
    print("-" * 60)
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=200, n_features=4, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, oob_score=True)
    rf_reg.fit(X_train, y_train)

    train_r2 = rf_reg.score(X_train, y_train)
    test_r2 = rf_reg.score(X_test, y_test)

    print(f"Training R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    print(f"OOB score: {rf_reg.oob_score_:.3f}")
    print(f"Number of trees: {len(rf_reg.trees_)}")
    print(f"Feature importances: {rf_reg.feature_importances_}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
