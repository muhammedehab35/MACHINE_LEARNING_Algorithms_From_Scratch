"""
Gradient Boosting Machine (GBM) - From Scratch Implementation

An ensemble method that builds models sequentially, where each new model
corrects the errors of the previous models by fitting to the residuals.

Key Concepts:
- Sequential ensemble learning
- Gradient descent in function space
- Residual fitting
- Shrinkage (learning rate)
- Multiple loss functions support
"""

import numpy as np
from typing import Optional, Literal
import sys
import os

# Import DecisionTree for base learners
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '09_DECISION_TREE'))
from decision_tree import DecisionTreeClassifier

# Import DecisionTreeRegressor from Random Forest folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '10_RANDOM_FOREST'))
from decision_tree_regressor import DecisionTreeRegressor


class GradientBoostingClassifier:
    """
    Gradient Boosting Classifier

    Builds an additive model in a forward stage-wise fashion by optimizing
    a differentiable loss function. Each new tree fits the negative gradient
    of the loss function.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting stages (trees) to perform

    learning_rate : float, default=0.1
        Shrinks the contribution of each tree. Lower values need more trees

    max_depth : int, default=3
        Maximum depth of the individual regression trees

    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node

    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node

    subsample : float, default=1.0
        Fraction of samples to be used for fitting individual base learners
        If < 1.0, results in Stochastic Gradient Boosting

    loss : {'log_loss', 'exponential'}, default='log_loss'
        Loss function to be optimized

    random_state : int, default=None
        Random seed for reproducibility

    verbose : int, default=0
        Controls the verbosity when fitting
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        loss: Literal['log_loss', 'exponential'] = 'log_loss',
        random_state: Optional[int] = None,
        verbose: int = 0
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.loss = loss
        self.random_state = random_state
        self.verbose = verbose

        self.trees_ = []
        self.init_prediction_ = None
        self.n_classes_ = None
        self.classes_ = None

        if random_state is not None:
            np.random.seed(random_state)

    def _log_loss_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute gradient of log loss (negative gradient for gradient descent)

        For binary classification with log loss:
        L(y, F) = -y*log(p) - (1-y)*log(1-p)
        where p = sigmoid(F)

        Gradient: dL/dF = p - y = sigmoid(F) - y
        Negative gradient: y - sigmoid(F)
        """
        prob = 1 / (1 + np.exp(-y_pred))
        return y_true - prob

    def _exponential_loss_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute gradient of exponential loss (AdaBoost loss)

        L(y, F) = exp(-y*F) where y in {-1, 1}
        Gradient: -y*exp(-y*F)
        Negative gradient: y*exp(-y*F)
        """
        # Convert {0, 1} to {-1, 1}
        y_signed = 2 * y_true - 1
        return y_signed * np.exp(-y_signed * y_pred)

    def _compute_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute negative gradient based on loss function"""
        if self.loss == 'log_loss':
            return self._log_loss_gradient(y_true, y_pred)
        elif self.loss == 'exponential':
            return self._exponential_loss_gradient(y_true, y_pred)
        else:
            raise ValueError(f"Unknown loss: {self.loss}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the gradient boosting model

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data

        y : array-like of shape (n_samples,)
            Target values (class labels)

        Returns
        -------
        self : object
            Fitted estimator
        """
        X = np.array(X)
        y = np.array(y)

        # Get unique classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ > 2:
            raise NotImplementedError("Multi-class classification not yet implemented")

        # Convert labels to {0, 1}
        y_binary = (y == self.classes_[1]).astype(float)

        # Initialize with constant prediction (log odds for binary classification)
        pos_count = np.sum(y_binary)
        neg_count = len(y_binary) - pos_count

        # Avoid log(0)
        if pos_count == 0 or neg_count == 0:
            self.init_prediction_ = 0.0
        else:
            # Log odds: log(p / (1-p))
            self.init_prediction_ = np.log(pos_count / neg_count)

        # Initialize predictions (in logit space)
        F = np.full(len(y_binary), self.init_prediction_)

        # Build trees sequentially
        self.trees_ = []
        n_samples = len(X)

        for i in range(self.n_estimators):
            # Compute negative gradient (pseudo-residuals)
            residuals = self._compute_gradient(y_binary, F)

            # Subsample if needed (Stochastic Gradient Boosting)
            if self.subsample < 1.0:
                sample_size = int(self.subsample * n_samples)
                indices = np.random.choice(n_samples, sample_size, replace=False)
                X_sample = X[indices]
                residuals_sample = residuals[indices]
            else:
                X_sample = X
                residuals_sample = residuals

            # Fit a tree to the negative gradient
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
            tree.fit(X_sample, residuals_sample)

            # Update predictions with learning rate
            tree_predictions = tree.predict(X)
            F += self.learning_rate * tree_predictions

            self.trees_.append(tree)

            if self.verbose > 0 and (i + 1) % 10 == 0:
                # Compute training accuracy for monitoring
                probs = 1 / (1 + np.exp(-F))
                y_pred = (probs > 0.5).astype(int)
                acc = np.mean(y_pred == y_binary)
                print(f"Iteration {i + 1}/{self.n_estimators}, Train Accuracy: {acc:.4f}")

        return self

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """
        Predict raw values (in logit space before applying sigmoid)

        F(x) = F_0 + learning_rate * sum(tree_i(x))
        """
        if not self.trees_:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.array(X)

        # Start with initial prediction
        F = np.full(len(X), self.init_prediction_)

        # Add contributions from all trees
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)

        return F

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities
        """
        # Get raw predictions (logits)
        F = self._predict_raw(X)

        # Apply sigmoid to get probabilities
        prob_class_1 = 1 / (1 + np.exp(-F))
        prob_class_0 = 1 - prob_class_1

        return np.column_stack([prob_class_0, prob_class_1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples

        y : array-like of shape (n_samples,)
            True labels

        Returns
        -------
        score : float
            Accuracy score
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def staged_predict_proba(self, X: np.ndarray):
        """
        Generator that yields predictions after each boosting iteration

        Useful for early stopping and monitoring
        """
        X = np.array(X)
        F = np.full(len(X), self.init_prediction_)

        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
            prob_class_1 = 1 / (1 + np.exp(-F))
            prob_class_0 = 1 - prob_class_1
            yield np.column_stack([prob_class_0, prob_class_1])

    def staged_predict(self, X: np.ndarray):
        """
        Generator that yields class predictions after each boosting iteration
        """
        for proba in self.staged_predict_proba(X):
            yield self.classes_[np.argmax(proba, axis=1)]


class GradientBoostingRegressor:
    """
    Gradient Boosting Regressor

    Builds an additive model by fitting regression trees to pseudo-residuals.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting stages

    learning_rate : float, default=0.1
        Shrinkage parameter

    max_depth : int, default=3
        Maximum depth of trees

    min_samples_split : int, default=2
        Minimum samples to split

    min_samples_leaf : int, default=1
        Minimum samples in leaf

    subsample : float, default=1.0
        Fraction of samples for fitting

    loss : {'squared_error', 'absolute_error', 'huber'}, default='squared_error'
        Loss function

    alpha : float, default=0.9
        Quantile for Huber loss

    random_state : int, default=None
        Random seed

    verbose : int, default=0
        Verbosity level
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        loss: Literal['squared_error', 'absolute_error', 'huber'] = 'squared_error',
        alpha: float = 0.9,
        random_state: Optional[int] = None,
        verbose: int = 0
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.loss = loss
        self.alpha = alpha
        self.random_state = random_state
        self.verbose = verbose

        self.trees_ = []
        self.init_prediction_ = None

        if random_state is not None:
            np.random.seed(random_state)

    def _squared_error_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Gradient for squared error: -(y - y_pred)"""
        return y_true - y_pred

    def _absolute_error_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Gradient for absolute error: -sign(y - y_pred)"""
        residuals = y_true - y_pred
        return np.sign(residuals)

    def _huber_loss_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Gradient for Huber loss"""
        residuals = y_true - y_pred
        abs_residuals = np.abs(residuals)
        delta = np.quantile(abs_residuals, self.alpha)

        gradient = np.where(
            abs_residuals <= delta,
            residuals,  # L2 region
            delta * np.sign(residuals)  # L1 region
        )
        return gradient

    def _compute_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute negative gradient based on loss"""
        if self.loss == 'squared_error':
            return self._squared_error_gradient(y_true, y_pred)
        elif self.loss == 'absolute_error':
            return self._absolute_error_gradient(y_true, y_pred)
        elif self.loss == 'huber':
            return self._huber_loss_gradient(y_true, y_pred)
        else:
            raise ValueError(f"Unknown loss: {self.loss}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the gradient boosting regressor"""
        X = np.array(X)
        y = np.array(y)

        # Initialize with mean prediction
        self.init_prediction_ = np.mean(y)
        F = np.full(len(y), self.init_prediction_)

        self.trees_ = []
        n_samples = len(X)

        for i in range(self.n_estimators):
            # Compute negative gradient
            residuals = self._compute_gradient(y, F)

            # Subsample if needed
            if self.subsample < 1.0:
                sample_size = int(self.subsample * n_samples)
                indices = np.random.choice(n_samples, sample_size, replace=False)
                X_sample = X[indices]
                residuals_sample = residuals[indices]
            else:
                X_sample = X
                residuals_sample = residuals

            # Fit tree to residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
            tree.fit(X_sample, residuals_sample)

            # Update predictions
            tree_predictions = tree.predict(X)
            F += self.learning_rate * tree_predictions

            self.trees_.append(tree)

            if self.verbose > 0 and (i + 1) % 10 == 0:
                mse = np.mean((y - F) ** 2)
                print(f"Iteration {i + 1}/{self.n_estimators}, MSE: {mse:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regression target"""
        if not self.trees_:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.array(X)
        F = np.full(len(X), self.init_prediction_)

        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)

        return F

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

    def staged_predict(self, X: np.ndarray):
        """Generator yielding predictions after each iteration"""
        X = np.array(X)
        F = np.full(len(X), self.init_prediction_)

        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
            yield F.copy()
