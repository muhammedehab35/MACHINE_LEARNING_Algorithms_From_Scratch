"""
AdaBoost (Adaptive Boosting) - From Scratch Implementation

An ensemble method that combines multiple weak learners (decision stumps)
into a strong classifier using adaptive sample weighting.

Key Concepts:
- Weak learners: Decision stumps (single-feature threshold classifiers)
- Adaptive weighting: Misclassified samples get higher weights each round
- Weighted voting: Each stump votes proportionally to its accuracy (alpha)
- Sequential training: Each stump corrects the errors of previous ones
"""

import numpy as np
from typing import Optional, List


class DecisionStump:
    """
    Decision Stump — single-level decision tree used as weak learner.

    Finds the best (feature, threshold, polarity) to minimize weighted error.
    Polarity handles both <= and > orientations.
    """

    def __init__(self):
        self.feature_index: Optional[int] = None
        self.threshold: Optional[float] = None
        self.polarity: int = 1      # +1 or -1
        self.alpha: float = 0.0     # stump weight in the ensemble

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions in {-1, +1}"""
        n_samples = X.shape[0]
        col = X[:, self.feature_index]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[col < self.threshold] = -1
        else:
            predictions[col >= self.threshold] = -1
        return predictions


class AdaBoostClassifier:
    """
    AdaBoost Binary Classifier

    Sequentially trains decision stumps. After each stump, sample weights
    are updated so that misclassified samples get higher focus next round.
    Final prediction is a weighted majority vote of all stumps.

    Parameters
    ----------
    n_estimators : int, default=50
        Number of boosting rounds (weak learners)
    learning_rate : float, default=1.0
        Shrinks the contribution of each stump (trade-off with n_estimators)
    random_state : int, optional
        Seed for reproducibility

    Attributes
    ----------
    stumps_ : list of DecisionStump
        Trained weak learners
    alphas_ : list of float
        Weight of each stump in the ensemble
    classes_ : ndarray of shape (2,)
        Original class labels
    feature_importances_ : ndarray of shape (n_features,)
        Sum of |alpha| for each feature, normalized to sum to 1
    training_errors_ : list of float
        Weighted error at each boosting iteration
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state: Optional[int] = None
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.stumps_: List[DecisionStump] = []
        self.alphas_: List[float] = []
        self.classes_: Optional[np.ndarray] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.training_errors_: List[float] = []
        self.n_features_: Optional[int] = None

        if random_state is not None:
            np.random.seed(random_state)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaBoostClassifier':
        """
        Fit AdaBoost classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,) — binary labels (any two values)
        """
        X = np.array(X, dtype=float)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.classes_ = np.unique(y)

        # Encode labels as {-1, +1}
        y_enc = self._encode(y)

        # Uniform initial weights
        w = np.full(n_samples, 1.0 / n_samples)

        importance_acc = np.zeros(n_features)
        self.stumps_.clear()
        self.alphas_.clear()
        self.training_errors_.clear()

        for _ in range(self.n_estimators):
            stump = self._fit_stump(X, y_enc, w)

            # Weighted error on this stump
            preds = stump.predict(X)
            error = np.sum(w[y_enc != preds])
            error = np.clip(error, 1e-10, 1 - 1e-10)

            # Stump weight
            alpha = self.learning_rate * 0.5 * np.log((1 - error) / error)
            stump.alpha = alpha

            # Update sample weights
            w *= np.exp(-alpha * y_enc * preds)
            w /= w.sum()

            importance_acc[stump.feature_index] += abs(alpha)
            self.stumps_.append(stump)
            self.alphas_.append(alpha)
            self.training_errors_.append(error)

        total = importance_acc.sum()
        self.feature_importances_ = (
            importance_acc / total if total > 0 else importance_acc
        )
        return self

    def _fit_stump(
        self,
        X: np.ndarray,
        y_enc: np.ndarray,
        w: np.ndarray
    ) -> DecisionStump:
        """Find the weighted-error-minimizing stump."""
        n_samples, n_features = X.shape
        best_stump = DecisionStump()
        min_error = float('inf')

        for feat in range(n_features):
            col = X[:, feat]
            thresholds = np.unique(col)

            for thresh in thresholds:
                for polarity in (1, -1):
                    preds = np.ones(n_samples)
                    if polarity == 1:
                        preds[col < thresh] = -1
                    else:
                        preds[col >= thresh] = -1

                    err = np.sum(w[y_enc != preds])

                    if err < min_error:
                        min_error = err
                        best_stump.feature_index = feat
                        best_stump.threshold = thresh
                        best_stump.polarity = polarity

        return best_stump

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Weighted sum of stump predictions (raw scores)."""
        X = np.array(X, dtype=float)
        scores = np.zeros(X.shape[0])
        for stump in self.stumps_:
            scores += stump.alpha * stump.predict(X)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        scores = self.decision_function(X)
        y_enc = np.sign(scores)
        y_enc[y_enc == 0] = 1
        return self._decode(y_enc)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities via sigmoid of decision function.

        Returns ndarray of shape (n_samples, 2)
        """
        scores = self.decision_function(X)
        prob_pos = 1.0 / (1.0 + np.exp(-2.0 * scores))
        return np.column_stack([1 - prob_pos, prob_pos])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy score."""
        return float(np.mean(self.predict(X) == np.array(y)))

    # ------------------------------------------------------------------
    # Staged predictions (for convergence analysis)
    # ------------------------------------------------------------------

    def staged_predict(self, X: np.ndarray):
        """Yield predictions after each boosting stage."""
        X = np.array(X, dtype=float)
        scores = np.zeros(X.shape[0])
        for stump in self.stumps_:
            scores += stump.alpha * stump.predict(X)
            y_enc = np.sign(scores)
            y_enc[y_enc == 0] = 1
            yield self._decode(y_enc)

    def staged_score(self, X: np.ndarray, y: np.ndarray):
        """Yield accuracy after each boosting stage."""
        y = np.array(y)
        for preds in self.staged_predict(X):
            yield float(np.mean(preds == y))

    # ------------------------------------------------------------------
    # Label helpers
    # ------------------------------------------------------------------

    def _encode(self, y: np.ndarray) -> np.ndarray:
        """Map original labels -> {-1, +1}."""
        pos = self.classes_[-1]
        return np.where(y == pos, 1, -1).astype(float)

    def _decode(self, y_enc: np.ndarray) -> np.ndarray:
        """Map {-1, +1} -> original labels."""
        pos, neg = self.classes_[-1], self.classes_[0]
        return np.where(y_enc == 1, pos, neg)
