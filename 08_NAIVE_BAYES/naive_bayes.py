"""
Naive Bayes Classifiers Implementation from Scratch
===================================================

Implements three variants of Naive Bayes classifiers:
1. GaussianNB: For continuous features with Gaussian distribution
2. MultinomialNB: For discrete/count features (e.g., text classification)
3. BernoulliNB: For binary features (e.g., document classification)

Mathematical Foundation:
-----------------------
Bayes' Theorem:
P(y|X) = P(X|y) * P(y) / P(X)

For classification, we compute:
y_pred = argmax_y [P(y) * ∏ P(x_i|y)]

Where:
- P(y) is the class prior probability
- P(x_i|y) is the likelihood of feature x_i given class y
- We assume features are conditionally independent given the class (naive assumption)

To avoid numerical underflow, we work in log space:
log P(y|X) = log P(y) + ∑ log P(x_i|y)
"""

import numpy as np
from typing import Optional, Union


class GaussianNB:
    """
    Gaussian Naive Bayes classifier for continuous features.

    Assumes features follow a Gaussian (normal) distribution within each class.

    Mathematical Model:
    ------------------
    For continuous features, the likelihood is modeled using Gaussian PDF:

    P(x_i|y) = (1 / √(2πσ²_y)) * exp(-(x_i - μ_y)² / (2σ²_y))

    Where:
    - μ_y is the mean of feature x_i for class y
    - σ²_y is the variance of feature x_i for class y

    Parameters
    ----------
    var_smoothing : float, default=1e-9
        Portion of the largest variance of all features that is added to
        variances for calculation stability (to avoid division by zero).

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier
    class_prior_ : ndarray of shape (n_classes,)
        Probability of each class P(y)
    theta_ : ndarray of shape (n_classes, n_features)
        Mean of each feature per class
    var_ : ndarray of shape (n_classes, n_features)
        Variance of each feature per class
    epsilon_ : float
        Absolute additive value to variances
    n_features_ : int
        Number of features seen during fit
    """

    def __init__(self, var_smoothing: float = 1e-9):
        """
        Initialize Gaussian Naive Bayes classifier.

        Parameters
        ----------
        var_smoothing : float, default=1e-9
            Variance smoothing parameter to ensure numerical stability
        """
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.class_prior_ = None
        self.theta_ = None
        self.var_ = None
        self.epsilon_ = None
        self.n_features_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNB':
        """
        Fit Gaussian Naive Bayes classifier.

        Computes:
        1. Class priors: P(y) = count(y) / n_samples
        2. Feature means per class: μ_y,i = mean(x_i | y)
        3. Feature variances per class: σ²_y,i = var(x_i | y)

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training vectors
        y : ndarray of shape (n_samples,)
            Target values

        Returns
        -------
        self : object
            Returns the instance itself
        """
        # Input validation
        X, y = self._validate_input(X, y)

        # Get unique classes and their counts
        self.classes_, y_counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # Calculate class priors: P(y)
        self.class_prior_ = y_counts / n_samples

        # Initialize arrays for means and variances
        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))

        # Calculate mean and variance for each class
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]  # Select samples belonging to class c
            self.theta_[idx, :] = X_c.mean(axis=0)
            self.var_[idx, :] = X_c.var(axis=0)

        # Add smoothing to variances to ensure numerical stability
        # epsilon = var_smoothing * max(var across all features)
        self.epsilon_ = self.var_smoothing * np.var(X, axis=0).max()
        self.var_ += self.epsilon_

        return self

    def _gaussian_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log likelihood using Gaussian PDF.

        Log of Gaussian PDF:
        log P(x_i|y) = -0.5 * [log(2π) + log(σ²) + (x_i - μ)² / σ²]

        For numerical stability, we compute in log space.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples

        Returns
        -------
        log_likelihood : ndarray of shape (n_samples, n_classes)
            Log likelihood of each sample for each class
        """
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # Initialize log likelihood matrix
        log_likelihood = np.zeros((n_samples, n_classes))

        for idx in range(n_classes):
            # For each class, compute log likelihood for all samples
            # log P(X|y) = ∑ log P(x_i|y) across all features

            # Vectorized computation:
            # (X - μ)² / σ²
            diff = X - self.theta_[idx, :]
            squared_diff = diff ** 2
            normalized = squared_diff / self.var_[idx, :]

            # -0.5 * [log(2πσ²) + (x - μ)² / σ²]
            log_likelihood[:, idx] = -0.5 * (
                np.sum(np.log(2 * np.pi * self.var_[idx, :])) +
                np.sum(normalized, axis=1)
            )

        return log_likelihood

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log posterior probabilities.

        log P(y|X) = log P(y) + log P(X|y)
                   = log P(y) + ∑ log P(x_i|y)

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples

        Returns
        -------
        log_proba : ndarray of shape (n_samples, n_classes)
            Log posterior probabilities for each class
        """
        self._check_is_fitted()
        X = self._validate_input(X)

        # log P(y)
        log_prior = np.log(self.class_prior_)

        # log P(X|y)
        log_likelihood = self._gaussian_log_likelihood(X)

        # log P(y|X) = log P(y) + log P(X|y)
        log_posterior = log_prior + log_likelihood

        return log_posterior

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute posterior probabilities.

        Converts log probabilities to probabilities and normalizes:
        P(y|X) = exp(log P(y|X)) / ∑ exp(log P(y'|X))

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Posterior probabilities for each class
        """
        log_proba = self.predict_log_proba(X)

        # Use log-sum-exp trick for numerical stability
        # P = exp(log_proba - log_sum_exp(log_proba))
        log_sum = self._logsumexp(log_proba, axis=1, keepdims=True)
        proba = np.exp(log_proba - log_sum)

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Returns the class with highest posterior probability:
        y_pred = argmax_y P(y|X)

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels
        """
        log_proba = self.predict_log_proba(X)
        # Return class with highest log probability
        return self.classes_[np.argmax(log_proba, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples,)
            True labels

        Returns
        -------
        score : float
            Accuracy score (fraction of correct predictions)
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def _validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Validate and convert input arrays."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")

        if y is not None:
            y = np.asarray(y).ravel()
            if X.shape[0] != len(y):
                raise ValueError(
                    f"X and y have inconsistent numbers of samples: "
                    f"{X.shape[0]} != {len(y)}"
                )
            return X, y

        return X

    def _check_is_fitted(self):
        """Check if the classifier has been fitted."""
        if self.classes_ is None:
            raise ValueError(
                "This GaussianNB instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

    @staticmethod
    def _logsumexp(arr: np.ndarray, axis: int = None, keepdims: bool = False) -> np.ndarray:
        """
        Compute log(sum(exp(arr))) in a numerically stable way.

        Uses the log-sum-exp trick:
        log(∑ exp(x_i)) = c + log(∑ exp(x_i - c))
        where c = max(x_i)
        """
        arr_max = np.max(arr, axis=axis, keepdims=True)
        out = arr_max + np.log(np.sum(np.exp(arr - arr_max), axis=axis, keepdims=True))
        if not keepdims:
            out = np.squeeze(out, axis=axis)
        return out


class MultinomialNB:
    """
    Multinomial Naive Bayes classifier for discrete/count features.

    Suitable for classification with discrete features (e.g., word counts for text
    classification). The multinomial distribution typically requires integer feature
    counts, though in practice, fractional counts may also work.

    Mathematical Model:
    ------------------
    For count data, the likelihood is modeled using multinomial distribution:

    P(x_i|y) = (N_yi + α) / (N_y + α * n_features)

    Where:
    - N_yi is the count of feature x_i in class y
    - N_y is the total count of all features in class y
    - α is the Laplace smoothing parameter (alpha)
    - n_features is the number of features

    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
        Higher values mean more smoothing.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier
    class_prior_ : ndarray of shape (n_classes,)
        Log probability of each class P(y)
    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        Empirical log probability of features given a class, P(x_i|y)
    n_features_ : int
        Number of features seen during fit
    """

    def __init__(self, alpha: float = 1.0):
        """
        Initialize Multinomial Naive Bayes classifier.

        Parameters
        ----------
        alpha : float, default=1.0
            Laplace smoothing parameter (must be non-negative)
        """
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        self.alpha = alpha
        self.classes_ = None
        self.class_prior_ = None
        self.feature_log_prob_ = None
        self.n_features_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultinomialNB':
        """
        Fit Multinomial Naive Bayes classifier.

        Computes:
        1. Class priors: P(y) = count(y) / n_samples
        2. Feature probabilities with Laplace smoothing:
           P(x_i|y) = (count(x_i, y) + α) / (count(y) + α * n_features)

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training vectors (should be non-negative counts/frequencies)
        y : ndarray of shape (n_samples,)
            Target values

        Returns
        -------
        self : object
            Returns the instance itself
        """
        # Input validation
        X, y = self._validate_input(X, y)

        # Check for negative values
        if np.any(X < 0):
            raise ValueError("Input X must contain non-negative values")

        # Get unique classes and their counts
        self.classes_, y_counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # Calculate class log priors: log P(y)
        self.class_prior_ = np.log(y_counts / n_samples)

        # Initialize feature count matrix
        feature_count = np.zeros((n_classes, n_features))

        # Count features for each class
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]  # Select samples belonging to class c
            feature_count[idx, :] = X_c.sum(axis=0)

        # Apply Laplace smoothing and compute log probabilities
        # log P(x_i|y) = log[(N_yi + α) / (N_y + α * n_features)]
        smoothed_fc = feature_count + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True)

        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)

        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log posterior probabilities.

        log P(y|X) = log P(y) + ∑ x_i * log P(x_i|y)

        For count data, each feature appears x_i times, so we multiply
        the log probability by the count.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples

        Returns
        -------
        log_proba : ndarray of shape (n_samples, n_classes)
            Log posterior probabilities for each class
        """
        self._check_is_fitted()
        X = self._validate_input(X)

        # log P(y|X) = log P(y) + X @ log P(x|y).T
        # Shape: (n_samples, n_classes)
        log_posterior = X @ self.feature_log_prob_.T + self.class_prior_

        return log_posterior

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute posterior probabilities.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Posterior probabilities for each class
        """
        log_proba = self.predict_log_proba(X)

        # Use log-sum-exp trick for numerical stability
        log_sum = self._logsumexp(log_proba, axis=1, keepdims=True)
        proba = np.exp(log_proba - log_sum)

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels
        """
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples,)
            True labels

        Returns
        -------
        score : float
            Accuracy score (fraction of correct predictions)
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def _validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Validate and convert input arrays."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")

        if y is not None:
            y = np.asarray(y).ravel()
            if X.shape[0] != len(y):
                raise ValueError(
                    f"X and y have inconsistent numbers of samples: "
                    f"{X.shape[0]} != {len(y)}"
                )
            return X, y

        return X

    def _check_is_fitted(self):
        """Check if the classifier has been fitted."""
        if self.classes_ is None:
            raise ValueError(
                "This MultinomialNB instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

    @staticmethod
    def _logsumexp(arr: np.ndarray, axis: int = None, keepdims: bool = False) -> np.ndarray:
        """Compute log(sum(exp(arr))) in a numerically stable way."""
        arr_max = np.max(arr, axis=axis, keepdims=True)
        out = arr_max + np.log(np.sum(np.exp(arr - arr_max), axis=axis, keepdims=True))
        if not keepdims:
            out = np.squeeze(out, axis=axis)
        return out


class BernoulliNB:
    """
    Bernoulli Naive Bayes classifier for binary features.

    Like MultinomialNB, this classifier is suitable for discrete data. The difference is
    that while MultinomialNB works with occurrence counts, BernoulliNB is designed for
    binary/boolean features (present/absent).

    Mathematical Model:
    ------------------
    For binary features, the likelihood is modeled using Bernoulli distribution:

    P(x_i|y) = P_yi^(x_i) * (1 - P_yi)^(1 - x_i)

    In log space:
    log P(x_i|y) = x_i * log(P_yi) + (1 - x_i) * log(1 - P_yi)

    Where:
    - P_yi is the probability that feature x_i appears in class y
    - x_i ∈ {0, 1} indicates presence/absence of feature

    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    binarize : float or None, default=0.0
        Threshold for binarizing (mapping to booleans) of sample features.
        If None, input is presumed to already consist of binary vectors.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier
    class_prior_ : ndarray of shape (n_classes,)
        Log probability of each class P(y)
    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        Empirical log probability of features given a class, log P(x_i=1|y)
    feature_log_prob_neg_ : ndarray of shape (n_classes, n_features)
        Empirical log probability of features not occurring, log P(x_i=0|y)
    n_features_ : int
        Number of features seen during fit
    """

    def __init__(self, alpha: float = 1.0, binarize: Optional[float] = 0.0):
        """
        Initialize Bernoulli Naive Bayes classifier.

        Parameters
        ----------
        alpha : float, default=1.0
            Laplace smoothing parameter (must be non-negative)
        binarize : float or None, default=0.0
            Threshold for binarizing features. If None, assumes input is already binary.
        """
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        self.alpha = alpha
        self.binarize = binarize
        self.classes_ = None
        self.class_prior_ = None
        self.feature_log_prob_ = None
        self.feature_log_prob_neg_ = None
        self.n_features_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BernoulliNB':
        """
        Fit Bernoulli Naive Bayes classifier.

        Computes:
        1. Class priors: P(y) = count(y) / n_samples
        2. Feature probabilities with Laplace smoothing:
           P(x_i=1|y) = (count(x_i=1, y) + α) / (count(y) + 2α)
           P(x_i=0|y) = 1 - P(x_i=1|y)

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training vectors (will be binarized if binarize is not None)
        y : ndarray of shape (n_samples,)
            Target values

        Returns
        -------
        self : object
            Returns the instance itself
        """
        # Input validation
        X, y = self._validate_input(X, y)

        # Binarize if threshold is set
        if self.binarize is not None:
            X = self._binarize_X(X)

        # Get unique classes and their counts
        self.classes_, y_counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # Calculate class log priors: log P(y)
        self.class_prior_ = np.log(y_counts / n_samples)

        # Initialize feature count matrix
        feature_count = np.zeros((n_classes, n_features))

        # Count occurrences of each feature for each class
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]  # Select samples belonging to class c
            feature_count[idx, :] = X_c.sum(axis=0)

        # Apply Laplace smoothing and compute log probabilities
        # For Bernoulli, we smooth with 2*alpha (for binary outcomes)
        # P(x_i=1|y) = (N_yi + α) / (N_y + 2α)
        smoothed_fc = feature_count + self.alpha
        smoothed_cc = np.array([y_counts]).T + 2 * self.alpha

        # log P(x_i=1|y)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)

        # log P(x_i=0|y) = log(1 - P(x_i=1|y))
        self.feature_log_prob_neg_ = np.log(1 - np.exp(self.feature_log_prob_))

        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log posterior probabilities.

        For Bernoulli:
        log P(y|X) = log P(y) + ∑ [x_i * log P(x_i=1|y) + (1-x_i) * log P(x_i=0|y)]

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples

        Returns
        -------
        log_proba : ndarray of shape (n_samples, n_classes)
            Log posterior probabilities for each class
        """
        self._check_is_fitted()
        X = self._validate_input(X)

        # Binarize if threshold is set
        if self.binarize is not None:
            X = self._binarize_X(X)

        # log P(y|X) = log P(y) + ∑ [x_i * log P(x_i=1|y) + (1-x_i) * log P(x_i=0|y)]
        # This can be rewritten as:
        # log P(y|X) = log P(y) + X @ log P(x=1|y).T + (1-X) @ log P(x=0|y).T
        # = log P(y) + X @ [log P(x=1|y) - log P(x=0|y)].T + ∑ log P(x=0|y)

        n_samples = X.shape[0]

        # Vectorized computation
        # Shape: (n_samples, n_classes)
        log_posterior = (
            X @ (self.feature_log_prob_ - self.feature_log_prob_neg_).T +
            self.feature_log_prob_neg_.sum(axis=1) +
            self.class_prior_
        )

        return log_posterior

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute posterior probabilities.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Posterior probabilities for each class
        """
        log_proba = self.predict_log_proba(X)

        # Use log-sum-exp trick for numerical stability
        log_sum = self._logsumexp(log_proba, axis=1, keepdims=True)
        proba = np.exp(log_proba - log_sum)

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels
        """
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples,)
            True labels

        Returns
        -------
        score : float
            Accuracy score (fraction of correct predictions)
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def _binarize_X(self, X: np.ndarray) -> np.ndarray:
        """Binarize input data based on threshold."""
        return (X > self.binarize).astype(np.float64)

    def _validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Validate and convert input arrays."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")

        if y is not None:
            y = np.asarray(y).ravel()
            if X.shape[0] != len(y):
                raise ValueError(
                    f"X and y have inconsistent numbers of samples: "
                    f"{X.shape[0]} != {len(y)}"
                )
            return X, y

        return X

    def _check_is_fitted(self):
        """Check if the classifier has been fitted."""
        if self.classes_ is None:
            raise ValueError(
                "This BernoulliNB instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

    @staticmethod
    def _logsumexp(arr: np.ndarray, axis: int = None, keepdims: bool = False) -> np.ndarray:
        """Compute log(sum(exp(arr))) in a numerically stable way."""
        arr_max = np.max(arr, axis=axis, keepdims=True)
        out = arr_max + np.log(np.sum(np.exp(arr - arr_max), axis=axis, keepdims=True))
        if not keepdims:
            out = np.squeeze(out, axis=axis)
        return out


if __name__ == "__main__":
    # Simple test for each classifier
    print("Testing Naive Bayes Classifiers")
    print("=" * 50)

    # Test GaussianNB
    print("\n1. GaussianNB Test")
    print("-" * 50)
    np.random.seed(42)
    X_gaussian = np.random.randn(100, 2)
    y_gaussian = (X_gaussian[:, 0] + X_gaussian[:, 1] > 0).astype(int)

    gnb = GaussianNB()
    gnb.fit(X_gaussian[:80], y_gaussian[:80])
    score_g = gnb.score(X_gaussian[80:], y_gaussian[80:])
    print(f"Accuracy: {score_g:.3f}")
    print(f"Class priors: {gnb.class_prior_}")

    # Test MultinomialNB
    print("\n2. MultinomialNB Test")
    print("-" * 50)
    X_multi = np.array([[2, 3, 0, 1],
                        [1, 0, 2, 3],
                        [0, 1, 3, 2],
                        [3, 2, 1, 0],
                        [2, 3, 1, 0],
                        [1, 0, 3, 2]])
    y_multi = np.array([0, 1, 1, 0, 0, 1])

    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_multi, y_multi)
    print(f"Accuracy: {mnb.score(X_multi, y_multi):.3f}")
    print(f"Class priors (log): {mnb.class_prior_}")

    # Test BernoulliNB
    print("\n3. BernoulliNB Test")
    print("-" * 50)
    X_bern = np.array([[0, 1, 0, 1],
                       [1, 0, 1, 0],
                       [0, 0, 1, 1],
                       [1, 1, 0, 0],
                       [0, 1, 1, 0],
                       [1, 0, 0, 1]])
    y_bern = np.array([0, 1, 1, 0, 1, 0])

    bnb = BernoulliNB(alpha=1.0)
    bnb.fit(X_bern, y_bern)
    print(f"Accuracy: {bnb.score(X_bern, y_bern):.3f}")
    print(f"Class priors (log): {bnb.class_prior_}")

    print("\n" + "=" * 50)
    print("All tests completed successfully!")
