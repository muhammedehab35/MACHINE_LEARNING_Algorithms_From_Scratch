"""
Bagging (Bootstrap Aggregating) — From Scratch Implementation

Leo Breiman (1994). "Bagging Predictors." Machine Learning, 24(2), 123-140.

Core idea: train B estimators on independent bootstrap samples of the
training data, then aggregate (average or majority vote) their predictions.
Reduces variance without increasing bias.
"""

import numpy as np
from copy import deepcopy
from typing import Optional, List, Union


# ---------------------------------------------------------------------------
# Bootstrap utilities
# ---------------------------------------------------------------------------

def bootstrap_sample(n: int, size: int, random_state=None) -> np.ndarray:
    """
    Draw a bootstrap sample of `size` indices from {0, ..., n-1} with replacement.

    Expected fraction of unique indices: 1 - (1 - 1/n)^n -> 1 - 1/e ~ 63.2%.
    Equivalently, ~36.8% of samples are out-of-bag (OOB) on average.
    """
    rng = np.random.RandomState(random_state) if random_state is not None else np.random
    return rng.choice(n, size=size, replace=True)


def oob_indices(in_bag: np.ndarray, n: int) -> np.ndarray:
    """Return indices not present in the bootstrap sample (OOB samples)."""
    mask = np.ones(n, dtype=bool)
    mask[np.unique(in_bag)] = False
    return np.where(mask)[0]


# ---------------------------------------------------------------------------
# BaggingClassifier
# ---------------------------------------------------------------------------

class BaggingClassifier:
    """
    Bagging ensemble for classification.

    Trains `n_estimators` base classifiers on independent bootstrap samples
    and aggregates via soft voting (average of predicted probabilities).

    Parameters
    ----------
    base_estimator   : sklearn-compatible classifier (default: DecisionTreeClassifier)
    n_estimators     : number of bootstrap estimators
    max_samples      : samples per bootstrap (float = fraction, int = count)
    max_features     : features per estimator  (float = fraction, int = count)
    bootstrap        : sample rows with replacement
    bootstrap_features : sample columns with replacement
    oob_score        : compute out-of-bag accuracy estimate
    random_state     : global seed

    Attributes
    ----------
    estimators_          : list of fitted base estimators
    estimators_samples_  : bootstrap row indices for each estimator
    estimators_features_ : column indices used by each estimator
    classes_             : unique class labels
    n_features_          : number of input features
    oob_score_           : OOB accuracy (if oob_score=True)
    oob_decision_function_ : OOB probability matrix, shape (n_train, n_classes)
    """

    def __init__(
        self,
        base_estimator=None,
        n_estimators: int = 10,
        max_samples: Union[int, float] = 1.0,
        max_features: Union[int, float] = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        oob_score: bool = False,
        random_state: Optional[int] = None,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.random_state = random_state

        self.estimators_: List = []
        self.estimators_samples_: List[np.ndarray] = []
        self.estimators_features_: List[np.ndarray] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.oob_score_: Optional[float] = None
        self.oob_decision_function_: Optional[np.ndarray] = None

        if random_state is not None:
            np.random.seed(random_state)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_estimator(self):
        if self.base_estimator is None:
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier()
        return deepcopy(self.base_estimator)

    def _n_samples(self, n: int) -> int:
        if isinstance(self.max_samples, float):
            return max(1, int(n * self.max_samples))
        return min(int(self.max_samples), n)

    def _n_features(self, p: int) -> int:
        if isinstance(self.max_features, float):
            return max(1, int(p * self.max_features))
        return min(int(self.max_features), p)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaggingClassifier':
        X = np.array(X, dtype=float)
        y = np.array(y)
        n, p = X.shape
        self.classes_ = np.unique(y)
        self.n_features_ = p
        n_cls = len(self.classes_)

        n_samp = self._n_samples(n)
        n_feat = self._n_features(p)

        self.estimators_.clear()
        self.estimators_samples_.clear()
        self.estimators_features_.clear()

        oob_proba = np.zeros((n, n_cls))
        oob_cnt   = np.zeros(n)

        for b in range(self.n_estimators):
            # ---- row sampling ----
            if self.bootstrap:
                row_idx = np.random.choice(n, size=n_samp, replace=True)
            else:
                row_idx = np.random.choice(n, size=n_samp, replace=False)

            # ---- column sampling ----
            if self.bootstrap_features:
                col_idx = np.random.choice(p, size=n_feat, replace=True)
            else:
                col_idx = np.random.choice(p, size=n_feat, replace=False)

            self.estimators_samples_.append(row_idx)
            self.estimators_features_.append(col_idx)

            est = self._make_estimator()
            est.fit(X[row_idx][:, col_idx], y[row_idx])
            self.estimators_.append(est)

            # ---- OOB accumulation ----
            if self.oob_score:
                oob_idx = oob_indices(row_idx, n)
                if len(oob_idx) == 0:
                    continue
                X_oob = X[oob_idx][:, col_idx]

                if hasattr(est, 'predict_proba'):
                    proba = est.predict_proba(X_oob)
                    est_classes = est.classes_
                else:
                    preds = est.predict(X_oob)
                    est_classes = np.unique(preds)
                    proba = np.zeros((len(oob_idx), len(est_classes)))
                    for j, c in enumerate(est_classes):
                        proba[:, j] = (preds == c).astype(float)

                for j, c in enumerate(est_classes):
                    global_j = np.searchsorted(self.classes_, c)
                    if global_j < n_cls and self.classes_[global_j] == c:
                        oob_proba[oob_idx, global_j] += proba[:, j]
                oob_cnt[oob_idx] += 1

        if self.oob_score:
            valid = oob_cnt > 0
            if valid.sum() == 0:
                self.oob_score_ = 0.0
            else:
                norm = np.where(oob_cnt[:, None] > 0, oob_cnt[:, None], 1)
                oob_norm = oob_proba / norm
                oob_pred_idx = np.argmax(oob_norm[valid], axis=1)
                oob_pred = self.classes_[oob_pred_idx]
                self.oob_score_ = float(np.mean(oob_pred == y[valid]))
            self.oob_decision_function_ = np.divide(
                oob_proba, np.where(oob_cnt[:, None] > 0, oob_cnt[:, None], 1)
            )

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        n_cls = len(self.classes_)
        agg = np.zeros((X.shape[0], n_cls))

        for est, col_idx in zip(self.estimators_, self.estimators_features_):
            X_sub = X[:, col_idx]
            if hasattr(est, 'predict_proba'):
                proba = est.predict_proba(X_sub)
                for j, c in enumerate(est.classes_):
                    gj = np.searchsorted(self.classes_, c)
                    if gj < n_cls and self.classes_[gj] == c:
                        agg[:, gj] += proba[:, j]
            else:
                preds = est.predict(X_sub)
                for i, c in enumerate(preds):
                    gj = np.searchsorted(self.classes_, c)
                    if gj < n_cls and self.classes_[gj] == c:
                        agg[i, gj] += 1

        return agg / self.n_estimators

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.array(y)))

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        """Average feature importances across estimators (if available)."""
        imps = []
        for est, col_idx in zip(self.estimators_, self.estimators_features_):
            if hasattr(est, 'feature_importances_'):
                fi = np.asarray(est.feature_importances_)
                imp = np.zeros(self.n_features_)
                for local_i, global_i in enumerate(col_idx):
                    if local_i < len(fi):
                        imp[global_i] += fi[local_i]
                imps.append(imp)
        if not imps:
            return None
        avg = np.mean(imps, axis=0)
        total = avg.sum()
        return avg / total if total > 0 else avg


# ---------------------------------------------------------------------------
# BaggingRegressor
# ---------------------------------------------------------------------------

class BaggingRegressor:
    """
    Bagging ensemble for regression.

    Aggregates base regressors' predictions by averaging.

    Parameters
    ----------
    Same as BaggingClassifier (no oob_decision_function, oob_score uses R2).

    Attributes
    ----------
    estimators_          : fitted base regressors
    estimators_samples_  : bootstrap row indices
    estimators_features_ : column indices
    n_features_          : number of input features
    oob_score_           : OOB R2 score (if oob_score=True)
    oob_prediction_      : OOB predictions, shape (n_train,)
    """

    def __init__(
        self,
        base_estimator=None,
        n_estimators: int = 10,
        max_samples: Union[int, float] = 1.0,
        max_features: Union[int, float] = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        oob_score: bool = False,
        random_state: Optional[int] = None,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.random_state = random_state

        self.estimators_: List = []
        self.estimators_samples_: List[np.ndarray] = []
        self.estimators_features_: List[np.ndarray] = []
        self.n_features_: Optional[int] = None
        self.oob_score_: Optional[float] = None
        self.oob_prediction_: Optional[np.ndarray] = None

        if random_state is not None:
            np.random.seed(random_state)

    def _make_estimator(self):
        if self.base_estimator is None:
            from sklearn.tree import DecisionTreeRegressor
            return DecisionTreeRegressor()
        return deepcopy(self.base_estimator)

    def _n_samples(self, n):
        return max(1, int(n * self.max_samples)) if isinstance(self.max_samples, float) else min(int(self.max_samples), n)

    def _n_features(self, p):
        return max(1, int(p * self.max_features)) if isinstance(self.max_features, float) else min(int(self.max_features), p)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaggingRegressor':
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n, p = X.shape
        self.n_features_ = p

        n_samp = self._n_samples(n)
        n_feat = self._n_features(p)

        self.estimators_.clear()
        self.estimators_samples_.clear()
        self.estimators_features_.clear()

        oob_sum = np.zeros(n)
        oob_cnt = np.zeros(n)

        for _ in range(self.n_estimators):
            row_idx = (np.random.choice(n, n_samp, replace=True)
                       if self.bootstrap
                       else np.random.choice(n, n_samp, replace=False))
            col_idx = (np.random.choice(p, n_feat, replace=True)
                       if self.bootstrap_features
                       else np.random.choice(p, n_feat, replace=False))

            self.estimators_samples_.append(row_idx)
            self.estimators_features_.append(col_idx)

            est = self._make_estimator()
            est.fit(X[row_idx][:, col_idx], y[row_idx])
            self.estimators_.append(est)

            if self.oob_score:
                oob_idx = oob_indices(row_idx, n)
                if len(oob_idx) > 0:
                    preds = est.predict(X[oob_idx][:, col_idx])
                    oob_sum[oob_idx] += preds
                    oob_cnt[oob_idx] += 1

        if self.oob_score:
            valid = oob_cnt > 0
            oob_pred = np.where(oob_cnt > 0, oob_sum / np.where(oob_cnt > 0, oob_cnt, 1), 0.0)
            self.oob_prediction_ = oob_pred
            if valid.sum() == 0:
                self.oob_score_ = 0.0
            else:
                ss_res = np.sum((y[valid] - oob_pred[valid]) ** 2)
                ss_tot = np.sum((y[valid] - y[valid].mean()) ** 2)
                self.oob_score_ = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        preds = np.zeros(X.shape[0])
        for est, col_idx in zip(self.estimators_, self.estimators_features_):
            preds += est.predict(X[:, col_idx])
        return preds / self.n_estimators

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.array(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        imps = []
        for est, col_idx in zip(self.estimators_, self.estimators_features_):
            if hasattr(est, 'feature_importances_'):
                fi = np.asarray(est.feature_importances_)
                imp = np.zeros(self.n_features_)
                for li, gi in enumerate(col_idx):
                    if li < len(fi):
                        imp[gi] += fi[li]
                imps.append(imp)
        if not imps:
            return None
        avg = np.mean(imps, axis=0)
        total = avg.sum()
        return avg / total if total > 0 else avg
