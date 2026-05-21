"""
CatBoost (Categorical Boosting) — From Scratch Implementation

Yandex's gradient boosting library (Prokhorenkova et al., NeurIPS 2018).

Three core innovations over XGBoost / LightGBM:
1. Symmetric (Oblivious) Trees   — same split at every node of a given depth
2. Ordered Boosting              — unbiased gradient estimation, no prediction shift
3. Ordered Target Statistics     — leak-free categorical feature encoding
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Split condition (one level of an oblivious tree)
# ---------------------------------------------------------------------------

@dataclass
class SplitCondition:
    """A single (feature, threshold) pair used at one depth level."""
    feature_index: int = -1
    threshold: float = 0.0
    gain: float = 0.0


# ---------------------------------------------------------------------------
# Oblivious (Symmetric) Decision Tree
# ---------------------------------------------------------------------------

class ObliviousTree:
    """
    Symmetric decision tree: all nodes at the same depth share one split.

    A depth-d oblivious tree has exactly 2^d leaves.
    Prediction is O(depth): check d binary conditions, index into leaf array.

    Advantages
    ----------
    - Built-in regularization via the symmetric constraint (fewer effective parameters)
    - Cache-friendly evaluation (same feature accessed for all samples at each level)
    - O(depth) prediction vs O(n_nodes) for asymmetric trees
    """

    def __init__(
        self,
        max_depth: int = 6,
        min_samples_leaf: int = 1,
        reg_lambda: float = 3.0,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda

        self.splits: List[SplitCondition] = []
        self.leaf_values: np.ndarray = np.array([0.0])

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
    ) -> 'ObliviousTree':
        """
        Grow the oblivious tree greedily level by level.

        At each depth d, find the single (feature, threshold) that maximises
        the total gain summed across all 2^d current nodes.
        """
        n_samples, n_features = X.shape
        node_assignments = np.zeros(n_samples, dtype=int)  # which leaf each sample is in

        for depth in range(self.max_depth):
            n_nodes = 2 ** depth
            best_gain, best_feat, best_thresh = self._find_best_split(
                X, gradients, hessians, node_assignments, n_nodes, n_features
            )

            if best_feat < 0 or best_gain <= 0.0:
                break

            self.splits.append(SplitCondition(best_feat, best_thresh, best_gain))

            # Update leaf assignments: left child -> 2*node, right child -> 2*node + 1
            go_right = (X[:, best_feat] > best_thresh).astype(int)
            node_assignments = node_assignments * 2 + go_right

        # Compute optimal leaf weights  w* = -G / (H + lambda)
        n_leaves = max(1, 2 ** len(self.splits))
        G_leaves = np.zeros(n_leaves)
        H_leaves = np.zeros(n_leaves)
        np.add.at(G_leaves, node_assignments, gradients)
        np.add.at(H_leaves, node_assignments, hessians)

        self.leaf_values = np.where(
            (H_leaves + self.reg_lambda) > 0,
            -G_leaves / (H_leaves + self.reg_lambda),
            0.0
        )
        return self

    def _find_best_split(
        self,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        node_assignments: np.ndarray,
        n_nodes: int,
        n_features: int,
    ) -> Tuple[float, int, float]:
        """
        Find the (feature, threshold) maximising total gain across all current nodes.

        For each candidate split, we sum the gain over every existing node:
            total_gain = sum_{node k} [ G_L_k^2/(H_L_k + lambda)
                                       + G_R_k^2/(H_R_k + lambda)
                                       - G_k^2/(H_k + lambda) ]
        """
        best_gain = -np.inf
        best_feat = -1
        best_thresh = 0.0

        for feat in range(n_features):
            vals = X[:, feat]
            unique_vals = np.unique(vals)
            if len(unique_vals) < 2:
                continue

            cut_points = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thresh in cut_points:
                go_right = vals > thresh
                total_gain = 0.0
                valid = True

                for node_id in range(n_nodes):
                    in_node = np.where(node_assignments == node_id)[0]
                    if len(in_node) == 0:
                        continue

                    left_idx = in_node[~go_right[in_node]]
                    right_idx = in_node[go_right[in_node]]

                    if (len(left_idx) < self.min_samples_leaf or
                            len(right_idx) < self.min_samples_leaf):
                        valid = False
                        break

                    G_L = gradients[left_idx].sum()
                    H_L = hessians[left_idx].sum()
                    G_R = gradients[right_idx].sum()
                    H_R = hessians[right_idx].sum()
                    G   = G_L + G_R
                    H   = H_L + H_R

                    node_gain = (
                        G_L ** 2 / (H_L + self.reg_lambda) +
                        G_R ** 2 / (H_R + self.reg_lambda) -
                        G   ** 2 / (H   + self.reg_lambda)
                    )
                    total_gain += node_gain

                if valid and total_gain > best_gain:
                    best_gain = total_gain
                    best_feat = feat
                    best_thresh = thresh

        return best_gain, best_feat, best_thresh

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict leaf values in O(depth) per sample via binary indexing."""
        n = X.shape[0]
        leaf_indices = np.zeros(n, dtype=int)

        for split in self.splits:
            go_right = (X[:, split.feature_index] > split.threshold).astype(int)
            leaf_indices = leaf_indices * 2 + go_right

        leaf_indices = np.clip(leaf_indices, 0, len(self.leaf_values) - 1)
        return self.leaf_values[leaf_indices]

    def get_feature_importances(self, n_features: int) -> np.ndarray:
        """Gain-based importance: sum of gain contributed at each split level."""
        imp = np.zeros(n_features)
        for split in self.splits:
            if split.feature_index >= 0:
                imp[split.feature_index] += max(0.0, split.gain)
        return imp


# ---------------------------------------------------------------------------
# Ordered Target Statistics (leak-free categorical encoding)
# ---------------------------------------------------------------------------

def ordered_target_statistics(
    categories: np.ndarray,
    targets: np.ndarray,
    prior: float = 1.0,
    prior_strength: float = 1.0,
) -> np.ndarray:
    """
    Compute ordered (leak-free) target statistics for one categorical feature.

    For sample i in permutation order, with category c:

        x_hat_i = (sum_{j < i : cat_j = c} y_j  +  prior * prior_strength)
                  / (count_{j < i : cat_j = c}   +  prior_strength)

    Crucially, sample i's own target y_i is NOT included when encoding x_hat_i,
    which eliminates the target leakage produced by plain mean encoding.

    Parameters
    ----------
    categories      : integer category indices for each sample
    targets         : regression or binary classification targets
    prior           : fallback value when a category has no history
    prior_strength  : weight of the prior (higher = stronger smoothing)

    Returns
    -------
    encoded : float array of shape (n_samples,)
    """
    n = len(categories)
    cat_sum = {}    # cumulative sum of targets per category (before current sample)
    cat_cnt = {}    # cumulative count per category
    encoded = np.zeros(n)

    for i in range(n):
        c = int(categories[i])
        s   = cat_sum.get(c, 0.0)
        cnt = cat_cnt.get(c, 0)
        encoded[i] = (s + prior * prior_strength) / (cnt + prior_strength)
        # Update AFTER encoding to avoid leaking y_i into x_hat_i
        cat_sum[c] = s + float(targets[i])
        cat_cnt[c] = cnt + 1

    return encoded


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------

class _CatBoostBase:
    """Shared gradient-boosting loop for Regressor and Classifier."""

    def __init__(
        self,
        iterations: int = 100,
        learning_rate: float = 0.1,
        depth: int = 6,
        reg_lambda: float = 3.0,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        colsample_bylevel: float = 1.0,
        use_ordered_boosting: bool = False,
        n_ordered_folds: int = 4,
        random_state: Optional[int] = None,
    ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.reg_lambda = reg_lambda
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample_bylevel = colsample_bylevel
        self.use_ordered_boosting = use_ordered_boosting
        self.n_ordered_folds = n_ordered_folds
        self.random_state = random_state

        self.trees_: List[ObliviousTree] = []
        self.base_score_: float = 0.0
        self.feature_importances_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.selected_features_: List[np.ndarray] = []

        if random_state is not None:
            np.random.seed(random_state)

    # Subclasses must override these two
    def _compute_gradients(self, F: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _base_score(self, y: np.ndarray) -> float:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Ordered boosting: cross-fold unbiased gradient estimation
    # ------------------------------------------------------------------

    def _ordered_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simplified ordered boosting via leave-fold-out gradient estimation.

        Idea: divide data into K ordered folds. For fold k, compute gradients
        using the current ensemble's predictions on fold k — but the ensemble
        was built before seeing this round's data, so the gradient estimate for
        fold k is computed with a model that (approximately) did not memorise
        fold k in the current round.

        This captures the spirit of CatBoost ordered boosting: each sample's
        gradient is computed using a model that was not trained on that sample
        in the current boosting step.
        """
        n = len(y)
        K = self.n_ordered_folds
        perm = np.random.permutation(n)
        fold_size = max(1, n // K)

        ordered_g = np.zeros(n)
        ordered_h = np.zeros(n)

        for k in range(K):
            start = k * fold_size
            end = min(n, (k + 1) * fold_size) if k < K - 1 else n
            fold_idx = perm[start:end]

            # Predict on fold_idx using ONLY trees built so far (before this round)
            F_fold = np.full(len(fold_idx), self.base_score_)
            for tree, sel in zip(self.trees_, self.selected_features_):
                F_fold += self.learning_rate * tree.predict(X[fold_idx][:, sel])

            g_k, h_k = self._compute_gradients(F_fold, y[fold_idx])
            ordered_g[fold_idx] = g_k
            ordered_h[fold_idx] = h_k

        return ordered_g, ordered_h

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------

    def _fit_core(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.base_score_ = self._base_score(y)
        F = np.full(n_samples, self.base_score_)

        importance_acc = np.zeros(n_features)
        self.trees_.clear()
        self.selected_features_.clear()

        for _ in range(self.iterations):
            if self.use_ordered_boosting:
                gradients, hessians = self._ordered_gradients(X, y)
            else:
                gradients, hessians = self._compute_gradients(F, y)

            # Row subsampling
            if self.subsample < 1.0:
                row_idx = np.random.choice(n_samples, int(n_samples * self.subsample), replace=False)
                X_s, g_s, h_s = X[row_idx], gradients[row_idx], hessians[row_idx]
            else:
                X_s, g_s, h_s = X, gradients, hessians

            # Column subsampling (at-level in CatBoost; here applied per tree)
            if self.colsample_bylevel < 1.0:
                n_sel = max(1, int(n_features * self.colsample_bylevel))
                sel_feats = np.random.choice(n_features, n_sel, replace=False)
            else:
                sel_feats = np.arange(n_features)
            self.selected_features_.append(sel_feats)

            tree = ObliviousTree(
                max_depth=self.depth,
                min_samples_leaf=self.min_samples_leaf,
                reg_lambda=self.reg_lambda,
            )
            tree.fit(X_s[:, sel_feats], g_s, h_s)

            # Update predictions on full dataset
            F += self.learning_rate * tree.predict(X[:, sel_feats])

            # Accumulate importances
            local_imp = tree.get_feature_importances(len(sel_feats))
            for local_i, global_i in enumerate(sel_feats):
                importance_acc[global_i] += local_imp[local_i]

            self.trees_.append(tree)

        total = importance_acc.sum()
        self.feature_importances_ = importance_acc / total if total > 0 else importance_acc

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        F = np.full(X.shape[0], self.base_score_)
        for tree, sel in zip(self.trees_, self.selected_features_):
            F += self.learning_rate * tree.predict(X[:, sel])
        return F


# ---------------------------------------------------------------------------
# CatBoost Regressor
# ---------------------------------------------------------------------------

class CatBoostRegressor(_CatBoostBase):
    """
    CatBoost for regression using squared-error loss.

    Parameters
    ----------
    iterations           : number of boosting rounds
    learning_rate        : shrinkage factor
    depth                : max depth of each oblivious tree
    reg_lambda           : L2 regularization on leaf weights
    min_samples_leaf     : minimum samples required in a leaf
    subsample            : fraction of rows sampled per round
    colsample_bylevel    : fraction of features considered per split level
    use_ordered_boosting : enable cross-fold ordered gradient estimation
    n_ordered_folds      : number of folds for ordered boosting
    random_state         : seed for reproducibility
    """

    def _base_score(self, y: np.ndarray) -> float:
        return float(np.mean(y))

    def _compute_gradients(self, F, y):
        return F - y, np.ones(len(y))

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CatBoostRegressor':
        self._fit_core(np.array(X, dtype=float), np.array(y, dtype=float))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._predict_raw(np.array(X, dtype=float))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.array(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ---------------------------------------------------------------------------
# CatBoost Classifier
# ---------------------------------------------------------------------------

class CatBoostClassifier(_CatBoostBase):
    """
    CatBoost for binary classification using log loss.

    Parameters
    ----------
    Same as CatBoostRegressor.
    """

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x))
        )

    def _base_score(self, y: np.ndarray) -> float:
        pos = np.sum(y == 1)
        neg = len(y) - pos
        return float(np.log(pos / neg)) if pos > 0 and neg > 0 else 0.0

    def _compute_gradients(self, F, y):
        p = self._sigmoid(F)
        g = p - y
        h = np.maximum(p * (1.0 - p), 1e-6)
        return g, h

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CatBoostClassifier':
        self._fit_core(np.array(X, dtype=float), np.array(y, dtype=float))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self._sigmoid(self._predict_raw(np.array(X, dtype=float)))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.array(y)))
