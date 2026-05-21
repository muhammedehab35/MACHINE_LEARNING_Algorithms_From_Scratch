"""
LightGBM (Light Gradient Boosting Machine) - From Scratch Implementation

Microsoft's highly optimized gradient boosting framework that improves
upon XGBoost with three key innovations:

1. Histogram-based split finding  — O(B) instead of O(n log n)
2. Leaf-wise tree growth          — always splits the best leaf
3. GOSS sampling                  — keeps high-gradient samples, subsamples rest

Key References:
- Ke et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree."
  NeurIPS 2017.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Histogram builder
# ---------------------------------------------------------------------------

class FeatureHistogram:
    """
    Bins a single feature into at most `n_bins` discrete buckets.

    For each bin we accumulate:
        - sum of gradients  G_b
        - sum of hessians   H_b
        - count of samples  cnt_b

    Split scan is then O(B) instead of O(n log n).
    """

    def __init__(self, n_bins: int = 255):
        self.n_bins = n_bins
        self.bin_edges: Optional[np.ndarray] = None

    def fit(self, values: np.ndarray) -> 'FeatureHistogram':
        unique = np.unique(values)
        if len(unique) <= self.n_bins:
            self.bin_edges = unique
        else:
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            self.bin_edges = np.unique(np.percentile(values, percentiles))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Map continuous values -> bin indices (0-based)."""
        return np.searchsorted(self.bin_edges, values, side='right') - 1

    def build(
        self,
        bin_values: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return per-bin (G, H, cnt) arrays for samples given by bin_values.
        """
        n_bins = len(self.bin_edges)
        G = np.zeros(n_bins)
        H = np.zeros(n_bins)
        cnt = np.zeros(n_bins, dtype=int)
        np.add.at(G, bin_values, gradients)
        np.add.at(H, bin_values, hessians)
        np.add.at(cnt, bin_values, 1)
        return G, H, cnt


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    """Node in a LightGBM tree."""
    is_leaf: bool = True
    leaf_value: float = 0.0

    feature_index: int = -1
    bin_threshold: int = -1          # split: bin_idx <= bin_threshold -> left
    real_threshold: float = 0.0      # for human-readable output

    left:  Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None
    gain:  float = 0.0


# ---------------------------------------------------------------------------
# Leaf-wise tree
# ---------------------------------------------------------------------------

class LGBMTree:
    """
    Single decision tree for LightGBM.

    Uses:
    - Histogram-based split finding (O(B) per leaf per feature)
    - Leaf-wise growth (always splits the leaf with maximum gain)
    - L1 + L2 regularization on leaf weights
    - gamma (min split gain) pruning
    """

    def __init__(
        self,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_weight: float = 1e-3,
        min_data_in_leaf: int = 20,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        gamma: float = 0.0,
        n_bins: int = 255,
    ):
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.min_data_in_leaf = min_data_in_leaf
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.n_bins = n_bins

        self.root: Optional[TreeNode] = None
        self.histograms: Optional[List[FeatureHistogram]] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_binned: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        histograms: List[FeatureHistogram],
    ) -> 'LGBMTree':
        """
        Build tree using leaf-wise growth.

        Parameters
        ----------
        X_binned   : (n_samples, n_features) int array of bin indices
        gradients  : (n_samples,) first-order gradients
        hessians   : (n_samples,) second-order gradients
        histograms : list of FeatureHistogram (one per feature, already fit)
        """
        self.histograms = histograms
        n_samples, n_features = X_binned.shape

        root_node = TreeNode(is_leaf=True)
        root_indices = np.arange(n_samples)

        # Each entry: (node, sample_indices, depth)
        leaves: List[Tuple[TreeNode, np.ndarray, int]] = [
            (root_node, root_indices, 0)
        ]

        n_leaves = 1
        self.root = root_node

        # Set leaf values for root
        G_root = gradients[root_indices].sum()
        H_root = hessians[root_indices].sum()
        root_node.leaf_value = self._leaf_weight(G_root, H_root)

        while n_leaves < self.num_leaves:
            # Find the best split across all current leaves
            best_gain = -np.inf
            best_leaf_idx = -1
            best_split = None  # (feature_idx, bin_threshold, real_threshold, G_L, H_L, G_R, H_R, left_idx, right_idx)

            for leaf_idx, (node, indices, depth) in enumerate(leaves):
                if len(indices) < 2 * self.min_data_in_leaf:
                    continue
                if self.max_depth > 0 and depth >= self.max_depth:
                    continue

                split = self._find_best_split(
                    X_binned, gradients, hessians, indices, histograms, n_features
                )
                if split is not None and split[0] > best_gain:
                    best_gain = split[0]
                    best_leaf_idx = leaf_idx
                    best_split = split

            if best_leaf_idx == -1 or best_gain <= self.gamma:
                break  # No valid split found

            # Apply the best split
            gain, feat, bin_thresh, real_thresh, G_L, H_L, G_R, H_R, left_idx, right_idx = best_split
            node, indices, depth = leaves[best_leaf_idx]

            node.is_leaf = False
            node.feature_index = feat
            node.bin_threshold = bin_thresh
            node.real_threshold = real_thresh
            node.gain = gain

            left_node = TreeNode(
                is_leaf=True,
                leaf_value=self._leaf_weight(G_L, H_L)
            )
            right_node = TreeNode(
                is_leaf=True,
                leaf_value=self._leaf_weight(G_R, H_R)
            )
            node.left = left_node
            node.right = right_node

            leaves.pop(best_leaf_idx)
            leaves.append((left_node,  left_idx,  depth + 1))
            leaves.append((right_node, right_idx, depth + 1))
            n_leaves += 1

        return self

    def _find_best_split(
        self,
        X_binned: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        indices: np.ndarray,
        histograms: List[FeatureHistogram],
        n_features: int,
    ) -> Optional[Tuple]:
        """Find the best (gain, feature, bin_threshold, ...) for a leaf."""
        G_node = gradients[indices].sum()
        H_node = hessians[indices].sum()

        best_gain = -np.inf
        result = None

        for feat in range(n_features):
            hist = histograms[feat]
            bin_vals = X_binned[indices, feat]
            G_bins, H_bins, cnt_bins = hist.build(bin_vals, gradients[indices], hessians[indices])

            G_left = 0.0
            H_left = 0.0
            cnt_left = 0

            for b in range(len(hist.bin_edges) - 1):
                G_left += G_bins[b]
                H_left += H_bins[b]
                cnt_left += cnt_bins[b]
                cnt_right = len(indices) - cnt_left

                if cnt_left < self.min_data_in_leaf or cnt_right < self.min_data_in_leaf:
                    continue
                if H_left < self.min_child_weight or (H_node - H_left) < self.min_child_weight:
                    continue

                G_right = G_node - G_left
                H_right = H_node - H_left

                gain = self._split_gain(G_left, H_left, G_right, H_right, G_node, H_node)

                if gain > best_gain:
                    best_gain = gain
                    # Recover sample indices
                    mask_left = bin_vals <= b
                    left_idx = indices[mask_left]
                    right_idx = indices[~mask_left]
                    real_thresh = float(hist.bin_edges[b])
                    result = (gain, feat, b, real_thresh,
                              G_left, H_left, G_right, H_right,
                              left_idx, right_idx)

        return result if result is not None and best_gain > self.gamma else None

    def _split_gain(
        self,
        G_L: float, H_L: float,
        G_R: float, H_R: float,
        G:   float, H:   float,
    ) -> float:
        """
        Gain = 0.5 * [score(L) + score(R) - score(parent)] - gamma

        score(G, H) = G^2 / (H + lambda)
        """
        def score(g, h):
            return g ** 2 / (h + self.reg_lambda)

        return 0.5 * (score(G_L, H_L) + score(G_R, H_R) - score(G, H)) - self.gamma

    def _leaf_weight(self, G: float, H: float) -> float:
        """
        Optimal leaf weight with L1 + L2 regularization.

        w* = -clip(G, alpha) / (H + lambda)
        """
        if self.reg_alpha > 0:
            if G > self.reg_alpha:
                G_reg = G - self.reg_alpha
            elif G < -self.reg_alpha:
                G_reg = G + self.reg_alpha
            else:
                G_reg = 0.0
        else:
            G_reg = G
        return -G_reg / (H + self.reg_lambda)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X_binned: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x, self.root) for x in X_binned])

    def _predict_one(self, x: np.ndarray, node: TreeNode) -> float:
        if node.is_leaf:
            return node.leaf_value
        if x[node.feature_index] <= node.bin_threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def get_feature_importances(self, n_features: int) -> np.ndarray:
        imp = np.zeros(n_features)
        self._collect_gains(self.root, imp)
        return imp

    def _collect_gains(self, node: Optional[TreeNode], imp: np.ndarray):
        if node is None or node.is_leaf:
            return
        imp[node.feature_index] += node.gain
        self._collect_gains(node.left, imp)
        self._collect_gains(node.right, imp)


# ---------------------------------------------------------------------------
# GOSS sampler
# ---------------------------------------------------------------------------

def goss_sample(
    gradients: np.ndarray,
    top_rate: float = 0.2,
    other_rate: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gradient-based One-Side Sampling (GOSS).

    Keep all samples with large gradients (top_rate fraction),
    randomly sample from the rest (other_rate fraction).
    Scale the small-gradient samples to preserve the distribution.

    Returns
    -------
    indices : selected sample indices
    weights : importance weights (1.0 for large, scale for small)
    """
    n = len(gradients)
    abs_grads = np.abs(gradients)
    sorted_idx = np.argsort(abs_grads)[::-1]

    n_large = max(1, int(n * top_rate))
    n_small = max(1, int(n * other_rate))

    large_idx = sorted_idx[:n_large]
    small_pool = sorted_idx[n_large:]
    small_idx = np.random.choice(small_pool, size=min(n_small, len(small_pool)), replace=False)

    scale = (1.0 - top_rate) / other_rate if other_rate > 0 else 1.0

    indices = np.concatenate([large_idx, small_idx])
    weights = np.ones(len(indices))
    weights[n_large:] = scale

    return indices, weights


# ---------------------------------------------------------------------------
# LightGBM Regressor
# ---------------------------------------------------------------------------

class LightGBMRegressor:
    """
    LightGBM for regression (squared error loss).

    Innovations over XGBoost:
    - Histogram-based splits (faster)
    - Leaf-wise tree growth (more accurate per tree)
    - Optional GOSS sampling

    Parameters
    ----------
    n_estimators     : number of trees
    learning_rate    : shrinkage factor eta
    num_leaves       : max leaves per tree (leaf-wise control)
    max_depth        : hard depth limit (-1 = unlimited)
    min_child_weight : min sum of hessians in a leaf
    min_data_in_leaf : min samples in a leaf
    reg_lambda       : L2 regularization
    reg_alpha        : L1 regularization
    gamma            : min gain to split
    n_bins           : number of histogram bins (default 255)
    subsample        : row subsampling fraction (< 1 = stochastic)
    colsample_bytree : feature subsampling fraction
    use_goss         : enable GOSS sampling
    goss_top_rate    : fraction of large-gradient samples to keep
    goss_other_rate  : fraction of small-gradient samples to keep
    random_state     : seed
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_weight: float = 1e-3,
        min_data_in_leaf: int = 20,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        gamma: float = 0.0,
        n_bins: int = 255,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        use_goss: bool = False,
        goss_top_rate: float = 0.2,
        goss_other_rate: float = 0.1,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.min_data_in_leaf = min_data_in_leaf
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.n_bins = n_bins
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.use_goss = use_goss
        self.goss_top_rate = goss_top_rate
        self.goss_other_rate = goss_other_rate
        self.random_state = random_state

        self.trees_: List[LGBMTree] = []
        self.histograms_: Optional[List[FeatureHistogram]] = None
        self.base_score_: float = 0.0
        self.feature_importances_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.selected_features_: Optional[List[np.ndarray]] = None

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LightGBMRegressor':
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # Build global histograms (fit once on full data)
        self.histograms_ = [
            FeatureHistogram(self.n_bins).fit(X[:, f]) for f in range(n_features)
        ]
        X_binned = np.column_stack([
            self.histograms_[f].transform(X[:, f]) for f in range(n_features)
        ]).astype(int)

        self.base_score_ = float(np.mean(y))
        F = np.full(n_samples, self.base_score_)

        importance_acc = np.zeros(n_features)
        self.trees_.clear()
        self.selected_features_ = []

        for _ in range(self.n_estimators):
            # Gradients and hessians for squared error
            gradients = F - y          # dL/dF = F - y
            hessians  = np.ones(n_samples)  # d²L/dF² = 1

            # Row sampling
            if self.use_goss:
                row_idx, goss_w = goss_sample(gradients, self.goss_top_rate, self.goss_other_rate)
                gradients_s = gradients[row_idx] * goss_w
                hessians_s  = hessians[row_idx]  * goss_w
                X_s = X_binned[row_idx]
            elif self.subsample < 1.0:
                row_idx = np.random.choice(n_samples, int(n_samples * self.subsample), replace=False)
                gradients_s = gradients[row_idx]
                hessians_s  = hessians[row_idx]
                X_s = X_binned[row_idx]
            else:
                row_idx = np.arange(n_samples)
                gradients_s = gradients
                hessians_s  = hessians
                X_s = X_binned

            # Feature sampling
            if self.colsample_bytree < 1.0:
                n_sel = max(1, int(n_features * self.colsample_bytree))
                sel_feats = np.random.choice(n_features, n_sel, replace=False)
            else:
                sel_feats = np.arange(n_features)
            self.selected_features_.append(sel_feats)

            hists_sel = [self.histograms_[f] for f in sel_feats]
            X_s_sel = X_s[:, sel_feats]

            tree = LGBMTree(
                num_leaves=self.num_leaves,
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                min_data_in_leaf=self.min_data_in_leaf,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
                gamma=self.gamma,
                n_bins=self.n_bins,
            )
            tree.fit(X_s_sel, gradients_s, hessians_s, hists_sel)

            # Predict on full data using selected features
            preds = tree.predict(X_binned[:, sel_feats])
            F += self.learning_rate * preds

            # Accumulate importances (map back to original feature indices)
            local_imp = tree.get_feature_importances(len(sel_feats))
            for local_i, global_i in enumerate(sel_feats):
                importance_acc[global_i] += local_imp[local_i]

            self.trees_.append(tree)

        total = importance_acc.sum()
        self.feature_importances_ = importance_acc / total if total > 0 else importance_acc
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        X_binned = np.column_stack([
            self.histograms_[f].transform(X[:, f]) for f in range(self.n_features_)
        ]).astype(int)

        F = np.full(X.shape[0], self.base_score_)
        for tree, sel_feats in zip(self.trees_, self.selected_features_):
            F += self.learning_rate * tree.predict(X_binned[:, sel_feats])
        return F

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ---------------------------------------------------------------------------
# LightGBM Classifier
# ---------------------------------------------------------------------------

class LightGBMClassifier:
    """
    LightGBM for binary classification (log loss).

    Same innovations as LightGBMRegressor but uses sigmoid + log loss.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_weight: float = 1e-3,
        min_data_in_leaf: int = 20,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        gamma: float = 0.0,
        n_bins: int = 255,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        use_goss: bool = False,
        goss_top_rate: float = 0.2,
        goss_other_rate: float = 0.1,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.min_data_in_leaf = min_data_in_leaf
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.n_bins = n_bins
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.use_goss = use_goss
        self.goss_top_rate = goss_top_rate
        self.goss_other_rate = goss_other_rate
        self.random_state = random_state

        self.trees_: List[LGBMTree] = []
        self.histograms_: Optional[List[FeatureHistogram]] = None
        self.base_score_: float = 0.0
        self.feature_importances_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.selected_features_: Optional[List[np.ndarray]] = None

        if random_state is not None:
            np.random.seed(random_state)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0,
                        1.0 / (1.0 + np.exp(-x)),
                        np.exp(x) / (1.0 + np.exp(x)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LightGBMClassifier':
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # Global histograms
        self.histograms_ = [
            FeatureHistogram(self.n_bins).fit(X[:, f]) for f in range(n_features)
        ]
        X_binned = np.column_stack([
            self.histograms_[f].transform(X[:, f]) for f in range(n_features)
        ]).astype(int)

        # Base score: log-odds
        pos = np.sum(y == 1)
        neg = n_samples - pos
        self.base_score_ = float(np.log(pos / neg)) if pos > 0 and neg > 0 else 0.0
        F = np.full(n_samples, self.base_score_)

        importance_acc = np.zeros(n_features)
        self.trees_.clear()
        self.selected_features_ = []

        for _ in range(self.n_estimators):
            probs = self._sigmoid(F)
            gradients = probs - y          # dL/dF
            hessians  = probs * (1 - probs)  # d²L/dF²
            hessians  = np.maximum(hessians, 1e-6)

            # Row sampling
            if self.use_goss:
                row_idx, goss_w = goss_sample(gradients, self.goss_top_rate, self.goss_other_rate)
                gradients_s = gradients[row_idx] * goss_w
                hessians_s  = hessians[row_idx]  * goss_w
                X_s = X_binned[row_idx]
            elif self.subsample < 1.0:
                row_idx = np.random.choice(n_samples, int(n_samples * self.subsample), replace=False)
                gradients_s = gradients[row_idx]
                hessians_s  = hessians[row_idx]
                X_s = X_binned[row_idx]
            else:
                gradients_s = gradients
                hessians_s  = hessians
                X_s = X_binned

            # Feature sampling
            if self.colsample_bytree < 1.0:
                n_sel = max(1, int(n_features * self.colsample_bytree))
                sel_feats = np.random.choice(n_features, n_sel, replace=False)
            else:
                sel_feats = np.arange(n_features)
            self.selected_features_.append(sel_feats)

            hists_sel = [self.histograms_[f] for f in sel_feats]
            X_s_sel = X_s[:, sel_feats]

            tree = LGBMTree(
                num_leaves=self.num_leaves,
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                min_data_in_leaf=self.min_data_in_leaf,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
                gamma=self.gamma,
                n_bins=self.n_bins,
            )
            tree.fit(X_s_sel, gradients_s, hessians_s, hists_sel)

            F += self.learning_rate * tree.predict(X_binned[:, sel_feats])

            local_imp = tree.get_feature_importances(len(sel_feats))
            for local_i, global_i in enumerate(sel_feats):
                importance_acc[global_i] += local_imp[local_i]

            self.trees_.append(tree)

        total = importance_acc.sum()
        self.feature_importances_ = importance_acc / total if total > 0 else importance_acc
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        X_binned = np.column_stack([
            self.histograms_[f].transform(X[:, f]) for f in range(self.n_features_)
        ]).astype(int)

        F = np.full(X.shape[0], self.base_score_)
        for tree, sel_feats in zip(self.trees_, self.selected_features_):
            F += self.learning_rate * tree.predict(X_binned[:, sel_feats])

        prob_pos = self._sigmoid(F)
        return np.column_stack([1 - prob_pos, prob_pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.array(y)))
