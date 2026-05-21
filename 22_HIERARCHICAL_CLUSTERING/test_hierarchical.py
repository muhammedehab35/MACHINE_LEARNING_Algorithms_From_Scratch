"""
Comprehensive Tests for Agglomerative Hierarchical Clustering
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hierarchical_scratch import (
    AgglomerativeClustering, linkage, fcluster,
    cophenetic_correlation, pairwise_distances
)


def test_linkage_matrix_shape():
    """Linkage matrix must have shape (n-1, 4)."""
    print("Test 1: Linkage Matrix Shape")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(10, 2)
    for method in ['single', 'complete', 'average', 'ward']:
        Z = linkage(X, method=method)
        assert Z.shape == (9, 4), f"{method}: expected (9,4), got {Z.shape}"
        print(f"   [OK] {method}: Z.shape = {Z.shape}")
    return True


def test_linkage_indices_valid():
    """Cluster indices in Z must be valid (0 to 2n-2)."""
    print("\nTest 2: Linkage Indices Valid")
    print("-" * 70)

    np.random.seed(1)
    X = np.random.randn(8, 2)
    Z = linkage(X, method='ward')
    n = len(X)
    for k in range(n - 1):
        ci, cj = int(Z[k, 0]), int(Z[k, 1])
        assert 0 <= ci < 2 * n - 1
        assert 0 <= cj < 2 * n - 1
        assert ci != cj
    print(f"   [OK] All {n-1} merge indices are valid")
    return True


def test_merge_distances_non_decreasing():
    """Merge distances must be non-decreasing for valid dendrograms."""
    print("\nTest 3: Non-Decreasing Merge Distances")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(15, 2)
    for method in ['single', 'complete', 'average', 'ward']:
        Z = linkage(X, method=method)
        heights = Z[:, 2]
        for i in range(1, len(heights)):
            assert heights[i] >= heights[i-1] - 1e-9, \
                f"{method}: heights not non-decreasing at step {i}: {heights[i-1]:.4f} > {heights[i]:.4f}"
        print(f"   [OK] {method}: heights non-decreasing")
    return True


def test_cluster_sizes():
    """Cluster sizes in Z must be cumulative merges."""
    print("\nTest 4: Cluster Sizes in Z")
    print("-" * 70)

    np.random.seed(0)
    X = np.random.randn(6, 2)
    Z = linkage(X, method='ward')
    n = len(X)
    # Track sizes
    sizes = {i: 1 for i in range(n)}
    for k in range(n - 1):
        ci, cj, _, sz = int(Z[k, 0]), int(Z[k, 1]), Z[k, 2], int(Z[k, 3])
        expected = sizes[ci] + sizes[cj]
        assert sz == expected, f"Step {k}: expected size {expected}, got {sz}"
        sizes[n + k] = sz
    print(f"   [OK] All {n-1} cluster sizes correct")
    return True


def test_fcluster_maxclust():
    """fcluster with maxclust=K returns exactly K unique labels."""
    print("\nTest 5: fcluster maxclust")
    print("-" * 70)

    np.random.seed(42)
    X = np.vstack([np.random.randn(20, 2) + c for c in [[3,3], [-3,3], [0,-3]]])
    Z = linkage(X, method='ward')

    for k in [2, 3, 4]:
        labels = fcluster(Z, k, criterion='maxclust')
        assert labels.shape == (60,)
        n_unique = len(np.unique(labels))
        assert n_unique == k, f"Expected {k} clusters, got {n_unique}"
        print(f"   [OK] maxclust={k}: {n_unique} clusters, shape={labels.shape}")
    return True


def test_fcluster_distance():
    """fcluster with distance criterion returns labels cut at height t."""
    print("\nTest 6: fcluster distance criterion")
    print("-" * 70)

    np.random.seed(42)
    X = np.vstack([np.random.randn(15, 2) * 0.3 + c for c in [[5,5], [-5,-5]]])
    Z = linkage(X, method='ward')

    # Very large height: one cluster
    labels_high = fcluster(Z, 1e9, criterion='distance')
    assert len(np.unique(labels_high)) == 1
    print(f"   [OK] Very high cut: {len(np.unique(labels_high))} cluster")

    # Very small height: n clusters
    labels_low = fcluster(Z, 0.0, criterion='distance')
    assert len(np.unique(labels_low)) == 30
    print(f"   [OK] Very low cut: {len(np.unique(labels_low))} clusters (singletons)")
    return True


def test_ward_blobs():
    """Ward linkage should correctly cluster 3 Gaussian blobs."""
    print("\nTest 7: Ward on 3 Blobs")
    print("-" * 70)

    np.random.seed(42)
    centers = [[5, 5], [-5, 5], [0, -5]]
    X = np.vstack([np.random.randn(40, 2) * 0.5 + c for c in centers])
    true_labels = np.repeat([0, 1, 2], 40)

    ac = AgglomerativeClustering(n_clusters=3, linkage_method='ward')
    ac.fit(X)

    # Check that the clustering matches true labels (up to permutation)
    contingency = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            contingency[i, j] = int(np.sum((true_labels == i) & (ac.labels_ == j)))
    # Each true cluster should map mostly to one predicted cluster
    for i in range(3):
        assert contingency[i].max() > 35, f"True cluster {i} not well recovered"
    print(f"   [OK] Ward correctly clusters 3 blobs")
    return True


def test_single_linkage_chaining():
    """Single linkage should exhibit chaining on elongated clusters."""
    print("\nTest 8: Single Linkage Chaining")
    print("-" * 70)

    np.random.seed(42)
    # Two elongated chains
    X1 = np.c_[np.linspace(0, 5, 20), np.zeros(20)]
    X2 = np.c_[np.linspace(0, 5, 20), np.ones(20) * 3]
    X = np.vstack([X1, X2])

    ac_single = AgglomerativeClustering(n_clusters=2, linkage_method='single')
    ac_ward = AgglomerativeClustering(n_clusters=2, linkage_method='ward')
    ac_single.fit(X)
    ac_ward.fit(X)

    # Both should find 2 clusters; single linkage may chain differently
    assert len(np.unique(ac_single.labels_)) == 2
    assert len(np.unique(ac_ward.labels_)) == 2
    print(f"   [OK] Single linkage: {len(np.unique(ac_single.labels_))} clusters")
    print(f"   [OK] Ward linkage:   {len(np.unique(ac_ward.labels_))} clusters")
    return True


def test_all_linkage_methods():
    """All linkage methods produce valid outputs."""
    print("\nTest 9: All Linkage Methods")
    print("-" * 70)

    np.random.seed(42)
    X = np.vstack([np.random.randn(20, 2) + c for c in [[3,3], [-3,-3]]])

    for method in ['single', 'complete', 'average', 'ward', 'centroid', 'median']:
        ac = AgglomerativeClustering(n_clusters=2, linkage_method=method)
        ac.fit(X)
        assert len(np.unique(ac.labels_)) == 2
        print(f"   [OK] {method}: 2 clusters found")
    return True


def test_cophenetic_correlation_good():
    """Cophenetic correlation should be high for well-separated clusters."""
    print("\nTest 10: Cophenetic Correlation")
    print("-" * 70)

    np.random.seed(42)
    X_good = np.vstack([np.random.randn(30, 2) * 0.2 + c
                        for c in [[5, 5], [-5, 5], [0, -5]]])
    X_bad = np.random.randn(90, 2)

    Z_good = linkage(X_good, method='average')
    Z_bad = linkage(X_bad, method='average')

    coph_good = cophenetic_correlation(X_good, Z_good)
    coph_bad = cophenetic_correlation(X_bad, Z_bad)

    print(f"   [OK] Cophenetic (well-separated): {coph_good:.4f}")
    print(f"   [OK] Cophenetic (random noise):   {coph_bad:.4f}")
    assert coph_good > coph_bad, "Well-separated data should have higher cophenetic correlation"
    assert coph_good > 0.7
    return True


def test_fit_predict():
    """fit_predict returns same as fit().labels_."""
    print("\nTest 11: fit_predict")
    print("-" * 70)

    np.random.seed(3)
    X = np.random.randn(30, 2)
    ac = AgglomerativeClustering(n_clusters=3, linkage_method='ward')
    labels = ac.fit_predict(X)
    assert np.array_equal(labels, ac.labels_)
    print("   [OK] fit_predict == fit().labels_")
    return True


def test_manhattan_metric():
    """Hierarchical clustering works with Manhattan distance."""
    print("\nTest 12: Manhattan Metric")
    print("-" * 70)

    np.random.seed(42)
    X = np.vstack([np.random.randn(25, 2) + c for c in [[4, 0], [-4, 0]]])
    ac = AgglomerativeClustering(n_clusters=2, linkage_method='single',
                                 metric='manhattan')
    ac.fit(X)
    assert len(np.unique(ac.labels_)) == 2
    print(f"   [OK] Manhattan: 2 clusters found")
    return True


def test_n_clusters_range():
    """Cutting at k=1..n all produce valid partitions."""
    print("\nTest 13: n_clusters Range")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(20, 2)
    Z = linkage(X, method='ward')

    for k in [1, 2, 5, 10, 20]:
        labels = fcluster(Z, k, criterion='maxclust')
        n_unique = len(np.unique(labels))
        assert 1 <= n_unique <= k
        print(f"   [OK] maxclust={k:2d}: {n_unique} clusters")
    return True


def test_ward_minimises_variance():
    """Ward linkage should give lower inertia than single linkage on blobs."""
    print("\nTest 14: Ward Minimises Inertia")
    print("-" * 70)

    np.random.seed(42)
    X = np.vstack([np.random.randn(30, 2) + c for c in [[3,3], [-3,3], [0,-3]]])

    def inertia(X, labels):
        total = 0.0
        for k in np.unique(labels):
            Xk = X[labels == k]
            total += float(np.sum((Xk - Xk.mean(axis=0)) ** 2))
        return total

    ac_ward = AgglomerativeClustering(n_clusters=3, linkage_method='ward').fit(X)
    ac_single = AgglomerativeClustering(n_clusters=3, linkage_method='single').fit(X)

    inert_ward = inertia(X, ac_ward.labels_)
    inert_single = inertia(X, ac_single.labels_)

    print(f"   [OK] Ward inertia:   {inert_ward:.2f}")
    print(f"   [OK] Single inertia: {inert_single:.2f}")
    assert inert_ward <= inert_single
    return True


def test_dendrogram_total_members():
    """The final merge should contain all n points."""
    print("\nTest 15: Final Merge Contains All Points")
    print("-" * 70)

    np.random.seed(7)
    n = 12
    X = np.random.randn(n, 2)
    Z = linkage(X, method='complete')

    final_size = int(Z[-1, 3])
    assert final_size == n, f"Final cluster size {final_size} != n={n}"
    print(f"   [OK] Final merge size = {final_size} = n")
    return True


def run_all_tests():
    print("Testing Agglomerative Hierarchical Clustering")
    print("=" * 70)

    tests = [
        test_linkage_matrix_shape,
        test_linkage_indices_valid,
        test_merge_distances_non_decreasing,
        test_cluster_sizes,
        test_fcluster_maxclust,
        test_fcluster_distance,
        test_ward_blobs,
        test_single_linkage_chaining,
        test_all_linkage_methods,
        test_cophenetic_correlation_good,
        test_fit_predict,
        test_manhattan_metric,
        test_n_clusters_range,
        test_ward_minimises_variance,
        test_dendrogram_total_members,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"   [FAIL] {test.__name__}: {e}")

    print("\n" + "=" * 70)
    print(f"Tests Passed: {passed}/{len(tests)}")
    print(f"Tests Failed: {failed}/{len(tests)}")
    if failed == 0:
        print("\n[OK] All tests passed successfully!")
        return True
    else:
        print(f"\n[FAIL] {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
