"""
Comprehensive Tests for DBSCAN From-Scratch Implementation
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dbscan_scratch import DBSCAN, pairwise_distances, k_dist, cluster_stats


def test_two_blobs():
    """Well-separated blobs should yield exactly 2 clusters, no noise."""
    print("Test 1: Two Well-Separated Blobs")
    print("-" * 70)

    np.random.seed(42)
    X = np.vstack([
        np.random.randn(50, 2) * 0.3 + [5, 5],
        np.random.randn(50, 2) * 0.3 + [-5, -5],
    ])

    db = DBSCAN(eps=0.8, min_samples=5)
    db.fit(X)

    assert db.n_clusters_ == 2, f"Expected 2 clusters, got {db.n_clusters_}"
    assert (db.labels_ == -1).sum() == 0, "Expected no noise"
    print(f"   [OK] Found {db.n_clusters_} clusters, 0 noise points")
    return True


def test_noise_detection():
    """Isolated points should be labelled as noise (-1)."""
    print("\nTest 2: Noise Detection")
    print("-" * 70)

    np.random.seed(0)
    X_cluster = np.random.randn(40, 2) * 0.3
    X_noise = np.array([[10, 10], [-10, -10], [10, -10], [0, 15]])
    X = np.vstack([X_cluster, X_noise])

    db = DBSCAN(eps=0.6, min_samples=4)
    db.fit(X)

    noise_mask = db.labels_ == -1
    # The 4 isolated points should all be noise
    assert noise_mask[-4:].all(), "Isolated points should be noise"
    print(f"   [OK] {noise_mask.sum()} noise points detected (expected >= 4)")
    return True


def test_circles():
    """DBSCAN should separate concentric circles where K-Means fails."""
    print("\nTest 3: Concentric Circles")
    print("-" * 70)

    from sklearn.datasets import make_circles
    X, y = make_circles(n_samples=200, factor=0.5, noise=0.05, random_state=42)

    db = DBSCAN(eps=0.2, min_samples=5)
    db.fit(X)

    assert db.n_clusters_ == 2, f"Expected 2 clusters, got {db.n_clusters_}"
    # Check that labels correlate with true structure
    lab0 = db.labels_[y == 0]
    lab1 = db.labels_[y == 1]
    # Each true ring should be mostly one cluster
    dominant_0 = np.bincount(lab0[lab0 >= 0]).max() if (lab0 >= 0).any() else 0
    dominant_1 = np.bincount(lab1[lab1 >= 0]).max() if (lab1 >= 0).any() else 0
    assert dominant_0 > 0.8 * len(lab0)
    assert dominant_1 > 0.8 * len(lab1)
    print(f"   [OK] Found {db.n_clusters_} clusters on circles (K-Means would fail)")
    return True


def test_moons():
    """DBSCAN should separate two moons."""
    print("\nTest 4: Two Moons")
    print("-" * 70)

    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=200, noise=0.07, random_state=42)

    db = DBSCAN(eps=0.25, min_samples=5)
    db.fit(X)

    assert db.n_clusters_ == 2, f"Expected 2 clusters, got {db.n_clusters_}"
    print(f"   [OK] Found {db.n_clusters_} clusters on moons")
    return True


def test_labels_shape():
    """labels_ has correct shape and valid values."""
    print("\nTest 5: Labels Shape and Values")
    print("-" * 70)

    np.random.seed(1)
    X = np.random.randn(80, 3)
    db = DBSCAN(eps=1.0, min_samples=5)
    db.fit(X)

    assert db.labels_.shape == (80,)
    assert db.labels_.min() >= -1
    assert db.labels_.max() == db.n_clusters_ - 1
    print(f"   [OK] labels_ shape {db.labels_.shape}, range [{db.labels_.min()}, {db.labels_.max()}]")
    return True


def test_core_sample_indices():
    """core_sample_indices_ contains only true core points."""
    print("\nTest 6: Core Sample Indices")
    print("-" * 70)

    np.random.seed(42)
    X = np.vstack([np.random.randn(60, 2) * 0.4 + c for c in [[3, 3], [-3, -3]]])
    db = DBSCAN(eps=0.8, min_samples=5)
    db.fit(X)

    D = pairwise_distances(X)
    for idx in db.core_sample_indices_:
        n_nbrs = (D[idx] <= db.eps).sum()
        assert n_nbrs >= db.min_samples, f"Point {idx} claimed as core but has {n_nbrs} nbrs"

    # Non-core points should not be in the list
    all_idx = set(range(len(X)))
    non_core = all_idx - set(db.core_sample_indices_)
    for idx in non_core:
        n_nbrs = (D[idx] <= db.eps).sum()
        assert n_nbrs < db.min_samples

    print(f"   [OK] {len(db.core_sample_indices_)} core points verified")
    return True


def test_fit_predict():
    """fit_predict returns same as fit().labels_."""
    print("\nTest 7: fit_predict")
    print("-" * 70)

    np.random.seed(3)
    X = np.random.randn(60, 2)
    db = DBSCAN(eps=0.6, min_samples=4)
    labels = db.fit_predict(X)
    assert np.array_equal(labels, db.labels_)
    print("   [OK] fit_predict == fit().labels_")
    return True


def test_eps_effect():
    """Smaller eps means more noise; larger eps merges clusters."""
    print("\nTest 8: Eps Effect")
    print("-" * 70)

    np.random.seed(42)
    X = np.vstack([np.random.randn(40, 2) * 0.5 + c for c in [[3, 3], [-3, -3]]])

    # Tiny eps: most points become noise
    db_tiny = DBSCAN(eps=0.05, min_samples=3).fit(X)
    # Large eps: all merges into 1 cluster
    db_large = DBSCAN(eps=20.0, min_samples=3).fit(X)

    assert db_tiny.n_clusters_ <= 2
    assert (db_tiny.labels_ == -1).sum() > 0   # expect noise
    assert db_large.n_clusters_ == 1            # merged

    print(f"   [OK] eps=0.05: {db_tiny.n_clusters_} clusters, {(db_tiny.labels_==-1).sum()} noise")
    print(f"   [OK] eps=20.0: {db_large.n_clusters_} clusters, {(db_large.labels_==-1).sum()} noise")
    return True


def test_min_samples_effect():
    """Higher min_samples creates more noise."""
    print("\nTest 9: min_samples Effect")
    print("-" * 70)

    np.random.seed(42)
    X = np.vstack([np.random.randn(50, 2) * 0.5 + c for c in [[4, 0], [-4, 0]]])

    noise_counts = []
    for ms in [2, 5, 15, 30]:
        db = DBSCAN(eps=0.8, min_samples=ms).fit(X)
        noise_counts.append((db.labels_ == -1).sum())
        print(f"   [OK] min_samples={ms:2d}: noise={noise_counts[-1]}")

    assert noise_counts[-1] >= noise_counts[0], "More min_samples -> more noise"
    return True


def test_manhattan_metric():
    """DBSCAN works with Manhattan distance."""
    print("\nTest 10: Manhattan Metric")
    print("-" * 70)

    np.random.seed(42)
    X = np.vstack([np.random.randn(40, 2) * 0.4 + c for c in [[4, 0], [-4, 0]]])
    db = DBSCAN(eps=1.0, min_samples=4, metric='manhattan').fit(X)

    assert db.n_clusters_ >= 1
    print(f"   [OK] Manhattan metric: {db.n_clusters_} clusters, {(db.labels_==-1).sum()} noise")
    return True


def test_custom_metric():
    """DBSCAN accepts a callable distance metric."""
    print("\nTest 11: Custom Callable Metric")
    print("-" * 70)

    np.random.seed(42)
    X = np.vstack([np.random.randn(30, 2) * 0.4 + c for c in [[3, 0], [-3, 0]]])

    def my_l2(u, v):
        return float(np.sqrt(np.sum((u - v) ** 2)))

    db = DBSCAN(eps=0.8, min_samples=4, metric=my_l2).fit(X)
    assert db.n_clusters_ >= 1
    print(f"   [OK] Custom callable metric: {db.n_clusters_} clusters")
    return True


def test_single_cluster():
    """Dense uniform blob yields exactly 1 cluster."""
    print("\nTest 12: Single Dense Cluster")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(60, 2) * 0.2
    db = DBSCAN(eps=0.5, min_samples=4).fit(X)

    assert db.n_clusters_ == 1
    assert (db.labels_ == -1).sum() == 0
    print(f"   [OK] 1 cluster, 0 noise from dense uniform blob")
    return True


def test_cluster_stats():
    """cluster_stats reports correct counts."""
    print("\nTest 13: Cluster Stats")
    print("-" * 70)

    np.random.seed(42)
    X_cluster = np.vstack([np.random.randn(50, 2) * 0.3 + c for c in [[4, 4], [-4, -4]]])
    X_noise = np.array([[20, 20], [-20, 20]])
    X = np.vstack([X_cluster, X_noise])

    db = DBSCAN(eps=0.7, min_samples=4).fit(X)
    stats = cluster_stats(db.labels_)

    assert stats['n_clusters'] == 2
    assert stats['n_noise'] >= 2
    print(f"   [OK] Stats: {stats['n_clusters']} clusters, {stats['n_noise']} noise, "
          f"sizes={stats['cluster_sizes']}")
    return True


def test_k_dist():
    """k_dist returns sorted descending array of length n."""
    print("\nTest 14: k-dist Graph")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(80, 2)
    kd = k_dist(X, k=4)

    assert len(kd) == 80
    # Should be sorted descending
    assert np.all(kd[:-1] >= kd[1:] - 1e-10)
    print(f"   [OK] k-dist length={len(kd)}, max={kd[0]:.3f}, min={kd[-1]:.3f}")
    return True


def test_arbitrary_shape():
    """DBSCAN handles arbitrary-shaped clusters."""
    print("\nTest 15: Arbitrary Shape (S-curve)")
    print("-" * 70)

    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 150)
    X1 = np.c_[np.sin(t), t / (2 * np.pi)] + np.random.randn(150, 2) * 0.06
    X2 = np.c_[np.sin(t) + 3, t / (2 * np.pi)] + np.random.randn(150, 2) * 0.06
    X = np.vstack([X1, X2])

    db = DBSCAN(eps=0.2, min_samples=5).fit(X)
    assert db.n_clusters_ >= 2
    noise_frac = (db.labels_ == -1).mean()
    print(f"   [OK] S-curve: {db.n_clusters_} clusters, noise={noise_frac:.2%}")
    return True


def run_all_tests():
    print("Testing DBSCAN From-Scratch Implementation")
    print("=" * 70)

    tests = [
        test_two_blobs,
        test_noise_detection,
        test_circles,
        test_moons,
        test_labels_shape,
        test_core_sample_indices,
        test_fit_predict,
        test_eps_effect,
        test_min_samples_effect,
        test_manhattan_metric,
        test_custom_metric,
        test_single_cluster,
        test_cluster_stats,
        test_k_dist,
        test_arbitrary_shape,
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
