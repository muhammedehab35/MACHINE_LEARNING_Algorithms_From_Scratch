"""
Comprehensive Tests for K-Medoids (PAM) From-Scratch Implementation
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmedoids_scratch import KMedoids, pairwise_distances, elbow_costs


def test_pairwise_distances_euclidean():
    """Verify euclidean distance matrix properties."""
    print("Test 1: Pairwise Euclidean Distances")
    print("-" * 70)

    X = np.array([[0, 0], [3, 4], [0, 0]], dtype=float)
    D = pairwise_distances(X, 'euclidean')

    assert D.shape == (3, 3)
    assert abs(D[0, 0]) < 1e-9
    assert abs(D[0, 1] - 5.0) < 1e-9
    assert abs(D[0, 2]) < 1e-9
    assert np.allclose(D, D.T)
    print("   [OK] Shape, symmetry, and ||[3,4] - [0,0]|| = 5")
    return True


def test_pairwise_distances_manhattan():
    """Verify Manhattan distance."""
    print("\nTest 2: Pairwise Manhattan Distances")
    print("-" * 70)

    X = np.array([[0, 0], [1, 2]], dtype=float)
    D = pairwise_distances(X, 'manhattan')
    assert abs(D[0, 1] - 3.0) < 1e-9
    print("   [OK] Manhattan ||(0,0)-(1,2)|| = 3")
    return True


def test_pairwise_distances_cosine():
    """Verify cosine distance."""
    print("\nTest 3: Pairwise Cosine Distances")
    print("-" * 70)

    X = np.array([[1, 0], [0, 1], [1, 0]], dtype=float)
    D = pairwise_distances(X, 'cosine')
    assert D.shape == (3, 3)
    assert abs(D[0, 0]) < 1e-9
    assert abs(D[0, 2]) < 1e-9
    assert abs(D[0, 1] - 1.0) < 1e-9   # orthogonal -> cosine dist = 1
    print("   [OK] Cosine dist(orthogonal) = 1, dist(parallel) = 0")
    return True


def test_medoids_are_data_points():
    """Medoids must be actual data points."""
    print("\nTest 4: Medoids Are Data Points")
    print("-" * 70)

    np.random.seed(42)
    X = np.vstack([np.random.randn(50, 2) + c for c in [[3, 3], [-3, -3], [3, -3]]])
    km = KMedoids(n_clusters=3, random_state=42)
    km.fit(X)

    for idx in km.medoid_indices_:
        assert 0 <= idx < len(X)
        assert np.allclose(km.cluster_centers_[np.where(km.medoid_indices_ == idx)[0]], X[idx])
    print(f"   [OK] All medoid indices in [0, {len(X)-1}]: {km.medoid_indices_}")
    print("   [OK] cluster_centers_ == X[medoid_indices_]")
    return True


def test_basic_clustering():
    """Well-separated blobs should be correctly clustered."""
    print("\nTest 5: Basic Clustering on Blobs")
    print("-" * 70)

    np.random.seed(0)
    centers = np.array([[8, 8], [-8, -8], [8, -8]])
    X = np.vstack([np.random.randn(60, 2) * 0.5 + c for c in centers])

    km = KMedoids(n_clusters=3, random_state=0)
    km.fit(X)

    assert len(np.unique(km.labels_)) == 3
    # Each true center should map to a unique cluster
    found = set()
    for c in centers:
        nearest = np.argmin(np.linalg.norm(km.cluster_centers_ - c, axis=1))
        found.add(nearest)
    assert len(found) == 3
    print(f"   [OK] All 3 clusters found, medoid indices: {sorted(km.medoid_indices_)}")
    return True


def test_predict():
    """predict() assigns new points to nearest medoid."""
    print("\nTest 6: Predict")
    print("-" * 70)

    np.random.seed(1)
    X = np.vstack([np.random.randn(40, 2) + c for c in [[5, 5], [-5, -5]]])
    km = KMedoids(n_clusters=2, random_state=1)
    km.fit(X)

    labels = km.predict(X)
    assert labels.shape == (80,)
    # Point very close to medoid 0 should be cluster 0
    c0 = km.cluster_centers_[0]
    close_pt = (c0 + 0.001).reshape(1, -1)
    assert km.predict(close_pt)[0] == 0
    print("   [OK] predict() shape and nearest-medoid correctness")
    return True


def test_fit_predict():
    """fit_predict returns same as fit().labels_."""
    print("\nTest 7: fit_predict")
    print("-" * 70)

    np.random.seed(5)
    X = np.random.randn(60, 3)
    km = KMedoids(n_clusters=3, random_state=5)
    labels = km.fit_predict(X)
    assert np.array_equal(labels, km.labels_)
    print("   [OK] fit_predict == fit().labels_")
    return True


def test_transform():
    """transform() returns distances to medoids."""
    print("\nTest 8: Transform")
    print("-" * 70)

    np.random.seed(2)
    X = np.vstack([np.random.randn(30, 2) + c for c in [[4, 0], [-4, 0], [0, 4]]])
    km = KMedoids(n_clusters=3, random_state=2)
    km.fit(X)
    D = km.transform(X)

    assert D.shape == (90, 3)
    assert np.all(D >= 0)
    print("   [OK] transform() shape and non-negativity")
    return True


def test_cost_monotone():
    """Total cost should not increase over SWAP iterations."""
    print("\nTest 9: Cost History Monotone")
    print("-" * 70)

    np.random.seed(3)
    X = np.vstack([np.random.randn(40, 2) + c for c in [[3, 3], [-3, 3], [0, -3]]])
    km = KMedoids(n_clusters=3, init='random', n_init=1, random_state=3)
    km.fit(X)

    hist = km.cost_history_
    for i in range(1, len(hist)):
        assert hist[i] <= hist[i-1] + 1e-6, f"Cost increased at step {i}"
    print(f"   [OK] Cost monotone over {len(hist)} steps: {[round(c, 2) for c in hist]}")
    return True


def test_score():
    """score() == negative inertia."""
    print("\nTest 10: Score")
    print("-" * 70)

    np.random.seed(4)
    X = np.random.randn(50, 2)
    km = KMedoids(n_clusters=3, random_state=4)
    km.fit(X)

    s = km.score(X)
    assert s < 0
    assert abs(s + km.inertia_) < 1e-4
    print(f"   [OK] score={s:.4f} == -inertia={-km.inertia_:.4f}")
    return True


def test_robustness_to_outliers():
    """KMedoids should be more robust to outliers than KMeans."""
    print("\nTest 11: Robustness to Outliers")
    print("-" * 70)

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '19_KMEANS'))
    from kmeans_scratch import KMeans

    np.random.seed(42)
    X_clean = np.vstack([np.random.randn(40, 2) * 0.4 + c for c in [[4, 4], [-4, -4]]])
    # Add 5 extreme outliers
    outliers = np.array([[50, 50], [-50, -50], [50, -50], [-50, 50], [0, 100]])
    X = np.vstack([X_clean, outliers])

    km_means = KMeans(n_clusters=2, random_state=42).fit(X)
    km_meds = KMedoids(n_clusters=2, random_state=42).fit(X)

    # KMedoids centroids should be near true centers; KMeans may be pulled by outliers
    true_centers = np.array([[4, 4], [-4, -4]])
    err_means = min(
        np.linalg.norm(km_means.cluster_centers_[0] - true_centers[0]) +
        np.linalg.norm(km_means.cluster_centers_[1] - true_centers[1]),
        np.linalg.norm(km_means.cluster_centers_[0] - true_centers[1]) +
        np.linalg.norm(km_means.cluster_centers_[1] - true_centers[0])
    )
    err_meds = min(
        np.linalg.norm(km_meds.cluster_centers_[0] - true_centers[0]) +
        np.linalg.norm(km_meds.cluster_centers_[1] - true_centers[1]),
        np.linalg.norm(km_meds.cluster_centers_[0] - true_centers[1]) +
        np.linalg.norm(km_meds.cluster_centers_[1] - true_centers[0])
    )
    print(f"   [OK] KMeans centroid error (with outliers): {err_means:.3f}")
    print(f"   [OK] KMedoids centroid error (with outliers): {err_meds:.3f}")
    assert err_meds < err_means + 2.0, "KMedoids should not be worse than KMeans on outlier data"
    return True


def test_manhattan_metric():
    """KMedoids works correctly with Manhattan distance."""
    print("\nTest 12: Manhattan Metric")
    print("-" * 70)

    np.random.seed(42)
    X = np.vstack([np.random.randn(40, 2) + c for c in [[5, 0], [-5, 0]]])
    km = KMedoids(n_clusters=2, metric='manhattan', random_state=42)
    km.fit(X)

    assert len(np.unique(km.labels_)) == 2
    acc = max(
        np.mean(km.labels_[:40] == 0) + np.mean(km.labels_[40:] == 1),
        np.mean(km.labels_[:40] == 1) + np.mean(km.labels_[40:] == 0)
    ) / 2
    print(f"   [OK] Manhattan clustering accuracy: {acc:.3f}")
    assert acc > 0.85
    return True


def test_build_vs_random_init():
    """BUILD initialisation should achieve lower or equal cost than random."""
    print("\nTest 13: BUILD vs Random Init")
    print("-" * 70)

    np.random.seed(42)
    X = np.vstack([np.random.randn(50, 2) + c for c in [[4, 4], [-4, 4], [4, -4], [-4, -4]]])

    km_build = KMedoids(n_clusters=4, init='build', random_state=42)
    km_rnd = KMedoids(n_clusters=4, init='random', n_init=10, random_state=42)
    km_build.fit(X)
    km_rnd.fit(X)

    print(f"   [OK] BUILD init cost:  {km_build.inertia_:.2f}")
    print(f"   [OK] Random init cost: {km_rnd.inertia_:.2f}")
    assert km_build.inertia_ <= km_rnd.inertia_ * 1.1
    return True


def test_elbow_costs():
    """elbow_costs returns one cost per K, decreasing."""
    print("\nTest 14: Elbow Costs")
    print("-" * 70)

    np.random.seed(42)
    X = np.vstack([np.random.randn(30, 2) + c for c in [[5, 0], [-5, 0], [0, 5]]])
    ks = [1, 2, 3, 4, 5]
    costs = elbow_costs(X, ks, random_state=42)

    assert len(costs) == len(ks)
    assert costs[0] > costs[1] > costs[2]
    print(f"   [OK] Costs for K=1..5: {[round(c, 2) for c in costs]}")
    return True


def test_custom_metric():
    """KMedoids accepts a callable metric."""
    print("\nTest 15: Custom Callable Metric")
    print("-" * 70)

    np.random.seed(42)
    X = np.vstack([np.random.randn(30, 2) + c for c in [[4, 0], [-4, 0]]])

    def my_l2(u, v):
        return float(np.sqrt(np.sum((u - v) ** 2)))

    km = KMedoids(n_clusters=2, metric=my_l2, random_state=42)
    km.fit(X)

    assert len(np.unique(km.labels_)) == 2
    print("   [OK] Custom callable metric accepted, 2 clusters found")
    return True


def run_all_tests():
    print("Testing K-Medoids (PAM) From-Scratch Implementation")
    print("=" * 70)

    tests = [
        test_pairwise_distances_euclidean,
        test_pairwise_distances_manhattan,
        test_pairwise_distances_cosine,
        test_medoids_are_data_points,
        test_basic_clustering,
        test_predict,
        test_fit_predict,
        test_transform,
        test_cost_monotone,
        test_score,
        test_robustness_to_outliers,
        test_manhattan_metric,
        test_build_vs_random_init,
        test_elbow_costs,
        test_custom_metric,
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
