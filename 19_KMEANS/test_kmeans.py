"""
Comprehensive Tests for K-Means From-Scratch Implementation
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmeans_scratch import (
    KMeans, euclidean_distances, inertia,
    silhouette_score, davies_bouldin_score, elbow_scores
)


def test_euclidean_distances():
    """Verify squared distance computation."""
    print("Test 1: Euclidean Distances")
    print("-" * 70)

    X = np.array([[0, 0], [3, 4]], dtype=float)
    C = np.array([[0, 0]], dtype=float)
    d = euclidean_distances(X, C)
    assert d.shape == (2, 1)
    assert abs(d[0, 0]) < 1e-9
    assert abs(d[1, 0] - 25.0) < 1e-9
    print("   [OK] ||[3,4] - [0,0]||^2 = 25")

    X2 = np.random.randn(50, 10)
    C2 = np.random.randn(5, 10)
    D = euclidean_distances(X2, C2)
    assert D.shape == (50, 5)
    assert np.all(D >= 0)
    print("   [OK] Shape and non-negativity")
    return True


def test_kmeans_basic():
    """Fit on well-separated Gaussian blobs."""
    print("\nTest 2: Basic KMeans on Blobs")
    print("-" * 70)

    np.random.seed(42)
    centers = np.array([[5, 5], [-5, -5], [5, -5]])
    X = np.vstack([np.random.randn(100, 2) + c for c in centers])

    km = KMeans(n_clusters=3, random_state=42)
    km.fit(X)

    assert km.cluster_centers_.shape == (3, 2)
    assert km.labels_.shape == (300,)
    assert len(np.unique(km.labels_)) == 3
    print(f"   [OK] Centroids shape: {km.cluster_centers_.shape}")

    # Check each cluster is near one of the true centers
    found_centers = set()
    for true_c in centers:
        dists = np.linalg.norm(km.cluster_centers_ - true_c, axis=1)
        closest = np.argmin(dists)
        assert dists[closest] < 2.0
        found_centers.add(closest)
    assert len(found_centers) == 3
    print("   [OK] Centroids near true centers")
    return True


def test_predict():
    """predict() assigns to nearest centroid."""
    print("\nTest 3: Predict")
    print("-" * 70)

    np.random.seed(0)
    X = np.vstack([np.random.randn(50, 2) + c for c in [[4, 4], [-4, -4]]])
    km = KMeans(n_clusters=2, random_state=0)
    km.fit(X)

    labels = km.predict(X)
    assert labels.shape == (100,)
    assert len(np.unique(labels)) == 2
    print("   [OK] predict() returns correct shape and labels")

    # New point very close to centroid 0 should be assigned to cluster 0
    c0 = km.cluster_centers_[0]
    test_pt = (c0 + 0.01).reshape(1, -1)
    assert km.predict(test_pt)[0] == 0
    print("   [OK] New point near centroid 0 -> cluster 0")
    return True


def test_fit_predict():
    """fit_predict returns same as fit().labels_."""
    print("\nTest 4: fit_predict")
    print("-" * 70)

    np.random.seed(7)
    X = np.random.randn(80, 3)
    km = KMeans(n_clusters=4, random_state=7)
    labels = km.fit_predict(X)
    assert np.array_equal(labels, km.labels_)
    print("   [OK] fit_predict == fit().labels_")
    return True


def test_transform():
    """transform() returns distances to each centroid."""
    print("\nTest 5: Transform")
    print("-" * 70)

    np.random.seed(1)
    X = np.vstack([np.random.randn(40, 2) + c for c in [[3, 0], [-3, 0], [0, 3]]])
    km = KMeans(n_clusters=3, random_state=1)
    km.fit(X)
    D = km.transform(X)

    assert D.shape == (120, 3)
    assert np.all(D >= 0)
    # Distance from centroid to itself should be near 0
    for j in range(3):
        c = km.cluster_centers_[j:j+1]
        d = km.transform(c)[0, j]
        assert d < 0.01
    print("   [OK] transform() shape and centroid self-distance ~ 0")
    return True


def test_inertia():
    """Inertia decreases with more clusters."""
    print("\nTest 6: Inertia vs K")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(200, 2)
    prev_inertia = np.inf

    for k in [1, 2, 4, 8]:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        assert km.inertia_ < prev_inertia + 1e-3
        prev_inertia = km.inertia_
        print(f"   [OK] K={k:2d}  inertia={km.inertia_:.2f}")
    return True


def test_kmeans_plus_plus_vs_random():
    """K-Means++ should achieve lower or equal inertia vs random init."""
    print("\nTest 7: KMeans++ vs Random Init")
    print("-" * 70)

    np.random.seed(42)
    X = np.vstack([np.random.randn(80, 2) + c for c in [[4, 0], [-4, 0], [0, 4], [0, -4]]])

    km_pp = KMeans(n_clusters=4, init='k-means++', n_init=5, random_state=42)
    km_rnd = KMeans(n_clusters=4, init='random', n_init=5, random_state=42)
    km_pp.fit(X)
    km_rnd.fit(X)

    print(f"   [OK] KMeans++ inertia: {km_pp.inertia_:.2f}")
    print(f"   [OK] Random   inertia: {km_rnd.inertia_:.2f}")
    # K-Means++ should find at least as good a solution
    assert km_pp.inertia_ <= km_rnd.inertia_ * 1.1, "KMeans++ should not be much worse than random"
    return True


def test_convergence():
    """Algorithm converges (n_iter_ < max_iter on easy data)."""
    print("\nTest 8: Convergence")
    print("-" * 70)

    np.random.seed(0)
    X = np.vstack([np.random.randn(50, 2) + c for c in [[10, 0], [-10, 0]]])
    km = KMeans(n_clusters=2, max_iter=300, tol=1e-4, random_state=0)
    km.fit(X)

    print(f"   [OK] Converged in {km.n_iter_} iterations (max 300)")
    assert km.n_iter_ < 300
    assert len(km.inertia_history_) == km.n_iter_
    print("   [OK] inertia_history_ length matches n_iter_")
    return True


def test_inertia_monotone():
    """Inertia should be non-increasing over iterations."""
    print("\nTest 9: Inertia History Monotone")
    print("-" * 70)

    np.random.seed(5)
    X = np.vstack([np.random.randn(60, 2) + c for c in [[3, 3], [-3, 3], [0, -3]]])
    km = KMeans(n_clusters=3, n_init=1, random_state=5)
    km.fit(X)

    hist = km.inertia_history_
    for i in range(1, len(hist)):
        assert hist[i] <= hist[i-1] + 1e-6, f"Inertia increased at step {i}"
    print(f"   [OK] Inertia monotonically non-increasing over {len(hist)} steps")
    return True


def test_score():
    """score() returns negative inertia."""
    print("\nTest 10: Score")
    print("-" * 70)

    np.random.seed(3)
    X = np.random.randn(100, 2)
    km = KMeans(n_clusters=3, random_state=3)
    km.fit(X)

    s = km.score(X)
    assert s < 0
    assert abs(s + km.inertia_) < 1e-6
    print(f"   [OK] score={s:.4f} == -inertia={-km.inertia_:.4f}")
    return True


def test_silhouette():
    """Silhouette score is higher for well-separated clusters."""
    print("\nTest 11: Silhouette Score")
    print("-" * 70)

    np.random.seed(42)
    X_sep = np.vstack([np.random.randn(40, 2) * 0.3 + c for c in [[5, 0], [-5, 0]]])
    X_mix = np.random.randn(80, 2)

    km_sep = KMeans(n_clusters=2, random_state=42).fit(X_sep)
    km_mix = KMeans(n_clusters=2, random_state=42).fit(X_mix)

    sil_sep = silhouette_score(X_sep, km_sep.labels_)
    sil_mix = silhouette_score(X_mix, km_mix.labels_)

    print(f"   [OK] Silhouette (separated): {sil_sep:.3f}")
    print(f"   [OK] Silhouette (mixed):     {sil_mix:.3f}")
    assert sil_sep > sil_mix, "Separated clusters should have higher silhouette"
    assert sil_sep > 0.5
    return True


def test_davies_bouldin():
    """Davies-Bouldin is lower for better-separated clusters."""
    print("\nTest 12: Davies-Bouldin Score")
    print("-" * 70)

    np.random.seed(42)
    X_sep = np.vstack([np.random.randn(40, 2) * 0.3 + c for c in [[6, 0], [-6, 0], [0, 6]]])
    X_mix = np.random.randn(120, 2)

    km_sep = KMeans(n_clusters=3, random_state=42).fit(X_sep)
    km_mix = KMeans(n_clusters=3, random_state=42).fit(X_mix)

    db_sep = davies_bouldin_score(X_sep, km_sep.labels_, km_sep.cluster_centers_)
    db_mix = davies_bouldin_score(X_mix, km_mix.labels_, km_mix.cluster_centers_)

    print(f"   [OK] DB (separated): {db_sep:.3f}")
    print(f"   [OK] DB (mixed):     {db_mix:.3f}")
    assert db_sep < db_mix, "Separated clusters should have lower DB index"
    return True


def test_n_init():
    """Multiple runs (n_init) yields better inertia than single run."""
    print("\nTest 13: Multiple Runs (n_init)")
    print("-" * 70)

    np.random.seed(9)
    X = np.vstack([np.random.randn(50, 2) + c for c in [[3, 3], [-3, 3], [3, -3], [-3, -3]]])

    km1 = KMeans(n_clusters=4, n_init=1, init='random', random_state=9)
    km10 = KMeans(n_clusters=4, n_init=10, init='random', random_state=9)
    km1.fit(X)
    km10.fit(X)

    print(f"   [OK] n_init=1   inertia: {km1.inertia_:.2f}")
    print(f"   [OK] n_init=10  inertia: {km10.inertia_:.2f}")
    assert km10.inertia_ <= km1.inertia_ + 1e-3
    return True


def test_elbow_scores():
    """elbow_scores() returns one inertia per K."""
    print("\nTest 14: Elbow Scores")
    print("-" * 70)

    np.random.seed(42)
    X = np.vstack([np.random.randn(40, 2) + c for c in [[4, 0], [-4, 0], [0, 4]]])
    ks = [1, 2, 3, 4, 5, 6]
    scores = elbow_scores(X, ks, random_state=42)

    assert len(scores) == len(ks)
    # Inertia should decrease from K=1 to K=3 (true K)
    assert scores[0] > scores[1] > scores[2]
    print(f"   [OK] Inertia for K=1..6: {[round(s, 1) for s in scores]}")
    return True


def test_high_dimensional():
    """KMeans works on high-dimensional data."""
    print("\nTest 15: High Dimensional (50D)")
    print("-" * 70)

    np.random.seed(0)
    centers = np.random.randn(5, 50) * 10
    X = np.vstack([np.random.randn(30, 50) + c for c in centers])

    km = KMeans(n_clusters=5, random_state=0, n_init=3)
    km.fit(X)

    assert km.cluster_centers_.shape == (5, 50)
    assert len(np.unique(km.labels_)) == 5
    print(f"   [OK] Converged in {km.n_iter_} iterations on 150x50 data")
    return True


def run_all_tests():
    print("Testing K-Means From-Scratch Implementation")
    print("=" * 70)

    tests = [
        test_euclidean_distances,
        test_kmeans_basic,
        test_predict,
        test_fit_predict,
        test_transform,
        test_inertia,
        test_kmeans_plus_plus_vs_random,
        test_convergence,
        test_inertia_monotone,
        test_score,
        test_silhouette,
        test_davies_bouldin,
        test_n_init,
        test_elbow_scores,
        test_high_dimensional,
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
