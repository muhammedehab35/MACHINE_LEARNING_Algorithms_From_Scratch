"""
Comprehensive Tests for UMAP
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from umap_scratch import (
    UMAP, compute_fuzzy_simplicial_set, find_ab_params,
    _knn_distances, _smooth_knn_dist, _pairwise_sq_distances,
    umap_optimize_layout,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blobs(centers, n=30, std=0.3, seed=42):
    np.random.seed(seed)
    X = np.vstack([np.random.randn(n, 2) * std + c for c in centers])
    y = np.repeat(np.arange(len(centers)), n)
    return X, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_shape():
    """embedding_ must have shape (n, n_components)."""
    print("Test 1: Output Shape")
    print("-" * 70)

    np.random.seed(0)
    X = np.random.randn(40, 5)
    for d in [1, 2]:
        umap = UMAP(n_components=d, n_neighbors=8, n_epochs=50, random_state=0)
        Z = umap.fit_transform(X)
        assert Z.shape == (40, d), f"Expected (40, {d}), got {Z.shape}"
        print(f"   [OK] n_components={d}: shape={Z.shape}")
    return True


def test_embedding_attribute():
    """embedding_ attribute must exist and equal fit_transform output."""
    print("\nTest 2: embedding_ Attribute")
    print("-" * 70)

    np.random.seed(1)
    X = np.random.randn(30, 4)
    umap = UMAP(n_components=2, n_neighbors=6, n_epochs=30, random_state=1)
    Z = umap.fit_transform(X)
    assert hasattr(umap, 'embedding_')
    assert np.array_equal(Z, umap.embedding_)
    print(f"   [OK] embedding_ shape: {umap.embedding_.shape}")
    return True


def test_pairwise_distances():
    """D must be symmetric, have zero diagonal, and be non-negative."""
    print("\nTest 3: Pairwise Distance Matrix")
    print("-" * 70)

    np.random.seed(2)
    X = np.random.randn(20, 4)
    D = _pairwise_sq_distances(X)
    assert D.shape == (20, 20)
    assert np.allclose(D, D.T, atol=1e-10)
    assert np.allclose(np.diag(D), 0.0, atol=1e-10)
    assert np.all(D >= 0)
    print(f"   [OK] D symmetric, zero diagonal, non-negative")
    return True


def test_knn_distances():
    """k-NN distances must be sorted, positive, and not include self."""
    print("\nTest 4: k-NN Distances")
    print("-" * 70)

    np.random.seed(3)
    X = np.random.randn(25, 3)
    k = 5
    dists, idx = _knn_distances(X, k)
    assert dists.shape == (25, k)
    assert idx.shape == (25, k)
    assert np.all(np.diff(dists, axis=1) >= -1e-10), "Distances not sorted ascending"
    assert np.all(dists > 0), "All k-NN distances must be positive"
    for i in range(25):
        assert i not in idx[i], f"Point {i} is its own neighbour"
    print(f"   [OK] distances sorted, positive, self excluded")
    return True


def test_smooth_knn_dist():
    """sigma > 0; rho = nearest-neighbour distance; sum of memberships ~ log2(k)."""
    print("\nTest 5: Smooth kNN Distance (rho, sigma)")
    print("-" * 70)

    np.random.seed(4)
    X = np.random.randn(30, 4)
    k = 8
    dists, _ = _knn_distances(X, k)
    rho, sigma = _smooth_knn_dist(dists, k)
    assert rho.shape == (30,) and sigma.shape == (30,)
    assert np.all(sigma > 0), "sigma must be positive"
    assert np.allclose(rho, dists[:, 0]), "rho should equal nearest-neighbour distance"

    target = np.log2(k)
    for i in range(30):
        shifted = np.maximum(dists[i] - rho[i], 0.0)
        val = np.sum(np.exp(-shifted / sigma[i]))
        assert abs(val - target) < 0.1, f"point {i}: sum={val:.3f} != {target:.3f}"
    print(f"   [OK] rho=nearest-neighbour dist, sigma>0, sum~log2(k)={target:.3f}")
    return True


def test_find_ab_params():
    """a, b must be positive; q(0)~1 and q at large d must be small."""
    print("\nTest 6: find_ab_params")
    print("-" * 70)

    a, b = find_ab_params(spread=1.0, min_dist=0.1)
    assert a > 0 and b > 0, f"a={a}, b={b} must both be positive"
    q0 = 1.0 / (1.0 + a * (1e-10) ** (2 * b))
    assert q0 > 0.99, f"q(0)={q0:.6f} should be ~1"
    q_large = 1.0 / (1.0 + a * 10.0 ** (2 * b))
    assert q_large < 0.1, f"q(10)={q_large:.6f} should be small"
    print(f"   [OK] a={a:.4f}, b={b:.4f}, q(0)={q0:.6f}, q(10)={q_large:.8f}")
    return True


def test_fuzzy_graph_symmetry():
    """W must be symmetric with values in [0, 1]."""
    print("\nTest 7: Fuzzy Graph Symmetry")
    print("-" * 70)

    np.random.seed(5)
    X = np.random.randn(25, 4)
    W = compute_fuzzy_simplicial_set(X, n_neighbors=6)
    assert np.allclose(W, W.T, atol=1e-10), "W must be symmetric"
    assert np.all(W >= 0), "W must be non-negative"
    assert np.all(W <= 1.0 + 1e-10), "W must be <= 1 (fuzzy membership)"
    print(f"   [OK] W symmetric, values in [0,1], shape={W.shape}")
    return True


def test_fuzzy_graph_connectivity():
    """Each point must have at least n_neighbors non-zero edges (due to symmetrisation)."""
    print("\nTest 8: Fuzzy Graph Connectivity")
    print("-" * 70)

    np.random.seed(6)
    X = np.random.randn(30, 3)
    k = 7
    W = compute_fuzzy_simplicial_set(X, n_neighbors=k)
    counts = (W > 0).sum(axis=1)
    assert np.all(counts >= k), \
        f"Some points have < {k} connections: min={counts.min()}"
    print(f"   [OK] min connections per point = {counts.min()} >= {k}")
    return True


def test_random_state_reproducibility():
    """Same random_state must give identical embeddings."""
    print("\nTest 9: Random State Reproducibility")
    print("-" * 70)

    np.random.seed(7)
    X = np.random.randn(30, 4)
    u1 = UMAP(n_components=2, n_neighbors=8, n_epochs=50, random_state=42)
    u2 = UMAP(n_components=2, n_neighbors=8, n_epochs=50, random_state=42)
    Z1 = u1.fit_transform(X)
    Z2 = u2.fit_transform(X)
    assert np.allclose(Z1, Z2, atol=1e-12), "Same seed must give identical results"
    print(f"   [OK] Two runs with seed=42 are identical")
    return True


def test_cluster_separation():
    """UMAP must separate well-separated 3-blob clusters."""
    print("\nTest 10: Cluster Separation on 3-Blob Data")
    print("-" * 70)

    centers = [[8, 0], [-8, 0], [0, 8]]
    X, true = _blobs(centers, n=30, std=0.3, seed=42)

    umap = UMAP(n_components=2, n_neighbors=8, n_epochs=300,
                learning_rate=1.0, random_state=42, init='spectral')
    Z = umap.fit_transform(X)

    centroids = np.array([Z[true == k].mean(axis=0) for k in range(3)])
    inter = np.min([np.linalg.norm(centroids[i] - centroids[j])
                    for i in range(3) for j in range(i + 1, 3)])
    intra = np.max([np.mean(np.linalg.norm(Z[true == k] - centroids[k], axis=1))
                    for k in range(3)])

    assert inter > intra, \
        f"Clusters not separated: inter={inter:.3f}, intra={intra:.3f}"
    print(f"   [OK] inter-cluster dist={inter:.3f} >> intra={intra:.3f}")
    return True


def test_small_dataset():
    """UMAP must work on n < 2*n_neighbors."""
    print("\nTest 11: Small Dataset (n=12, n_neighbors=4)")
    print("-" * 70)

    np.random.seed(8)
    X = np.random.randn(12, 3)
    umap = UMAP(n_components=2, n_neighbors=4, n_epochs=50, random_state=8)
    Z = umap.fit_transform(X)
    assert Z.shape == (12, 2)
    assert np.all(np.isfinite(Z)), "Embedding contains non-finite values"
    print(f"   [OK] shape={Z.shape}, all finite")
    return True


def test_different_n_components():
    """Works for n_components = 1, 2, 3."""
    print("\nTest 12: Different n_components")
    print("-" * 70)

    np.random.seed(9)
    X = np.random.randn(35, 6)
    for d in [1, 2, 3]:
        umap = UMAP(n_components=d, n_neighbors=8, n_epochs=30, random_state=9)
        Z = umap.fit_transform(X)
        assert Z.shape == (35, d), f"Expected (35,{d}), got {Z.shape}"
        assert np.all(np.isfinite(Z))
        print(f"   [OK] n_components={d}: shape={Z.shape}")
    return True


def test_init_pca():
    """init='pca' must produce valid finite output."""
    print("\nTest 13: PCA Initialization")
    print("-" * 70)

    np.random.seed(10)
    X = np.random.randn(30, 6)
    umap = UMAP(n_components=2, n_neighbors=7, n_epochs=30,
                init='pca', random_state=10)
    Z = umap.fit_transform(X)
    assert Z.shape == (30, 2)
    assert np.all(np.isfinite(Z))
    print(f"   [OK] PCA init shape={Z.shape}, all finite")
    return True


def test_graph_attribute():
    """graph_ must be set after fit: symmetric n x n matrix."""
    print("\nTest 14: graph_ Attribute")
    print("-" * 70)

    np.random.seed(11)
    X = np.random.randn(20, 3)
    umap = UMAP(n_components=2, n_neighbors=5, n_epochs=20, random_state=11)
    umap.fit(X)
    assert hasattr(umap, 'graph_') and umap.graph_ is not None
    assert umap.graph_.shape == (20, 20)
    assert np.allclose(umap.graph_, umap.graph_.T, atol=1e-10)
    print(f"   [OK] graph_ shape={umap.graph_.shape}, symmetric")
    return True


def test_ab_params_positive():
    """a_ and b_ must be set and positive after fit."""
    print("\nTest 15: a_ and b_ Parameters")
    print("-" * 70)

    np.random.seed(12)
    X = np.random.randn(25, 4)
    umap = UMAP(n_components=2, n_neighbors=6, n_epochs=20,
                min_dist=0.1, spread=1.0, random_state=12)
    umap.fit(X)
    assert umap.a_ is not None and umap.b_ is not None
    assert umap.a_ > 0 and umap.b_ > 0, f"a_={umap.a_}, b_={umap.b_}"
    print(f"   [OK] a_={umap.a_:.4f}, b_={umap.b_:.4f}")
    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    print("Testing UMAP")
    print("=" * 70)

    tests = [
        test_output_shape,
        test_embedding_attribute,
        test_pairwise_distances,
        test_knn_distances,
        test_smooth_knn_dist,
        test_find_ab_params,
        test_fuzzy_graph_symmetry,
        test_fuzzy_graph_connectivity,
        test_random_state_reproducibility,
        test_cluster_separation,
        test_small_dataset,
        test_different_n_components,
        test_init_pca,
        test_graph_attribute,
        test_ab_params_positive,
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
