"""
Comprehensive Tests for Principal Component Analysis
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pca_scratch import PCA, pca_svd


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fit_shapes():
    """All attribute shapes must match n_components and p."""
    print("Test 1: Fit Shapes")
    print("-" * 70)

    np.random.seed(0)
    X = np.random.randn(100, 8)

    for d in [1, 3, 8]:
        pca = PCA(n_components=d).fit(X)
        assert pca.components_.shape == (d, 8), f"d={d}: bad components shape"
        assert pca.explained_variance_.shape == (d,)
        assert pca.explained_variance_ratio_.shape == (d,)
        assert pca.singular_values_.shape == (d,)
        assert pca.mean_.shape == (8,)
        assert pca.n_components_ == d
        print(f"   [OK] n_components={d}: all shapes correct")
    return True


def test_explained_variance_ratio_sums_to_one():
    """All explained_variance_ratio_ must sum to 1 when keeping all PCs."""
    print("\nTest 2: Explained Variance Ratio Sums to 1")
    print("-" * 70)

    np.random.seed(1)
    X = np.random.randn(50, 6)
    pca = PCA(n_components=None).fit(X)
    total = pca.explained_variance_ratio_.sum()
    assert abs(total - 1.0) < 1e-10, f"Sum = {total}"
    print(f"   [OK] Sum of all ratios = {total:.12f}")
    return True


def test_transform_shape():
    """transform() output shape must be (n, n_components)."""
    print("\nTest 3: Transform Shape")
    print("-" * 70)

    np.random.seed(2)
    X_train = np.random.randn(80, 10)
    X_test = np.random.randn(20, 10)
    pca = PCA(n_components=4).fit(X_train)
    Z = pca.transform(X_test)
    assert Z.shape == (20, 4), f"Expected (20, 4), got {Z.shape}"
    print(f"   [OK] transform shape: {Z.shape}")
    return True


def test_perfect_reconstruction():
    """Full PCA (d=p) must reconstruct X exactly."""
    print("\nTest 4: Perfect Reconstruction (d = p)")
    print("-" * 70)

    np.random.seed(3)
    X = np.random.randn(50, 5)
    pca = PCA(n_components=5)
    Z = pca.fit_transform(X)
    X_rec = pca.inverse_transform(Z)
    err = np.max(np.abs(X - X_rec))
    assert err < 1e-10, f"Max reconstruction error: {err:.2e}"
    print(f"   [OK] Max reconstruction error = {err:.2e}")
    return True


def test_partial_reconstruction_error():
    """Fewer PCs must give higher reconstruction error."""
    print("\nTest 5: Partial Reconstruction Error Increases")
    print("-" * 70)

    np.random.seed(4)
    X = np.random.randn(100, 10)
    errors = []
    for d in [1, 3, 5, 10]:
        pca = PCA(n_components=d).fit(X)
        X_rec = pca.inverse_transform(pca.transform(X))
        errors.append(float(np.mean((X - X_rec) ** 2)))

    for i in range(1, len(errors)):
        assert errors[i] <= errors[i - 1] + 1e-12, \
            f"Error at d={[1,3,5,10][i]} not smaller than d={[1,3,5,10][i-1]}"
    print(f"   [OK] MSE errors: {[f'{e:.4f}' for e in errors]}")
    return True


def test_mean_centering():
    """mean_ must equal the per-feature mean of training data."""
    print("\nTest 6: Mean Centering")
    print("-" * 70)

    np.random.seed(5)
    X = np.random.randn(60, 7) + np.arange(7)
    pca = PCA(n_components=3).fit(X)
    assert np.allclose(pca.mean_, X.mean(axis=0), atol=1e-12)
    print(f"   [OK] mean_ matches X.mean(axis=0)")
    return True


def test_orthonormality():
    """Principal components must be orthonormal (W W^T = I)."""
    print("\nTest 7: Orthonormality of Components")
    print("-" * 70)

    np.random.seed(6)
    X = np.random.randn(80, 8)
    pca = PCA(n_components=5).fit(X)
    W = pca.components_                           # (5, 8)
    gram = W @ W.T                                # should be I_5
    assert np.allclose(gram, np.eye(5), atol=1e-10), \
        f"Max off-diag: {np.max(np.abs(gram - np.eye(5))):.2e}"
    print(f"   [OK] W W^T = I_5 (max error {np.max(np.abs(gram-np.eye(5))):.2e})")
    return True


def test_1d_in_2d():
    """PCA must identify the direction of maximum variance in 2D data."""
    print("\nTest 8: 1D Direction in 2D Data")
    print("-" * 70)

    np.random.seed(7)
    t = np.linspace(0, 5, 200)
    X = np.column_stack([t, 2.0 * t + np.random.randn(200) * 0.02])
    pca = PCA(n_components=2).fit(X)

    # First PC should explain > 99.9% of variance
    assert pca.explained_variance_ratio_[0] > 0.999, \
        f"PC1 ratio = {pca.explained_variance_ratio_[0]:.4f}"

    # PC direction should be proportional to [1, 2] / sqrt(5)
    expected = np.array([1.0, 2.0]) / np.sqrt(5.0)
    pc1 = pca.components_[0]
    cos_sim = abs(pc1 @ expected)
    assert cos_sim > 0.999, f"cos(PC1, true dir) = {cos_sim:.4f}"
    print(f"   [OK] PC1 explains {pca.explained_variance_ratio_[0]*100:.2f}% variance")
    print(f"   [OK] cos(PC1, true direction) = {cos_sim:.6f}")
    return True


def test_variance_decreasing():
    """Explained variance values must be non-increasing."""
    print("\nTest 9: Variance Non-Increasing")
    print("-" * 70)

    np.random.seed(8)
    X = np.random.randn(100, 12)
    pca = PCA().fit(X)
    ev = pca.explained_variance_
    for i in range(1, len(ev)):
        assert ev[i] <= ev[i - 1] + 1e-12, \
            f"Variance increased at component {i}"
    print(f"   [OK] {len(ev)} eigenvalues are non-increasing")
    return True


def test_n_components_none():
    """n_components=None must retain min(n, p) components."""
    print("\nTest 10: n_components=None Keeps All")
    print("-" * 70)

    np.random.seed(9)
    X = np.random.randn(40, 10)
    pca = PCA(n_components=None).fit(X)
    expected = min(40, 10)
    assert pca.n_components_ == expected, \
        f"Expected {expected}, got {pca.n_components_}"
    print(f"   [OK] n_components_ = {pca.n_components_} = min(n=40, p=10)")
    return True


def test_n_components_float():
    """Float n_components must keep enough PCs for that fraction of variance."""
    print("\nTest 11: Float n_components (Variance Threshold)")
    print("-" * 70)

    np.random.seed(10)
    # Correlated data: first few PCs dominate
    A = np.random.randn(100, 3)
    X = A @ np.random.randn(3, 8)  + np.random.randn(100, 8) * 0.1

    for frac in [0.80, 0.95, 0.99]:
        pca = PCA(n_components=frac).fit(X)
        total_ratio = pca.explained_variance_ratio_.sum()
        assert total_ratio >= frac, \
            f"frac={frac}: only {total_ratio:.4f} variance explained"
        print(f"   [OK] threshold={frac}: {pca.n_components_} PCs, "
              f"{total_ratio*100:.2f}% variance")
    return True


def test_whitening():
    """Whitened scores must have unit variance along each PC."""
    print("\nTest 12: Whitening")
    print("-" * 70)

    np.random.seed(11)
    X = np.random.randn(200, 5) @ np.array([
        [3, 1, 0, 0, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0.5, 0],
        [0, 0, 0, 0, 0.2],
    ])
    pca = PCA(n_components=5, whiten=True).fit(X)
    Z = pca.transform(X)
    var_z = Z.var(axis=0, ddof=1)
    assert np.allclose(var_z, 1.0, atol=0.05), f"Variances: {var_z}"
    print(f"   [OK] Whitened PC variances: {var_z.round(3)} (all approx 1.0)")
    return True


def test_fit_transform_equals_transform():
    """fit_transform(X) must equal fit(X).transform(X)."""
    print("\nTest 13: fit_transform == fit().transform()")
    print("-" * 70)

    np.random.seed(12)
    X = np.random.randn(60, 7)
    Z1 = PCA(n_components=4).fit_transform(X)
    Z2 = PCA(n_components=4).fit(X).transform(X)
    assert np.allclose(Z1, Z2, atol=1e-12)
    print(f"   [OK] Max difference: {np.max(np.abs(Z1 - Z2)):.2e}")
    return True


def test_score_increases_with_components():
    """More components must give a higher (less negative) score."""
    print("\nTest 14: Score Increases with n_components")
    print("-" * 70)

    np.random.seed(13)
    X = np.random.randn(80, 10)
    prev_score = -np.inf
    for d in [1, 3, 5, 8, 10]:
        pca = PCA(n_components=d).fit(X)
        s = pca.score(X)
        assert s >= prev_score - 1e-12, f"Score decreased at d={d}"
        prev_score = s
        print(f"   [OK] d={d:2d}: score = {s:.6f}")
    return True


def test_noise_variance():
    """noise_variance_ must be non-negative and zero when d=p."""
    print("\nTest 15: Noise Variance")
    print("-" * 70)

    np.random.seed(14)
    X = np.random.randn(50, 6)

    pca_partial = PCA(n_components=3).fit(X)
    assert pca_partial.noise_variance_ >= 0
    print(f"   [OK] noise_variance_ (d=3) = {pca_partial.noise_variance_:.6f} >= 0")

    pca_full = PCA(n_components=6).fit(X)
    assert pca_full.noise_variance_ == 0.0
    print(f"   [OK] noise_variance_ (d=6) = {pca_full.noise_variance_}")
    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    print("Testing Principal Component Analysis")
    print("=" * 70)

    tests = [
        test_fit_shapes,
        test_explained_variance_ratio_sums_to_one,
        test_transform_shape,
        test_perfect_reconstruction,
        test_partial_reconstruction_error,
        test_mean_centering,
        test_orthonormality,
        test_1d_in_2d,
        test_variance_decreasing,
        test_n_components_none,
        test_n_components_float,
        test_whitening,
        test_fit_transform_equals_transform,
        test_score_increases_with_components,
        test_noise_variance,
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
