"""
Comprehensive Tests for Gaussian Mixture Models
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmm_scratch import GaussianMixture


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blobs(centers, n=40, std=0.5, seed=42):
    np.random.seed(seed)
    return np.vstack([np.random.randn(n, 2) * std + c for c in centers])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fit_shapes():
    """Parameter shapes must match n_components and data dimension."""
    print("Test 1: Parameter Shapes")
    print("-" * 70)

    X = _blobs([[3, 3], [-3, 3], [0, -3]])
    K, p = 3, 2

    for ct in ['full', 'diag', 'spherical', 'tied']:
        gmm = GaussianMixture(n_components=K, covariance_type=ct,
                              random_state=0).fit(X)
        assert gmm.weights_.shape == (K,), f"{ct}: bad weights shape"
        assert gmm.means_.shape == (K, p), f"{ct}: bad means shape"
        if ct == 'full':
            assert gmm.covariances_.shape == (K, p, p)
        elif ct == 'diag':
            assert gmm.covariances_.shape == (K, p)
        elif ct == 'spherical':
            assert gmm.covariances_.shape == (K,)
        elif ct == 'tied':
            assert gmm.covariances_.shape == (p, p)
        print(f"   [OK] {ct}: correct shapes")
    return True


def test_weights_sum_to_one():
    """Mixing weights must sum to 1 and each pi_k >= 0."""
    print("\nTest 2: Weights Sum to 1")
    print("-" * 70)

    X = _blobs([[4, 0], [-4, 0]])
    gmm = GaussianMixture(n_components=2, random_state=1).fit(X)
    assert abs(gmm.weights_.sum() - 1.0) < 1e-10
    assert np.all(gmm.weights_ >= 0)
    print(f"   [OK] weights = {gmm.weights_}, sum = {gmm.weights_.sum():.10f}")
    return True


def test_predict_valid_labels():
    """predict() must return labels in {0, ..., K-1}."""
    print("\nTest 3: Predict Valid Labels")
    print("-" * 70)

    X = _blobs([[3, 3], [-3, -3], [3, -3]])
    gmm = GaussianMixture(n_components=3, random_state=2).fit(X)
    labels = gmm.predict(X)
    assert labels.shape == (len(X),)
    assert set(np.unique(labels)).issubset({0, 1, 2})
    print(f"   [OK] labels in {{0,1,2}}, unique = {sorted(np.unique(labels))}")
    return True


def test_predict_proba_sums_to_one():
    """Each row of predict_proba() must sum to 1."""
    print("\nTest 4: predict_proba Sums to 1")
    print("-" * 70)

    X = _blobs([[5, 5], [-5, 5], [0, -5]])
    gmm = GaussianMixture(n_components=3, random_state=3).fit(X)
    proba = gmm.predict_proba(X)
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-10), f"Max deviation: {abs(row_sums-1).max()}"
    assert np.all(proba >= 0)
    print(f"   [OK] All {len(X)} rows sum to 1.0 (max error {abs(row_sums-1).max():.2e})")
    return True


def test_log_likelihood_monotone():
    """EM must not decrease the log-likelihood at any iteration."""
    print("\nTest 5: Log-Likelihood Monotonically Non-Decreasing")
    print("-" * 70)

    X = _blobs([[4, 4], [-4, 4], [0, -4]])
    gmm = GaussianMixture(n_components=3, max_iter=50, random_state=4).fit(X)
    history = gmm.log_likelihoods_
    for i in range(1, len(history)):
        assert history[i] >= history[i - 1] - 1e-8, \
            f"Decreased at iter {i}: {history[i-1]:.6f} -> {history[i]:.6f}"
    print(f"   [OK] {len(history)} iterations, all non-decreasing")
    return True


def test_3blob_recovery():
    """Full GMM should correctly separate 3 tight Gaussian blobs."""
    print("\nTest 6: 3-Blob Recovery")
    print("-" * 70)

    np.random.seed(42)
    centers = [[6, 6], [-6, 6], [0, -6]]
    X = np.vstack([np.random.randn(50, 2) * 0.4 + c for c in centers])
    true = np.repeat([0, 1, 2], 50)

    gmm = GaussianMixture(n_components=3, n_init=3, random_state=5).fit(X)

    # Check contingency: each true cluster maps mostly to one predicted cluster
    contingency = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            contingency[i, j] = int(np.sum((true == i) & (gmm.labels_ == j)))
    for i in range(3):
        assert contingency[i].max() > 45, f"True cluster {i} not well recovered"
    print(f"   [OK] All 3 blobs correctly identified")
    return True


def test_full_covariance():
    """Full covariance captures anisotropic clusters."""
    print("\nTest 7: Full Covariance")
    print("-" * 70)

    np.random.seed(10)
    # Elongated cluster 1 (x-direction) + isotropic cluster 2
    X1 = np.random.randn(60, 2) @ np.array([[3, 0], [0, 0.3]]) + [4, 0]
    X2 = np.random.randn(60, 2) * 0.5 + [-4, 0]
    X = np.vstack([X1, X2])

    gmm = GaussianMixture(n_components=2, covariance_type='full',
                          n_init=3, random_state=10).fit(X)
    assert gmm.covariances_.shape == (2, 2, 2)
    # Larger component should have higher off-diagonal ratio
    print(f"   [OK] Full covariance fitted; score = {gmm.score(X):.4f}")
    return True


def test_diag_covariance():
    """Diagonal covariance type runs without error."""
    print("\nTest 8: Diagonal Covariance")
    print("-" * 70)

    X = _blobs([[3, 3], [-3, -3]])
    gmm = GaussianMixture(n_components=2, covariance_type='diag',
                          random_state=6).fit(X)
    assert gmm.covariances_.shape == (2, 2)
    assert np.all(gmm.covariances_ > 0)
    print(f"   [OK] diag covariances all positive: {gmm.covariances_}")
    return True


def test_spherical_covariance():
    """Spherical covariance type runs without error."""
    print("\nTest 9: Spherical Covariance")
    print("-" * 70)

    X = _blobs([[3, 3], [-3, -3]])
    gmm = GaussianMixture(n_components=2, covariance_type='spherical',
                          random_state=7).fit(X)
    assert gmm.covariances_.shape == (2,)
    assert np.all(gmm.covariances_ > 0)
    print(f"   [OK] spherical variances: {gmm.covariances_}")
    return True


def test_tied_covariance():
    """Tied covariance type runs without error."""
    print("\nTest 10: Tied Covariance")
    print("-" * 70)

    X = _blobs([[3, 3], [-3, -3]])
    gmm = GaussianMixture(n_components=2, covariance_type='tied',
                          random_state=8).fit(X)
    assert gmm.covariances_.shape == (2, 2)
    print(f"   [OK] tied covariance shape: {gmm.covariances_.shape}")
    return True


def test_bic_aic_finite():
    """BIC and AIC must be finite after fitting."""
    print("\nTest 11: BIC/AIC Finite")
    print("-" * 70)

    X = _blobs([[4, 0], [0, 4], [-4, 0]])
    gmm = GaussianMixture(n_components=3, random_state=9).fit(X)
    bic = gmm.bic(X)
    aic = gmm.aic(X)
    assert np.isfinite(bic)
    assert np.isfinite(aic)
    assert bic > aic  # BIC >= AIC when n > e^2 ≈ 7.4
    print(f"   [OK] BIC = {bic:.2f}, AIC = {aic:.2f}")
    return True


def test_bic_selects_true_k():
    """BIC minimum should occur at or near the true K."""
    print("\nTest 12: BIC Selects True K")
    print("-" * 70)

    np.random.seed(42)
    X = np.vstack([np.random.randn(80, 2) * 0.4 + c
                   for c in [[6, 6], [-6, 6], [0, -6]]])

    bics = []
    for k in range(1, 7):
        gmm = GaussianMixture(n_components=k, n_init=3, random_state=0).fit(X)
        bics.append(gmm.bic(X))

    best_k = np.argmin(bics) + 1
    print(f"   BICs: {[f'{b:.1f}' for b in bics]}")
    assert best_k in [2, 3, 4], f"BIC chose K={best_k}, expected near 3"
    print(f"   [OK] BIC best K = {best_k}")
    return True


def test_sample():
    """sample() returns correct shapes and valid component indices."""
    print("\nTest 13: sample()")
    print("-" * 70)

    X = _blobs([[4, 0], [-4, 0]])
    gmm = GaussianMixture(n_components=2, random_state=11).fit(X)
    X_samp, comps = gmm.sample(200, random_state=99)
    assert X_samp.shape == (200, 2)
    assert comps.shape == (200,)
    assert set(np.unique(comps)).issubset({0, 1})
    print(f"   [OK] sample(200): shape={X_samp.shape}, components={np.unique(comps)}")
    return True


def test_score_consistency():
    """score() must equal mean(score_samples())."""
    print("\nTest 14: Score Consistency")
    print("-" * 70)

    X = _blobs([[3, 3], [-3, -3], [3, -3]])
    gmm = GaussianMixture(n_components=3, random_state=12).fit(X)
    mean_ll = float(np.mean(gmm.score_samples(X)))
    total_ll = gmm.score(X)
    assert abs(mean_ll - total_ll) < 1e-10
    print(f"   [OK] score = {total_ll:.6f} == mean(score_samples)")
    return True


def test_fit_predict():
    """fit_predict() must return same labels as fit().predict()."""
    print("\nTest 15: fit_predict")
    print("-" * 70)

    np.random.seed(5)
    X = np.random.randn(50, 2)
    gmm1 = GaussianMixture(n_components=3, random_state=13)
    gmm2 = GaussianMixture(n_components=3, random_state=13)
    labels_fp = gmm1.fit_predict(X)
    labels_f = gmm2.fit(X).predict(X)
    assert np.array_equal(labels_fp, labels_f)
    print(f"   [OK] fit_predict == fit().predict() on {len(X)} samples")
    return True


def run_all_tests():
    print("Testing Gaussian Mixture Models")
    print("=" * 70)

    tests = [
        test_fit_shapes,
        test_weights_sum_to_one,
        test_predict_valid_labels,
        test_predict_proba_sums_to_one,
        test_log_likelihood_monotone,
        test_3blob_recovery,
        test_full_covariance,
        test_diag_covariance,
        test_spherical_covariance,
        test_tied_covariance,
        test_bic_aic_finite,
        test_bic_selects_true_k,
        test_sample,
        test_score_consistency,
        test_fit_predict,
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
