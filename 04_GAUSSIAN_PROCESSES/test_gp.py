"""
Comprehensive Test Suite for Gaussian Processes

Tests: GaussianProcessRegressor, Kernels, Metrics, Utils

Author: ML Algorithms from Scratch
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gp import GaussianProcessRegressor
from kernels import (
    RBFKernel, MaternKernel, RationalQuadraticKernel,
    PeriodicKernel, LinearKernel, KernelSum, KernelProduct
)
from metrics import (
    mean_squared_error, r2_score, regression_report,
    negative_log_predictive_density, prediction_interval_coverage
)
from utils import StandardScaler, train_test_split, make_regression


def test_rbf_kernel():
    """Test 1: RBF Kernel computation."""
    print("\n" + "=" * 70)
    print("TEST 1: RBF Kernel Computation")
    print("=" * 70)

    X1 = np.array([[0, 0], [1, 1]])
    X2 = np.array([[0, 0], [2, 2]])

    kernel = RBFKernel(length_scale=1.0, variance=1.0)
    K = kernel(X1, X2)

    print(f"Kernel matrix shape: {K.shape}")
    print(f"K[0,0] (same point): {K[0,0]:.4f}")
    print(f"K[0,1] (different points): {K[0,1]:.4f}")

    # K[0,0] should be 1.0 (same point with variance=1.0)
    assert np.isclose(K[0, 0], 1.0), "Self-kernel should be 1.0"
    assert K.shape == (2, 2), "Kernel matrix shape incorrect"

    print("[PASS] Test 1")


def test_matern_kernel():
    """Test 2: Matern Kernel computation."""
    print("\n" + "=" * 70)
    print("TEST 2: Matern Kernel Computation")
    print("=" * 70)

    X = np.array([[0], [1], [2]])

    for nu in [0.5, 1.5, 2.5]:
        kernel = MaternKernel(length_scale=1.0, nu=nu)
        K = kernel(X, X)

        print(f"Matern nu={nu}: shape={K.shape}, K[0,0]={K[0,0]:.4f}")

        assert K.shape == (3, 3), f"Matern nu={nu} shape incorrect"
        assert np.isclose(K[0, 0], 1.0), f"Matern nu={nu} self-kernel should be 1.0"

    print("[PASS] Test 2")


def test_periodic_kernel():
    """Test 3: Periodic Kernel computation."""
    print("\n" + "=" * 70)
    print("TEST 3: Periodic Kernel Computation")
    print("=" * 70)

    X = np.array([[0], [1], [2], [3]])

    kernel = PeriodicKernel(period=2.0, length_scale=1.0)
    K = kernel(X, X)

    print(f"Kernel shape: {K.shape}")
    print(f"K[0,2] (distance=2, period=2): {K[0,2]:.4f}")

    # Points at period distance should have high correlation
    assert K.shape == (4, 4), "Periodic kernel shape incorrect"
    assert K[0, 2] > 0.5, "Periodic kernel should have high correlation at period"

    print("[PASS] Test 3")


def test_kernel_composition():
    """Test 4: Kernel sum and product."""
    print("\n" + "=" * 70)
    print("TEST 4: Kernel Composition")
    print("=" * 70)

    X = np.array([[1], [2], [3]])

    rbf = RBFKernel(length_scale=1.0)
    linear = LinearKernel(variance=0.5)

    # Sum
    kernel_sum = KernelSum(rbf, linear)
    K_sum = kernel_sum(X, X)

    # Product
    kernel_prod = KernelProduct(rbf, linear)
    K_prod = kernel_prod(X, X)

    print(f"Sum kernel shape: {K_sum.shape}")
    print(f"Product kernel shape: {K_prod.shape}")

    K_rbf = rbf(X, X)
    K_linear = linear(X, X)

    assert np.allclose(K_sum, K_rbf + K_linear), "Kernel sum incorrect"
    assert np.allclose(K_prod, K_rbf * K_linear), "Kernel product incorrect"

    print("[PASS] Test 4")


def test_gp_basic_fit_predict():
    """Test 5: Basic GP fit and predict."""
    print("\n" + "=" * 70)
    print("TEST 5: Basic GP Fit and Predict")
    print("=" * 70)

    np.random.seed(42)

    # Generate simple data
    X_train = np.array([1, 3, 5, 6, 8]).reshape(-1, 1)
    y_train = np.sin(X_train).ravel()

    X_test = np.array([2, 4, 7]).reshape(-1, 1)

    gp = GaussianProcessRegressor(kernel=RBFKernel(length_scale=1.0), alpha=0.01)
    gp.fit(X_train, y_train)

    y_mean, y_std = gp.predict(X_test, return_std=True)

    print(f"Predictions shape: {y_mean.shape}")
    print(f"Std shape: {y_std.shape}")
    print(f"Mean predictions: {y_mean}")
    print(f"Std predictions: {y_std}")

    assert y_mean.shape == (3,), "Prediction shape incorrect"
    assert y_std.shape == (3,), "Std shape incorrect"
    assert np.all(y_std > 0), "Std should be positive"

    print("[PASS] Test 5")


def test_gp_uncertainty():
    """Test 6: GP uncertainty quantification."""
    print("\n" + "=" * 70)
    print("TEST 6: GP Uncertainty Quantification")
    print("=" * 70)

    np.random.seed(42)

    X_train = np.array([0, 2, 4]).reshape(-1, 1)
    y_train = np.array([0, 1, 0])

    # Test points: near training data vs far
    X_near = np.array([[1.9]])  # Near training point
    X_far = np.array([[10.0]])  # Far from training data

    gp = GaussianProcessRegressor(kernel=RBFKernel(length_scale=1.0), alpha=0.01)
    gp.fit(X_train, y_train)

    _, std_near = gp.predict(X_near, return_std=True)
    _, std_far = gp.predict(X_far, return_std=True)

    print(f"Uncertainty near training data: {std_near[0]:.4f}")
    print(f"Uncertainty far from training data: {std_far[0]:.4f}")

    # Uncertainty should be higher far from training data
    assert std_far[0] > std_near[0], "Uncertainty should increase away from data"

    print("[PASS] Test 6")


def test_gp_normalization():
    """Test 7: GP with target normalization."""
    print("\n" + "=" * 70)
    print("TEST 7: GP with Target Normalization")
    print("=" * 70)

    np.random.seed(42)

    # Data with large values
    X_train = np.random.randn(20, 1)
    y_train = 1000 + 100 * np.sin(X_train).ravel()

    X_test = np.random.randn(10, 1)

    gp_no_norm = GaussianProcessRegressor(normalize_y=False, alpha=0.1)
    gp_norm = GaussianProcessRegressor(normalize_y=True, alpha=0.1)

    gp_no_norm.fit(X_train, y_train)
    gp_norm.fit(X_train, y_train)

    y_pred_no_norm = gp_no_norm.predict(X_test)
    y_pred_norm = gp_norm.predict(X_test)

    print(f"Without normalization - prediction range: [{y_pred_no_norm.min():.1f}, {y_pred_no_norm.max():.1f}]")
    print(f"With normalization - prediction range: [{y_pred_norm.min():.1f}, {y_pred_norm.max():.1f}]")

    # Both should give similar predictions
    assert np.allclose(y_pred_no_norm, y_pred_norm, rtol=0.5), "Normalization affects predictions significantly"

    print("[PASS] Test 7")


def test_gp_sampling():
    """Test 8: Posterior sampling."""
    print("\n" + "=" * 70)
    print("TEST 8: Posterior Sampling")
    print("=" * 70)

    np.random.seed(42)

    X_train = np.array([[0], [2], [4]])
    y_train = np.array([0, 1, 0])

    X_test = np.linspace(0, 4, 50).reshape(-1, 1)

    gp = GaussianProcessRegressor(kernel=RBFKernel(length_scale=1.0))
    gp.fit(X_train, y_train)

    samples = gp.sample_y(X_test, n_samples=5, random_state=42)

    print(f"Samples shape: {samples.shape}")
    print(f"Expected shape: (5, 50)")

    assert samples.shape == (5, 50), "Sample shape incorrect"

    print("[PASS] Test 8")


def test_log_marginal_likelihood():
    """Test 9: Log marginal likelihood."""
    print("\n" + "=" * 70)
    print("TEST 9: Log Marginal Likelihood")
    print("=" * 70)

    np.random.seed(42)

    X = np.random.randn(30, 1)
    y = np.sin(X).ravel() + 0.1 * np.random.randn(30)

    gp1 = GaussianProcessRegressor(kernel=RBFKernel(length_scale=1.0), alpha=0.1)
    gp2 = GaussianProcessRegressor(kernel=RBFKernel(length_scale=0.1), alpha=0.1)

    gp1.fit(X, y)
    gp2.fit(X, y)

    lml1 = gp1.log_marginal_likelihood()
    lml2 = gp2.log_marginal_likelihood()

    print(f"LML (length_scale=1.0): {lml1:.4f}")
    print(f"LML (length_scale=0.1): {lml2:.4f}")

    # LML should be a finite number
    assert np.isfinite(lml1), "LML should be finite"
    assert np.isfinite(lml2), "LML should be finite"

    print("[PASS] Test 9")


def test_metrics():
    """Test 10: Regression metrics."""
    print("\n" + "=" * 70)
    print("TEST 10: Regression Metrics")
    print("=" * 70)

    np.random.seed(42)

    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    y_std = np.array([0.2, 0.3, 0.2, 0.4, 0.3])

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    nlpd = negative_log_predictive_density(y_true, y_pred, y_std)
    coverage = prediction_interval_coverage(y_true, y_pred, y_std, confidence=0.95)

    print(f"MSE: {mse:.6f}")
    print(f"R²: {r2:.6f}")
    print(f"NLPD: {nlpd:.6f}")
    print(f"95% Coverage: {coverage:.6f}")

    assert 0 <= r2 <= 1, "R² should be between 0 and 1"
    assert 0 <= coverage <= 1, "Coverage should be between 0 and 1"

    print("[PASS] Test 10")


def test_utils():
    """Test 11: Utility functions."""
    print("\n" + "=" * 70)
    print("TEST 11: Utility Functions")
    print("=" * 70)

    # StandardScaler
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(np.mean(X_scaled, axis=0), 0, atol=1e-10), "Mean should be ~0"
    assert np.allclose(np.std(X_scaled, axis=0), 1, atol=1e-10), "Std should be ~1"

    print("  [OK] StandardScaler works")

    # train_test_split
    X = np.arange(100).reshape(-1, 1)
    y = np.arange(100)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    assert len(X_train) == 80, "Train size incorrect"
    assert len(X_test) == 20, "Test size incorrect"

    print("  [OK] train_test_split works")

    # make_regression
    X, y = make_regression(n_samples=50, n_features=1, noise=0.1, random_state=42)

    assert X.shape == (50, 1), "make_regression shape incorrect"
    assert y.shape == (50,), "make_regression y shape incorrect"

    print("  [OK] make_regression works")

    print("[PASS] Test 11")


def test_edge_cases():
    """Test 12: Edge cases."""
    print("\n" + "=" * 70)
    print("TEST 12: Edge Cases")
    print("=" * 70)

    np.random.seed(42)

    # Test 1: Prediction without fitting
    try:
        gp = GaussianProcessRegressor()
        X_test = np.random.randn(10, 1)
        gp.predict(X_test)
        assert False, "Should raise error for prediction without fitting"
    except ValueError as e:
        print(f"  Correctly raised error: {str(e)[:50]}...")

    # Test 2: Small dataset
    X_train = np.array([[1], [2], [3]])
    y_train = np.array([1, 2, 1])

    gp = GaussianProcessRegressor(kernel=RBFKernel(), alpha=0.01)
    gp.fit(X_train, y_train)

    predictions, std = gp.predict(X_train, return_std=True)

    assert len(predictions) == len(X_train), "Prediction length incorrect"
    print("  Small dataset works correctly")

    # Test 3: Single test point
    X_test = np.array([[1.5]])
    y_pred = gp.predict(X_test)

    assert y_pred.shape == (1,), "Single point prediction incorrect"
    print("  Single test point works correctly")

    print("[PASS] Test 12")


def test_real_world_scenario():
    """Test 13: Real-world regression scenario."""
    print("\n" + "=" * 70)
    print("TEST 13: Real-World Regression Scenario")
    print("=" * 70)

    np.random.seed(42)

    # Generate realistic data
    X_train = np.sort(np.random.uniform(0, 10, 50)).reshape(-1, 1)
    y_train = np.sin(X_train).ravel() + 0.2 * np.random.randn(50)

    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    y_test = np.sin(X_test).ravel()

    # Fit GP
    kernel = RBFKernel(length_scale=1.0, variance=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.05, normalize_y=True)
    gp.fit(X_train, y_train)

    # Predict
    y_mean, y_std = gp.predict(X_test, return_std=True)

    # Evaluate
    r2 = gp.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_mean)

    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"Log Marginal Likelihood: {gp.log_marginal_likelihood():.4f}")

    assert r2 > 0.5, f"R² too low: {r2}"
    assert mse < 1.0, f"MSE too high: {mse}"

    print("[PASS] Test 13")


def run_all_tests():
    """Run all test suites."""
    print("\n" + "=" * 70)
    print(" " * 18 + "GAUSSIAN PROCESS - TEST SUITE")
    print("=" * 70)

    tests = [
        test_rbf_kernel,
        test_matern_kernel,
        test_periodic_kernel,
        test_kernel_composition,
        test_gp_basic_fit_predict,
        test_gp_uncertainty,
        test_gp_normalization,
        test_gp_sampling,
        test_log_marginal_likelihood,
        test_metrics,
        test_utils,
        test_edge_cases,
        test_real_world_scenario
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n[FAIL] {test_func.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"\n[ERROR] {test_func.__name__}: {e}")
            failed += 1

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total Tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n*** ALL TESTS PASSED! ***")
    else:
        print(f"\n*** {failed} test(s) failed ***")

    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
