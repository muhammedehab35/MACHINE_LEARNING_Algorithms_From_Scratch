"""
Comprehensive Test Suite for Elastic Net Regression

Tests all components:
1. ElasticNet model
2. Metrics
3. Utils (preprocessing)

Author: ML Algorithms from Scratch
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from elastic_net import ElasticNet
from metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    adjusted_r2_score,
    evaluate_regression
)
from utils import StandardScaler, MinMaxScaler, train_test_split, polynomial_features


def test_elastic_net_basic():
    """Test 1: Basic ElasticNet functionality."""
    print("\n" + "=" * 70)
    print("TEST 1: Basic ElasticNet Functionality")
    print("=" * 70)

    # Simple linear data
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 3 * X.flatten() + 2 + np.random.randn(100) * 0.1

    # Train model
    model = ElasticNet(
        learning_rate=0.01,
        n_iterations=500,
        l1_ratio=0.5,
        alpha=0.1,
        early_stopping=False,
        random_state=42
    )

    model.fit(X, y)
    predictions = model.predict(X)

    # Evaluate
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)

    print(f"MSE: {mse:.6f}")
    print(f"R² Score: {r2:.6f}")
    print(f"Learned weight: {model.weights_[0]:.4f} (expected: ~3.0)")
    print(f"Learned bias: {model.bias_:.4f} (expected: ~2.0)")

    assert r2 > 0.95, f"R² too low: {r2}"
    assert abs(model.weights_[0] - 3.0) < 0.5, "Weight not close to expected value"

    print("[PASS] Test 1")


def test_elastic_net_multivariate():
    """Test 2: Multivariate regression."""
    print("\n" + "=" * 70)
    print("TEST 2: Multivariate Regression")
    print("=" * 70)

    np.random.seed(42)
    n_samples, n_features = 200, 5

    X = np.random.randn(n_samples, n_features)
    true_weights = np.array([2, -3, 1.5, 0, 0])  # Some weights are zero
    y = np.dot(X, true_weights) + 5 + np.random.randn(n_samples) * 0.5

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train with high L1 to induce sparsity
    model = ElasticNet(
        learning_rate=0.01,
        n_iterations=1000,
        l1_ratio=0.8,  # High L1 for sparsity
        alpha=1.0,
        early_stopping=False,
        random_state=42
    )

    model.fit(X_scaled, y)

    # Evaluate
    predictions = model.predict(X_scaled)
    r2 = r2_score(y, predictions)
    sparsity = model.get_sparsity()

    print(f"R² Score: {r2:.4f}")
    print(f"Sparsity: {sparsity:.2%}")
    print(f"Non-zero features: {len(model.get_nonzero_features())}/{n_features}")
    print(f"Weights: {model.weights_}")

    assert r2 > 0.75, f"R² too low: {r2}"
    assert sparsity > 0, "No sparsity achieved"

    print("[PASS] Test 2")


def test_elastic_net_pure_ridge():
    """Test 3: Pure Ridge (l1_ratio=0)."""
    print("\n" + "=" * 70)
    print("TEST 3: Pure Ridge Regression (l1_ratio=0)")
    print("=" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(100) * 0.2

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Pure Ridge
    model = ElasticNet(
        learning_rate=0.01,
        n_iterations=1000,
        l1_ratio=0.0,  # Pure Ridge
        alpha=0.5,
        early_stopping=False,
        random_state=42
    )

    model.fit(X_scaled, y)

    r2 = model.score(X_scaled, y)
    sparsity = model.get_sparsity()

    print(f"R² Score: {r2:.4f}")
    print(f"Sparsity: {sparsity:.2%}")
    print(f"L1 penalty: {model.l1_penalty_}")
    print(f"L2 penalty: {model.l2_penalty_}")

    assert r2 > 0.85, f"R² too low: {r2}"
    assert model.l1_penalty_ == 0, "L1 penalty should be 0 for pure Ridge"
    assert model.l2_penalty_ > 0, "L2 penalty should be > 0"
    assert sparsity == 0, "Ridge should not produce sparsity"

    print("[PASS] Test 3")


def test_elastic_net_pure_lasso():
    """Test 4: Pure Lasso (l1_ratio=1)."""
    print("\n" + "=" * 70)
    print("TEST 4: Pure Lasso Regression (l1_ratio=1)")
    print("=" * 70)

    np.random.seed(42)
    n_features = 10
    X = np.random.randn(150, n_features)

    # Only 3 features are relevant
    true_weights = np.zeros(n_features)
    true_weights[[0, 3, 7]] = [3, -2, 1.5]

    y = np.dot(X, true_weights) + np.random.randn(150) * 0.3

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Pure Lasso
    model = ElasticNet(
        learning_rate=0.01,
        n_iterations=1500,
        l1_ratio=1.0,  # Pure Lasso
        alpha=0.5,
        early_stopping=False,
        random_state=42
    )

    model.fit(X_scaled, y)

    r2 = model.score(X_scaled, y)
    sparsity = model.get_sparsity()
    n_selected = len(model.get_nonzero_features())

    print(f"R² Score: {r2:.4f}")
    print(f"Sparsity: {sparsity:.2%}")
    print(f"Selected features: {n_selected}/{n_features}")
    print(f"L1 penalty: {model.l1_penalty_}")
    print(f"L2 penalty: {model.l2_penalty_}")

    assert r2 > 0.85, f"R² too low: {r2}"
    assert model.l2_penalty_ == 0, "L2 penalty should be 0 for pure Lasso"
    assert model.l1_penalty_ > 0, "L1 penalty should be > 0"
    assert sparsity > 0.3, f"Lasso should produce significant sparsity, got {sparsity}"

    print("[PASS] Test 4")


def test_early_stopping():
    """Test 5: Early stopping functionality."""
    print("\n" + "=" * 70)
    print("TEST 5: Early Stopping")
    print("=" * 70)

    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = np.sum(X, axis=1) + np.random.randn(200) * 0.5

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = ElasticNet(
        learning_rate=0.01,
        n_iterations=5000,
        l1_ratio=0.5,
        alpha=0.1,
        early_stopping=True,
        patience=20,
        random_state=42
    )

    model.fit(X_scaled, y)

    print(f"Stopped at iteration: {model.n_iter_}/{model.n_iterations}")
    print(f"Final train loss: {model.loss_history_[-1]:.6f}")
    print(f"Final val loss: {model.val_loss_history_[-1]:.6f}")

    assert model.n_iter_ < model.n_iterations, "Early stopping should trigger"
    assert len(model.val_loss_history_) > 0, "Validation history should exist"

    print("[PASS] Test 5")


def test_batch_modes():
    """Test 6: Different batch modes (Batch, Mini-Batch, SGD)."""
    print("\n" + "=" * 70)
    print("TEST 6: Batch Modes")
    print("=" * 70)

    np.random.seed(42)
    X = np.random.randn(150, 3)
    y = 2*X[:, 0] - 3*X[:, 1] + X[:, 2] + np.random.randn(150) * 0.3

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    configs = [
        ("Batch GD", None),
        ("Mini-Batch GD", 32),
        ("SGD", 1)
    ]

    for name, batch_size in configs:
        model = ElasticNet(
            learning_rate=0.01,
            n_iterations=500,
            batch_size=batch_size,
            early_stopping=False,
            random_state=42
        )

        model.fit(X_scaled, y)
        r2 = model.score(X_scaled, y)

        print(f"{name:20} - R² Score: {r2:.4f}")

        assert r2 > 0.70, f"{name} failed: R² = {r2}"

    print("[PASS] Test 6")


def test_metrics():
    """Test 7: All metrics functions."""
    print("\n" + "=" * 70)
    print("TEST 7: Metrics Functions")
    print("=" * 70)

    y_true = np.array([3, -0.5, 2, 7, 4.2])
    y_pred = np.array([2.5, 0.0, 2, 8, 4.5])

    metrics = evaluate_regression(y_true, y_pred, n_features=3)

    print("Computed Metrics:")
    for name, value in metrics.items():
        print(f"  {name:20}: {value:.6f}")

    assert metrics['MSE'] > 0, "MSE should be positive"
    assert metrics['RMSE'] > 0, "RMSE should be positive"
    assert metrics['MAE'] > 0, "MAE should be positive"
    assert -1 <= metrics['R²'] <= 1, "R² should be between -1 and 1"

    print("[PASS] Test 7")

def test_utils():
    """Test 8: Utility functions."""
    print("\n" + "=" * 70)
    print("TEST 8: Utility Functions")
    print("=" * 70)

    # Test StandardScaler
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(np.mean(X_scaled, axis=0), 0, atol=1e-10), "Mean should be ~0"
    assert np.allclose(np.std(X_scaled, axis=0), 1, atol=1e-10), "Std should be ~1"

    X_recovered = scaler.inverse_transform(X_scaled)
    assert np.allclose(X, X_recovered), "Inverse transform failed"

    print("  [OK] StandardScaler works")

    # Test MinMaxScaler
    scaler_mm = MinMaxScaler(feature_range=(0, 1))
    X_scaled_mm = scaler_mm.fit_transform(X)

    assert np.allclose(np.min(X_scaled_mm, axis=0), 0), "Min should be 0"
    assert np.allclose(np.max(X_scaled_mm, axis=0), 1), "Max should be 1"

    print("  [OK] MinMaxScaler works")

    # Test train_test_split
    X = np.arange(100).reshape(-1, 1)
    y = np.arange(100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    assert len(X_train) == 80, "Train size incorrect"
    assert len(X_test) == 20, "Test size incorrect"
    assert len(X_train) + len(X_test) == len(X), "Split size mismatch"

    print("  [OK] train_test_split works")

    # Test polynomial_features
    X = np.array([[1], [2], [3]])
    X_poly = polynomial_features(X, degree=2)

    assert X_poly.shape == (3, 2), f"Polynomial features shape incorrect: {X_poly.shape}"

    print("  [OK] polynomial_features works")

    print("[PASS] Test 8")


def run_all_tests():
    """Run all test suites."""
    print("\n" + "=" * 70)
    print(" " * 15 + "ELASTIC NET - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    tests = [
        test_elastic_net_basic,
        test_elastic_net_multivariate,
        test_elastic_net_pure_ridge,
        test_elastic_net_pure_lasso,
        test_early_stopping,
        test_batch_modes,
        test_metrics,
        test_utils
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
