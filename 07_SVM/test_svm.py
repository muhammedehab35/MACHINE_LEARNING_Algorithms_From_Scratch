"""
Comprehensive Test Suite for Support Vector Machines

Tests: SVMClassifier, Kernels, Metrics, Utils

Author: ML Algorithms from Scratch
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from svm import SVMClassifier
from metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, classification_report
)
from utils import StandardScaler, train_test_split


def test_linear_svm_basic():
    """Test 1: Basic Linear SVM classification."""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Linear SVM Classification")
    print("=" * 70)

    np.random.seed(42)

    # Simple linearly separable data
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = SVMClassifier(kernel='linear', C=1.0, n_iterations=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Number of classes: {model.n_classes_}")

    assert accuracy > 0.75, f"Accuracy too low: {accuracy}"
    assert model.n_classes_ == 2, "Should have 2 classes"

    print("[PASS] Test 1")


def test_rbf_kernel():
    """Test 2: RBF kernel for non-linear data."""
    print("\n" + "=" * 70)
    print("TEST 2: RBF Kernel SVM")
    print("=" * 70)

    np.random.seed(42)

    # Non-linear data (circular)
    X = np.random.randn(200, 2)
    y = ((X[:, 0]**2 + X[:, 1]**2) < 1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale data (important for RBF)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVMClassifier(kernel='rbf', C=1.0, gamma='scale', n_iterations=500)
    model.fit(X_train_scaled, y_train)

    accuracy = model.score(X_test_scaled, y_test)
    print(f"RBF SVM Accuracy: {accuracy:.4f}")
    print(f"Gamma: {model.gamma_:.6f}")

    assert accuracy > 0.60, f"RBF accuracy too low: {accuracy}"

    print("[PASS] Test 2")


def test_polynomial_kernel():
    """Test 3: Polynomial kernel."""
    print("\n" + "=" * 70)
    print("TEST 3: Polynomial Kernel SVM")
    print("=" * 70)

    np.random.seed(42)
    X = np.random.randn(150, 2)
    y = ((X[:, 0]**2 + X[:, 1]) > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVMClassifier(kernel='poly', degree=2, C=1.0, n_iterations=500)
    model.fit(X_train_scaled, y_train)

    accuracy = model.score(X_test_scaled, y_test)
    print(f"Polynomial SVM Accuracy: {accuracy:.4f}")
    print(f"Degree: {model.degree}")

    assert accuracy > 0.60, f"Poly accuracy too low: {accuracy}"

    print("[PASS] Test 3")


def test_different_c_values():
    """Test 4: Different regularization parameter C values."""
    print("\n" + "=" * 70)
    print("TEST 4: Different C Values")
    print("=" * 70)

    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    c_values = [0.1, 1.0, 10.0]

    for c in c_values:
        model = SVMClassifier(kernel='linear', C=c, n_iterations=1000)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

        print(f"C={c:5.1f} - Accuracy: {acc:.4f}")

        assert acc > 0.70, f"C={c} failed: Accuracy = {acc}"

    print("[PASS] Test 4")


def test_decision_function():
    """Test 5: Decision function values."""
    print("\n" + "=" * 70)
    print("TEST 5: Decision Function")
    print("=" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] > 0).astype(int)

    model = SVMClassifier(kernel='linear', C=1.0)
    model.fit(X, y)

    decisions = model.decision_function(X[:10])

    print(f"Decision values shape: {decisions.shape}")
    print(f"Sample decision values: {decisions[:5]}")

    # Predictions should match sign of decision function
    predictions = model.predict(X[:10])
    expected = np.where(decisions > 0, model.classes_[1], model.classes_[0])

    assert np.array_equal(predictions, expected), "Predictions don't match decision function"

    print("[PASS] Test 5")


def test_gamma_options():
    """Test 6: Different gamma options for RBF."""
    print("\n" + "=" * 70)
    print("TEST 6: Gamma Options")
    print("=" * 70)

    np.random.seed(42)
    X = np.random.randn(150, 3)
    y = (np.sum(X, axis=1) > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    gamma_options = ['scale', 'auto', 0.1]

    for gamma in gamma_options:
        model = SVMClassifier(kernel='rbf', gamma=gamma, C=1.0, n_iterations=300)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

        gamma_str = str(gamma) if not isinstance(gamma, str) else gamma
        print(f"Gamma={gamma_str:10} - Accuracy: {acc:.4f}, Computed gamma: {model.gamma_:.6f}")

        assert acc > 0.55, f"Gamma={gamma} failed: Accuracy = {acc}"

    print("[PASS] Test 6")


def test_classification_metrics():
    """Test 7: Classification metrics."""
    print("\n" + "=" * 70)
    print("TEST 7: Classification Metrics")
    print("=" * 70)

    np.random.seed(42)
    X = np.random.randn(150, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = SVMClassifier(kernel='linear', C=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = classification_report(y_test, y_pred)

    print("Computed Metrics:")
    for name, value in metrics.items():
        print(f"  {name:20}: {value:.6f}")

    assert 0 <= metrics['Accuracy'] <= 1, "Accuracy out of range"
    assert 0 <= metrics['Precision'] <= 1, "Precision out of range"
    assert 0 <= metrics['Recall'] <= 1, "Recall out of range"
    assert 0 <= metrics['F1 Score'] <= 1, "F1 out of range"

    print("[PASS] Test 7")


def test_utils():
    """Test 8: Utility functions."""
    print("\n" + "=" * 70)
    print("TEST 8: Utility Functions")
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    assert len(X_train) == 80, "Train size incorrect"
    assert len(X_test) == 20, "Test size incorrect"

    print("  [OK] train_test_split works")

    print("[PASS] Test 8")


def test_edge_cases():
    """Test 9: Edge cases and error handling."""
    print("\n" + "=" * 70)
    print("TEST 9: Edge Cases")
    print("=" * 70)

    np.random.seed(42)

    # Test 1: Prediction without fitting
    try:
        model = SVMClassifier()
        X_test = np.random.randn(10, 2)
        model.predict(X_test)
        assert False, "Should raise error for prediction without fitting"
    except (ValueError, AttributeError) as e:
        print(f"  Correctly raised error: {str(e)[:50]}...")

    # Test 2: Mismatched X and y shapes
    try:
        model = SVMClassifier()
        X = np.random.randn(10, 2)
        y = np.random.randint(0, 2, 5)  # Wrong size
        model.fit(X, y)
        assert False, "Should raise ValueError for mismatched shapes"
    except ValueError as e:
        print(f"  Correctly raised ValueError: {str(e)[:50]}...")

    # Test 3: Small dataset should still work
    X = np.array([[0, 0], [1, 1], [2, 0], [0, 2]])
    y = np.array([0, 1, 0, 1])

    model = SVMClassifier(kernel='linear', n_iterations=100)
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(X), "Prediction length incorrect"
    print("  Small dataset works correctly")

    print("[PASS] Test 9")


def test_kernel_comparison():
    """Test 10: Compare all kernel types."""
    print("\n" + "=" * 70)
    print("TEST 10: Kernel Comparison")
    print("=" * 70)

    np.random.seed(42)

    # Mix of linear and non-linear patterns
    X = np.random.randn(200, 2)
    y = ((X[:, 0] + 0.5*X[:, 1]) > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    kernels = ['linear', 'rbf', 'poly']

    for kernel in kernels:
        model = SVMClassifier(kernel=kernel, C=1.0, n_iterations=500)
        model.fit(X_train_scaled, y_train)

        accuracy = model.score(X_test_scaled, y_test)
        print(f"{kernel:10} kernel - Accuracy: {accuracy:.4f}")

        assert accuracy > 0.60, f"{kernel} kernel failed: Accuracy = {accuracy}"

    print("[PASS] Test 10")


def run_all_tests():
    """Run all test suites."""
    print("\n" + "=" * 70)
    print(" " * 20 + "SVM - TEST SUITE")
    print("=" * 70)

    tests = [
        test_linear_svm_basic,
        test_rbf_kernel,
        test_polynomial_kernel,
        test_different_c_values,
        test_decision_function,
        test_gamma_options,
        test_classification_metrics,
        test_utils,
        test_edge_cases,
        test_kernel_comparison
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
