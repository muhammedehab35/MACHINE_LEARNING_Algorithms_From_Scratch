"""
Comprehensive Test Suite for Logistic Regression

Tests: Model, Metrics, Utils

Author: ML Algorithms from Scratch
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from logistic_regression import LogisticRegression
from metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from utils import StandardScaler, train_test_split


def test_basic_classification():
    """Test 1: Basic binary classification."""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Binary Classification")
    print("=" * 70)

    np.random.seed(42)

    # Simple linearly separable data
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = LogisticRegression(
        learning_rate=0.1,
        n_iterations=500,
        early_stopping=False,
        random_state=42
    )

    model.fit(X, y)
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    print(f"Accuracy: {accuracy:.4f}")

    assert accuracy > 0.85, f"Accuracy too low: {accuracy}"
    print("[PASS] Test 1")


def test_regularization():
    """Test 2: L2 regularization."""
    print("\n" + "=" * 70)
    print("TEST 2: L2 Regularization")
    print("=" * 70)

    np.random.seed(42)
    X = np.random.randn(150, 10)
    y = (np.sum(X[:, :3], axis=1) > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        learning_rate=0.01,
        n_iterations=1000,
        regularization='l2',
        alpha=0.1,
        early_stopping=False,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    assert test_acc > 0.70, f"Test accuracy too low: {test_acc}"
    print("[PASS] Test 2")


def test_probability_predictions():
    """Test 3: Probability predictions."""
    print("\n" + "=" * 70)
    print("TEST 3: Probability Predictions")
    print("=" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] > 0).astype(int)

    model = LogisticRegression(learning_rate=0.1, n_iterations=500, random_state=42)
    model.fit(X, y)

    proba = model.predict_proba(X)

    print(f"Proba shape: {proba.shape}")
    print(f"Proba sum per sample: {np.sum(proba, axis=1)[:5]}")

    assert proba.shape == (100, 2), "Proba shape incorrect"
    assert np.allclose(np.sum(proba, axis=1), 1.0), "Probabilities don't sum to 1"
    assert np.all((proba >= 0) & (proba <= 1)), "Probabilities out of range"

    print("[PASS] Test 3")


def test_early_stopping():
    """Test 4: Early stopping functionality."""
    print("\n" + "=" * 70)
    print("TEST 4: Early Stopping Functionality")
    print("=" * 70)

    np.random.seed(42)
    # Dataset with noise
    X = np.random.randn(300, 8)
    y = (np.sum(X[:, :3], axis=1) + 0.5 * np.random.randn(300) > 0).astype(int)

    model = LogisticRegression(
        learning_rate=0.001,
        n_iterations=2000,
        early_stopping=True,
        patience=15,
        random_state=42
    )

    model.fit(X, y)

    print(f"Stopped at iteration: {model.n_iter_}/{model.n_iterations}")
    print(f"Validation history length: {len(model.val_loss_history_)}")

    # Check that early stopping mechanism is working (validation split occurred)
    assert len(model.val_loss_history_) > 0, "Validation history should exist"
    assert model.n_iter_ > 0, "Model should have trained"

    # If stopped early, verify it makes sense
    if model.n_iter_ < model.n_iterations:
        print(f"  Early stopping triggered at iteration {model.n_iter_}")

    print("[PASS] Test 4")


def test_batch_modes():
    """Test 5: Batch modes."""
    print("\n" + "=" * 70)
    print("TEST 5: Batch Modes")
    print("=" * 70)

    np.random.seed(42)
    X = np.random.randn(150, 3)
    y = (X[:, 0] + X[:, 1] > X[:, 2]).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    configs = [
        ("Batch GD", None),
        ("Mini-Batch GD", 32),
        ("SGD", 1)
    ]

    for name, batch_size in configs:
        model = LogisticRegression(
            learning_rate=0.01,
            n_iterations=500,
            batch_size=batch_size,
            early_stopping=False,
            random_state=42
        )

        model.fit(X_scaled, y)
        acc = model.score(X_scaled, y)

        print(f"{name:20} - Accuracy: {acc:.4f}")

        assert acc > 0.65, f"{name} failed: Accuracy = {acc}"

    print("[PASS] Test 5")


def test_metrics():
    """Test 6: Classification metrics."""
    print("\n" + "=" * 70)
    print("TEST 6: Classification Metrics")
    print("=" * 70)

    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 1, 1, 0])
    y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.85, 0.6, 0.75, 0.95, 0.15])

    metrics = classification_report(y_true, y_pred, y_pred_proba)

    print("Computed Metrics:")
    for name, value in metrics.items():
        print(f"  {name:20}: {value:.6f}")

    assert 0 <= metrics['Accuracy'] <= 1, "Accuracy out of range"
    assert 0 <= metrics['Precision'] <= 1, "Precision out of range"
    assert 0 <= metrics['Recall'] <= 1, "Recall out of range"
    assert 0 <= metrics['F1 Score'] <= 1, "F1 out of range"

    print("[PASS] Test 6")


def test_utils():
    """Test 7: Utils functions."""
    print("\n" + "=" * 70)
    print("TEST 7: Utility Functions")
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

    print("[PASS] Test 7")


def test_multiclass_error():
    """Test 8: Error on non-binary labels."""
    print("\n" + "=" * 70)
    print("TEST 8: Non-Binary Label Error")
    print("=" * 70)

    X = np.random.randn(100, 2)
    y = np.random.randint(0, 3, 100)  # 3 classes

    model = LogisticRegression()

    try:
        model.fit(X, y)
        assert False, "Should have raised ValueError for non-binary labels"
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")

    print("[PASS] Test 8")


def run_all_tests():
    """Run all test suites."""
    print("\n" + "=" * 70)
    print(" " * 12 + "LOGISTIC REGRESSION - TEST SUITE")
    print("=" * 70)

    tests = [
        test_basic_classification,
        test_regularization,
        test_probability_predictions,
        test_early_stopping,
        test_batch_modes,
        test_metrics,
        test_utils,
        test_multiclass_error
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
