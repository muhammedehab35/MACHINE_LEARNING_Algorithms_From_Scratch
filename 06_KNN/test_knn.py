"""
Comprehensive Test Suite for K-Nearest Neighbors

Tests: KNNClassifier, KNNRegressor, Metrics, Utils

Author: ML Algorithms from Scratch
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from knn import KNNClassifier, KNNRegressor
from metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from utils import StandardScaler, train_test_split


def test_knn_classifier_basic():
    """Test 1: Basic KNN classification."""
    print("\n" + "=" * 70)
    print("TEST 1: Basic KNN Classification")
    print("=" * 70)

    np.random.seed(42)

    # Simple linearly separable data
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = KNNClassifier(n_neighbors=5, weights='uniform')
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Number of neighbors: {model.n_neighbors}")
    print(f"Number of classes: {model.n_classes_}")

    assert accuracy > 0.80, f"Accuracy too low: {accuracy}"
    assert model.n_classes_ == 2, "Should have 2 classes"

    print("[PASS] Test 1")


def test_distance_metrics():
    """Test 2: Different distance metrics."""
    print("\n" + "=" * 70)
    print("TEST 2: Distance Metrics")
    print("=" * 70)

    np.random.seed(42)
    X = np.random.randn(150, 3)
    y = (np.sum(X, axis=1) > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    metrics = ['euclidean', 'manhattan', 'minkowski', 'cosine']

    for metric in metrics:
        model = KNNClassifier(n_neighbors=5, metric=metric)
        model.fit(X_train_scaled, y_train)

        acc = model.score(X_test_scaled, y_test)
        print(f"{metric:15} - Accuracy: {acc:.4f}")

        assert acc > 0.60, f"{metric} failed: Accuracy = {acc}"

    print("[PASS] Test 2")


def test_weighted_voting():
    """Test 3: Weighted voting."""
    print("\n" + "=" * 70)
    print("TEST 3: Weighted Voting")
    print("=" * 70)

    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0]**2 + X[:, 1]**2 > 1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Uniform weights
    model_uniform = KNNClassifier(n_neighbors=7, weights='uniform')
    model_uniform.fit(X_train, y_train)
    acc_uniform = model_uniform.score(X_test, y_test)

    # Distance weights
    model_distance = KNNClassifier(n_neighbors=7, weights='distance')
    model_distance.fit(X_train, y_train)
    acc_distance = model_distance.score(X_test, y_test)

    print(f"Uniform weights:  {acc_uniform:.4f}")
    print(f"Distance weights: {acc_distance:.4f}")

    assert acc_uniform > 0.65, f"Uniform accuracy too low: {acc_uniform}"
    assert acc_distance > 0.65, f"Distance accuracy too low: {acc_distance}"

    print("[PASS] Test 3")


def test_probability_predictions():
    """Test 4: Probability predictions."""
    print("\n" + "=" * 70)
    print("TEST 4: Probability Predictions")
    print("=" * 70)

    np.random.seed(42)
    X = np.random.randn(150, 2)
    y = (X[:, 0] > 0).astype(int)

    model = KNNClassifier(n_neighbors=5)
    model.fit(X, y)

    proba = model.predict_proba(X[:10])

    print(f"Proba shape: {proba.shape}")
    print(f"Proba sum per sample: {np.sum(proba, axis=1)[:5]}")
    print(f"Sample probabilities:\n{proba[:3]}")

    assert proba.shape == (10, 2), "Proba shape incorrect"
    assert np.allclose(np.sum(proba, axis=1), 1.0), "Probabilities don't sum to 1"
    assert np.all((proba >= 0) & (proba <= 1)), "Probabilities out of range"

    print("[PASS] Test 4")


def test_different_k_values():
    """Test 5: Different values of k."""
    print("\n" + "=" * 70)
    print("TEST 5: Different K Values")
    print("=" * 70)

    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    k_values = [1, 3, 5, 10, 20]

    for k in k_values:
        model = KNNClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

        print(f"k={k:2d} - Accuracy: {acc:.4f}")

        assert acc > 0.70, f"k={k} failed: Accuracy = {acc}"

    print("[PASS] Test 5")


def test_knn_regressor():
    """Test 6: KNN Regressor."""
    print("\n" + "=" * 70)
    print("TEST 6: KNN Regressor")
    print("=" * 70)

    np.random.seed(42)

    # Generate regression data
    X = np.random.rand(200, 1) * 10
    y = 2 * X.ravel() + 3 + np.random.randn(200) * 0.5

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Test uniform weights
    model_uniform = KNNRegressor(n_neighbors=5, weights='uniform')
    model_uniform.fit(X_train, y_train)
    r2_uniform = model_uniform.score(X_test, y_test)

    # Test distance weights
    model_distance = KNNRegressor(n_neighbors=5, weights='distance')
    model_distance.fit(X_train, y_train)
    r2_distance = model_distance.score(X_test, y_test)

    print(f"R² (uniform):  {r2_uniform:.4f}")
    print(f"R² (distance): {r2_distance:.4f}")

    assert r2_uniform > 0.85, f"R² too low: {r2_uniform}"
    assert r2_distance > 0.85, f"R² too low: {r2_distance}"

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

    model = KNNClassifier(n_neighbors=5, weights='distance')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = classification_report(y_test, y_pred, y_proba)

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

    # Test 1: k > n_samples
    X = np.random.randn(10, 2)
    y = np.random.randint(0, 2, 10)

    try:
        model = KNNClassifier(n_neighbors=15)
        model.fit(X, y)
        assert False, "Should raise ValueError for k > n_samples"
    except ValueError as e:
        print(f"  Correctly raised ValueError: {str(e)[:50]}...")

    # Test 2: Prediction without fitting
    try:
        model = KNNClassifier(n_neighbors=5)
        model.predict(X)
        assert False, "Should raise ValueError for prediction without fitting"
    except ValueError as e:
        print(f"  Correctly raised ValueError: {str(e)[:50]}...")

    # Test 3: k=1 (should work)
    model = KNNClassifier(n_neighbors=1)
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(X), "Prediction length incorrect"
    print("  k=1 works correctly")

    print("[PASS] Test 9")


def test_multiclass_classification():
    """Test 10: Multi-class classification."""
    print("\n" + "=" * 70)
    print("TEST 10: Multi-class Classification")
    print("=" * 70)

    np.random.seed(42)

    # Generate 3-class data
    n_samples = 300
    X = np.random.randn(n_samples, 2)

    # Create 3 classes
    y = np.zeros(n_samples, dtype=int)
    y[(X[:, 0] > 1) & (X[:, 1] > 1)] = 1
    y[(X[:, 0] < -1) & (X[:, 1] < -1)] = 2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = KNNClassifier(n_neighbors=5, weights='distance')
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    proba = model.predict_proba(X_test)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Number of classes: {model.n_classes_}")
    print(f"Probability shape: {proba.shape}")

    assert model.n_classes_ == 3, "Should have 3 classes"
    assert proba.shape[1] == 3, "Probability should have 3 columns"
    assert accuracy > 0.70, f"Accuracy too low: {accuracy}"

    print("[PASS] Test 10")


def run_all_tests():
    """Run all test suites."""
    print("\n" + "=" * 70)
    print(" " * 20 + "KNN - TEST SUITE")
    print("=" * 70)

    tests = [
        test_knn_classifier_basic,
        test_distance_metrics,
        test_weighted_voting,
        test_probability_predictions,
        test_different_k_values,
        test_knn_regressor,
        test_classification_metrics,
        test_utils,
        test_edge_cases,
        test_multiclass_classification
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
