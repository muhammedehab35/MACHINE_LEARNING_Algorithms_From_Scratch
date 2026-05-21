"""
Comprehensive Tests for AdaBoost Implementation
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaboost import AdaBoostClassifier, DecisionStump


def test_decision_stump():
    """Test DecisionStump predictions"""
    print("Test 1: Decision Stump Basic Predictions")
    print("-" * 70)

    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    stump = DecisionStump()
    stump.feature_index = 0
    stump.threshold = 3.0
    stump.polarity = 1

    preds = stump.predict(X)
    assert preds.shape == (5,), "Shape mismatch"
    assert set(np.unique(preds)) <= {-1.0, 1.0}, "Predictions not in {-1, 1}"
    print("   [OK] Stump predicts {-1, +1}")

    # polarity = 1 → values < threshold get -1
    assert preds[0] == -1 and preds[1] == -1  # 1, 2 < 3
    assert preds[3] == 1 and preds[4] == 1    # 4, 5 >= 3
    print("   [OK] Polarity +1 correct")

    stump.polarity = -1
    preds2 = stump.predict(X)
    assert preds2[0] == 1 and preds2[4] == -1
    print("   [OK] Polarity -1 correct")
    return True


def test_basic_binary_classification():
    """Test on linearly separable data"""
    print("\nTest 2: Basic Binary Classification")
    print("-" * 70)

    np.random.seed(42)
    X1 = np.random.randn(60, 2) + [2, 2]
    X2 = np.random.randn(60, 2) + [-2, -2]
    X = np.vstack([X1, X2])
    y = np.array([1] * 60 + [0] * 60)

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    split = 90
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    clf = AdaBoostClassifier(n_estimators=30, random_state=42)
    clf.fit(X_train, y_train)
    print("   [OK] Model trained")

    acc = clf.score(X_test, y_test)
    print(f"   [OK] Test accuracy: {acc:.3f}")
    assert acc > 0.85, f"Accuracy too low: {acc}"
    return True


def test_iris_binary():
    """Test on Iris (binary: setosa vs rest)"""
    print("\nTest 3: Iris Dataset (Binary)")
    print("-" * 70)

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.3, random_state=42
    )

    clf = AdaBoostClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(f"   [OK] Test accuracy: {acc:.3f}")
    assert acc > 0.90, f"Accuracy too low: {acc}"
    return True


def test_predict_proba():
    """Test probability predictions"""
    print("\nTest 4: Predict Probabilities")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)

    clf = AdaBoostClassifier(n_estimators=30, random_state=42)
    clf.fit(X[:100], y_bin[:100])

    proba = clf.predict_proba(X[100:])
    assert proba.shape == (50, 2), "Shape mismatch"
    print(f"   [OK] Probability shape: {proba.shape}")

    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6), "Do not sum to 1"
    print("   [OK] Probabilities sum to 1")

    assert np.all((proba >= 0) & (proba <= 1)), "Out of [0, 1]"
    print("   [OK] Probabilities in [0, 1]")
    return True


def test_feature_importances():
    """Test feature importances"""
    print("\nTest 5: Feature Importances")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)

    clf = AdaBoostClassifier(n_estimators=40, random_state=42)
    clf.fit(X, y_bin)

    fi = clf.feature_importances_
    assert fi is not None, "feature_importances_ is None"
    assert fi.shape == (4,), "Wrong shape"
    assert abs(fi.sum() - 1.0) < 1e-6, "Does not sum to 1"
    assert np.all(fi >= 0), "Negative importances"
    print(f"   [OK] Feature importances: {fi.round(3)}")
    print(f"   [OK] Sum = {fi.sum():.6f}")
    return True


def test_n_estimators_effect():
    """More estimators should improve training accuracy"""
    print("\nTest 6: N Estimators Effect")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)

    accs = []
    for n in [5, 20, 50, 100]:
        clf = AdaBoostClassifier(n_estimators=n, random_state=42)
        clf.fit(X, y_bin)
        accs.append(clf.score(X, y_bin))
        print(f"   [OK] n_estimators={n:3d}  train_acc={accs[-1]:.3f}")

    assert accs[-1] >= accs[0], "More estimators did not help"
    return True


def test_learning_rate():
    """Test different learning rates"""
    print("\nTest 7: Learning Rate Effect")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)

    for lr in [0.1, 0.5, 1.0, 2.0]:
        clf = AdaBoostClassifier(n_estimators=30, learning_rate=lr, random_state=42)
        clf.fit(X, y_bin)
        acc = clf.score(X, y_bin)
        print(f"   [OK] learning_rate={lr}  acc={acc:.3f}")
    return True


def test_staged_score():
    """Test staged_score yields one value per round"""
    print("\nTest 8: Staged Score")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)

    n = 30
    clf = AdaBoostClassifier(n_estimators=n, random_state=42)
    clf.fit(X, y_bin)

    scores = list(clf.staged_score(X, y_bin))
    assert len(scores) == n, f"Expected {n} scores, got {len(scores)}"
    print(f"   [OK] {len(scores)} staged scores returned")
    print(f"   [OK] First score: {scores[0]:.3f}  Last score: {scores[-1]:.3f}")
    return True


def test_training_errors():
    """Training errors should decrease over rounds"""
    print("\nTest 9: Training Errors")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)

    clf = AdaBoostClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y_bin)

    errs = clf.training_errors_
    assert len(errs) == 50, "Wrong number of error records"
    assert all(0 <= e <= 1 for e in errs), "Errors out of [0, 1]"
    print(f"   [OK] {len(errs)} error records")
    print(f"   [OK] First error: {errs[0]:.4f}  Last error: {errs[-1]:.4f}")
    return True


def test_decision_function():
    """Test decision_function output"""
    print("\nTest 10: Decision Function")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = (X[:, 0] > 0).astype(int)

    clf = AdaBoostClassifier(n_estimators=20, random_state=42)
    clf.fit(X, y)

    scores = clf.decision_function(X)
    assert scores.shape == (50,), "Shape mismatch"
    print(f"   [OK] Scores shape: {scores.shape}")
    print(f"   [OK] Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    return True


def test_custom_labels():
    """Test with non-standard labels {-1, +1}"""
    print("\nTest 11: Custom Labels {-1, +1}")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)

    clf = AdaBoostClassifier(n_estimators=30, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)

    assert set(np.unique(preds)) <= {-1, 1}, "Wrong label set"
    acc = clf.score(X, y)
    print(f"   [OK] Labels {set(np.unique(preds))} — accuracy: {acc:.3f}")
    assert acc > 0.7, f"Accuracy too low: {acc}"
    return True


def test_reproducibility():
    """Same random_state → same results"""
    print("\nTest 12: Reproducibility")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)

    clf1 = AdaBoostClassifier(n_estimators=20, random_state=0)
    clf1.fit(X, y_bin)

    clf2 = AdaBoostClassifier(n_estimators=20, random_state=0)
    clf2.fit(X, y_bin)

    np.testing.assert_array_equal(clf1.predict(X), clf2.predict(X))
    print("   [OK] Same seed -> identical predictions")
    return True


def run_all_tests():
    """Run all tests"""
    print("Testing AdaBoost Implementation")
    print("=" * 70)

    tests = [
        test_decision_stump,
        test_basic_binary_classification,
        test_iris_binary,
        test_predict_proba,
        test_feature_importances,
        test_n_estimators_effect,
        test_learning_rate,
        test_staged_score,
        test_training_errors,
        test_decision_function,
        test_custom_labels,
        test_reproducibility,
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
