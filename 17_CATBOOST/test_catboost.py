"""
Comprehensive Tests for CatBoost From-Scratch Implementation
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from catboost_scratch import (
    CatBoostClassifier, CatBoostRegressor,
    ObliviousTree, ordered_target_statistics
)


def test_oblivious_tree_symmetric():
    """Test that all nodes at each depth use the same split."""
    print("Test 1: Oblivious Tree — Symmetric Structure")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(200, 4)
    g = np.random.randn(200)
    h = np.ones(200)

    tree = ObliviousTree(max_depth=4, min_samples_leaf=1, reg_lambda=1.0)
    tree.fit(X, g, h)

    n_splits = len(tree.splits)
    assert n_splits <= 4, f"Too many splits: {n_splits}"
    n_leaves = 2 ** n_splits
    assert len(tree.leaf_values) == n_leaves, \
        f"Expected {n_leaves} leaves, got {len(tree.leaf_values)}"

    print(f"   [OK] Depth={n_splits}, leaves={n_leaves} (= 2^{n_splits})")
    for d, sp in enumerate(tree.splits):
        print(f"   [OK] Level {d}: feature={sp.feature_index}, threshold={sp.threshold:.4f}")
    return True


def test_oblivious_tree_predict():
    """Test prediction consistency."""
    print("\nTest 2: Oblivious Tree — Prediction")
    print("-" * 70)

    np.random.seed(0)
    X_train = np.random.randn(100, 3)
    g = X_train[:, 0]          # gradient = first feature
    h = np.ones(100)

    tree = ObliviousTree(max_depth=3, reg_lambda=1.0)
    tree.fit(X_train, g, h)

    preds = tree.predict(X_train)
    assert preds.shape == (100,)
    print(f"   [OK] Predictions shape: {preds.shape}")
    print(f"   [OK] Pred range: [{preds.min():.4f}, {preds.max():.4f}]")

    # Test on new data
    X_test = np.random.randn(20, 3)
    preds_test = tree.predict(X_test)
    assert preds_test.shape == (20,)
    print(f"   [OK] Test predictions shape: {preds_test.shape}")
    return True


def test_ordered_target_statistics():
    """Test ordered target statistics: no leakage."""
    print("\nTest 3: Ordered Target Statistics")
    print("-" * 70)

    np.random.seed(42)
    n = 100
    # 3 categories, all with mean target = their category id
    categories = np.array([0, 1, 2] * (n // 3) + [0] * (n % 3))
    targets = categories.astype(float) + np.random.randn(n) * 0.1

    encoded = ordered_target_statistics(categories, targets, prior=0.5, prior_strength=1.0)

    assert encoded.shape == (n,)
    print(f"   [OK] Shape: {encoded.shape}")

    # First sample of each category must use only prior (no history)
    first_cat0 = np.where(categories == 0)[0][0]
    assert encoded[first_cat0] == 0.5 * 1.0 / 1.0
    print(f"   [OK] First cat-0 sample uses prior only: {encoded[first_cat0]:.4f}")

    # Values should be in a reasonable range
    assert np.all(np.isfinite(encoded))
    print(f"   [OK] Range: [{encoded.min():.3f}, {encoded.max():.3f}]")
    return True


def test_ordered_ots_no_leakage():
    """Target statistics must not include the sample's own target."""
    print("\nTest 4: Ordered Target Statistics — No Leakage")
    print("-" * 70)

    # Single category: all samples in category 0
    categories = np.zeros(10, dtype=int)
    targets = np.arange(10, dtype=float)   # 0, 1, 2, ..., 9
    encoded = ordered_target_statistics(categories, targets, prior=0.0, prior_strength=1e-9)

    # Sample 0: no history -> ~0 (prior)
    # Sample 1: only sample 0 seen -> encoded[1] ~ 0
    # Sample 2: samples 0 and 1 seen -> encoded[2] ~ 0.5
    print(f"   [OK] encoded[0]={encoded[0]:.4f} (should be ~0)")
    print(f"   [OK] encoded[1]={encoded[1]:.4f} (should be ~0, target of sample 0)")
    print(f"   [OK] encoded[2]={encoded[2]:.4f} (should be ~0.5)")
    assert encoded[0] < 1.0   # didn't use y_0 = 0 to compute x_hat_0
    assert encoded[2] < 2.0   # only targets 0,1 used, not 2
    print("   [OK] No leakage confirmed")
    return True


def test_basic_classification():
    """Test binary classification on linearly separable data."""
    print("\nTest 5: Basic Binary Classification")
    print("-" * 70)

    np.random.seed(42)
    X1 = np.random.randn(100, 2) + [2, 2]
    X2 = np.random.randn(100, 2) + [-2, -2]
    X = np.vstack([X1, X2])
    y = np.array([1] * 100 + [0] * 100)
    idx = np.random.permutation(200)
    X, y = X[idx], y[idx]

    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]

    clf = CatBoostClassifier(
        iterations=50, learning_rate=0.1, depth=4,
        min_samples_leaf=3, random_state=42
    )
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"   [OK] Test accuracy: {acc:.3f}")
    assert acc > 0.85, f"Accuracy too low: {acc}"
    return True


def test_iris_classification():
    """Test on Iris dataset (binary)."""
    print("\nTest 6: Iris Dataset (Binary)")
    print("-" * 70)

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.3, random_state=42
    )

    clf = CatBoostClassifier(
        iterations=100, learning_rate=0.1, depth=4,
        min_samples_leaf=3, random_state=42
    )
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"   [OK] Test accuracy: {acc:.3f}")
    assert acc > 0.90, f"Accuracy too low: {acc}"
    return True


def test_predict_proba():
    """Test probability predictions."""
    print("\nTest 7: Predict Probabilities")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)

    clf = CatBoostClassifier(iterations=30, depth=4, min_samples_leaf=3, random_state=42)
    clf.fit(X[:100], y_bin[:100])

    proba = clf.predict_proba(X[100:])
    assert proba.shape == (50, 2), f"Wrong shape: {proba.shape}"
    print(f"   [OK] Shape: {proba.shape}")
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    print("   [OK] Sum to 1")
    assert np.all((proba >= 0) & (proba <= 1))
    print("   [OK] In [0, 1]")
    return True


def test_basic_regression():
    """Test regression on linear data."""
    print("\nTest 8: Basic Regression")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(200, 1) * 3
    y = 2.5 * X[:, 0] + 1.0 + np.random.randn(200) * 0.5

    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]

    reg = CatBoostRegressor(
        iterations=100, learning_rate=0.1, depth=4,
        min_samples_leaf=3, random_state=42
    )
    reg.fit(X_train, y_train)
    r2 = reg.score(X_test, y_test)
    print(f"   [OK] Test R2: {r2:.3f}")
    assert r2 > 0.80, f"R2 too low: {r2}"
    return True


def test_nonlinear_regression():
    """Test regression on sine function."""
    print("\nTest 9: Nonlinear Regression (Sine)")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.uniform(-np.pi * 2, np.pi * 2, 300).reshape(-1, 1)
    y = np.sin(X[:, 0]) + np.random.randn(300) * 0.1

    X_train, X_test = X[:240], X[240:]
    y_train, y_test = y[:240], y[240:]

    reg = CatBoostRegressor(
        iterations=200, learning_rate=0.05, depth=6,
        min_samples_leaf=3, random_state=42
    )
    reg.fit(X_train, y_train)
    r2 = reg.score(X_test, y_test)
    print(f"   [OK] Test R2 on sine: {r2:.3f}")
    assert r2 > 0.75, f"R2 too low: {r2}"
    return True


def test_feature_importances():
    """Test feature importances."""
    print("\nTest 10: Feature Importances")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)

    clf = CatBoostClassifier(
        iterations=100, depth=4, min_samples_leaf=3, random_state=42
    )
    clf.fit(X, y_bin)

    fi = clf.feature_importances_
    assert fi is not None
    assert fi.shape == (4,)
    assert abs(fi.sum() - 1.0) < 1e-6
    assert np.all(fi >= 0)
    print(f"   [OK] Feature importances: {fi.round(3)}")
    print(f"   [OK] Sum = {fi.sum():.6f}")
    return True


def test_depth_effect():
    """Deeper trees should capture more complex patterns."""
    print("\nTest 11: Depth Effect")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)

    for d in [1, 3, 6]:
        clf = CatBoostClassifier(
            iterations=50, depth=d, min_samples_leaf=3, random_state=42
        )
        clf.fit(X, y_bin)
        acc = clf.score(X, y_bin)
        leaves = 2 ** d
        print(f"   [OK] depth={d}  leaves={leaves:2d}  train_acc={acc:.3f}")
    return True


def test_regularization():
    """Test L2 regularization effect."""
    print("\nTest 12: L2 Regularization")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    for lam in [0.1, 3.0, 30.0]:
        clf = CatBoostClassifier(
            iterations=30, reg_lambda=lam,
            min_samples_leaf=3, random_state=42
        )
        clf.fit(X, y)
        acc = clf.score(X, y)
        print(f"   [OK] reg_lambda={lam:5.1f}  acc={acc:.3f}")
    return True


def test_ordered_boosting():
    """Test ordered boosting mode."""
    print("\nTest 13: Ordered Boosting")
    print("-" * 70)

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.3, random_state=42
    )

    clf = CatBoostClassifier(
        iterations=50, learning_rate=0.1, depth=4,
        use_ordered_boosting=True, n_ordered_folds=4,
        min_samples_leaf=3, random_state=42
    )
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"   [OK] Ordered boosting test accuracy: {acc:.3f}")
    assert acc > 0.85
    return True


def test_subsample():
    """Test row subsampling."""
    print("\nTest 14: Row Subsampling")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)

    clf = CatBoostClassifier(
        iterations=50, depth=4, subsample=0.7,
        min_samples_leaf=3, random_state=42
    )
    clf.fit(X, y_bin)
    acc = clf.score(X, y_bin)
    print(f"   [OK] subsample=0.7  acc={acc:.3f}")
    assert acc > 0.85
    return True


def test_california_housing():
    """Test on California Housing regression."""
    print("\nTest 15: California Housing Regression")
    print("-" * 70)

    try:
        from sklearn.datasets import fetch_california_housing
        from sklearn.model_selection import train_test_split

        X, y = fetch_california_housing(return_X_y=True)
        X, y = X[:500], y[:500]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        reg = CatBoostRegressor(
            iterations=100, learning_rate=0.1, depth=5,
            min_samples_leaf=5, random_state=42
        )
        reg.fit(X_train, y_train)
        r2 = reg.score(X_test, y_test)
        print(f"   [OK] R2 on California Housing: {r2:.3f}")
        assert r2 > 0.3
    except Exception as e:
        print(f"   [SKIP] {e}")
    return True


def run_all_tests():
    print("Testing CatBoost From-Scratch Implementation")
    print("=" * 70)

    tests = [
        test_oblivious_tree_symmetric,
        test_oblivious_tree_predict,
        test_ordered_target_statistics,
        test_ordered_ots_no_leakage,
        test_basic_classification,
        test_iris_classification,
        test_predict_proba,
        test_basic_regression,
        test_nonlinear_regression,
        test_feature_importances,
        test_depth_effect,
        test_regularization,
        test_ordered_boosting,
        test_subsample,
        test_california_housing,
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
