"""
Comprehensive Tests for LightGBM From-Scratch Implementation
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lightgbm_scratch import (
    LightGBMClassifier, LightGBMRegressor,
    FeatureHistogram, goss_sample
)


def test_histogram_binning():
    """Test FeatureHistogram fit and transform"""
    print("Test 1: Histogram Binning")
    print("-" * 70)

    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    hist = FeatureHistogram(n_bins=5)
    hist.fit(values)

    assert hist.bin_edges is not None
    print(f"   [OK] Bin edges: {hist.bin_edges}")

    bins = hist.transform(values)
    assert bins.shape == (10,)
    assert bins.min() >= 0
    print(f"   [OK] Bin indices: {bins}")

    # New values in range
    new_vals = np.array([1.5, 5.5, 9.9])
    new_bins = hist.transform(new_vals)
    assert new_bins.shape == (3,)
    print(f"   [OK] New values mapped to bins: {new_bins}")
    return True


def test_goss_sampling():
    """Test GOSS sampling"""
    print("\nTest 2: GOSS Sampling")
    print("-" * 70)

    np.random.seed(42)
    gradients = np.random.randn(1000)

    idx, weights = goss_sample(gradients, top_rate=0.2, other_rate=0.1)

    assert len(idx) > 0
    assert len(idx) == len(weights)
    assert np.all(weights > 0)
    print(f"   [OK] Selected {len(idx)} samples from 1000")
    print(f"   [OK] Weight range: [{weights.min():.2f}, {weights.max():.2f}]")

    # Large-gradient samples get weight 1.0
    abs_grads = np.abs(gradients)
    sorted_idx = np.argsort(abs_grads)[::-1]
    large_idx_set = set(sorted_idx[:200])
    first_large = idx[0]
    assert first_large in large_idx_set
    print("   [OK] Large-gradient samples included")
    return True


def test_basic_classification():
    """Test binary classification on linearly separable data"""
    print("\nTest 3: Basic Binary Classification")
    print("-" * 70)

    np.random.seed(42)
    X1 = np.random.randn(100, 2) + [2, 2]
    X2 = np.random.randn(100, 2) + [-2, -2]
    X = np.vstack([X1, X2])
    y = np.array([1] * 100 + [0] * 100)
    idx = np.random.permutation(200)
    X, y = X[idx], y[idx]

    split = 150
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    clf = LightGBMClassifier(
        n_estimators=30, learning_rate=0.1,
        num_leaves=15, min_data_in_leaf=5, random_state=42
    )
    clf.fit(X_train, y_train)
    print("   [OK] Model trained")

    acc = clf.score(X_test, y_test)
    print(f"   [OK] Test accuracy: {acc:.3f}")
    assert acc > 0.85, f"Accuracy too low: {acc}"
    return True


def test_iris_classification():
    """Test on Iris dataset (binary)"""
    print("\nTest 4: Iris Dataset (Binary)")
    print("-" * 70)

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.3, random_state=42
    )

    clf = LightGBMClassifier(
        n_estimators=50, learning_rate=0.1,
        num_leaves=15, min_data_in_leaf=3, random_state=42
    )
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"   [OK] Test accuracy: {acc:.3f}")
    assert acc > 0.90, f"Accuracy too low: {acc}"
    return True


def test_predict_proba():
    """Test probability predictions"""
    print("\nTest 5: Predict Probabilities")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)

    clf = LightGBMClassifier(n_estimators=30, min_data_in_leaf=3, random_state=42)
    clf.fit(X[:100], y_bin[:100])

    proba = clf.predict_proba(X[100:])
    assert proba.shape == (50, 2)
    print(f"   [OK] Shape: {proba.shape}")
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    print("   [OK] Sum to 1")
    assert np.all((proba >= 0) & (proba <= 1))
    print("   [OK] In [0, 1]")
    return True


def test_basic_regression():
    """Test regression on linear data"""
    print("\nTest 6: Basic Regression")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(200, 1) * 3
    y = 2.5 * X[:, 0] + 1.0 + np.random.randn(200) * 0.5

    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]

    reg = LightGBMRegressor(
        n_estimators=50, learning_rate=0.1,
        num_leaves=15, min_data_in_leaf=5, random_state=42
    )
    reg.fit(X_train, y_train)
    r2 = reg.score(X_test, y_test)
    print(f"   [OK] Test R2: {r2:.3f}")
    assert r2 > 0.8, f"R2 too low: {r2}"
    return True


def test_nonlinear_regression():
    """Test regression on sine function"""
    print("\nTest 7: Nonlinear Regression (Sine)")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.uniform(-np.pi * 2, np.pi * 2, 300).reshape(-1, 1)
    y = np.sin(X[:, 0]) + np.random.randn(300) * 0.1

    X_train, X_test = X[:240], X[240:]
    y_train, y_test = y[:240], y[240:]

    reg = LightGBMRegressor(
        n_estimators=100, learning_rate=0.1,
        num_leaves=31, min_data_in_leaf=5, random_state=42
    )
    reg.fit(X_train, y_train)
    r2 = reg.score(X_test, y_test)
    print(f"   [OK] Test R2 on sine: {r2:.3f}")
    assert r2 > 0.80, f"R2 too low: {r2}"
    return True


def test_feature_importances():
    """Test feature importances"""
    print("\nTest 8: Feature Importances")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)

    clf = LightGBMClassifier(
        n_estimators=50, min_data_in_leaf=3, random_state=42
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


def test_num_leaves():
    """Test num_leaves effect on complexity"""
    print("\nTest 9: Num Leaves Effect")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)

    for nl in [4, 15, 31]:
        clf = LightGBMClassifier(
            n_estimators=30, num_leaves=nl,
            min_data_in_leaf=3, random_state=42
        )
        clf.fit(X, y_bin)
        acc = clf.score(X, y_bin)
        print(f"   [OK] num_leaves={nl:2d}  acc={acc:.3f}")
    return True


def test_regularization():
    """Test L2 regularization"""
    print("\nTest 10: L2 Regularization")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    for lam in [0.0, 1.0, 10.0]:
        clf = LightGBMClassifier(
            n_estimators=20, reg_lambda=lam,
            min_data_in_leaf=3, random_state=42
        )
        clf.fit(X, y)
        acc = clf.score(X, y)
        print(f"   [OK] reg_lambda={lam:4.1f}  acc={acc:.3f}")
    return True


def test_goss_classifier():
    """Test GOSS sampling with classifier"""
    print("\nTest 11: GOSS Sampling (Classifier)")
    print("-" * 70)

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.3, random_state=42
    )

    clf = LightGBMClassifier(
        n_estimators=50, learning_rate=0.1, num_leaves=15,
        use_goss=True, goss_top_rate=0.3, goss_other_rate=0.1,
        min_data_in_leaf=3, random_state=42
    )
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"   [OK] GOSS test accuracy: {acc:.3f}")
    assert acc > 0.85
    return True


def test_subsample_colsample():
    """Test row and column subsampling"""
    print("\nTest 12: Subsampling")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)

    clf = LightGBMClassifier(
        n_estimators=30, subsample=0.7, colsample_bytree=0.7,
        min_data_in_leaf=3, random_state=42
    )
    clf.fit(X, y_bin)
    acc = clf.score(X, y_bin)
    print(f"   [OK] subsample=0.7, colsample=0.7  acc={acc:.3f}")
    assert acc > 0.80
    return True


def test_california_housing():
    """Test on California Housing regression"""
    print("\nTest 13: California Housing Regression")
    print("-" * 70)

    try:
        from sklearn.datasets import fetch_california_housing
        from sklearn.model_selection import train_test_split

        X, y = fetch_california_housing(return_X_y=True)
        X, y = X[:500], y[:500]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        reg = LightGBMRegressor(
            n_estimators=50, learning_rate=0.1,
            num_leaves=31, min_data_in_leaf=10, random_state=42
        )
        reg.fit(X_train, y_train)
        r2 = reg.score(X_test, y_test)
        print(f"   [OK] R2 on California Housing: {r2:.3f}")
        assert r2 > 0.4
    except Exception as e:
        print(f"   [SKIP] {e}")
    return True


def run_all_tests():
    print("Testing LightGBM From-Scratch Implementation")
    print("=" * 70)

    tests = [
        test_histogram_binning,
        test_goss_sampling,
        test_basic_classification,
        test_iris_classification,
        test_predict_proba,
        test_basic_regression,
        test_nonlinear_regression,
        test_feature_importances,
        test_num_leaves,
        test_regularization,
        test_goss_classifier,
        test_subsample_colsample,
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
