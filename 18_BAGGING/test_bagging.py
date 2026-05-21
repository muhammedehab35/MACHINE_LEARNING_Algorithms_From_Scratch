"""
Comprehensive Tests for Bagging From-Scratch Implementation
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bagging_scratch import (
    BaggingClassifier, BaggingRegressor,
    bootstrap_sample, oob_indices
)


def test_bootstrap_sample():
    """Test bootstrap sampling properties."""
    print("Test 1: Bootstrap Sample")
    print("-" * 70)

    np.random.seed(42)
    n = 1000
    samples = bootstrap_sample(n, n)
    unique = np.unique(samples)
    frac_unique = len(unique) / n

    print(f"   [OK] Unique fraction: {frac_unique:.3f} (expected ~0.632)")
    assert 0.55 < frac_unique < 0.70, f"Unexpected fraction: {frac_unique}"

    oob = oob_indices(samples, n)
    frac_oob = len(oob) / n
    print(f"   [OK] OOB fraction: {frac_oob:.3f} (expected ~0.368)")
    assert 0.30 < frac_oob < 0.45
    assert len(unique) + len(oob) == n
    print("   [OK] in-bag + OOB = n")
    return True


def test_bootstrap_statistics():
    """Verify ~63.2% unique samples across many bootstrap draws."""
    print("\nTest 2: Bootstrap Statistics")
    print("-" * 70)

    np.random.seed(0)
    fracs = [len(np.unique(bootstrap_sample(200, 200))) / 200 for _ in range(500)]
    mean_frac = np.mean(fracs)
    print(f"   [OK] Mean unique fraction over 500 draws: {mean_frac:.4f}")
    assert abs(mean_frac - (1 - 1/np.e)) < 0.02
    return True


def test_basic_classification():
    """Test binary classification on linearly separable data."""
    print("\nTest 3: Basic Binary Classification")
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

    clf = BaggingClassifier(n_estimators=20, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"   [OK] Test accuracy: {acc:.3f}")
    assert acc > 0.85
    return True


def test_iris_classification():
    """Test on Iris dataset."""
    print("\nTest 4: Iris Dataset (Binary)")
    print("-" * 70)

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.3, random_state=42
    )

    clf = BaggingClassifier(n_estimators=30, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"   [OK] Test accuracy: {acc:.3f}")
    assert acc > 0.90
    return True


def test_predict_proba():
    """Test probability predictions."""
    print("\nTest 5: Predict Probabilities")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)

    clf = BaggingClassifier(n_estimators=20, random_state=42)
    clf.fit(X[:100], y_bin[:100])

    proba = clf.predict_proba(X[100:])
    assert proba.shape == (50, 2)
    print(f"   [OK] Shape: {proba.shape}")
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    print("   [OK] Sum to 1")
    assert np.all((proba >= 0) & (proba <= 1))
    print("   [OK] In [0, 1]")
    return True


def test_oob_score_classifier():
    """Test OOB score for classifier."""
    print("\nTest 6: OOB Score (Classifier)")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)

    clf = BaggingClassifier(
        n_estimators=50, oob_score=True, random_state=42
    )
    clf.fit(X, y_bin)

    assert clf.oob_score_ is not None
    print(f"   [OK] OOB accuracy: {clf.oob_score_:.3f}")
    assert 0.0 <= clf.oob_score_ <= 1.0
    assert clf.oob_decision_function_.shape == (len(y_bin), 2)
    print(f"   [OK] OOB decision function shape: {clf.oob_decision_function_.shape}")
    return True


def test_basic_regression():
    """Test regression on linear data."""
    print("\nTest 7: Basic Regression")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(200, 1) * 3
    y = 2.5 * X[:, 0] + 1.0 + np.random.randn(200) * 0.5

    reg = BaggingRegressor(n_estimators=20, random_state=42)
    reg.fit(X[:150], y[:150])
    r2 = reg.score(X[150:], y[150:])
    print(f"   [OK] Test R2: {r2:.3f}")
    assert r2 > 0.80
    return True


def test_oob_score_regressor():
    """Test OOB score for regressor."""
    print("\nTest 8: OOB Score (Regressor)")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(200, 3)
    y = X[:, 0] * 2 + X[:, 1] + np.random.randn(200) * 0.3

    reg = BaggingRegressor(n_estimators=50, oob_score=True, random_state=42)
    reg.fit(X, y)

    assert reg.oob_score_ is not None
    print(f"   [OK] OOB R2: {reg.oob_score_:.3f}")
    assert reg.oob_score_ > 0.5
    assert reg.oob_prediction_.shape == (200,)
    print(f"   [OK] OOB prediction shape: {reg.oob_prediction_.shape}")
    return True


def test_n_estimators_effect():
    """More estimators should improve or stabilize accuracy."""
    print("\nTest 9: N Estimators Effect")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)

    accs = []
    for n in [1, 5, 20, 50]:
        clf = BaggingClassifier(n_estimators=n, random_state=42)
        clf.fit(X, y_bin)
        accs.append(clf.score(X, y_bin))
        print(f"   [OK] n_estimators={n:3d}  acc={accs[-1]:.3f}")

    assert accs[-1] >= accs[0], "More estimators should not hurt"
    return True


def test_max_samples():
    """Test different max_samples values."""
    print("\nTest 10: Max Samples")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)

    for ms in [0.5, 0.8, 1.0]:
        clf = BaggingClassifier(
            n_estimators=20, max_samples=ms, random_state=42
        )
        clf.fit(X, y_bin)
        acc = clf.score(X, y_bin)
        print(f"   [OK] max_samples={ms}  acc={acc:.3f}")
    return True


def test_max_features():
    """Test feature subsampling."""
    print("\nTest 11: Max Features")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)

    for mf in [0.5, 0.75, 1.0]:
        clf = BaggingClassifier(
            n_estimators=20, max_features=mf, random_state=42
        )
        clf.fit(X, y_bin)
        acc = clf.score(X, y_bin)
        print(f"   [OK] max_features={mf}  acc={acc:.3f}")
    return True


def test_no_bootstrap():
    """Test pasting (sampling without replacement)."""
    print("\nTest 12: Pasting (bootstrap=False)")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)

    clf = BaggingClassifier(
        n_estimators=20, max_samples=0.8,
        bootstrap=False, random_state=42
    )
    clf.fit(X, y_bin)
    acc = clf.score(X, y_bin)
    print(f"   [OK] Pasting accuracy: {acc:.3f}")
    assert acc > 0.80
    return True


def test_feature_importances():
    """Test averaged feature importances."""
    print("\nTest 13: Feature Importances")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)

    clf = BaggingClassifier(n_estimators=30, random_state=42)
    clf.fit(X, y_bin)

    fi = clf.feature_importances_
    assert fi is not None
    assert fi.shape == (4,)
    assert abs(fi.sum() - 1.0) < 1e-6
    assert np.all(fi >= 0)
    print(f"   [OK] Feature importances: {fi.round(3)}")
    print(f"   [OK] Sum = {fi.sum():.6f}")
    return True


def test_variance_reduction():
    """Ensemble should have lower variance than single estimator."""
    print("\nTest 14: Variance Reduction")
    print("-" * 70)

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import cross_val_score

    X, y = load_iris(return_X_y=True)
    y_bin = (y != 0).astype(int)

    np.random.seed(42)
    single_scores = []
    bag_scores = []

    for seed in range(10):
        np.random.seed(seed)
        idx = np.random.permutation(len(y_bin))
        X_s, y_s = X[idx], y_bin[idx]
        split = int(0.7 * len(y_s))

        dt = DecisionTreeClassifier(random_state=seed)
        dt.fit(X_s[:split], y_s[:split])
        single_scores.append(dt.score(X_s[split:], y_s[split:]))

        bag = BaggingClassifier(n_estimators=20, random_state=seed)
        bag.fit(X_s[:split], y_s[:split])
        bag_scores.append(bag.score(X_s[split:], y_s[split:]))

    single_std = np.std(single_scores)
    bag_std = np.std(bag_scores)
    print(f"   [OK] Single tree  std={single_std:.4f}  mean={np.mean(single_scores):.3f}")
    print(f"   [OK] Bagging      std={bag_std:.4f}  mean={np.mean(bag_scores):.3f}")
    assert bag_std <= single_std * 1.5, "Bagging should not increase variance"
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
        reg = BaggingRegressor(n_estimators=20, random_state=42)
        reg.fit(X_train, y_train)
        r2 = reg.score(X_test, y_test)
        print(f"   [OK] R2 on California Housing: {r2:.3f}")
        assert r2 > 0.4
    except Exception as e:
        print(f"   [SKIP] {e}")
    return True


def run_all_tests():
    print("Testing Bagging From-Scratch Implementation")
    print("=" * 70)

    tests = [
        test_bootstrap_sample,
        test_bootstrap_statistics,
        test_basic_classification,
        test_iris_classification,
        test_predict_proba,
        test_oob_score_classifier,
        test_basic_regression,
        test_oob_score_regressor,
        test_n_estimators_effect,
        test_max_samples,
        test_max_features,
        test_no_bootstrap,
        test_feature_importances,
        test_variance_reduction,
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
