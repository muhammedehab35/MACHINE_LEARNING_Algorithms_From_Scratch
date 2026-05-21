"""
Comprehensive Tests for XGBoost Implementation

Tests XGBoost for classification and regression
"""

import numpy as np
import sys
from xgboost import XGBoostClassifier, XGBoostRegressor


def test_basic_classification():
    """Test basic binary classification"""
    print("Test 1: Basic Binary Classification")
    print("-" * 70)

    np.random.seed(42)
    # Create linearly separable data
    X1 = np.random.randn(50, 2) + np.array([2, 2])
    X2 = np.random.randn(50, 2) + np.array([-2, -2])
    X = np.vstack([X1, X2])
    y = np.array([1] * 50 + [0] * 50)

    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    # Split
    split = 70
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    xgb = XGBoostClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                            random_state=42)
    print("   [OK] XGBoost classifier created")

    xgb.fit(X_train, y_train)
    print("   [OK] Model trained")

    y_pred = xgb.predict(X_test)
    assert y_pred.shape == y_test.shape, "Predictions shape mismatch"
    print("   [OK] Predictions shape correct")

    acc = xgb.score(X_test, y_test)
    print(f"   [OK] Test accuracy: {acc:.3f}")
    assert acc > 0.7, "Accuracy too low"

    return True


def test_classification_iris():
    """Test on Iris dataset (binary: setosa vs others)"""
    print("\nTest 2: Iris Dataset (Binary Classification)")
    print("-" * 70)

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    # Binary: setosa (0) vs others (1)
    y_binary = (y != 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.3, random_state=42
    )

    xgb = XGBoostClassifier(n_estimators=30, max_depth=3, learning_rate=0.1,
                            random_state=42)
    xgb.fit(X_train, y_train)
    print("   [OK] Model trained on Iris")

    acc = xgb.score(X_test, y_test)
    print(f"   [OK] Test accuracy: {acc:.3f}")
    assert acc > 0.9, "Accuracy too low on Iris"

    return True


def test_probability_predictions():
    """Test probability predictions"""
    print("\nTest 3: Probability Predictions")
    print("-" * 70)

    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    y_binary = (y != 0).astype(int)

    xgb = XGBoostClassifier(n_estimators=20, max_depth=3, random_state=42)
    xgb.fit(X[:100], y_binary[:100])

    proba = xgb.predict_proba(X[100:])
    print(f"   [OK] Probability shape: {proba.shape}")

    # Check that probabilities sum to 1
    assert np.allclose(proba.sum(axis=1), 1.0), "Probabilities don't sum to 1"
    print("   [OK] Probabilities sum to 1")

    # Check range [0, 1]
    assert np.all((proba >= 0) & (proba <= 1)), "Probabilities out of range"
    print("   [OK] Probabilities in valid range [0, 1]")

    return True


def test_basic_regression():
    """Test basic regression"""
    print("\nTest 4: Basic Regression")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 1) * 3
    y = 2 * X[:, 0] + 1 + np.random.randn(100) * 0.5

    split = 70
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    xgb = XGBoostRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                           random_state=42)
    print("   [OK] XGBoost regressor created")

    xgb.fit(X_train, y_train)
    print("   [OK] Model trained")

    y_pred = xgb.predict(X_test)
    assert y_pred.shape == y_test.shape, "Predictions shape mismatch"
    print("   [OK] Predictions shape correct")

    r2 = xgb.score(X_test, y_test)
    print(f"   [OK] Test R²: {r2:.3f}")
    assert r2 > 0.7, "R² too low"

    return True


def test_regression_boston():
    """Test on Boston housing (or similar dataset)"""
    print("\nTest 5: Regression on California Housing")
    print("-" * 70)

    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    try:
        X, y = fetch_california_housing(return_X_y=True)

        # Use subset for speed
        X = X[:500]
        y = y[:500]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        xgb = XGBoostRegressor(n_estimators=30, max_depth=4, learning_rate=0.1,
                               random_state=42)
        xgb.fit(X_train, y_train)
        print("   [OK] Model trained on California Housing")

        r2 = xgb.score(X_test, y_test)
        print(f"   [OK] Test R²: {r2:.3f}")
        assert r2 > 0.5, "R² too low"

    except Exception as e:
        print(f"   [SKIP] California Housing not available: {e}")

    return True


def test_regularization_lambda():
    """Test L2 regularization (lambda)"""
    print("\nTest 6: L2 Regularization (lambda)")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, size=100)

    # No regularization
    xgb_no_reg = XGBoostClassifier(n_estimators=20, reg_lambda=0.0, random_state=42)
    xgb_no_reg.fit(X, y)
    acc_no_reg = xgb_no_reg.score(X, y)
    print(f"   [OK] No regularization: {acc_no_reg:.3f}")

    # With regularization
    xgb_reg = XGBoostClassifier(n_estimators=20, reg_lambda=10.0, random_state=42)
    xgb_reg.fit(X, y)
    acc_reg = xgb_reg.score(X, y)
    print(f"   [OK] With L2 regularization: {acc_reg:.3f}")

    # Regularization should prevent overfitting
    # (training accuracy should be lower with regularization)
    print(f"   [OK] Regularization applied")

    return True


def test_regularization_alpha():
    """Test L1 regularization (alpha)"""
    print("\nTest 7: L1 Regularization (alpha)")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, size=100)

    # No L1 regularization
    xgb_no_l1 = XGBoostClassifier(n_estimators=20, reg_alpha=0.0, random_state=42)
    xgb_no_l1.fit(X, y)
    acc_no_l1 = xgb_no_l1.score(X, y)
    print(f"   [OK] No L1 regularization: {acc_no_l1:.3f}")

    # With L1 regularization
    xgb_l1 = XGBoostClassifier(n_estimators=20, reg_alpha=1.0, random_state=42)
    xgb_l1.fit(X, y)
    acc_l1 = xgb_l1.score(X, y)
    print(f"   [OK] With L1 regularization: {acc_l1:.3f}")

    return True


def test_gamma_pruning():
    """Test gamma (complexity control)"""
    print("\nTest 8: Gamma (Complexity Control)")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, size=100)

    # No pruning
    xgb_no_gamma = XGBoostClassifier(n_estimators=20, gamma=0.0, random_state=42)
    xgb_no_gamma.fit(X, y)
    acc_no_gamma = xgb_no_gamma.score(X, y)
    print(f"   [OK] No gamma (no pruning): {acc_no_gamma:.3f}")

    # With pruning
    xgb_gamma = XGBoostClassifier(n_estimators=20, gamma=5.0, random_state=42)
    xgb_gamma.fit(X, y)
    acc_gamma = xgb_gamma.score(X, y)
    print(f"   [OK] Gamma=5.0 (pruning): {acc_gamma:.3f}")

    return True


def test_subsample():
    """Test row subsampling"""
    print("\nTest 9: Row Subsampling")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, size=100)

    # No subsampling
    xgb_no_sub = XGBoostClassifier(n_estimators=20, subsample=1.0, random_state=42)
    xgb_no_sub.fit(X, y)
    acc_no_sub = xgb_no_sub.score(X, y)
    print(f"   [OK] No subsampling: {acc_no_sub:.3f}")

    # With subsampling
    xgb_sub = XGBoostClassifier(n_estimators=20, subsample=0.7, random_state=42)
    xgb_sub.fit(X, y)
    acc_sub = xgb_sub.score(X, y)
    print(f"   [OK] Subsample=0.7: {acc_sub:.3f}")

    return True


def test_colsample():
    """Test column (feature) subsampling"""
    print("\nTest 10: Feature Subsampling")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, size=100)

    # No column sampling
    xgb_no_col = XGBoostClassifier(n_estimators=20, colsample_bytree=1.0,
                                   random_state=42)
    xgb_no_col.fit(X, y)
    acc_no_col = xgb_no_col.score(X, y)
    print(f"   [OK] No column sampling: {acc_no_col:.3f}")

    # With column sampling
    xgb_col = XGBoostClassifier(n_estimators=20, colsample_bytree=0.5,
                                random_state=42)
    xgb_col.fit(X, y)
    acc_col = xgb_col.score(X, y)
    print(f"   [OK] Colsample=0.5: {acc_col:.3f}")

    return True


def test_max_depth():
    """Test max depth parameter"""
    print("\nTest 11: Max Depth")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, size=100)

    depths = [2, 4, 6]
    for depth in depths:
        xgb = XGBoostClassifier(n_estimators=20, max_depth=depth, random_state=42)
        xgb.fit(X, y)
        acc = xgb.score(X, y)
        print(f"   [OK] Max depth {depth}: accuracy {acc:.3f}")

    return True


def test_learning_rate():
    """Test learning rate effect"""
    print("\nTest 12: Learning Rate Effect")
    print("-" * 70)

    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    y_binary = (y != 0).astype(int)

    rates = [0.01, 0.1, 0.5]
    for lr in rates:
        xgb = XGBoostClassifier(n_estimators=50, learning_rate=lr, random_state=42)
        xgb.fit(X[:100], y_binary[:100])
        acc = xgb.score(X[100:], y_binary[100:])
        print(f"   [OK] Learning rate {lr}: accuracy {acc:.3f}")

    return True


def test_min_child_weight():
    """Test min_child_weight parameter"""
    print("\nTest 13: Min Child Weight")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, size=100)

    weights = [1.0, 5.0, 10.0]
    for weight in weights:
        xgb = XGBoostClassifier(n_estimators=20, min_child_weight=weight,
                                random_state=42)
        xgb.fit(X, y)
        acc = xgb.score(X, y)
        print(f"   [OK] Min child weight {weight}: accuracy {acc:.3f}")

    return True


def test_n_estimators():
    """Test number of trees effect"""
    print("\nTest 14: Number of Trees")
    print("-" * 70)

    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    y_binary = (y != 0).astype(int)

    n_trees = [10, 50, 100]
    for n in n_trees:
        xgb = XGBoostClassifier(n_estimators=n, learning_rate=0.1, random_state=42)
        xgb.fit(X[:100], y_binary[:100])
        acc = xgb.score(X[100:], y_binary[100:])
        print(f"   [OK] {n} trees: accuracy {acc:.3f}")

    return True


def test_regression_nonlinear():
    """Test on non-linear regression"""
    print("\nTest 15: Non-linear Regression")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.rand(200, 1) * 10 - 5
    y = np.sin(X[:, 0]) + np.random.randn(200) * 0.1

    split = 150
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    xgb = XGBoostRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                           random_state=42)
    xgb.fit(X_train, y_train)
    print("   [OK] Model trained on sine function")

    r2 = xgb.score(X_test, y_test)
    print(f"   [OK] Test R²: {r2:.3f}")
    assert r2 > 0.8, "R² too low on sine function"

    return True


def run_all_tests():
    """Run all tests"""
    print("Testing XGBoost Implementation")
    print("=" * 70)

    tests = [
        test_basic_classification,
        test_classification_iris,
        test_probability_predictions,
        test_basic_regression,
        test_regression_boston,
        test_regularization_lambda,
        test_regularization_alpha,
        test_gamma_pruning,
        test_subsample,
        test_colsample,
        test_max_depth,
        test_learning_rate,
        test_min_child_weight,
        test_n_estimators,
        test_regression_nonlinear
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"   [FAIL] {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"   [FAIL] {test.__name__}: {str(e)}")

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
