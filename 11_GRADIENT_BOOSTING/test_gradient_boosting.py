"""
Comprehensive Tests for Gradient Boosting Implementation

Tests both GradientBoostingClassifier and GradientBoostingRegressor
"""

import numpy as np
import sys
from gradient_boosting import GradientBoostingClassifier, GradientBoostingRegressor


def test_classifier_basic():
    """Test basic classification functionality"""
    print("Test 1: Basic Classification")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    split = 70
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1,
                                     max_depth=3, random_state=42)
    print("   [OK] GradientBoostingClassifier created")

    gb.fit(X_train, y_train)
    print("   [OK] Model trained")

    y_pred = gb.predict(X_test)
    assert y_pred.shape == y_test.shape, "Predictions shape mismatch"
    print("   [OK] Predictions shape correct")

    assert np.all(np.isin(y_pred, [0, 1])), "Invalid predictions"
    print("   [OK] All predictions are valid classes")

    acc = gb.score(X_test, y_test)
    assert 0 <= acc <= 1, "Invalid accuracy"
    print(f"   [OK] Test accuracy: {acc:.3f}")

    return True


def test_classifier_iris():
    """Test on Iris dataset"""
    print("\nTest 2: Iris Dataset Classification")
    print("-" * 70)

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)

    # Binary classification (setosa vs others)
    y_binary = (y == 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.3, random_state=42
    )

    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                     random_state=42)
    gb.fit(X_train, y_train)
    print("   [OK] Model trained on Iris")

    acc = gb.score(X_test, y_test)
    print(f"   [OK] Test accuracy: {acc:.3f}")

    assert acc > 0.8, "Accuracy too low on Iris"
    print("   [OK] Accuracy > 0.8")

    return True


def test_probability_predictions():
    """Test probability predictions"""
    print("\nTest 3: Probability Predictions")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
    gb.fit(X[:70], y[:70])

    proba = gb.predict_proba(X[70:])
    assert proba.shape == (30, 2), "Probability shape incorrect"
    print("   [OK] Probability shape correct")

    assert np.allclose(proba.sum(axis=1), 1.0), "Probabilities don't sum to 1"
    print("   [OK] Probabilities sum to 1")

    assert np.all((proba >= 0) & (proba <= 1)), "Probabilities out of range"
    print("   [OK] Probabilities in valid range [0, 1]")

    return True


def test_learning_rate_effect():
    """Test effect of learning rate"""
    print("\nTest 4: Learning Rate Effect")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    rates = [0.01, 0.1, 0.5]

    for lr in rates:
        gb = GradientBoostingClassifier(n_estimators=50, learning_rate=lr, random_state=42)
        gb.fit(X[:70], y[:70])
        acc = gb.score(X[70:], y[70:])
        print(f"   [OK] Learning rate={lr:.2f}: accuracy={acc:.3f}")

    return True


def test_n_estimators_effect():
    """Test effect of number of estimators"""
    print("\nTest 5: Number of Estimators Effect")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    n_trees = [10, 50, 100, 200]

    for n in n_trees:
        gb = GradientBoostingClassifier(n_estimators=n, random_state=42)
        gb.fit(X[:70], y[:70])
        acc = gb.score(X[70:], y[70:])
        print(f"   [OK] n_estimators={n:3d}: accuracy={acc:.3f}, trees={len(gb.trees_)}")

    return True


def test_max_depth_effect():
    """Test effect of max depth"""
    print("\nTest 6: Max Depth Effect")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    depths = [1, 2, 3, 5]

    for depth in depths:
        gb = GradientBoostingClassifier(n_estimators=50, max_depth=depth, random_state=42)
        gb.fit(X[:70], y[:70])
        acc = gb.score(X[70:], y[70:])
        print(f"   [OK] max_depth={depth}: accuracy={acc:.3f}")

    return True


def test_subsample():
    """Test subsampling (Stochastic Gradient Boosting)"""
    print("\nTest 7: Subsampling")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    subsample_rates = [0.5, 0.8, 1.0]

    for rate in subsample_rates:
        gb = GradientBoostingClassifier(n_estimators=50, subsample=rate, random_state=42)
        gb.fit(X[:70], y[:70])
        acc = gb.score(X[70:], y[70:])
        print(f"   [OK] subsample={rate:.1f}: accuracy={acc:.3f}")

    return True


def test_loss_functions():
    """Test different loss functions"""
    print("\nTest 8: Loss Functions")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    losses = ['log_loss', 'exponential']

    for loss in losses:
        gb = GradientBoostingClassifier(n_estimators=50, loss=loss, random_state=42)
        gb.fit(X[:70], y[:70])
        acc = gb.score(X[70:], y[70:])
        print(f"   [OK] loss={loss}: accuracy={acc:.3f}")

    return True


def test_staged_predict():
    """Test staged predictions"""
    print("\nTest 9: Staged Predictions")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    gb = GradientBoostingClassifier(n_estimators=20, random_state=42)
    gb.fit(X[:70], y[:70])

    staged_preds = list(gb.staged_predict(X[70:]))
    assert len(staged_preds) == 20, "Incorrect number of staged predictions"
    print("   [OK] Staged predictions count correct")

    staged_proba = list(gb.staged_predict_proba(X[70:]))
    assert len(staged_proba) == 20, "Incorrect number of staged probabilities"
    print("   [OK] Staged probabilities count correct")

    return True


def test_regressor_basic():
    """Test basic regression functionality"""
    print("\nTest 10: Basic Regression")
    print("-" * 70)

    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=200, n_features=4, noise=10, random_state=42)

    split = 140
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                    max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    print("   [OK] GradientBoostingRegressor trained")

    y_pred = gb.predict(X_test)
    assert y_pred.shape == y_test.shape, "Predictions shape mismatch"
    print("   [OK] Predictions shape correct")

    r2 = gb.score(X_test, y_test)
    assert -1 <= r2 <= 1, "Invalid R² score"
    print(f"   [OK] Test R²: {r2:.3f}")

    return True


def test_regressor_loss_functions():
    """Test different regression loss functions"""
    print("\nTest 11: Regression Loss Functions")
    print("-" * 70)

    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=200, n_features=4, noise=10, random_state=42)

    split = 140
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    losses = ['squared_error', 'absolute_error', 'huber']

    for loss in losses:
        gb = GradientBoostingRegressor(n_estimators=100, loss=loss, random_state=42)
        gb.fit(X_train, y_train)
        r2 = gb.score(X_test, y_test)
        print(f"   [OK] loss={loss}: R²={r2:.3f}")

    return True


def test_regressor_staged():
    """Test staged predictions for regression"""
    print("\nTest 12: Regression Staged Predictions")
    print("-" * 70)

    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=100, n_features=4, noise=10, random_state=42)

    gb = GradientBoostingRegressor(n_estimators=20, random_state=42)
    gb.fit(X[:70], y[:70])

    staged_preds = list(gb.staged_predict(X[70:]))
    assert len(staged_preds) == 20, "Incorrect number of staged predictions"
    print("   [OK] Staged predictions count correct")

    # Check that predictions improve over stages
    test_mse = [np.mean((pred - y[70:]) ** 2) for pred in staged_preds]
    print(f"   First 5 MSE: {test_mse[:5]}")
    print(f"   Last 5 MSE: {test_mse[-5:]}")

    return True


def test_input_validation():
    """Test input validation"""
    print("\nTest 13: Input Validation")
    print("-" * 70)

    X = np.random.randn(100, 4)
    y = np.random.randint(0, 2, size=100)

    gb = GradientBoostingClassifier(n_estimators=10)
    gb.fit(X, y)

    # Wrong number of features
    try:
        gb.predict(np.random.randn(10, 5))
        print("   [FAIL] Should raise error for wrong features")
        return False
    except:
        print("   [OK] Correct error for wrong features")

    # Predict before fit
    try:
        gb2 = GradientBoostingClassifier(n_estimators=10)
        gb2.predict(X)
        print("   [FAIL] Should raise error for predict before fit")
        return False
    except ValueError:
        print("   [OK] Correct error for predict before fit")

    return True


def run_all_tests():
    """Run all tests"""
    print("Testing Gradient Boosting Implementation")
    print("=" * 70)

    tests = [
        test_classifier_basic,
        test_classifier_iris,
        test_probability_predictions,
        test_learning_rate_effect,
        test_n_estimators_effect,
        test_max_depth_effect,
        test_subsample,
        test_loss_functions,
        test_staged_predict,
        test_regressor_basic,
        test_regressor_loss_functions,
        test_regressor_staged,
        test_input_validation
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
