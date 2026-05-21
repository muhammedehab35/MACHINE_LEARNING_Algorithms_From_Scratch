"""
Comprehensive Tests for Random Forest Implementation

Tests both RandomForestClassifier and RandomForestRegressor
"""

import numpy as np
import sys
from random_forest import RandomForestClassifier, RandomForestRegressor


def test_classification_basic():
    """Test basic classification functionality"""
    print("Test 1: Basic Classification")
    print("-" * 70)

    # Create simple dataset
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Split data
    split = 70
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train model
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    print("   [OK] RandomForestClassifier created successfully")

    rf.fit(X_train, y_train)
    print("   [OK] Model trained successfully")

    # Test predictions
    y_pred = rf.predict(X_test)
    assert y_pred.shape == y_test.shape, "Predictions shape mismatch"
    print("   [OK] Predictions shape correct")

    assert np.all(np.isin(y_pred, [0, 1])), "Invalid predictions"
    print("   [OK] All predictions are valid classes")

    # Test accuracy
    acc = rf.score(X_test, y_test)
    assert 0 <= acc <= 1, "Invalid accuracy"
    print(f"   [OK] Test accuracy: {acc:.3f}")

    return True


def test_classification_multiclass():
    """Test multiclass classification"""
    print("\nTest 2: Multiclass Classification")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)

    # Split data
    split = 100
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    print("   [OK] Multiclass model trained")

    # Test predictions
    y_pred = rf.predict(X_test)
    assert np.all(np.isin(y_pred, [0, 1, 2])), "Invalid class predictions"
    print("   [OK] All predictions are valid classes")

    acc = rf.score(X_test, y_test)
    print(f"   [OK] Test accuracy: {acc:.3f}")

    return True


def test_probability_predictions():
    """Test probability predictions"""
    print("\nTest 3: Probability Predictions")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X[:70], y[:70])

    # Test probabilities
    proba = rf.predict_proba(X[70:])
    assert proba.shape == (30, 2), "Probability shape incorrect"
    print("   [OK] Probability predictions shape correct")

    # Check probabilities sum to 1
    assert np.allclose(proba.sum(axis=1), 1.0), "Probabilities don't sum to 1"
    print("   [OK] Probabilities sum to 1")

    # Check range
    assert np.all((proba >= 0) & (proba <= 1)), "Probabilities out of range"
    print("   [OK] Probabilities in valid range [0, 1]")

    return True


def test_feature_importance():
    """Test feature importance calculation"""
    print("\nTest 4: Feature Importance")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Check feature importances
    assert rf.feature_importances_ is not None, "Feature importances not computed"
    print("   [OK] Feature importances computed")

    assert rf.feature_importances_.shape == (4,), "Wrong feature importances shape"
    print("   [OK] Feature importances shape correct")

    assert np.allclose(rf.feature_importances_.sum(), 1.0), "Importances don't sum to 1"
    print("   [OK] Feature importances sum to 1")

    assert np.all(rf.feature_importances_ >= 0), "Negative importance values"
    print("   [OK] All importances non-negative")

    print(f"   Feature importances: {rf.feature_importances_}")

    return True


def test_oob_score():
    """Test out-of-bag score"""
    print("\nTest 5: OOB Score")
    print("-" * 70)

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)

    # Check OOB score exists
    assert rf.oob_score_ is not None, "OOB score not computed"
    print(f"   [OK] OOB score computed: {rf.oob_score_:.3f}")

    # Compare with test score
    test_score = rf.score(X_test, y_test)
    print(f"   [OK] Test score: {test_score:.3f}")

    # OOB should be reasonably close to test score
    diff = abs(rf.oob_score_ - test_score)
    print(f"   Difference between OOB and test: {diff:.3f}")

    return True


def test_regression():
    """Test regression functionality"""
    print("\nTest 6: Regression")
    print("-" * 70)

    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=200, n_features=4, noise=10, random_state=42)

    # Split data
    split = 140
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train model
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    print("   [OK] RandomForestRegressor trained")

    # Test predictions
    y_pred = rf.predict(X_test)
    assert y_pred.shape == y_test.shape, "Predictions shape mismatch"
    print("   [OK] Predictions shape correct")

    # Test R² score
    r2 = rf.score(X_test, y_test)
    assert -1 <= r2 <= 1, "Invalid R² score"
    print(f"   [OK] Test R²: {r2:.3f}")

    return True


def test_regression_oob():
    """Test regression with OOB score"""
    print("\nTest 7: Regression OOB Score")
    print("-" * 70)

    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=200, n_features=4, noise=10, random_state=42)

    # Split data
    split = 140
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train with OOB
    rf = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)

    assert rf.oob_score_ is not None, "OOB score not computed"
    print(f"   [OK] OOB R²: {rf.oob_score_:.3f}")

    test_r2 = rf.score(X_test, y_test)
    print(f"   [OK] Test R²: {test_r2:.3f}")

    return True


def test_max_features():
    """Test different max_features settings"""
    print("\nTest 8: Max Features Settings")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)

    # Test different settings
    settings = ['sqrt', 'log2', None, 2]

    for setting in settings:
        rf = RandomForestClassifier(n_estimators=50, max_features=setting, random_state=42)
        rf.fit(X[:100], y[:100])
        acc = rf.score(X[100:], y[100:])
        print(f"   [OK] max_features={setting}: accuracy={acc:.3f}")

    return True


def test_bootstrap():
    """Test bootstrap parameter"""
    print("\nTest 9: Bootstrap Parameter")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)

    # With bootstrap
    rf_with = RandomForestClassifier(n_estimators=50, bootstrap=True, random_state=42)
    rf_with.fit(X[:100], y[:100])
    acc_with = rf_with.score(X[100:], y[100:])
    print(f"   [OK] With bootstrap: accuracy={acc_with:.3f}")

    # Without bootstrap
    rf_without = RandomForestClassifier(n_estimators=50, bootstrap=False, random_state=42)
    rf_without.fit(X[:100], y[:100])
    acc_without = rf_without.score(X[100:], y[100:])
    print(f"   [OK] Without bootstrap: accuracy={acc_without:.3f}")

    return True


def test_n_estimators_effect():
    """Test effect of number of trees"""
    print("\nTest 10: Number of Trees Effect")
    print("-" * 70)

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)

    n_trees = [10, 50, 100, 200]

    for n in n_trees:
        rf = RandomForestClassifier(n_estimators=n, random_state=42)
        rf.fit(X[:100], y[:100])
        acc = rf.score(X[100:], y[100:])
        print(f"   [OK] n_estimators={n:3d}: accuracy={acc:.3f}, trees={len(rf.trees_)}")

    return True


def test_edge_cases():
    """Test edge cases"""
    print("\nTest 11: Edge Cases")
    print("-" * 70)

    # Single feature
    X = np.random.randn(100, 1)
    y = (X[:, 0] > 0).astype(int)

    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X[:70], y[:70])
    acc = rf.score(X[70:], y[70:])
    print(f"   [OK] Single feature: accuracy={acc:.3f}")

    # Binary features
    X = np.random.randint(0, 2, size=(100, 5))
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X[:70], y[:70])
    acc = rf.score(X[70:], y[70:])
    print(f"   [OK] Binary features: accuracy={acc:.3f}")

    return True


def test_input_validation():
    """Test input validation"""
    print("\nTest 12: Input Validation")
    print("-" * 70)

    X = np.random.randn(100, 4)
    y = np.random.randint(0, 2, size=100)

    rf = RandomForestClassifier(n_estimators=10)
    rf.fit(X, y)

    # Wrong number of features
    try:
        rf.predict(np.random.randn(10, 5))
        print("   [FAIL] Should raise error for wrong features")
        return False
    except ValueError:
        print("   [OK] Correct error for wrong features")

    # Predict before fit
    try:
        rf2 = RandomForestClassifier(n_estimators=10)
        rf2.predict(X)
        print("   [FAIL] Should raise error for predict before fit")
        return False
    except ValueError:
        print("   [OK] Correct error for predict before fit")

    return True


def run_all_tests():
    """Run all tests"""
    print("Testing Random Forest Implementation")
    print("=" * 70)

    tests = [
        test_classification_basic,
        test_classification_multiclass,
        test_probability_predictions,
        test_feature_importance,
        test_oob_score,
        test_regression,
        test_regression_oob,
        test_max_features,
        test_bootstrap,
        test_n_estimators_effect,
        test_edge_cases,
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
