"""
Comprehensive Tests for Linear Discriminant Analysis Implementation

Tests LDA for classification and dimensionality reduction
"""

import numpy as np
import sys
from lda import LinearDiscriminantAnalysis


def test_basic_binary_classification():
    """Test basic binary classification"""
    print("Test 1: Basic Binary Classification")
    print("-" * 70)

    np.random.seed(42)
    # Create linearly separable data
    X1 = np.random.randn(50, 2) + np.array([2, 2])
    X2 = np.random.randn(50, 2) + np.array([-2, -2])
    X = np.vstack([X1, X2])
    y = np.array([0] * 50 + [1] * 50)

    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    # Split
    split = 70
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    lda = LinearDiscriminantAnalysis()
    print("   [OK] LDA created")

    lda.fit(X_train, y_train)
    print("   [OK] Model trained")

    y_pred = lda.predict(X_test)
    assert y_pred.shape == y_test.shape, "Predictions shape mismatch"
    print("   [OK] Predictions shape correct")

    acc = lda.score(X_test, y_test)
    print(f"   [OK] Test accuracy: {acc:.3f}")
    assert acc > 0.7, "Accuracy too low"

    return True


def test_multiclass_iris():
    """Test on Iris dataset (3 classes)"""
    print("\nTest 2: Iris Dataset (Multi-class)")
    print("-" * 70)

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X_train, y_train)
    print("   [OK] Model trained on Iris")

    # Check n_components
    assert lda.n_components == 2, "n_components incorrect"
    print(f"   [OK] n_components = {lda.n_components}")

    # Check that we have 3 class means
    assert len(lda.classes_) == 3, "Should have 3 classes"
    print(f"   [OK] Number of classes: {len(lda.classes_)}")

    acc = lda.score(X_test, y_test)
    print(f"   [OK] Test accuracy: {acc:.3f}")
    assert acc > 0.9, "Accuracy too low on Iris"

    return True


def test_dimensionality_reduction():
    """Test dimensionality reduction (transform)"""
    print("\nTest 3: Dimensionality Reduction")
    print("-" * 70)

    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X, y)

    X_lda = lda.transform(X)
    print(f"   [OK] Original shape: {X.shape}")
    print(f"   [OK] Transformed shape: {X_lda.shape}")

    assert X_lda.shape == (len(X), 2), "Transformed shape incorrect"
    print("   [OK] Dimensionality reduced correctly")

    # Check explained variance ratio
    assert hasattr(lda, 'explained_variance_ratio_'), "Missing explained_variance_ratio_"
    print(f"   [OK] Explained variance ratio: {lda.explained_variance_ratio_}")

    # Check that variance ratios sum to <= 1
    assert np.sum(lda.explained_variance_ratio_) <= 1.0, "Variance ratios sum > 1"
    print("   [OK] Variance ratios valid")

    return True


def test_fit_transform():
    """Test fit_transform method"""
    print("\nTest 4: fit_transform Method")
    print("-" * 70)

    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X, y)

    print(f"   [OK] fit_transform shape: {X_lda.shape}")
    assert X_lda.shape == (len(X), 2), "fit_transform shape incorrect"

    # Compare with separate fit and transform
    lda2 = LinearDiscriminantAnalysis(n_components=2)
    lda2.fit(X, y)
    X_lda2 = lda2.transform(X)

    assert np.allclose(X_lda, X_lda2), "fit_transform != fit + transform"
    print("   [OK] fit_transform equivalent to fit + transform")

    return True


def test_probability_predictions():
    """Test probability predictions"""
    print("\nTest 5: Probability Predictions")
    print("-" * 70)

    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X[:100], y[:100])

    proba = lda.predict_proba(X[100:])
    print(f"   [OK] Probability shape: {proba.shape}")

    # Check that probabilities sum to 1
    assert np.allclose(proba.sum(axis=1), 1.0), "Probabilities don't sum to 1"
    print("   [OK] Probabilities sum to 1")

    # Check range [0, 1]
    assert np.all((proba >= 0) & (proba <= 1)), "Probabilities out of range"
    print("   [OK] Probabilities in valid range [0, 1]")

    return True


def test_decision_function():
    """Test decision function"""
    print("\nTest 6: Decision Function")
    print("-" * 70)

    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X[:100], y[:100])

    scores = lda.decision_function(X[100:])
    print(f"   [OK] Decision function shape: {scores.shape}")

    # Check that prediction matches argmax of scores
    y_pred = lda.predict(X[100:])
    y_pred_from_scores = lda.classes_[np.argmax(scores, axis=1)]

    assert np.array_equal(y_pred, y_pred_from_scores), "Prediction mismatch"
    print("   [OK] Predictions match argmax of decision function")

    return True


def test_priors():
    """Test class priors"""
    print("\nTest 7: Class Priors")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = np.array([0] * 30 + [1] * 70)  # Imbalanced

    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)

    print(f"   [OK] Class priors: {lda.priors_}")

    # Check that priors match class proportions
    expected_priors = np.array([0.3, 0.7])
    assert np.allclose(lda.priors_, expected_priors), "Priors incorrect"
    print("   [OK] Priors match class proportions")

    # Check that priors sum to 1
    assert np.allclose(np.sum(lda.priors_), 1.0), "Priors don't sum to 1"
    print("   [OK] Priors sum to 1")

    return True


def test_n_components_auto():
    """Test automatic n_components selection"""
    print("\nTest 8: Automatic n_components")
    print("-" * 70)

    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    # For 3 classes, max n_components = 2
    lda = LinearDiscriminantAnalysis(n_components=None)
    lda.fit(X, y)

    print(f"   [OK] Auto n_components: {lda.n_components}")
    assert lda.n_components == 2, "Auto n_components should be 2 for 3 classes"

    # Test with binary classification
    y_binary = (y == 0).astype(int)
    lda_binary = LinearDiscriminantAnalysis(n_components=None)
    lda_binary.fit(X, y_binary)

    print(f"   [OK] Binary n_components: {lda_binary.n_components}")
    assert lda_binary.n_components == 1, "Binary should have 1 component"

    return True


def test_solver_comparison():
    """Test different solvers"""
    print("\nTest 9: Solver Comparison")
    print("-" * 70)

    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    # SVD solver
    lda_svd = LinearDiscriminantAnalysis(solver='svd')
    lda_svd.fit(X, y)
    acc_svd = lda_svd.score(X, y)
    print(f"   [OK] SVD solver accuracy: {acc_svd:.3f}")

    # Eigen solver
    lda_eigen = LinearDiscriminantAnalysis(solver='eigen')
    lda_eigen.fit(X, y)
    acc_eigen = lda_eigen.score(X, y)
    print(f"   [OK] Eigen solver accuracy: {acc_eigen:.3f}")

    # Both should give similar results
    assert abs(acc_svd - acc_eigen) < 0.05, "Solvers give different results"
    print("   [OK] Solvers give similar results")

    return True


def test_shrinkage():
    """Test shrinkage regularization"""
    print("\nTest 10: Shrinkage Regularization")
    print("-" * 70)

    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    # Without shrinkage
    lda_no_shrink = LinearDiscriminantAnalysis(shrinkage=None)
    lda_no_shrink.fit(X, y)
    acc_no_shrink = lda_no_shrink.score(X, y)
    print(f"   [OK] No shrinkage accuracy: {acc_no_shrink:.3f}")

    # With shrinkage
    lda_shrink = LinearDiscriminantAnalysis(shrinkage=0.5)
    lda_shrink.fit(X, y)
    acc_shrink = lda_shrink.score(X, y)
    print(f"   [OK] Shrinkage=0.5 accuracy: {acc_shrink:.3f}")

    # Both should work reasonably well
    assert acc_no_shrink > 0.9, "No shrinkage accuracy too low"
    assert acc_shrink > 0.85, "Shrinkage accuracy too low"

    return True


def test_covariance_storage():
    """Test covariance matrix storage"""
    print("\nTest 11: Covariance Storage")
    print("-" * 70)

    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    # Without storing covariance
    lda_no_cov = LinearDiscriminantAnalysis(store_covariance=False)
    lda_no_cov.fit(X, y)
    assert lda_no_cov.covariance_ is None, "Covariance should be None"
    print("   [OK] Covariance not stored when store_covariance=False")

    # With storing covariance
    lda_cov = LinearDiscriminantAnalysis(store_covariance=True)
    lda_cov.fit(X, y)
    assert lda_cov.covariance_ is not None, "Covariance should be stored"
    print(f"   [OK] Covariance stored, shape: {lda_cov.covariance_.shape}")

    # Check covariance is symmetric
    assert np.allclose(lda_cov.covariance_, lda_cov.covariance_.T), "Covariance not symmetric"
    print("   [OK] Covariance matrix is symmetric")

    return True


def test_single_feature():
    """Test with single feature"""
    print("\nTest 12: Single Feature")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = (X[:, 0] > 0).astype(int)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    print("   [OK] Model trained on single feature")

    acc = lda.score(X, y)
    print(f"   [OK] Accuracy: {acc:.3f}")

    # n_components should be 1
    assert lda.n_components == 1, "n_components should be 1"
    print("   [OK] n_components = 1 for single feature")

    return True


def test_high_dimensional():
    """Test with high-dimensional data"""
    print("\nTest 13: High-Dimensional Data")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 20)  # 20 features
    y = np.random.randint(0, 3, size=100)  # 3 classes

    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X, y)
    print(f"   [OK] Model trained, input: {X.shape}")

    X_lda = lda.transform(X)
    print(f"   [OK] Transformed to: {X_lda.shape}")

    assert X_lda.shape == (100, 2), "Transform shape incorrect"
    print("   [OK] Dimensionality reduction successful")

    return True


def test_input_validation():
    """Test input validation"""
    print("\nTest 14: Input Validation")
    print("-" * 70)

    X = np.random.randn(100, 4)
    y = np.random.randint(0, 2, size=100)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)

    # Predict before fit
    try:
        lda2 = LinearDiscriminantAnalysis()
        lda2.predict(X)
        print("   [FAIL] Should raise error for predict before fit")
        return False
    except ValueError:
        print("   [OK] Correct error for predict before fit")

    # Single class error
    try:
        lda3 = LinearDiscriminantAnalysis()
        lda3.fit(X, np.zeros(len(X)))
        print("   [FAIL] Should raise error for single class")
        return False
    except ValueError:
        print("   [OK] Correct error for single class")

    return True


def test_comparison_with_sklearn():
    """Compare results with sklearn (if available)"""
    print("\nTest 15: Comparison with sklearn")
    print("-" * 70)

    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA
        from sklearn.datasets import load_iris

        X, y = load_iris(return_X_y=True)

        # Our implementation
        our_lda = LinearDiscriminantAnalysis(n_components=2)
        our_lda.fit(X, y)
        our_acc = our_lda.score(X, y)

        # Sklearn
        sklearn_lda = SklearnLDA(n_components=2)
        sklearn_lda.fit(X, y)
        sklearn_acc = sklearn_lda.score(X, y)

        print(f"   [OK] Our LDA accuracy: {our_acc:.3f}")
        print(f"   [OK] Sklearn LDA accuracy: {sklearn_acc:.3f}")

        # Should be close
        assert abs(our_acc - sklearn_acc) < 0.1, "Results differ significantly from sklearn"
        print("   [OK] Results close to sklearn")

    except ImportError:
        print("   [SKIP] sklearn not available for comparison")

    return True


def run_all_tests():
    """Run all tests"""
    print("Testing Linear Discriminant Analysis Implementation")
    print("=" * 70)

    tests = [
        test_basic_binary_classification,
        test_multiclass_iris,
        test_dimensionality_reduction,
        test_fit_transform,
        test_probability_predictions,
        test_decision_function,
        test_priors,
        test_n_components_auto,
        test_solver_comparison,
        test_shrinkage,
        test_covariance_storage,
        test_single_feature,
        test_high_dimensional,
        test_input_validation,
        test_comparison_with_sklearn
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
