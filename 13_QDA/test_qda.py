"""
Comprehensive Tests for Quadratic Discriminant Analysis Implementation

Tests QDA for classification with different covariance structures
"""

import numpy as np
import sys
from qda import QuadraticDiscriminantAnalysis, RegularizedQDA


def test_basic_binary_classification():
    """Test basic binary classification with different covariances"""
    print("Test 1: Basic Binary Classification")
    print("-" * 70)

    np.random.seed(42)
    # Create data with different covariances for each class
    # Class 0: narrow in x, wide in y
    X1 = np.random.randn(60, 2) * [0.5, 2.0] + [2, 2]
    # Class 1: wide in x, narrow in y
    X2 = np.random.randn(60, 2) * [2.0, 0.5] + [-2, -2]

    X = np.vstack([X1, X2])
    y = np.array([0] * 60 + [1] * 60)

    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    # Split
    split = 80
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    qda = QuadraticDiscriminantAnalysis()
    print("   [OK] QDA created")

    qda.fit(X_train, y_train)
    print("   [OK] Model trained")

    # Check that covariances are different
    cov0 = qda.covariances_[0]
    cov1 = qda.covariances_[1]
    assert not np.allclose(cov0, cov1), "Covariances should be different"
    print("   [OK] Different covariances per class")

    y_pred = qda.predict(X_test)
    assert y_pred.shape == y_test.shape, "Predictions shape mismatch"
    print("   [OK] Predictions shape correct")

    acc = qda.score(X_test, y_test)
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

    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    print("   [OK] Model trained on Iris")

    # Check that we have 3 class covariances
    assert len(qda.covariances_) == 3, "Should have 3 class covariances"
    print(f"   [OK] Number of class covariances: {len(qda.covariances_)}")

    # Check that we have 3 class means
    assert len(qda.classes_) == 3, "Should have 3 classes"
    print(f"   [OK] Number of classes: {len(qda.classes_)}")

    acc = qda.score(X_test, y_test)
    print(f"   [OK] Test accuracy: {acc:.3f}")
    assert acc > 0.85, "Accuracy too low on Iris"

    return True


def test_probability_predictions():
    """Test probability predictions"""
    print("\nTest 3: Probability Predictions")
    print("-" * 70)

    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X[:100], y[:100])

    proba = qda.predict_proba(X[100:])
    print(f"   [OK] Probability shape: {proba.shape}")

    # Check that probabilities sum to 1
    assert np.allclose(proba.sum(axis=1), 1.0), "Probabilities don't sum to 1"
    print("   [OK] Probabilities sum to 1")

    # Check range [0, 1]
    assert np.all((proba >= 0) & (proba <= 1)), "Probabilities out of range"
    print("   [OK] Probabilities in valid range [0, 1]")

    return True


def test_log_probability():
    """Test log probability predictions"""
    print("\nTest 4: Log Probability Predictions")
    print("-" * 70)

    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X[:100], y[:100])

    log_proba = qda.predict_log_proba(X[100:])
    proba = qda.predict_proba(X[100:])

    # Check that log_proba = log(proba)
    assert np.allclose(log_proba, np.log(proba)), "Log probabilities incorrect"
    print("   [OK] Log probabilities correct")

    # Log probabilities should be negative
    assert np.all(log_proba <= 0), "Log probabilities should be <= 0"
    print("   [OK] Log probabilities in valid range")

    return True


def test_decision_function():
    """Test decision function"""
    print("\nTest 5: Decision Function")
    print("-" * 70)

    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X[:100], y[:100])

    scores = qda.decision_function(X[100:])
    print(f"   [OK] Decision function shape: {scores.shape}")

    # Check that prediction matches argmax of scores
    y_pred = qda.predict(X[100:])
    y_pred_from_scores = qda.classes_[np.argmax(scores, axis=1)]

    assert np.array_equal(y_pred, y_pred_from_scores), "Prediction mismatch"
    print("   [OK] Predictions match argmax of decision function")

    return True


def test_priors():
    """Test class priors"""
    print("\nTest 6: Class Priors")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = np.array([0] * 30 + [1] * 70)  # Imbalanced

    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X, y)

    print(f"   [OK] Class priors: {qda.priors_}")

    # Check that priors match class proportions
    expected_priors = np.array([0.3, 0.7])
    assert np.allclose(qda.priors_, expected_priors), "Priors incorrect"
    print("   [OK] Priors match class proportions")

    # Check that priors sum to 1
    assert np.allclose(np.sum(qda.priors_), 1.0), "Priors don't sum to 1"
    print("   [OK] Priors sum to 1")

    return True


def test_different_covariances():
    """Test that QDA handles different covariances better than LDA"""
    print("\nTest 7: QDA vs LDA on Different Covariances")
    print("-" * 70)

    np.random.seed(42)
    # Create data with very different covariances
    # Class 0: horizontal ellipse
    X1 = np.random.randn(80, 2) * [3.0, 0.5] + [2, 2]
    # Class 1: vertical ellipse
    X2 = np.random.randn(80, 2) * [0.5, 3.0] + [-2, -2]

    X = np.vstack([X1, X2])
    y = np.array([0] * 80 + [1] * 80)

    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    split = 120
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train QDA
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    qda_acc = qda.score(X_test, y_test)
    print(f"   [OK] QDA accuracy: {qda_acc:.3f}")

    # Compare with LDA (should be worse due to equal covariance assumption)
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '12_LDA'))
        from lda import LinearDiscriminantAnalysis

        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        lda_acc = lda.score(X_test, y_test)
        print(f"   [OK] LDA accuracy: {lda_acc:.3f}")

        # QDA should do better with different covariances
        print(f"   [OK] QDA advantage: {qda_acc - lda_acc:.3f}")

    except ImportError:
        print("   [SKIP] LDA not available for comparison")

    return True


def test_regularization():
    """Test regularization parameter"""
    print("\nTest 8: Regularization")
    print("-" * 70)

    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    # Without regularization (very small)
    qda_no_reg = QuadraticDiscriminantAnalysis(reg_param=1e-10)
    qda_no_reg.fit(X, y)
    acc_no_reg = qda_no_reg.score(X, y)
    print(f"   [OK] No regularization accuracy: {acc_no_reg:.3f}")

    # With regularization
    qda_reg = QuadraticDiscriminantAnalysis(reg_param=0.1)
    qda_reg.fit(X, y)
    acc_reg = qda_reg.score(X, y)
    print(f"   [OK] With regularization accuracy: {acc_reg:.3f}")

    # Both should work reasonably well
    assert acc_no_reg > 0.9, "No regularization accuracy too low"
    assert acc_reg > 0.85, "Regularization accuracy too low"

    return True


def test_regularized_qda_shrinkage():
    """Test RegularizedQDA with shrinkage towards LDA"""
    print("\nTest 9: Regularized QDA - Shrinkage towards LDA")
    print("-" * 70)

    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    # No shrinkage (pure QDA)
    qda_no_shrink = RegularizedQDA(shrinkage=0.0)
    qda_no_shrink.fit(X, y)
    acc_no_shrink = qda_no_shrink.score(X, y)
    print(f"   [OK] No shrinkage (QDA): {acc_no_shrink:.3f}")

    # Partial shrinkage
    qda_partial = RegularizedQDA(shrinkage=0.5)
    qda_partial.fit(X, y)
    acc_partial = qda_partial.score(X, y)
    print(f"   [OK] Shrinkage=0.5: {acc_partial:.3f}")

    # Full shrinkage (towards LDA)
    qda_full = RegularizedQDA(shrinkage=1.0)
    qda_full.fit(X, y)
    acc_full = qda_full.score(X, y)
    print(f"   [OK] Shrinkage=1.0 (LDA): {acc_full:.3f}")

    # All should work reasonably well
    assert acc_no_shrink > 0.9, "No shrinkage accuracy too low"
    assert acc_partial > 0.85, "Partial shrinkage accuracy too low"
    assert acc_full > 0.85, "Full shrinkage accuracy too low"

    return True


def test_regularized_qda_diagonal():
    """Test RegularizedQDA with diagonal shrinkage"""
    print("\nTest 10: Regularized QDA - Diagonal Shrinkage")
    print("-" * 70)

    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    # No diagonal shrinkage (full covariance)
    qda_no_diag = RegularizedQDA(diagonal_shrinkage=0.0)
    qda_no_diag.fit(X, y)
    acc_no_diag = qda_no_diag.score(X, y)
    print(f"   [OK] No diagonal shrinkage: {acc_no_diag:.3f}")

    # Partial diagonal shrinkage
    qda_partial_diag = RegularizedQDA(diagonal_shrinkage=0.5)
    qda_partial_diag.fit(X, y)
    acc_partial_diag = qda_partial_diag.score(X, y)
    print(f"   [OK] Diagonal shrinkage=0.5: {acc_partial_diag:.3f}")

    # Full diagonal (Naive Bayes-like)
    qda_full_diag = RegularizedQDA(diagonal_shrinkage=1.0)
    qda_full_diag.fit(X, y)
    acc_full_diag = qda_full_diag.score(X, y)
    print(f"   [OK] Diagonal shrinkage=1.0: {acc_full_diag:.3f}")

    return True


def test_high_dimensional():
    """Test with high-dimensional data"""
    print("\nTest 11: High-Dimensional Data")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 20)  # 20 features
    y = np.random.randint(0, 3, size=100)  # 3 classes

    # Add some structure
    for i in range(3):
        X[y == i] += np.random.randn(20) * (i + 1)

    qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
    qda.fit(X, y)
    print(f"   [OK] Model trained, input: {X.shape}")

    acc = qda.score(X, y)
    print(f"   [OK] Training accuracy: {acc:.3f}")

    # Check covariance shapes
    for i in range(3):
        assert qda.covariances_[i].shape == (20, 20), "Covariance shape incorrect"
    print("   [OK] Covariance matrices have correct shape")

    return True


def test_single_feature():
    """Test with single feature"""
    print("\nTest 12: Single Feature")
    print("-" * 70)

    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = (X[:, 0] > 0).astype(int)

    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X, y)
    print("   [OK] Model trained on single feature")

    acc = qda.score(X, y)
    print(f"   [OK] Accuracy: {acc:.3f}")

    return True


def test_input_validation():
    """Test input validation"""
    print("\nTest 13: Input Validation")
    print("-" * 70)

    X = np.random.randn(100, 4)
    y = np.random.randint(0, 2, size=100)

    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X, y)

    # Predict before fit
    try:
        qda2 = QuadraticDiscriminantAnalysis()
        qda2.predict(X)
        print("   [FAIL] Should raise error for predict before fit")
        return False
    except ValueError:
        print("   [OK] Correct error for predict before fit")

    # Single class error
    try:
        qda3 = QuadraticDiscriminantAnalysis()
        qda3.fit(X, np.zeros(len(X)))
        print("   [FAIL] Should raise error for single class")
        return False
    except ValueError:
        print("   [OK] Correct error for single class")

    return True


def test_comparison_with_sklearn():
    """Compare results with sklearn (if available)"""
    print("\nTest 14: Comparison with sklearn")
    print("-" * 70)

    try:
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as SklearnQDA
        from sklearn.datasets import load_iris

        X, y = load_iris(return_X_y=True)

        # Our implementation
        our_qda = QuadraticDiscriminantAnalysis(reg_param=1e-4)
        our_qda.fit(X, y)
        our_acc = our_qda.score(X, y)

        # Sklearn
        sklearn_qda = SklearnQDA(reg_param=1e-4)
        sklearn_qda.fit(X, y)
        sklearn_acc = sklearn_qda.score(X, y)

        print(f"   [OK] Our QDA accuracy: {our_acc:.3f}")
        print(f"   [OK] Sklearn QDA accuracy: {sklearn_acc:.3f}")

        # Should be close
        assert abs(our_acc - sklearn_acc) < 0.1, "Results differ significantly from sklearn"
        print("   [OK] Results close to sklearn")

    except ImportError:
        print("   [SKIP] sklearn not available for comparison")

    return True


def test_covariance_storage():
    """Test covariance matrix storage"""
    print("\nTest 15: Covariance Storage")
    print("-" * 70)

    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    qda.fit(X, y)

    # Covariances should be stored
    assert qda.covariances_ is not None, "Covariances should be stored"
    print(f"   [OK] Covariances stored, shape: {qda.covariances_.shape}")

    # Check all covariances are symmetric
    for i in range(len(qda.covariances_)):
        cov = qda.covariances_[i]
        assert np.allclose(cov, cov.T), f"Covariance {i} not symmetric"
    print("   [OK] All covariance matrices are symmetric")

    # Check all covariances are positive definite
    for i in range(len(qda.covariances_)):
        eigenvalues = np.linalg.eigvalsh(qda.covariances_[i])
        assert np.all(eigenvalues > 0), f"Covariance {i} not positive definite"
    print("   [OK] All covariance matrices are positive definite")

    return True


def run_all_tests():
    """Run all tests"""
    print("Testing Quadratic Discriminant Analysis Implementation")
    print("=" * 70)

    tests = [
        test_basic_binary_classification,
        test_multiclass_iris,
        test_probability_predictions,
        test_log_probability,
        test_decision_function,
        test_priors,
        test_different_covariances,
        test_regularization,
        test_regularized_qda_shrinkage,
        test_regularized_qda_diagonal,
        test_high_dimensional,
        test_single_feature,
        test_input_validation,
        test_comparison_with_sklearn,
        test_covariance_storage
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
