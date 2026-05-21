"""
Test Suite for Polynomial Regression
====================================

Comprehensive tests for the Polynomial Regression implementation.
"""

import numpy as np
import sys
from pathlib import Path

# Fix encoding for Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from polynomial_regression import PolynomialRegression
from metrics import mean_squared_error, r2_score, mean_absolute_error
from utils import StandardScaler, train_test_split


def test_simple_polynomial():
    """Test on simple polynomial data (degree 2)."""
    print("\n" + "="*70)
    print("TEST 1: Simple Polynomial Regression (y = 2x - 3x^2)")
    print("="*70)

    # Generate data: y = 2x - 3x^2
    np.random.seed(42)
    X = np.linspace(-2, 2, 100).reshape(-1, 1)
    y = 2*X.flatten() - 3*(X.flatten()**2) + np.random.randn(100)*0.5

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = PolynomialRegression(
        degree=2,
        learning_rate=0.01,
        n_iterations=2000,
        verbose=False
    )
    model.fit(X_scaled, y)

    # Predictions
    y_pred = model.predict(X_scaled)

    # Evaluate
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"Degree: {model.degree}")
    print(f"Number of polynomial features: {len(model.weights_)}")
    print(f"MSE: {mse:.6f}")
    print(f"R² Score: {r2:.6f}")
    print(f"Training iterations: {model.n_iterations_used_}")

    assert r2 > 0.95, "R² should be > 0.95 for polynomial data"
    print("✓ Test passed!")
    return model


def test_degree_comparison():
    """Test different polynomial degrees."""
    print("\n" + "="*70)
    print("TEST 2: Polynomial Degree Comparison (1 to 6)")
    print("="*70)

    # Generate data with noise
    np.random.seed(42)
    X = np.linspace(-3, 3, 80).reshape(-1, 1)
    y_true = 1 + 2*X.flatten() - X.flatten()**2 + 0.5*X.flatten()**3
    y = y_true + np.random.randn(80)*2

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    degrees = [1, 2, 3, 4, 5, 6]
    train_scores = []
    test_scores = []

    print("\nTesting different degrees:")
    for d in degrees:
        model = PolynomialRegression(
            degree=d,
            learning_rate=0.01,
            n_iterations=1500,
            verbose=False
        )
        model.fit(X_train_scaled, y_train)

        train_r2 = model.score(X_train_scaled, y_train)
        test_r2 = model.score(X_test_scaled, y_test)

        train_scores.append(train_r2)
        test_scores.append(test_r2)

        print(f"  Degree {d}: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}")

    # Find best degree (highest test score)
    best_idx = np.argmax(test_scores)
    best_degree = degrees[best_idx]

    print(f"\nBest degree: {best_degree} (Test R²={test_scores[best_idx]:.4f})")

    # Test scores should generally increase then decrease (overfitting)
    assert test_scores[0] < test_scores[2], "Degree 3 should be better than linear"
    print("✓ Test passed!")


def test_overfitting_detection():
    """Test overfitting with very high degree."""
    print("\n" + "="*70)
    print("TEST 3: Overfitting Detection (High Degree)")
    print("="*70)

    # Small dataset with noise
    np.random.seed(42)
    X = np.linspace(-2, 2, 30).reshape(-1, 1)
    y = X.flatten()**2 + np.random.randn(30)*1

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train with very high degree (should overfit without regularization)
    model_overfit = PolynomialRegression(
        degree=10,
        learning_rate=0.01,
        n_iterations=2000,
        regularization=0.0,  # No regularization
        verbose=False
    )
    model_overfit.fit(X_train_scaled, y_train)

    train_r2_overfit = model_overfit.score(X_train_scaled, y_train)
    test_r2_overfit = model_overfit.score(X_test_scaled, y_test)

    print(f"Degree 10 without regularization:")
    print(f"  Train R²: {train_r2_overfit:.4f}")
    print(f"  Test R²: {test_r2_overfit:.4f}")
    print(f"  Gap: {train_r2_overfit - test_r2_overfit:.4f}")

    # With regularization
    model_reg = PolynomialRegression(
        degree=10,
        learning_rate=0.01,
        n_iterations=2000,
        regularization=0.5,  # Strong regularization
        verbose=False
    )
    model_reg.fit(X_train_scaled, y_train)

    train_r2_reg = model_reg.score(X_train_scaled, y_train)
    test_r2_reg = model_reg.score(X_test_scaled, y_test)

    print(f"\nDegree 10 WITH regularization (λ=0.5):")
    print(f"  Train R²: {train_r2_reg:.4f}")
    print(f"  Test R²: {test_r2_reg:.4f}")
    print(f"  Gap: {train_r2_reg - test_r2_reg:.4f}")

    # Regularization should reduce gap
    gap_overfit = train_r2_overfit - test_r2_overfit
    gap_reg = train_r2_reg - test_r2_reg

    assert gap_reg < gap_overfit, "Regularization should reduce train-test gap"
    print("\n✓ Regularization reduces overfitting!")


def test_regularization_effect():
    """Test L2 regularization effectiveness."""
    print("\n" + "="*70)
    print("TEST 4: L2 Regularization Effect")
    print("="*70)

    # Generate data
    np.random.seed(42)
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = X.flatten()**3 - 2*X.flatten()**2 + X.flatten() + np.random.randn(100)*2

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Without regularization
    model_no_reg = PolynomialRegression(degree=5, learning_rate=0.01, n_iterations=2000, regularization=0.0)
    model_no_reg.fit(X_scaled, y)

    # With regularization
    model_reg = PolynomialRegression(degree=5, learning_rate=0.01, n_iterations=2000, regularization=0.1)
    model_reg.fit(X_scaled, y)

    weight_norm_no_reg = np.linalg.norm(model_no_reg.weights_)
    weight_norm_reg = np.linalg.norm(model_reg.weights_)

    print(f"Weight norm without regularization: {weight_norm_no_reg:.4f}")
    print(f"Weight norm with regularization: {weight_norm_reg:.4f}")
    print(f"Reduction: {((weight_norm_no_reg - weight_norm_reg) / weight_norm_no_reg * 100):.1f}%")

    assert weight_norm_reg < weight_norm_no_reg, "Regularization should reduce weight norm"
    print("✓ Regularization works correctly!")


def test_multivariate_polynomial():
    """Test polynomial features with multiple input features."""
    print("\n" + "="*70)
    print("TEST 5: Multivariate Polynomial Regression (2 features)")
    print("="*70)

    # Generate data with 2 features
    np.random.seed(42)
    X = np.random.randn(100, 2)
    # y = x1 + 2*x2 + x1^2 - x1*x2 + noise
    y = X[:, 0] + 2*X[:, 1] + X[:, 0]**2 - X[:, 0]*X[:, 1] + np.random.randn(100)*0.5

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    model = PolynomialRegression(degree=2, learning_rate=0.01, n_iterations=2000, verbose=False)
    model.fit(X_train_scaled, y_train)

    train_r2 = model.score(X_train_scaled, y_train)
    test_r2 = model.score(X_test_scaled, y_test)

    print(f"Input features: 2")
    print(f"Polynomial features generated: {len(model.weights_)}")
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")

    assert test_r2 > 0.85, "Should capture multivariate polynomial patterns"
    print("✓ Multivariate polynomial regression works!")


def test_gradient_descent_variants():
    """Test different gradient descent methods."""
    print("\n" + "="*70)
    print("TEST 6: Gradient Descent Variants Comparison")
    print("="*70)

    # Generate data
    np.random.seed(42)
    X = np.linspace(-3, 3, 150).reshape(-1, 1)
    y = X.flatten()**2 + np.random.randn(150)*1

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    methods = ['batch', 'mini-batch', 'sgd']
    results = {}

    for method in methods:
        model = PolynomialRegression(
            degree=2,
            learning_rate=0.01,
            n_iterations=500,
            method=method,
            batch_size=32,
            random_state=42,
            verbose=False
        )
        model.fit(X_scaled, y)

        r2 = model.score(X_scaled, y)
        results[method] = r2

        print(f"\n{method.upper()}:")
        print(f"  R²: {r2:.6f}")
        print(f"  Final loss: {model.losses_[-1]:.6f}")

    # All methods should achieve good performance
    for method, r2 in results.items():
        assert r2 > 0.85, f"{method} should achieve R² > 0.85"

    print("\n✓ All gradient descent variants work correctly!")


def test_early_stopping():
    """Test early stopping functionality."""
    print("\n" + "="*70)
    print("TEST 7: Early Stopping")
    print("="*70)

    # Generate data that converges quickly
    np.random.seed(42)
    X = np.linspace(-2, 2, 100).reshape(-1, 1)
    y = X.flatten()**2 + 0.1

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train with early stopping
    model = PolynomialRegression(
        degree=2,
        learning_rate=0.05,
        n_iterations=5000,
        early_stopping=True,
        patience=50,
        tolerance=1e-6,
        verbose=False
    )
    model.fit(X_scaled, y)

    print(f"Stopped at iteration: {model.n_iterations_used_} (out of {model.n_iterations})")
    print(f"Final loss: {model.losses_[-1]:.8f}")
    print(f"R² Score: {model.score(X_scaled, y):.6f}")

    assert model.n_iterations_used_ < model.n_iterations, "Should stop early"
    print("✓ Early stopping works correctly!")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*70)
    print("TEST 8: Edge Cases and Error Handling")
    print("="*70)

    # Test 1: Prediction without fitting
    print("\n1. Testing prediction without fitting...")
    model = PolynomialRegression(degree=2)
    try:
        model.predict(np.array([[1, 2]]))
        print("   ❌ Should have raised RuntimeError")
        assert False
    except RuntimeError as e:
        print(f"   ✓ Correctly raised RuntimeError: {e}")

    # Test 2: Wrong input dimensions
    print("\n2. Testing invalid input shapes...")
    X = np.array([1, 2, 3])  # 1D instead of 2D
    y = np.array([1, 2, 3])

    model = PolynomialRegression(degree=2)
    # Should auto-reshape 1D to 2D
    model.fit(X, y)
    print("   ✓ Handles 1D input correctly")

    # Test 3: Mismatched X and y sizes
    print("\n3. Testing mismatched X and y sizes...")
    X = np.array([[1], [2]])
    y = np.array([1, 2, 3])

    model = PolynomialRegression(degree=2)
    try:
        model.fit(X, y)
        print("   ❌ Should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"   ✓ Correctly raised ValueError: {e}")

    # Test 4: Wrong number of features in prediction
    print("\n4. Testing wrong number of features in prediction...")
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([1, 2])
    X_test = np.array([[1, 2, 3]])  # 3 features instead of 2

    model = PolynomialRegression(degree=2)
    model.fit(X_train, y_train)
    try:
        model.predict(X_test)
        print("   ❌ Should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"   ✓ Correctly raised ValueError: {e}")

    print("\n✓ All edge cases handled correctly!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("POLYNOMIAL REGRESSION - COMPREHENSIVE TEST SUITE")
    print("="*70)

    tests = [
        ("Test 1: Simple Polynomial", test_simple_polynomial),
        ("Test 2: Degree Comparison", test_degree_comparison),
        ("Test 3: Overfitting Detection", test_overfitting_detection),
        ("Test 4: Regularization Effect", test_regularization_effect),
        ("Test 5: Multivariate Polynomial", test_multivariate_polynomial),
        ("Test 6: Gradient Descent Variants", test_gradient_descent_variants),
        ("Test 7: Early Stopping", test_early_stopping),
        ("Test 8: Edge Cases", test_edge_cases),
    ]

    for name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n" + "="*70)
    print("ALL TESTS PASSED! ✓")
    print("="*70)
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
