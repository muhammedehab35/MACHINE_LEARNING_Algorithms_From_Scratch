"""
Test Suite for Linear Regression
=================================

Comprehensive tests for the Linear Regression implementation.
"""

import numpy as np
import sys
from pathlib import Path

# Fix encoding for Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from linear_regression import LinearRegression
from metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    adjusted_r2_score
)
from utils import StandardScaler, MinMaxScaler, train_test_split


def test_simple_regression():
    """Test on a simple linear dataset."""
    print("\n" + "="*70)
    print("TEST 1: Simple Linear Regression (y = 2x + 3)")
    print("="*70)

    # Generate perfect linear data: y = 2x + 3
    np.random.seed(42)
    X = np.random.randn(100, 1) * 5
    y = 2 * X.flatten() + 3

    # Train model
    model = LinearRegression(
        learning_rate=0.01,
        n_iterations=1000,
        method='batch',
        verbose=False
    )
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)

    # Evaluate
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"Expected weights: [2.0], bias: 3.0")
    print(f"Learned weights: {model.weights_}, bias: {model.bias_:.4f}")
    print(f"MSE: {mse:.6f}")
    print(f"R² Score: {r2:.6f}")
    print(f"Training iterations: {model.n_iterations_used_}")

    # Check if model learned correctly (with tolerance)
    assert np.abs(model.weights_[0] - 2.0) < 0.01, "Weight should be close to 2.0"
    assert np.abs(model.bias_ - 3.0) < 0.01, "Bias should be close to 3.0"
    assert r2 > 0.99, "R² should be very close to 1.0 for perfect linear data"

    print("✓ Test passed!")
    return model


def test_multivariate_regression():
    """Test on multiple features."""
    print("\n" + "="*70)
    print("TEST 2: Multivariate Regression (3 features)")
    print("="*70)

    # Generate data: y = 1*x1 + 2*x2 + 3*x3 + 4
    np.random.seed(42)
    X = np.random.randn(200, 3)
    true_weights = np.array([1.0, 2.0, 3.0])
    true_bias = 4.0
    y = np.dot(X, true_weights) + true_bias + np.random.randn(200) * 0.1  # Add small noise

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LinearRegression(
        learning_rate=0.1,
        n_iterations=2000,
        method='batch',
        verbose=False
    )
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Evaluate
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_rmse = root_mean_squared_error(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    print(f"Expected weights: {true_weights}, bias: {true_bias}")
    print(f"Learned weights: {model.weights_} (on scaled data)")
    print(f"Learned bias: {model.bias_:.4f}")
    print(f"\nTraining R²: {train_r2:.6f}")
    print(f"Test R²: {test_r2:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")

    assert train_r2 > 0.95, "Training R² should be high"
    assert test_r2 > 0.95, "Test R² should be high"

    print("✓ Test passed!")
    return model


def test_gradient_descent_variants():
    """Test different gradient descent methods."""
    print("\n" + "="*70)
    print("TEST 3: Gradient Descent Variants Comparison")
    print("="*70)

    # Generate noisy data
    np.random.seed(42)
    X = np.random.randn(300, 2)
    y = 3 * X[:, 0] + 5 * X[:, 1] + 7 + np.random.randn(300) * 0.5

    methods = ['batch', 'mini-batch', 'sgd']
    results = {}

    for method in methods:
        print(f"\nTesting {method.upper()} gradient descent...")

        model = LinearRegression(
            learning_rate=0.01,
            n_iterations=500,
            method=method,
            batch_size=32,
            random_state=42,
            verbose=False
        )
        model.fit(X, y)

        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)

        results[method] = {
            'r2': r2,
            'mse': mse,
            'iterations': model.n_iterations_used_
        }

        print(f"  R²: {r2:.6f}")
        print(f"  MSE: {mse:.6f}")
        print(f"  Iterations: {model.n_iterations_used_}")

        assert r2 > 0.90, f"{method} should achieve good R²"

    print("\n✓ All gradient descent variants work correctly!")
    return results


def test_regularization():
    """Test L2 regularization (Ridge regression)."""
    print("\n" + "="*70)
    print("TEST 4: L2 Regularization (Ridge Regression)")
    print("="*70)

    # Generate data with many features (some irrelevant)
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    X = np.random.randn(n_samples, n_features)

    # Only first 5 features are relevant
    true_weights = np.zeros(n_features)
    true_weights[:5] = [1, 2, 3, 4, 5]
    y = np.dot(X, true_weights) + np.random.randn(n_samples) * 0.5

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train without regularization
    model_no_reg = LinearRegression(
        learning_rate=0.01,
        n_iterations=1000,
        regularization=0.0,
        verbose=False
    )
    model_no_reg.fit(X_scaled, y)

    # Train with regularization
    model_with_reg = LinearRegression(
        learning_rate=0.01,
        n_iterations=1000,
        regularization=0.1,
        verbose=False
    )
    model_with_reg.fit(X_scaled, y)

    # Compare weight magnitudes
    weights_no_reg = np.linalg.norm(model_no_reg.weights_)
    weights_with_reg = np.linalg.norm(model_with_reg.weights_)

    print(f"Weight norm without regularization: {weights_no_reg:.4f}")
    print(f"Weight norm with regularization: {weights_with_reg:.4f}")

    # Regularization should reduce weight magnitudes
    assert weights_with_reg < weights_no_reg, "Regularization should reduce weight magnitudes"

    # Both should still fit reasonably well
    r2_no_reg = model_no_reg.score(X_scaled, y)
    r2_with_reg = model_with_reg.score(X_scaled, y)

    print(f"R² without regularization: {r2_no_reg:.6f}")
    print(f"R² with regularization: {r2_with_reg:.6f}")

    print("✓ Regularization works correctly!")
    return model_no_reg, model_with_reg


def test_early_stopping():
    """Test early stopping functionality."""
    print("\n" + "="*70)
    print("TEST 5: Early Stopping")
    print("="*70)

    # Generate data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + 1

    # Train with early stopping
    model = LinearRegression(
        learning_rate=0.1,
        n_iterations=5000,
        early_stopping=True,
        patience=50,
        tolerance=1e-6,
        verbose=False
    )
    model.fit(X, y)

    print(f"Stopped at iteration: {model.n_iterations_used_} (out of {model.n_iterations})")
    print(f"Final loss: {model.losses_[-1]:.8f}")

    # Early stopping should stop before max iterations
    assert model.n_iterations_used_ < model.n_iterations, "Should stop early"

    # Should still achieve good fit
    r2 = model.score(X, y)
    print(f"R² Score: {r2:.6f}")
    assert r2 > 0.99, "Should achieve excellent fit"

    print("✓ Early stopping works correctly!")
    return model


def test_learning_rate_decay():
    """Test learning rate decay."""
    print("\n" + "="*70)
    print("TEST 6: Learning Rate Decay")
    print("="*70)

    # Generate data
    np.random.seed(42)
    X = np.random.randn(200, 3)
    y = np.dot(X, [1, 2, 3]) + 5 + np.random.randn(200) * 0.1

    # Train with learning rate decay
    model = LinearRegression(
        learning_rate=0.5,  # Start with high learning rate
        n_iterations=1000,
        learning_rate_decay=0.001,
        verbose=False
    )
    model.fit(X, y)

    print(f"Initial learning rate: 0.5")
    print(f"Final learning rate: {model.learning_rate:.6f}")
    print(f"Final R²: {model.score(X, y):.6f}")

    # Learning rate should have decreased
    assert model.learning_rate < 0.5, "Learning rate should decay"

    # Should still converge well
    assert model.score(X, y) > 0.95, "Should achieve good fit with decay"

    print("✓ Learning rate decay works correctly!")
    return model


def test_scalers():
    """Test StandardScaler and MinMaxScaler."""
    print("\n" + "="*70)
    print("TEST 7: Feature Scaling (StandardScaler and MinMaxScaler)")
    print("="*70)

    # Generate data
    np.random.seed(42)
    X = np.random.randn(100, 3) * 10 + 50  # Mean around 50, std around 10

    print(f"Original data - Mean: {np.mean(X, axis=0)}")
    print(f"Original data - Std: {np.std(X, axis=0)}")
    print(f"Original data - Min: {np.min(X, axis=0)}")
    print(f"Original data - Max: {np.max(X, axis=0)}")

    # Test StandardScaler
    print("\nStandardScaler:")
    std_scaler = StandardScaler()
    X_std_scaled = std_scaler.fit_transform(X)

    print(f"  Scaled data - Mean: {np.mean(X_std_scaled, axis=0)}")
    print(f"  Scaled data - Std: {np.std(X_std_scaled, axis=0)}")

    # Check standardization
    assert np.allclose(np.mean(X_std_scaled, axis=0), 0, atol=1e-10), "Mean should be ~0"
    assert np.allclose(np.std(X_std_scaled, axis=0), 1, atol=1e-10), "Std should be ~1"

    # Test inverse transform
    X_reversed = std_scaler.inverse_transform(X_std_scaled)
    assert np.allclose(X, X_reversed), "Inverse transform should recover original data"

    # Test MinMaxScaler
    print("\nMinMaxScaler:")
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    X_minmax_scaled = minmax_scaler.fit_transform(X)

    print(f"  Scaled data - Min: {np.min(X_minmax_scaled, axis=0)}")
    print(f"  Scaled data - Max: {np.max(X_minmax_scaled, axis=0)}")

    # Check min-max scaling
    assert np.allclose(np.min(X_minmax_scaled, axis=0), 0, atol=1e-10), "Min should be 0"
    assert np.allclose(np.max(X_minmax_scaled, axis=0), 1, atol=1e-10), "Max should be 1"

    # Test inverse transform
    X_reversed = minmax_scaler.inverse_transform(X_minmax_scaled)
    assert np.allclose(X, X_reversed), "Inverse transform should recover original data"

    print("\n✓ Both scalers work correctly!")
    return std_scaler, minmax_scaler


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*70)
    print("TEST 8: Edge Cases and Error Handling")
    print("="*70)

    # Test 1: Prediction without fitting
    print("\n1. Testing prediction without fitting...")
    model = LinearRegression()
    try:
        model.predict(np.array([[1, 2]]))
        assert False, "Should raise error"
    except RuntimeError as e:
        print(f"   ✓ Correctly raised RuntimeError: {e}")

    # Test 2: Invalid input shapes
    print("\n2. Testing invalid input shapes...")
    try:
        X = np.array([1, 2, 3])  # 1D instead of 2D
        y = np.array([1, 2, 3])
        model.fit(X, y)
        assert False, "Should raise error"
    except ValueError as e:
        print(f"   ✓ Correctly raised ValueError: {e}")

    # Test 3: Mismatched X and y sizes
    print("\n3. Testing mismatched X and y sizes...")
    try:
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2, 3])  # Wrong size
        model.fit(X, y)
        assert False, "Should raise error"
    except ValueError as e:
        print(f"   ✓ Correctly raised ValueError: {e}")

    # Test 4: Wrong number of features in prediction
    print("\n4. Testing wrong number of features in prediction...")
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])
    model.fit(X, y)

    try:
        model.predict(np.array([[1, 2, 3]]))  # 3 features instead of 2
        assert False, "Should raise error"
    except ValueError as e:
        print(f"   ✓ Correctly raised ValueError: {e}")

    print("\n✓ All edge cases handled correctly!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("LINEAR REGRESSION - COMPREHENSIVE TEST SUITE")
    print("="*70)

    try:
        test_simple_regression()
        test_multivariate_regression()
        test_gradient_descent_variants()
        test_regularization()
        test_early_stopping()
        test_learning_rate_decay()
        test_scalers()
        test_edge_cases()

        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
