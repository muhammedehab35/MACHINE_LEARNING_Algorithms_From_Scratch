"""Linear Regression Examples with Visualizations."""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import sys
from pathlib import Path

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Create images directory
IMAGES_DIR = Path(__file__).parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

from linear_regression import LinearRegression
from metrics import mean_squared_error, r2_score, mean_absolute_error
from utils import StandardScaler, MinMaxScaler, train_test_split


def example_1_simple_linear_regression():
    """Example 1: Simple Linear Regression with 1 feature."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Linear Regression")
    print("="*70)

    # Generate data: y = 3x + 5 + noise
    np.random.seed(42)
    X = np.random.uniform(-10, 10, (100, 1))
    y = 3 * X.flatten() + 5 + np.random.randn(100) * 2

    # Train model
    model = LinearRegression(learning_rate=0.01, n_iterations=1000, verbose=False)
    model.fit(X, y)

    # Predictions
    y_pred = model.predict(X)

    # Metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"True equation: y = 3x + 5")
    print(f"Learned equation: y = {model.weights_[0]:.2f}x + {model.bias_:.2f}")
    print(f"MSE: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Visualization
    plt.figure(figsize=(12, 5))

    # Plot 1: Data and Regression Line
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.5, label='Data points')
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, 'r-', linewidth=2, label='Regression line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Loss History
    plt.subplot(1, 2, 2)
    plt.plot(model.losses_, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss Over Time')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'example_1_simple_regression.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization saved as 'images/example_1_simple_regression.png'")


def example_2_gradient_descent_comparison():
    """Example 2: Compare different gradient descent methods."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Gradient Descent Methods Comparison")
    print("="*70)

    # Generate data
    np.random.seed(42)
    X = np.random.randn(500, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + 5 + np.random.randn(500) * 0.5

    methods = ['batch', 'mini-batch', 'sgd']
    colors = ['blue', 'green', 'red']
    models = {}

    plt.figure(figsize=(15, 5))

    for idx, (method, color) in enumerate(zip(methods, colors)):
        print(f"\nTraining with {method.upper()} method...")

        model = LinearRegression(
            learning_rate=0.01,
            n_iterations=200,
            method=method,
            batch_size=32,
            random_state=42,
            verbose=False
        )
        model.fit(X, y)

        models[method] = model

        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        print(f"  R² Score: {r2:.6f}")
        print(f"  Final Loss: {model.losses_[-1]:.6f}")

        # Plot loss history
        plt.subplot(1, 3, idx + 1)
        plt.plot(model.losses_, color=color, linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Loss (MSE)')
        plt.title(f'{method.upper()}\nR² = {r2:.4f}')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'example_2_gradient_descent_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Visualization saved as 'images/example_2_gradient_descent_comparison.png'")


def example_3_regularization_effect():
    """Example 3: Effect of L2 Regularization."""
    print("\n" + "="*70)
    print("EXAMPLE 3: L2 Regularization (Ridge Regression)")
    print("="*70)

    # Generate data with many features (some irrelevant)
    np.random.seed(42)
    n_samples = 100
    n_features = 20

    X = np.random.randn(n_samples, n_features)

    # Only first 3 features are relevant
    true_weights = np.zeros(n_features)
    true_weights[:3] = [5, 3, 2]
    y = np.dot(X, true_weights) + np.random.randn(n_samples) * 0.5

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train with different regularization strengths
    lambdas = [0.0, 0.01, 0.1, 1.0, 10.0]
    models = []

    plt.figure(figsize=(15, 10))

    for idx, lam in enumerate(lambdas):
        model = LinearRegression(
            learning_rate=0.01,
            n_iterations=1000,
            regularization=lam,
            verbose=False
        )
        model.fit(X_scaled, y)
        models.append(model)

        y_pred = model.predict(X_scaled)
        r2 = r2_score(y, y_pred)

        print(f"\nλ = {lam:6.2f}:")
        print(f"  R² Score: {r2:.6f}")
        print(f"  Weight Norm: {np.linalg.norm(model.weights_):.4f}")

        # Plot weights
        plt.subplot(2, 3, idx + 1)
        plt.bar(range(n_features), model.weights_, color='steelblue')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        plt.xlabel('Feature Index')
        plt.ylabel('Weight Value')
        plt.title(f'λ = {lam} (R² = {r2:.4f})')
        plt.grid(True, alpha=0.3, axis='y')

    # Plot weight norms vs lambda
    plt.subplot(2, 3, 6)
    weight_norms = [np.linalg.norm(m.weights_) for m in models]
    plt.plot(lambdas, weight_norms, 'o-', color='darkred', linewidth=2, markersize=8)
    plt.xlabel('Regularization λ')
    plt.ylabel('Weight Norm ||w||')
    plt.title('Effect of Regularization on Weights')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'example_3_regularization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Visualization saved as 'images/example_3_regularization.png'")


def example_4_learning_rate_effect():
    """Example 4: Effect of Learning Rate."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Learning Rate Effect")
    print("="*70)

    # Generate data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + 5 + np.random.randn(100) * 0.5

    learning_rates = [0.001, 0.01, 0.1, 0.5]
    colors = ['blue', 'green', 'orange', 'red']

    plt.figure(figsize=(15, 5))

    # Plot 1: Loss histories
    plt.subplot(1, 2, 1)
    for lr, color in zip(learning_rates, colors):
        model = LinearRegression(
            learning_rate=lr,
            n_iterations=200,
            verbose=False
        )
        model.fit(X, y)

        plt.plot(model.losses_, color=color, linewidth=2, label=f'lr = {lr}')

        print(f"\nLearning Rate = {lr}:")
        print(f"  Final Loss: {model.losses_[-1]:.6f}")
        print(f"  R² Score: {model.score(X, y):.6f}")

    plt.xlabel('Iteration')
    plt.ylabel('Loss (MSE)')
    plt.title('Loss History for Different Learning Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Plot 2: Final R² scores
    plt.subplot(1, 2, 2)
    r2_scores = []
    for lr in learning_rates:
        model = LinearRegression(learning_rate=lr, n_iterations=200, verbose=False)
        model.fit(X, y)
        r2_scores.append(model.score(X, y))

    plt.bar([str(lr) for lr in learning_rates], r2_scores, color=colors)
    plt.xlabel('Learning Rate')
    plt.ylabel('R² Score')
    plt.title('Final R² Score vs Learning Rate')
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim([0, 1])

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'example_4_learning_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Visualization saved as 'images/example_4_learning_rate.png'")


def example_5_feature_scaling():
    """Example 5: Importance of Feature Scaling."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Feature Scaling Importance")
    print("="*70)

    # Generate data with different scales
    np.random.seed(42)
    n_samples = 100

    # Feature 1: small scale (0-10)
    X1 = np.random.uniform(0, 10, n_samples)

    # Feature 2: large scale (0-1000)
    X2 = np.random.uniform(0, 1000, n_samples)

    X = np.column_stack([X1, X2])
    y = 2 * X1 + 0.5 * X2 + 10 + np.random.randn(n_samples) * 5

    print(f"\nOriginal feature scales:")
    print(f"  Feature 1: min={X1.min():.2f}, max={X1.max():.2f}")
    print(f"  Feature 2: min={X2.min():.2f}, max={X2.max():.2f}")

    # Train without scaling
    print("\nTraining WITHOUT scaling...")
    model_unscaled = LinearRegression(
        learning_rate=0.0001,  # Very small LR needed!
        n_iterations=2000,
        verbose=False
    )
    model_unscaled.fit(X, y)

    # Train with scaling
    print("Training WITH scaling...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model_scaled = LinearRegression(
        learning_rate=0.1,  # Much larger LR possible
        n_iterations=2000,
        verbose=False
    )
    model_scaled.fit(X_scaled, y)

    # Compare
    r2_unscaled = model_unscaled.score(X, y)
    r2_scaled = model_scaled.score(X_scaled, y)

    print(f"\nResults WITHOUT scaling:")
    print(f"  R² Score: {r2_unscaled:.6f}")
    print(f"  Final Loss: {model_unscaled.losses_[-1]:.6f}")

    print(f"\nResults WITH scaling:")
    print(f"  R² Score: {r2_scaled:.6f}")
    print(f"  Final Loss: {model_scaled.losses_[-1]:.6f}")

    # Visualization
    plt.figure(figsize=(15, 5))

    # Plot 1: Loss history comparison
    plt.subplot(1, 3, 1)
    plt.plot(model_unscaled.losses_, 'r-', linewidth=2, label='Without scaling')
    plt.plot(model_scaled.losses_, 'g-', linewidth=2, label='With scaling')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (MSE)')
    plt.title('Loss History Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Plot 2: Feature distribution before scaling
    plt.subplot(1, 3, 2)
    plt.boxplot([X[:, 0], X[:, 1]], labels=['Feature 1', 'Feature 2'])
    plt.ylabel('Value')
    plt.title('Original Feature Scales')
    plt.grid(True, alpha=0.3, axis='y')

    # Plot 3: Feature distribution after scaling
    plt.subplot(1, 3, 3)
    plt.boxplot([X_scaled[:, 0], X_scaled[:, 1]], labels=['Feature 1', 'Feature 2'])
    plt.ylabel('Value')
    plt.title('Scaled Feature Distributions')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'example_5_feature_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Visualization saved as 'images/example_5_feature_scaling.png'")


def example_6_prediction_intervals():
    """Example 6: Predictions with Residual Analysis."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Predictions and Residual Analysis")
    print("="*70)

    # Generate data
    np.random.seed(42)
    X = np.random.uniform(-5, 5, (200, 1))
    y = 2 * X.flatten() + 3 + np.random.randn(200) * 1.5

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = LinearRegression(learning_rate=0.01, n_iterations=1000, verbose=False)
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Residuals
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred

    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print(f"\nTraining Performance:")
    print(f"  R² Score: {train_r2:.6f}")

    print(f"\nTest Performance:")
    print(f"  R² Score: {test_r2:.6f}")
    print(f"  MSE: {test_mse:.6f}")
    print(f"  MAE: {test_mae:.6f}")

    # Visualization
    plt.figure(figsize=(15, 10))

    # Plot 1: Training data
    plt.subplot(2, 3, 1)
    plt.scatter(X_train, y_train, alpha=0.5, label='Training data')
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, 'r-', linewidth=2, label='Regression line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Training Data\nR² = {train_r2:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Test data
    plt.subplot(2, 3, 2)
    plt.scatter(X_test, y_test, alpha=0.5, color='orange', label='Test data')
    plt.plot(X_line, y_line, 'r-', linewidth=2, label='Regression line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Test Data\nR² = {test_r2:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Predicted vs Actual
    plt.subplot(2, 3, 3)
    plt.scatter(y_test, y_test_pred, alpha=0.5, color='green')
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect predictions')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 4: Training residuals
    plt.subplot(2, 3, 4)
    plt.scatter(y_train_pred, train_residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Training Residuals')
    plt.grid(True, alpha=0.3)

    # Plot 5: Test residuals
    plt.subplot(2, 3, 5)
    plt.scatter(y_test_pred, test_residuals, alpha=0.5, color='orange')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Test Residuals')
    plt.grid(True, alpha=0.3)

    # Plot 6: Residual distribution
    plt.subplot(2, 3, 6)
    plt.hist(test_residuals, bins=20, edgecolor='black', alpha=0.7, color='lightblue')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'example_6_predictions_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Visualization saved as 'images/example_6_predictions_residuals.png'")


def run_all_examples():
    """Run all examples."""
    print("\n" + "="*70)
    print("LINEAR REGRESSION - VISUALIZATION EXAMPLES")
    print("="*70)

    examples = [
        ("Example 1: Simple Linear Regression", example_1_simple_linear_regression),
        ("Example 2: Gradient Descent Comparison", example_2_gradient_descent_comparison),
        ("Example 3: L2 Regularization", example_3_regularization_effect),
        ("Example 4: Learning Rate Effect", example_4_learning_rate_effect),
        ("Example 5: Feature Scaling", example_5_feature_scaling),
        ("Example 6: Predictions & Residuals", example_6_prediction_intervals),
    ]

    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED! ✓")
    print("="*70)
    print("\n📊 All visualizations have been saved in the 'images/' directory:")
    print("   • images/example_1_simple_regression.png")
    print("   • images/example_2_gradient_descent_comparison.png")
    print("   • images/example_3_regularization.png")
    print("   • images/example_4_learning_rate.png")
    print("   • images/example_5_feature_scaling.png")
    print("   • images/example_6_predictions_residuals.png")
    print("\n💡 Open these files with any image viewer to see the results!")


if __name__ == "__main__":
    run_all_examples()
