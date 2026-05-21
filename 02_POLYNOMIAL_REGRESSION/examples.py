"""Polynomial Regression Examples with Visualizations."""

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

from polynomial_regression import PolynomialRegression
from metrics import mean_squared_error, r2_score, mean_absolute_error
from utils import StandardScaler, train_test_split


def example_1_polynomial_degrees():
    """Example 1: Compare different polynomial degrees."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Polynomial Degrees Comparison (1 to 8)")
    print("="*70)

    # Generate non-linear data
    np.random.seed(42)
    X = np.linspace(-3, 3, 100)
    y_true = 0.5*X**3 - 2*X**2 + X + 5
    y = y_true + np.random.randn(100)*3

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, 1))

    # Test degrees 1-8
    degrees = [1, 2, 3, 4, 5, 6, 7, 8]
    models = []

    for d in degrees:
        model = PolynomialRegression(degree=d, learning_rate=0.01, n_iterations=2000, verbose=False)
        model.fit(X_scaled, y)
        models.append(model)
        print(f"Degree {d}: R² = {model.score(X_scaled, y):.4f}")

    # Visualize
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
    X_plot_scaled = scaler.transform(X_plot)

    for idx, (d, model) in enumerate(zip(degrees, models)):
        ax = axes[idx]

        # Predictions
        y_pred = model.predict(X_plot_scaled)

        # Plot
        ax.scatter(X, y, alpha=0.5, s=20, label='Data')
        ax.plot(X_plot, y_pred, 'r-', linewidth=2, label=f'Degree {d} fit')
        ax.plot(X_plot, 0.5*X_plot**3 - 2*X_plot**2 + X_plot + 5, 'g--', linewidth=1, alpha=0.7, label='True function')

        r2 = model.score(X_scaled, y)
        ax.set_title(f'Degree {d}\nR² = {r2:.4f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'example_1_polynomial_degrees.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Visualization saved as 'images/example_1_polynomial_degrees.png'")


def example_2_overfitting():
    """Example 2: Demonstrate overfitting with high degree."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Overfitting with High-Degree Polynomials")
    print("="*70)

    # Small dataset
    np.random.seed(42)
    X_train = np.linspace(-3, 3, 20)
    y_train = X_train**2 + np.random.randn(20)*2

    X_test = np.linspace(-3, 3, 100)
    y_test = X_test**2 + np.random.randn(100)*2

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1))
    X_test_scaled = scaler.transform(X_test.reshape(-1, 1))

    # Train different degrees
    degrees = [2, 5, 10, 15]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    X_plot = np.linspace(-3.5, 3.5, 300).reshape(-1, 1)
    X_plot_scaled = scaler.transform(X_plot)

    for idx, d in enumerate(degrees):
        model = PolynomialRegression(degree=d, learning_rate=0.01, n_iterations=3000, verbose=False)
        model.fit(X_train_scaled, y_train)

        train_r2 = model.score(X_train_scaled, y_train)
        test_r2 = model.score(X_test_scaled, y_test)

        y_pred = model.predict(X_plot_scaled)

        ax = axes[idx]
        ax.scatter(X_train, y_train, alpha=0.7, s=60, c='blue', label='Train data')
        ax.scatter(X_test, y_test, alpha=0.3, s=20, c='green', label='Test data')
        ax.plot(X_plot, y_pred, 'r-', linewidth=2, label=f'Degree {d} fit')
        ax.plot(X_plot, X_plot**2, 'k--', linewidth=1, alpha=0.5, label='True function')

        ax.set_title(f'Degree {d}\nTrain R²={train_r2:.3f}, Test R²={test_r2:.3f}',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-5, 20])

        if d >= 10:
            ax.text(0.05, 0.95, 'OVERFITTING!', transform=ax.transAxes,
                   fontsize=14, color='red', fontweight='bold',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        print(f"Degree {d}: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}, Gap={train_r2-test_r2:.4f}")

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'example_2_overfitting.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Visualization saved as 'images/example_2_overfitting.png'")


def example_3_regularization():
    """Example 3: Effect of L2 regularization."""
    print("\n" + "="*70)
    print("EXAMPLE 3: L2 Regularization Effect")
    print("="*70)

    # Generate data
    np.random.seed(42)
    X = np.linspace(-3, 3, 50)
    y = X**2 + np.random.randn(50)*2

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1), y, test_size=0.3, random_state=42)

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Test different regularization strengths
    lambdas = [0.0, 0.01, 0.1, 1.0]
    degree = 10

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    X_plot = np.linspace(-3.5, 3.5, 300).reshape(-1, 1)
    X_plot_scaled = scaler.transform(X_plot)

    results = []

    for idx, lam in enumerate(lambdas):
        model = PolynomialRegression(
            degree=degree,
            learning_rate=0.01,
            n_iterations=3000,
            regularization=lam,
            verbose=False
        )
        model.fit(X_train_scaled, y_train)

        train_r2 = model.score(X_train_scaled, y_train)
        test_r2 = model.score(X_test_scaled, y_test)
        weight_norm = np.linalg.norm(model.weights_)

        results.append((lam, train_r2, test_r2, weight_norm))

        y_pred = model.predict(X_plot_scaled)

        ax = axes[idx]
        ax.scatter(X_train, y_train, alpha=0.7, s=60, c='blue', label='Train')
        ax.scatter(X_test, y_test, alpha=0.5, s=40, c='green', label='Test')
        ax.plot(X_plot, y_pred, 'r-', linewidth=2, label=f'λ={lam}')
        ax.plot(X_plot, X_plot**2, 'k--', linewidth=1, alpha=0.5, label='True')

        ax.set_title(f'λ = {lam}\nTrain R²={train_r2:.3f}, Test R²={test_r2:.3f}\n||w||={weight_norm:.2f}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-5, 20])

        print(f"λ={lam:5.2f}: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}, ||w||={weight_norm:.2f}")

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'example_3_regularization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Visualization saved as 'images/example_3_regularization.png'")


def example_4_learning_curves():
    """Example 4: Learning curves (train vs test error)."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Learning Curves by Degree")
    print("="*70)

    # Generate data
    np.random.seed(42)
    X = np.linspace(-3, 3, 100)
    y = 1 + X - 0.5*X**2 + np.random.randn(100)*1.5

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1), y, test_size=0.3, random_state=42)

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Test degrees
    degrees = list(range(1, 13))
    train_scores = []
    test_scores = []
    train_mse = []
    test_mse = []

    for d in degrees:
        model = PolynomialRegression(degree=d, learning_rate=0.01, n_iterations=2000, verbose=False)
        model.fit(X_train_scaled, y_train)

        train_r2 = model.score(X_train_scaled, y_train)
        test_r2 = model.score(X_test_scaled, y_test)

        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        train_scores.append(train_r2)
        test_scores.append(test_r2)
        train_mse.append(mean_squared_error(y_train, y_train_pred))
        test_mse.append(mean_squared_error(y_test, y_test_pred))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # R² scores
    ax1.plot(degrees, train_scores, 'o-', linewidth=2, markersize=8, label='Train R²', color='blue')
    ax1.plot(degrees, test_scores, 's-', linewidth=2, markersize=8, label='Test R²', color='red')
    ax1.axvline(x=degrees[np.argmax(test_scores)], color='green', linestyle='--', alpha=0.7, label='Best degree')
    ax1.set_xlabel('Polynomial Degree', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Learning Curve: R² Score vs Degree', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(degrees)

    # Annotate zones
    ax1.axvspan(1, 3, alpha=0.1, color='yellow', label='Underfitting')
    ax1.axvspan(6, 12, alpha=0.1, color='red', label='Overfitting')

    # MSE
    ax2.plot(degrees, train_mse, 'o-', linewidth=2, markersize=8, label='Train MSE', color='blue')
    ax2.plot(degrees, test_mse, 's-', linewidth=2, markersize=8, label='Test MSE', color='red')
    ax2.axvline(x=degrees[np.argmin(test_mse)], color='green', linestyle='--', alpha=0.7, label='Best degree')
    ax2.set_xlabel('Polynomial Degree', fontsize=12)
    ax2.set_ylabel('Mean Squared Error', fontsize=12)
    ax2.set_title('Learning Curve: MSE vs Degree', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(degrees)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'example_4_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    best_degree = degrees[np.argmax(test_scores)]
    print(f"\nBest polynomial degree: {best_degree} (Test R²={max(test_scores):.4f})")
    print("✓ Visualization saved as 'images/example_4_learning_curves.png'")


def example_5_feature_interactions():
    """Example 5: Polynomial features with 2 input features."""
    print("\n" + "="*70)
    print("EXAMPLE 5: 2D Polynomial Features (Feature Interactions)")
    print("="*70)

    # Generate 2D data
    np.random.seed(42)
    n = 200
    x1 = np.random.uniform(-2, 2, n)
    x2 = np.random.uniform(-2, 2, n)
    X = np.column_stack([x1, x2])

    # True function: y = x1 + 2*x2 + x1*x2 - x1^2
    y_true = x1 + 2*x2 + x1*x2 - x1**2
    y = y_true + np.random.randn(n)*0.5

    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = PolynomialRegression(degree=2, learning_rate=0.01, n_iterations=2000, verbose=False)
    model.fit(X_scaled, y)

    print(f"Number of polynomial features: {len(model.weights_)}")
    print(f"R² Score: {model.score(X_scaled, y):.4f}")

    # Create grid for visualization
    grid_size = 50
    x1_grid = np.linspace(-2, 2, grid_size)
    x2_grid = np.linspace(-2, 2, grid_size)
    X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)

    X_grid = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
    X_grid_scaled = scaler.transform(X_grid)
    y_pred_grid = model.predict(X_grid_scaled).reshape(grid_size, grid_size)
    y_true_grid = (X1_grid + 2*X2_grid + X1_grid*X2_grid - X1_grid**2)

    # Visualize
    fig = plt.figure(figsize=(18, 6))

    # True function
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X1_grid, X2_grid, y_true_grid, cmap='viridis', alpha=0.6)
    ax1.scatter(x1, x2, y, c='red', s=10, alpha=0.5)
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.set_zlabel('y')
    ax1.set_title('True Function\ny = x₁ + 2x₂ + x₁x₂ - x₁²', fontweight='bold')

    # Predicted function
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X1_grid, X2_grid, y_pred_grid, cmap='plasma', alpha=0.6)
    ax2.scatter(x1, x2, y, c='red', s=10, alpha=0.5)
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.set_zlabel('y')
    ax2.set_title(f'Polynomial Model (degree=2)\nR²={model.score(X_scaled, y):.4f}', fontweight='bold')

    # Residuals
    y_pred = model.predict(X_scaled)
    residuals = y - y_pred

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(x1, x2, residuals, c=residuals, cmap='coolwarm', s=20, alpha=0.7)
    ax3.set_xlabel('x₁')
    ax3.set_ylabel('x₂')
    ax3.set_zlabel('Residuals')
    ax3.set_title('Residuals\n(y - ŷ)', fontweight='bold')

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'example_5_feature_interactions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization saved as 'images/example_5_feature_interactions.png'")


def example_6_real_world():
    """Example 6: Real-world scenario - housing prices vs area."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Real-World Example (Simulated)")
    print("="*70)

    # Simulate housing data
    np.random.seed(42)
    area = np.random.uniform(50, 300, 150)  # House area (m²)
    # Price increases non-linearly with area
    price = 50000 + 800*area + 2*area**2 - 0.003*area**3 + np.random.randn(150)*20000

    X = area.reshape(-1, 1)
    y = price / 1000  # Convert to thousands

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Compare linear vs polynomial
    models_data = [
        ("Linear (degree=1)", 1, 'blue'),
        ("Quadratic (degree=2)", 2, 'green'),
        ("Cubic (degree=3)", 3, 'red'),
        ("High-degree (degree=6)", 6, 'purple')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    X_plot = np.linspace(40, 310, 300).reshape(-1, 1)
    X_plot_scaled = scaler.transform(X_plot)

    for idx, (name, degree, color) in enumerate(models_data):
        model = PolynomialRegression(
            degree=degree,
            learning_rate=0.01,
            n_iterations=2000,
            regularization=0.01 if degree > 3 else 0.0,
            verbose=False
        )
        model.fit(X_train_scaled, y_train)

        train_r2 = model.score(X_train_scaled, y_train)
        test_r2 = model.score(X_test_scaled, y_test)
        test_mse = mean_squared_error(y_test, model.predict(X_test_scaled))

        y_pred = model.predict(X_plot_scaled)

        ax = axes[idx]
        ax.scatter(X_train, y_train, alpha=0.6, s=50, c='lightblue', edgecolors='black', label='Train')
        ax.scatter(X_test, y_test, alpha=0.6, s=50, c='lightgreen', edgecolors='black', label='Test')
        ax.plot(X_plot, y_pred, color=color, linewidth=3, label=name)

        ax.set_xlabel('House Area (m²)', fontsize=11)
        ax.set_ylabel('Price (thousands $)', fontsize=11)
        ax.set_title(f'{name}\nTrain R²={train_r2:.3f}, Test R²={test_r2:.3f}, MSE={test_mse:.1f}',
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        print(f"{name}: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}")

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'example_6_real_world.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Visualization saved as 'images/example_6_real_world.png'")


def run_all_examples():
    """Run all examples."""
    print("\n" + "="*70)
    print("POLYNOMIAL REGRESSION - VISUALIZATION EXAMPLES")
    print("="*70)

    examples = [
        ("Example 1: Polynomial Degrees", example_1_polynomial_degrees),
        ("Example 2: Overfitting", example_2_overfitting),
        ("Example 3: Regularization", example_3_regularization),
        ("Example 4: Learning Curves", example_4_learning_curves),
        ("Example 5: Feature Interactions", example_5_feature_interactions),
        ("Example 6: Real-World", example_6_real_world),
    ]

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED! ✓")
    print("="*70)
    print("\n📊 All visualizations have been saved in the 'images/' directory:")
    print("   • images/example_1_polynomial_degrees.png")
    print("   • images/example_2_overfitting.png")
    print("   • images/example_3_regularization.png")
    print("   • images/example_4_learning_curves.png")
    print("   • images/example_5_feature_interactions.png")
    print("   • images/example_6_real_world.png")
    print("\n💡 Open these files with any image viewer to see the results!")


if __name__ == "__main__":
    run_all_examples()
