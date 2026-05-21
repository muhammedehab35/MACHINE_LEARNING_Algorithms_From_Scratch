"""
Elastic Net Regression - Visual Examples

This script generates 6 comprehensive visualizations:
1. L1 Ratio Comparison (0.0 to 1.0)
2. Sparsity vs L1 Ratio
3. Regularization Strength Effect
4. Feature Selection Demonstration
5. Comparison: Ridge vs Lasso vs Elastic Net
6. Real-World Example with Cross-Validation

Author: ML Algorithms from Scratch
"""

import numpy as np
import matplotlib.pyplot as plt
from elastic_net import ElasticNet
from utils import StandardScaler, train_test_split
from metrics import r2_score, mean_squared_error

# Set style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')


def example_1_l1_ratio_comparison():
    """Example 1: Compare different l1_ratio values."""
    print("\n" + "=" * 70)
    print("Example 1: L1 Ratio Comparison")
    print("=" * 70)

    np.random.seed(42)

    # Generate data
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = 2 * X.flatten()**2 + X.flatten() + np.random.randn(100) * 2

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Test different l1_ratios
    l1_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    labels = ['Ridge (0.0)', 'Mix (0.25)', 'Balanced (0.5)', 'Mix (0.75)', 'Lasso (1.0)']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (l1_ratio, label) in enumerate(zip(l1_ratios, labels)):
        model = ElasticNet(
            learning_rate=0.01,
            n_iterations=1000,
            l1_ratio=l1_ratio,
            alpha=1.0,
            early_stopping=False,
            random_state=42
        )

        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)

        r2 = r2_score(y, y_pred)
        sparsity = model.get_sparsity()

        axes[idx].scatter(X, y, alpha=0.5, label='Data')
        axes[idx].plot(X, y_pred, 'r-', linewidth=2, label='Prediction')
        axes[idx].set_title(f'{label}\nR²={r2:.3f}, Sparsity={sparsity:.1%}')
        axes[idx].set_xlabel('X')
        axes[idx].set_ylabel('y')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    # Hide last subplot
    axes[5].axis('off')

    plt.tight_layout()
    plt.savefig('images/example_1_l1_ratio_comparison.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: images/example_1_l1_ratio_comparison.png")
    plt.close()


def example_2_sparsity_analysis():
    """Example 2: Sparsity vs L1 Ratio."""
    print("\n" + "=" * 70)
    print("Example 2: Sparsity Analysis")
    print("=" * 70)

    np.random.seed(42)

    # Generate data with many features
    X = np.random.randn(200, 20)
    true_weights = np.zeros(20)
    true_weights[[0, 5, 10, 15]] = [3, -2, 1.5, -1]
    y = np.dot(X, true_weights) + np.random.randn(200) * 0.5

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    l1_ratios = np.linspace(0, 1, 21)
    sparsities = []
    r2_scores = []

    for l1_ratio in l1_ratios:
        model = ElasticNet(
            learning_rate=0.01,
            n_iterations=1000,
            l1_ratio=l1_ratio,
            alpha=1.0,
            early_stopping=False,
            random_state=42
        )

        model.fit(X_scaled, y)
        sparsities.append(model.get_sparsity())
        r2_scores.append(model.score(X_scaled, y))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Sparsity plot
    ax1.plot(l1_ratios, sparsities, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('L1 Ratio', fontsize=12)
    ax1.set_ylabel('Sparsity (% of zero coefficients)', fontsize=12)
    ax1.set_title('Sparsity vs L1 Ratio', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Balanced')
    ax1.legend()

    # R² plot
    ax2.plot(l1_ratios, r2_scores, 'g-o', linewidth=2, markersize=6)
    ax2.set_xlabel('L1 Ratio', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('Model Performance vs L1 Ratio', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Target R²=0.9')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('images/example_2_sparsity_analysis.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: images/example_2_sparsity_analysis.png")
    plt.close()


def example_3_regularization_strength():
    """Example 3: Effect of regularization strength (alpha)."""
    print("\n" + "=" * 70)
    print("Example 3: Regularization Strength Effect")
    print("=" * 70)

    np.random.seed(42)

    X = np.random.randn(150, 10)
    true_weights = np.array([3, -2, 1.5, 0, 0, 1, 0, 0, -1, 0])
    y = np.dot(X, true_weights) + np.random.randn(150) * 0.5

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, alpha in enumerate(alphas):
        model = ElasticNet(
            learning_rate=0.01,
            n_iterations=1000,
            l1_ratio=0.5,
            alpha=alpha,
            early_stopping=False,
            random_state=42
        )

        model.fit(X_scaled, y)

        # Plot coefficient values
        axes[idx].bar(range(10), model.weights_, color='steelblue', alpha=0.7)
        axes[idx].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[idx].set_title(f'Alpha = {alpha}\nSparsity: {model.get_sparsity():.1%}')
        axes[idx].set_xlabel('Feature Index')
        axes[idx].set_ylabel('Coefficient Value')
        axes[idx].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('images/example_3_regularization_strength.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: images/example_3_regularization_strength.png")
    plt.close()


def example_4_feature_selection():
    """Example 4: Feature Selection Demonstration."""
    print("\n" + "=" * 70)
    print("Example 4: Feature Selection")
    print("=" * 70)

    np.random.seed(42)

    # 30 features, only 5 are relevant
    n_features = 30
    X = np.random.randn(200, n_features)

    true_weights = np.zeros(n_features)
    relevant_features = [2, 7, 12, 18, 25]
    true_weights[relevant_features] = [5, -3, 2, -4, 3]

    y = np.dot(X, true_weights) + np.random.randn(200) * 0.8

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = ElasticNet(
        learning_rate=0.01,
        n_iterations=2000,
        l1_ratio=0.7,
        alpha=1.0,
        early_stopping=False,
        random_state=42
    )

    model.fit(X_scaled, y)

    selected_features = model.get_nonzero_features()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # All coefficients
    colors = ['green' if i in relevant_features else 'gray' for i in range(n_features)]
    ax1.bar(range(n_features), np.abs(model.weights_), color=colors, alpha=0.7)
    ax1.set_xlabel('Feature Index', fontsize=12)
    ax1.set_ylabel('|Coefficient|', fontsize=12)
    ax1.set_title(f'Feature Selection\n{len(selected_features)}/{n_features} features selected',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Comparison
    ax2.scatter(relevant_features, [1]*len(relevant_features), s=200, marker='o',
                color='green', alpha=0.6, label=f'True Relevant ({len(relevant_features)})')
    ax2.scatter(selected_features, [0.5]*len(selected_features), s=200, marker='s',
                color='blue', alpha=0.6, label=f'Selected ({len(selected_features)})')

    # Show overlap
    overlap = set(relevant_features) & set(selected_features)
    if overlap:
        ax2.scatter(list(overlap), [0.75]*len(overlap), s=300, marker='*',
                    color='red', label=f'Correct ({len(overlap)})')

    ax2.set_ylim(0, 1.5)
    ax2.set_xlabel('Feature Index', fontsize=12)
    ax2.set_title('Feature Selection Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('images/example_4_feature_selection.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: images/example_4_feature_selection.png")
    plt.close()


def example_5_comparison():
    """Example 5: Ridge vs Lasso vs Elastic Net Comparison."""
    print("\n" + "=" * 70)
    print("Example 5: Ridge vs Lasso vs Elastic Net")
    print("=" * 70)

    np.random.seed(42)

    X = np.random.randn(150, 15)
    true_weights = np.array([3, -2, 1.5, 0, 0, 2, 0, 0, -1, 0, 1, 0, 0, 0, -1.5])
    y = np.dot(X, true_weights) + np.random.randn(150) * 0.5

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Ridge (L2)': ElasticNet(l1_ratio=0.0, alpha=1.0, n_iterations=1000,
                                  early_stopping=False, random_state=42),
        'Lasso (L1)': ElasticNet(l1_ratio=1.0, alpha=1.0, n_iterations=1000,
                                  early_stopping=False, random_state=42),
        'Elastic Net': ElasticNet(l1_ratio=0.5, alpha=1.0, n_iterations=1000,
                                   early_stopping=False, random_state=42)
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (name, model) in enumerate(models.items()):
        model.fit(X_train_scaled, y_train)

        train_r2 = model.score(X_train_scaled, y_train)
        test_r2 = model.score(X_test_scaled, y_test)
        sparsity = model.get_sparsity()

        axes[idx].bar(range(15), model.weights_, color='steelblue', alpha=0.7)
        axes[idx].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[idx].set_title(f'{name}\nTrain R²={train_r2:.3f}, Test R²={test_r2:.3f}\nSparsity={sparsity:.1%}',
                             fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Feature Index')
        axes[idx].set_ylabel('Coefficient Value')
        axes[idx].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('images/example_5_comparison.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: images/example_5_comparison.png")
    plt.close()


def example_6_real_world():
    """Example 6: Real-world scenario with cross-validation."""
    print("\n" + "=" * 70)
    print("Example 6: Real-World Scenario")
    print("=" * 70)

    np.random.seed(42)

    # Simulate real-world data with noise and irrelevant features
    n_samples = 300
    n_features = 50
    n_informative = 10

    X = np.random.randn(n_samples, n_features)
    true_weights = np.zeros(n_features)
    informative_idx = np.random.choice(n_features, n_informative, replace=False)
    true_weights[informative_idx] = np.random.randn(n_informative) * 3

    y = np.dot(X, true_weights) + np.random.randn(n_samples) * 2

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Test different alpha values
    alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    train_scores = []
    test_scores = []
    sparsities = []

    for alpha in alphas:
        model = ElasticNet(
            learning_rate=0.01,
            n_iterations=1500,
            l1_ratio=0.5,
            alpha=alpha,
            early_stopping=False,
            random_state=42
        )

        model.fit(X_train_scaled, y_train)

        train_scores.append(model.score(X_train_scaled, y_train))
        test_scores.append(model.score(X_test_scaled, y_test))
        sparsities.append(model.get_sparsity())

    # Find best alpha
    best_idx = np.argmax(test_scores)
    best_alpha = alphas[best_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Learning curves
    ax1.plot(alphas, train_scores, 'b-o', linewidth=2, markersize=8, label='Train R²')
    ax1.plot(alphas, test_scores, 'r-o', linewidth=2, markersize=8, label='Test R²')
    ax1.axvline(x=best_alpha, color='green', linestyle='--', alpha=0.7,
                label=f'Best alpha={best_alpha}')
    ax1.set_xscale('log')
    ax1.set_xlabel('Alpha (log scale)', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Model Selection: Train vs Test Performance', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Sparsity
    ax2.plot(alphas, sparsities, 'g-o', linewidth=2, markersize=8)
    ax2.axvline(x=best_alpha, color='green', linestyle='--', alpha=0.7,
                label=f'Best alpha={best_alpha}')
    ax2.set_xscale('log')
    ax2.set_xlabel('Alpha (log scale)', fontsize=12)
    ax2.set_ylabel('Sparsity (%)', fontsize=12)
    ax2.set_title('Sparsity vs Regularization Strength', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('images/example_6_real_world.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: images/example_6_real_world.png")
    print(f"\nBest Alpha: {best_alpha}")
    print(f"Best Test R2: {test_scores[best_idx]:.4f}")
    print(f"Sparsity at best alpha: {sparsities[best_idx]:.1%}")
    plt.close()


def main():
    """Generate all examples."""
    print("\n" + "=" * 70)
    print(" " * 20 + "ELASTIC NET - EXAMPLES")
    print("=" * 70)

    # Create images directory if it doesn't exist
    import os
    os.makedirs('images', exist_ok=True)

    # Run all examples
    example_1_l1_ratio_comparison()
    example_2_sparsity_analysis()
    example_3_regularization_strength()
    example_4_feature_selection()
    example_5_comparison()
    example_6_real_world()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
