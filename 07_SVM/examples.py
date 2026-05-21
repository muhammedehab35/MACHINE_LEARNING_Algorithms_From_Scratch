"""
Support Vector Machine Examples with Visualizations

6 comprehensive examples demonstrating SVM capabilities:
1. Linear vs RBF kernel decision boundaries
2. Effect of regularization parameter C
3. Kernel comparison (Linear, RBF, Polynomial)
4. Support vectors visualization
5. Margin maximization
6. Real-world scenario: Cancer detection

Author: ML Algorithms from Scratch
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from svm import SVMClassifier
from metrics import accuracy_score, classification_report
from utils import StandardScaler, train_test_split


def plot_linear_vs_rbf():
    """
    Example 1: Linear vs RBF Kernel Decision Boundaries

    Demonstrates the difference between linear and RBF kernels on
    linearly separable and non-linear data.
    """
    print("\n1. Generating Linear vs RBF Kernel comparison...")

    np.random.seed(42)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Dataset 1: Linearly separable
    X_linear = np.random.randn(200, 2)
    y_linear = (X_linear[:, 0] + X_linear[:, 1] > 0).astype(int)

    # Dataset 2: Non-linear (circular)
    X_nonlinear = np.random.randn(200, 2) * 1.5
    y_nonlinear = ((X_nonlinear[:, 0]**2 + X_nonlinear[:, 1]**2) < 2).astype(int)

    datasets = [
        (X_linear, y_linear, "Linearly Separable Data"),
        (X_nonlinear, y_nonlinear, "Non-linear Data (Circular)")
    ]
    kernels = ['linear', 'rbf']

    for row, (X, y, title) in enumerate(datasets):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for col, kernel in enumerate(kernels):
            ax = axes[row, col]

            # Train SVM
            model = SVMClassifier(kernel=kernel, C=1.0, gamma='scale', n_iterations=500)
            model.fit(X_scaled, y)

            # Create mesh
            x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
            y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                 np.linspace(y_min, y_max, 100))

            # Get decision function
            Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # Plot decision boundary and margins
            ax.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.3, cmap='RdYlBu')
            ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'],
                      linestyles=['--', '-', '--'], linewidths=[2, 3, 2])

            # Plot data points
            ax.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1],
                      c='blue', label='Class 0', alpha=0.6, edgecolors='k', s=50)
            ax.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1],
                      c='red', label='Class 1', alpha=0.6, edgecolors='k', s=50)

            accuracy = accuracy_score(y, model.predict(X_scaled))
            kernel_name = kernel.upper() if kernel == 'rbf' else kernel.capitalize()
            ax.set_title(f'{title} - {kernel_name} Kernel\nAccuracy: {accuracy:.3f}',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Feature 1 (scaled)')
            ax.set_ylabel('Feature 2 (scaled)')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('images/01_linear_vs_rbf.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("   [OK] Saved: images/01_linear_vs_rbf.png")


def plot_c_effect():
    """
    Example 2: Effect of Regularization Parameter C

    Shows how different C values affect the decision boundary
    and model complexity (soft vs hard margin).
    """
    print("\n2. Generating C parameter effect visualization...")

    np.random.seed(42)

    # Generate data with some noise/overlap
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] + 0.5*np.random.randn(200) > 0).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    c_values = [0.1, 1.0, 10.0, 100.0]

    for idx, c in enumerate(c_values):
        ax = axes[idx // 2, idx % 2]

        # Train SVM
        model = SVMClassifier(kernel='linear', C=c, n_iterations=1000)
        model.fit(X_scaled, y)

        # Create mesh
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'],
                  linestyles=['--', '-', '--'], linewidths=[2, 3, 2])

        ax.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1],
                  c='blue', label='Class 0', alpha=0.6, edgecolors='k', s=50)
        ax.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1],
                  c='red', label='Class 1', alpha=0.6, edgecolors='k', s=50)

        accuracy = accuracy_score(y, model.predict(X_scaled))

        # Describe C effect
        if c < 1:
            desc = "(Large Margin, More Errors)"
        elif c == 1:
            desc = "(Balanced)"
        elif c <= 10:
            desc = "(Smaller Margin, Fewer Errors)"
        else:
            desc = "(Hard Margin, Minimize Errors)"

        ax.set_title(f'C = {c} {desc}\nAccuracy: {accuracy:.3f}',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature 1 (scaled)')
        ax.set_ylabel('Feature 2 (scaled)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Effect of Regularization Parameter C', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/02_c_parameter_effect.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("   [OK] Saved: images/02_c_parameter_effect.png")


def plot_kernel_comparison():
    """
    Example 3: Kernel Comparison

    Compares Linear, RBF, and Polynomial kernels on the same dataset.
    """
    print("\n3. Generating kernel comparison...")

    np.random.seed(42)

    # Complex non-linear data
    X = np.random.randn(300, 2)
    y = ((X[:, 0]**2 + X[:, 1]**2 < 2) & (X[:, 0] + X[:, 1] > -1)).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    kernels = [
        ('linear', {}, 'Linear Kernel'),
        ('rbf', {'gamma': 'scale'}, 'RBF Kernel'),
        ('poly', {'degree': 3}, 'Polynomial Kernel (degree=3)')
    ]

    for idx, (kernel, params, title) in enumerate(kernels):
        ax = axes[idx]

        # Train SVM
        model = SVMClassifier(kernel=kernel, C=1.0, n_iterations=500, **params)
        model.fit(X_scaled, y)

        # Create mesh
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot
        contour = ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu', levels=20)
        ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=3)

        ax.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1],
                  c='blue', label='Class 0', alpha=0.6, edgecolors='k', s=50)
        ax.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1],
                  c='red', label='Class 1', alpha=0.6, edgecolors='k', s=50)

        accuracy = accuracy_score(y, model.predict(X_scaled))
        ax.set_title(f'{title}\nAccuracy: {accuracy:.3f}',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature 1 (scaled)')
        ax.set_ylabel('Feature 2 (scaled)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.colorbar(contour, ax=ax, label='Decision Function')

    plt.tight_layout()
    plt.savefig('images/03_kernel_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("   [OK] Saved: images/03_kernel_comparison.png")


def plot_support_vectors():
    """
    Example 4: Support Vectors Visualization

    Highlights support vectors and shows margin boundaries.
    """
    print("\n4. Generating support vectors visualization...")

    np.random.seed(42)

    # Simple linearly separable data
    n_samples = 100
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + 0.5*X[:, 1] > 0).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    kernels = [('linear', 'Linear SVM'), ('rbf', 'RBF SVM')]

    for idx, (kernel, title) in enumerate(kernels):
        ax = axes[idx]

        # Train SVM
        model = SVMClassifier(kernel=kernel, C=1.0, gamma='scale', n_iterations=1000)
        model.fit(X_scaled, y)

        # Create mesh
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary and margins
        ax.contourf(xx, yy, Z, alpha=0.2, cmap='RdYlBu')
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'],
                  linestyles=['--', '-', '--'], linewidths=[2, 3, 2],
                  labels=['Margin -1', 'Decision Boundary', 'Margin +1'])

        # Plot all points
        ax.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1],
                  c='blue', label='Class 0', alpha=0.4, edgecolors='k', s=50)
        ax.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1],
                  c='red', label='Class 1', alpha=0.4, edgecolors='k', s=50)

        # Highlight support vectors (points close to decision boundary)
        if kernel == 'linear':
            margins = np.abs(model.decision_function(X_scaled))
            support_vectors = margins < 1.1  # Points within margin
        else:
            # For kernel SVM, use alpha values
            if hasattr(model, 'alpha_'):
                support_vectors = model.alpha_ > 1e-5
            else:
                margins = np.abs(model.decision_function(X_scaled))
                support_vectors = margins < 1.1

        ax.scatter(X_scaled[support_vectors, 0], X_scaled[support_vectors, 1],
                  s=200, linewidths=2, facecolors='none', edgecolors='green',
                  label=f'Support Vectors ({np.sum(support_vectors)})')

        accuracy = accuracy_score(y, model.predict(X_scaled))
        ax.set_title(f'{title}\nAccuracy: {accuracy:.3f}',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature 1 (scaled)')
        ax.set_ylabel('Feature 2 (scaled)')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Support Vectors and Decision Boundaries', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/04_support_vectors.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("   [OK] Saved: images/04_support_vectors.png")


def plot_margin_maximization():
    """
    Example 5: Margin Maximization

    Demonstrates how SVM maximizes the margin between classes.
    """
    print("\n5. Generating margin maximization visualization...")

    np.random.seed(42)

    # Create well-separated classes
    class_0 = np.random.randn(50, 2) - [2, 2]
    class_1 = np.random.randn(50, 2) + [2, 2]
    X = np.vstack([class_0, class_1])
    y = np.hstack([np.zeros(50), np.ones(50)]).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    c_values = [0.01, 1.0, 100.0]
    titles = ['Soft Margin (C=0.01)', 'Balanced (C=1.0)', 'Hard Margin (C=100.0)']

    for idx, (c, title) in enumerate(zip(c_values, titles)):
        ax = axes[idx]

        # Train SVM
        model = SVMClassifier(kernel='linear', C=c, n_iterations=1000)
        model.fit(X_scaled, y)

        # Create mesh
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot margins
        ax.contourf(xx, yy, Z, alpha=0.2, cmap='RdYlBu')
        margin_plot = ax.contour(xx, yy, Z, levels=[-1, 0, 1],
                                colors=['red', 'black', 'blue'],
                                linestyles=['--', '-', '--'],
                                linewidths=[2, 3, 2])

        # Calculate margin width
        if model.w_ is not None:
            margin_width = 2.0 / np.linalg.norm(model.w_)
        else:
            margin_width = 0.0

        # Plot data
        ax.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1],
                  c='blue', label='Class 0', alpha=0.6, edgecolors='k', s=60)
        ax.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1],
                  c='red', label='Class 1', alpha=0.6, edgecolors='k', s=60)

        # Highlight support vectors
        margins = np.abs(model.decision_function(X_scaled))
        support_vectors = margins < 1.1
        ax.scatter(X_scaled[support_vectors, 0], X_scaled[support_vectors, 1],
                  s=200, linewidths=2, facecolors='none', edgecolors='green')

        ax.set_title(f'{title}\nMargin Width: {margin_width:.3f}',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature 1 (scaled)')
        ax.set_ylabel('Feature 2 (scaled)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Margin Maximization with Different C Values', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/05_margin_maximization.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("   [OK] Saved: images/05_margin_maximization.png")


def plot_real_world_scenario():
    """
    Example 6: Real-World Scenario - Cancer Detection

    Simulates a cancer detection scenario with feature analysis.
    """
    print("\n6. Generating real-world cancer detection scenario...")

    np.random.seed(42)

    # Simulate medical data (tumor size, cell irregularity, etc.)
    n_samples = 300

    # Benign tumors
    benign = np.random.multivariate_normal([2, 2], [[1, 0.3], [0.3, 1]], n_samples//2)

    # Malignant tumors
    malignant = np.random.multivariate_normal([5, 5], [[1.5, 0.5], [0.5, 1.5]], n_samples//2)

    X = np.vstack([benign, malignant])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)]).astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    model_linear = SVMClassifier(kernel='linear', C=1.0, n_iterations=1000)
    model_linear.fit(X_train_scaled, y_train)

    model_rbf = SVMClassifier(kernel='rbf', C=1.0, gamma='scale', n_iterations=500)
    model_rbf.fit(X_train_scaled, y_train)

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Training data with Linear SVM
    ax1 = fig.add_subplot(gs[0, 0])
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    Z_linear = model_linear.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z_linear = Z_linear.reshape(xx.shape)

    ax1.contourf(xx, yy, Z_linear, alpha=0.3, cmap='RdYlBu')
    ax1.contour(xx, yy, Z_linear, levels=[0], colors='black', linewidths=3)
    ax1.scatter(X_train_scaled[y_train == 0, 0], X_train_scaled[y_train == 0, 1],
               c='blue', label='Benign', alpha=0.6, edgecolors='k', s=50)
    ax1.scatter(X_train_scaled[y_train == 1, 0], X_train_scaled[y_train == 1, 1],
               c='red', label='Malignant', alpha=0.6, edgecolors='k', s=50)
    ax1.set_title('Training Data - Linear SVM', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Tumor Size (scaled)')
    ax1.set_ylabel('Cell Irregularity (scaled)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training data with RBF SVM
    ax2 = fig.add_subplot(gs[0, 1])
    Z_rbf = model_rbf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z_rbf = Z_rbf.reshape(xx.shape)

    ax2.contourf(xx, yy, Z_rbf, alpha=0.3, cmap='RdYlBu')
    ax2.contour(xx, yy, Z_rbf, levels=[0], colors='black', linewidths=3)
    ax2.scatter(X_train_scaled[y_train == 0, 0], X_train_scaled[y_train == 0, 1],
               c='blue', label='Benign', alpha=0.6, edgecolors='k', s=50)
    ax2.scatter(X_train_scaled[y_train == 1, 0], X_train_scaled[y_train == 1, 1],
               c='red', label='Malignant', alpha=0.6, edgecolors='k', s=50)
    ax2.set_title('Training Data - RBF SVM', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Tumor Size (scaled)')
    ax2.set_ylabel('Cell Irregularity (scaled)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Model comparison
    ax3 = fig.add_subplot(gs[0, 2])
    models = [('Linear SVM', model_linear), ('RBF SVM', model_rbf)]
    metrics_data = []

    for name, model in models:
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred)
        metrics_data.append((name, report))

    x_pos = np.arange(len(models))
    accuracies = [metrics['Accuracy'] for _, metrics in metrics_data]
    precisions = [metrics['Precision'] for _, metrics in metrics_data]
    recalls = [metrics['Recall'] for _, metrics in metrics_data]
    f1s = [metrics['F1 Score'] for _, metrics in metrics_data]

    width = 0.2
    ax3.bar(x_pos - 1.5*width, accuracies, width, label='Accuracy', color='skyblue')
    ax3.bar(x_pos - 0.5*width, precisions, width, label='Precision', color='lightgreen')
    ax3.bar(x_pos + 0.5*width, recalls, width, label='Recall', color='lightcoral')
    ax3.bar(x_pos + 1.5*width, f1s, width, label='F1 Score', color='gold')

    ax3.set_ylabel('Score')
    ax3.set_title('Model Comparison on Test Set', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([name for name, _ in models])
    ax3.legend()
    ax3.set_ylim([0, 1.1])
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4-6: Confusion matrices and metrics
    for idx, (name, model) in enumerate(models):
        ax = fig.add_subplot(gs[1, idx])
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred)

        # Display metrics as text
        metrics_text = f"{name} Performance:\n\n"
        for metric, value in report.items():
            metrics_text += f"{metric:15}: {value:.4f}\n"

        ax.text(0.5, 0.5, metrics_text, ha='center', va='center',
               fontsize=11, family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.axis('off')

    # Overall summary
    ax_summary = fig.add_subplot(gs[1, 2])
    summary_text = "Clinical Application:\n\n"
    summary_text += "Task: Cancer Detection\n"
    summary_text += f"Samples: {len(X_test)} patients\n"
    summary_text += f"Features: 2 (tumor size, irregularity)\n\n"
    summary_text += "Best Model:\n"

    best_idx = np.argmax(accuracies)
    best_model_name = models[best_idx][0]
    best_accuracy = accuracies[best_idx]

    summary_text += f"{best_model_name}\n"
    summary_text += f"Accuracy: {best_accuracy:.4f}\n\n"
    summary_text += "Key Insight:\n"
    summary_text += "High recall is critical\n"
    summary_text += "to minimize false negatives\n"
    summary_text += "(missing malignant cases)"

    ax_summary.text(0.5, 0.5, summary_text, ha='center', va='center',
                   fontsize=10, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax_summary.axis('off')

    plt.suptitle('Real-World Scenario: Cancer Detection with SVM',
                fontsize=16, fontweight='bold')
    plt.savefig('images/06_real_world_cancer_detection.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("   [OK] Saved: images/06_real_world_cancer_detection.png")
    print(f"\n   Best Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print(" " * 15 + "SUPPORT VECTOR MACHINE - EXAMPLES")
    print("=" * 70)

    # Create output directory if needed
    Path("images").mkdir(exist_ok=True)

    try:
        plot_linear_vs_rbf()
        plot_c_effect()
        plot_kernel_comparison()
        plot_support_vectors()
        plot_margin_maximization()
        plot_real_world_scenario()

        print("\n" + "=" * 70)
        print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError generating visualizations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
