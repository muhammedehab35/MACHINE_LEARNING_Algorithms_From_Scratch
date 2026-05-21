"""
Examples and Visualizations for K-Nearest Neighbors

Demonstrates KNN capabilities through 6 comprehensive visualizations.

Author: ML Algorithms from Scratch
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from knn import KNNClassifier, KNNRegressor
from metrics import accuracy_score, classification_report
from utils import StandardScaler, train_test_split


def plot_decision_boundaries():
    """
    Example 1: Decision Boundaries for Different K Values

    Shows how decision boundaries change with different k values.
    """
    print("\n[1/6] Decision Boundaries Visualization...")

    np.random.seed(42)

    # Generate two-class data
    n_samples = 200
    X1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
    X2 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n_samples // 2), np.zeros(n_samples // 2)])

    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    k_values = [1, 5, 15, 50]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    for idx, k in enumerate(k_values):
        model = KNNClassifier(n_neighbors=k)
        model.fit(X, y)

        # Create mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot
        axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        axes[idx].scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Class 0', alpha=0.6, edgecolors='k')
        axes[idx].scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class 1', alpha=0.6, edgecolors='k')

        accuracy = accuracy_score(y, model.predict(X))
        axes[idx].set_title(f'k = {k} | Accuracy = {accuracy:.3f}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Feature 1')
        axes[idx].set_ylabel('Feature 2')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('images/01_decision_boundaries.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("   [OK] Saved: images/01_decision_boundaries.png")


def plot_distance_metrics():
    """
    Example 2: Comparison of Distance Metrics

    Compares Euclidean, Manhattan, Minkowski, and Cosine distances.
    """
    print("\n[2/6] Distance Metrics Comparison...")

    np.random.seed(42)

    # Generate circular data
    n_samples = 200
    theta = np.linspace(0, 2 * np.pi, n_samples)
    r1 = 2 + 0.5 * np.random.randn(n_samples // 2)
    r2 = 4 + 0.5 * np.random.randn(n_samples // 2)

    X1 = np.column_stack([r1 * np.cos(theta[:n_samples // 2]), r1 * np.sin(theta[:n_samples // 2])])
    X2 = np.column_stack([r2 * np.cos(theta[n_samples // 2:]), r2 * np.sin(theta[n_samples // 2:])])

    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    metrics = ['euclidean', 'manhattan', 'minkowski', 'cosine']
    metric_names = ['Euclidean', 'Manhattan', 'Minkowski (p=2)', 'Cosine']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        model = KNNClassifier(n_neighbors=5, metric=metric)
        model.fit(X_scaled, y)

        # Create mesh
        x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
        y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot
        axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        axes[idx].scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1], c='blue', label='Class 0', alpha=0.6, edgecolors='k')
        axes[idx].scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1], c='red', label='Class 1', alpha=0.6, edgecolors='k')

        accuracy = accuracy_score(y, model.predict(X_scaled))
        axes[idx].set_title(f'{name} | Accuracy = {accuracy:.3f}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Feature 1 (scaled)')
        axes[idx].set_ylabel('Feature 2 (scaled)')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('images/02_distance_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("   [OK] Saved: images/02_distance_metrics.png")


def plot_weighted_voting():
    """
    Example 3: Uniform vs Distance-Weighted Voting

    Compares uniform and distance-weighted voting strategies.
    """
    print("\n[3/6] Weighted Voting Comparison...")

    np.random.seed(42)

    # Generate overlapping data
    X = np.random.randn(300, 2)
    y = (X[:, 0]**2 + X[:, 1]**2 > 1.5).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    weights_types = ['uniform', 'distance']
    titles = ['Uniform Weights', 'Distance-Weighted']

    for idx, (weight, title) in enumerate(zip(weights_types, titles)):
        model = KNNClassifier(n_neighbors=10, weights=weight)
        model.fit(X_train, y_train)

        # Create mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot
        axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        axes[idx].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
                         c='blue', label='Train Class 0', alpha=0.4, edgecolors='k')
        axes[idx].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
                         c='red', label='Train Class 1', alpha=0.4, edgecolors='k')
        axes[idx].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
                         c='blue', marker='s', s=100, label='Test Class 0', edgecolors='k', linewidths=2)
        axes[idx].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
                         c='red', marker='s', s=100, label='Test Class 1', edgecolors='k', linewidths=2)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        axes[idx].set_title(f'{title}\nTrain Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}',
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Feature 1')
        axes[idx].set_ylabel('Feature 2')
        axes[idx].legend(loc='best', fontsize=8)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('images/03_weighted_voting.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("   [OK] Saved: images/03_weighted_voting.png")


def plot_k_optimization():
    """
    Example 4: Finding Optimal K Value

    Shows how to select the best k using cross-validation.
    """
    print("\n[4/6] K Value Optimization...")

    np.random.seed(42)

    # Generate data
    X = np.random.randn(300, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    k_range = range(1, 51)
    train_scores = []
    test_scores = []

    for k in k_range:
        model = KNNClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        train_scores.append(model.score(X_train, y_train))
        test_scores.append(model.score(X_test, y_test))

    best_k = k_range[np.argmax(test_scores)]
    best_score = max(test_scores)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Accuracy vs K
    ax1.plot(k_range, train_scores, 'b-', linewidth=2, label='Training Accuracy', marker='o', markersize=4)
    ax1.plot(k_range, test_scores, 'r-', linewidth=2, label='Test Accuracy', marker='s', markersize=4)
    ax1.axvline(best_k, color='green', linestyle='--', linewidth=2, label=f'Best k = {best_k}')
    ax1.set_xlabel('Number of Neighbors (k)', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Performance vs K', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Elbow Method - Error Rate
    train_error = [1 - score for score in train_scores]
    test_error = [1 - score for score in test_scores]

    ax2.plot(k_range, train_error, 'b-', linewidth=2, label='Training Error', marker='o', markersize=4)
    ax2.plot(k_range, test_error, 'r-', linewidth=2, label='Test Error (Elbow Method)', marker='s', markersize=4)
    ax2.axvline(best_k, color='green', linestyle='--', linewidth=2, label=f'Optimal k = {best_k} (Elbow)')
    ax2.axhline(test_error[best_k-1], color='green', linestyle=':', alpha=0.5)
    ax2.scatter([best_k], [test_error[best_k-1]], color='green', s=200, zorder=5, edgecolors='black', linewidth=2)
    ax2.set_xlabel('Number of Neighbors (k)', fontsize=12)
    ax2.set_ylabel('Error Rate', fontsize=12)
    ax2.set_title('Elbow Method for Optimal K Selection', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Add annotation for elbow point
    ax2.annotate('Elbow Point',
                xy=(best_k, test_error[best_k-1]),
                xytext=(best_k+10, test_error[best_k-1]+0.05),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=11, fontweight='bold', color='green')

    plt.tight_layout()
    plt.savefig('images/04_k_optimization.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   Best k: {best_k} with test accuracy: {best_score:.4f}")
    print("   [OK] Saved: images/04_k_optimization.png")


def plot_knn_regression():
    """
    Example 5: KNN for Regression

    Demonstrates KNN regression with different k values.
    """
    print("\n[5/6] KNN Regression...")

    np.random.seed(42)

    # Generate nonlinear data
    X = np.sort(np.random.rand(100, 1) * 10, axis=0)
    y = np.sin(X).ravel() + 0.2 * np.random.randn(100)

    X_test_plot = np.linspace(0, 10, 1000).reshape(-1, 1)

    k_values = [1, 3, 10, 30]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    for idx, k in enumerate(k_values):
        # Uniform weights
        model_uniform = KNNRegressor(n_neighbors=k, weights='uniform')
        model_uniform.fit(X, y)
        y_pred_uniform = model_uniform.predict(X_test_plot)

        # Distance weights
        model_distance = KNNRegressor(n_neighbors=k, weights='distance')
        model_distance.fit(X, y)
        y_pred_distance = model_distance.predict(X_test_plot)

        # Plot
        axes[idx].scatter(X, y, c='blue', s=30, alpha=0.6, edgecolors='k', label='Training Data')
        axes[idx].plot(X_test_plot, y_pred_uniform, 'r-', linewidth=2, label='Uniform Weights')
        axes[idx].plot(X_test_plot, y_pred_distance, 'g--', linewidth=2, label='Distance Weights')
        axes[idx].plot(X_test_plot, np.sin(X_test_plot), 'k:', linewidth=1, label='True Function', alpha=0.5)

        r2_uniform = model_uniform.score(X, y)
        r2_distance = model_distance.score(X, y)

        axes[idx].set_title(f'k = {k} | R²(uniform) = {r2_uniform:.3f}, R²(distance) = {r2_distance:.3f}',
                           fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('X')
        axes[idx].set_ylabel('y')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('images/05_knn_regression.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("   [OK] Saved: images/05_knn_regression.png")


def plot_multiclass_classification():
    """
    Example 6: Multi-class Classification

    Demonstrates KNN on a 3-class problem.
    """
    print("\n[6/6] Multi-class Classification...")

    np.random.seed(42)

    # Generate 3-class data
    n_samples = 300
    X = np.random.randn(n_samples, 2) * 2

    # Create 3 clusters
    y = np.zeros(n_samples, dtype=int)
    y[(X[:, 0] > 2) | (X[:, 1] > 2)] = 1
    y[(X[:, 0] < -2) | (X[:, 1] < -2)] = 2

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = KNNClassifier(n_neighbors=10, weights='distance')
    model.fit(X_train, y_train)

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Decision boundaries
    ax1 = fig.add_subplot(gs[0, :2])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax1.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

    colors = ['blue', 'red', 'green']
    for class_idx in range(3):
        ax1.scatter(X_train[y_train == class_idx, 0], X_train[y_train == class_idx, 1],
                   c=colors[class_idx], label=f'Class {class_idx}', alpha=0.6, edgecolors='k')

    accuracy = model.score(X_test, y_test)
    ax1.set_title(f'Decision Boundaries | Test Accuracy = {accuracy:.3f}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Confusion Matrix
    ax2 = fig.add_subplot(gs[0, 2])

    y_pred = model.predict(X_test)
    from metrics import confusion_matrix

    # Manual confusion matrix for 3 classes
    cm = np.zeros((3, 3), dtype=int)
    for true_label in range(3):
        for pred_label in range(3):
            cm[true_label, pred_label] = np.sum((y_test == true_label) & (y_pred == pred_label))

    im = ax2.imshow(cm, cmap='Blues', interpolation='nearest')
    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([0, 1, 2])
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Confusion Matrix', fontsize=12, fontweight='bold')

    for i in range(3):
        for j in range(3):
            ax2.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)

    # Plot 3: Class probabilities
    ax3 = fig.add_subplot(gs[1, :])

    # Get probabilities for test set
    proba = model.predict_proba(X_test)

    x_axis = np.arange(len(X_test))
    width = 0.25

    ax3.bar(x_axis - width, proba[:, 0], width, label='Class 0', color='blue', alpha=0.7)
    ax3.bar(x_axis, proba[:, 1], width, label='Class 1', color='red', alpha=0.7)
    ax3.bar(x_axis + width, proba[:, 2], width, label='Class 2', color='green', alpha=0.7)

    # Mark true labels
    for i, true_label in enumerate(y_test[:50]):  # Show first 50
        ax3.axvline(i, color=colors[true_label], alpha=0.2, linewidth=1)

    ax3.set_xlabel('Test Sample Index (showing first 50)')
    ax3.set_ylabel('Probability')
    ax3.set_title('Predicted Class Probabilities', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xlim(-1, 50)

    plt.suptitle('Multi-class Classification with KNN', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('images/06_multiclass_classification.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   Test Accuracy: {accuracy:.4f}")
    print("   [OK] Saved: images/06_multiclass_classification.png")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print(" " * 20 + "KNN - EXAMPLES")
    print("=" * 70)

    # Create output directory if needed
    Path("images").mkdir(exist_ok=True)

    try:
        plot_decision_boundaries()
        plot_distance_metrics()
        plot_weighted_voting()
        plot_k_optimization()
        plot_knn_regression()
        plot_multiclass_classification()

        print("\n" + "=" * 70)
        print("All visualizations created successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
