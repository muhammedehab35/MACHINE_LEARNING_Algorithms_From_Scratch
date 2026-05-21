"""
Examples and Visualizations for Logistic Regression

Demonstrates binary classification capabilities through 6 comprehensive visualizations.

Author: ML Algorithms from Scratch
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from logistic_regression import LogisticRegression
from metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from utils import StandardScaler, train_test_split


def plot_decision_boundary():
    """
    Example 1: Decision Boundary Visualization

    Shows how logistic regression creates linear decision boundaries.
    """
    print("\n[1/6] Decision Boundary Visualization...")

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

    # Train model
    model = LogisticRegression(
        learning_rate=0.1,
        n_iterations=1000,
        early_stopping=False,
        random_state=42
    )
    model.fit(X, y)

    # Create mesh for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Decision boundary
    ax1.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax1.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Class 0', alpha=0.6)
    ax1.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class 1', alpha=0.6)
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_title('Decision Boundary')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Probability contours
    Z_proba = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z_proba = Z_proba.reshape(xx.shape)

    contour = ax2.contourf(xx, yy, Z_proba, levels=20, cmap='RdYlBu', alpha=0.8)
    ax2.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Class 0', alpha=0.6)
    ax2.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class 1', alpha=0.6)
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_title('Probability Contours')
    ax2.legend()
    plt.colorbar(contour, ax=ax2, label='P(Class 1)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('images/01_decision_boundary.png', dpi=300, bbox_inches='tight')
    plt.close()

    accuracy = accuracy_score(y, model.predict(X))
    print(f"   Accuracy: {accuracy:.4f}")
    print("   [OK] Saved: images/01_decision_boundary.png")


def plot_roc_curve():
    """
    Example 2: ROC Curve Analysis

    Demonstrates the relationship between TPR and FPR at different thresholds.
    """
    print("\n[2/6] ROC Curve Analysis...")

    np.random.seed(42)

    # Generate imbalanced dataset
    n_samples = 500
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.3 * np.random.randn(n_samples) > 0).astype(int)

    # Train model
    model = LogisticRegression(
        learning_rate=0.1,
        n_iterations=500,
        random_state=42
    )
    model.fit(X, y)

    # Get probabilities
    y_proba = model.predict_proba(X)[:, 1]

    # Compute ROC curve manually
    thresholds = np.linspace(0, 1, 100)
    tpr_list = []
    fpr_list = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        tp = np.sum((y == 1) & (y_pred == 1))
        fn = np.sum((y == 1) & (y_pred == 0))
        tn = np.sum((y == 0) & (y_pred == 0))
        fp = np.sum((y == 0) & (y_pred == 1))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Calculate AUC
    auc = roc_auc_score(y, y_proba)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curve
    ax1.plot(fpr_list, tpr_list, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Probability distribution
    ax2.hist(y_proba[y == 0], bins=30, alpha=0.6, label='Class 0', color='blue')
    ax2.hist(y_proba[y == 1], bins=30, alpha=0.6, label='Class 1', color='red')
    ax2.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold = 0.5')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Probability Distribution by Class')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('images/02_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ROC AUC: {auc:.4f}")
    print("   [OK] Saved: images/02_roc_curve.png")


def plot_regularization_effect():
    """
    Example 3: Regularization Effect

    Shows how L2 regularization affects decision boundaries and prevents overfitting.
    """
    print("\n[3/6] Regularization Effect...")

    np.random.seed(42)

    # Generate noisy data
    n_samples = 150
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0]**2 + X[:, 1]**2 > 1.5).astype(int)

    # Add polynomial features
    from utils import polynomial_features
    X_poly = polynomial_features(X, degree=2)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    # Train with different regularization strengths
    alphas = [0.0, 0.01, 0.1, 1.0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    for idx, alpha in enumerate(alphas):
        model = LogisticRegression(
            learning_rate=0.01,
            n_iterations=1000,
            regularization='l2',
            alpha=alpha,
            early_stopping=False,
            random_state=42
        )
        model.fit(X_scaled, y)

        # Create mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))

        mesh_poly = polynomial_features(np.c_[xx.ravel(), yy.ravel()], degree=2)
        mesh_scaled = scaler.transform(mesh_poly)
        Z = model.predict(mesh_scaled)
        Z = Z.reshape(xx.shape)

        # Plot
        axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        axes[idx].scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Class 0', alpha=0.6)
        axes[idx].scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class 1', alpha=0.6)

        accuracy = accuracy_score(y, model.predict(X_scaled))
        n_nonzero = np.sum(np.abs(model.weights_) > 1e-4)

        axes[idx].set_title(f'Alpha = {alpha} | Acc = {accuracy:.3f} | Non-zero = {n_nonzero}')
        axes[idx].set_xlabel('Feature 1')
        axes[idx].set_ylabel('Feature 2')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('images/03_regularization_effect.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   [OK] Saved: images/03_regularization_effect.png")


def plot_convergence_comparison():
    """
    Example 4: Convergence Comparison

    Compares convergence behavior of different optimization methods.
    """
    print("\n[4/6] Convergence Comparison...")

    np.random.seed(42)

    # Generate dataset
    n_samples = 500
    X = np.random.randn(n_samples, 5)
    y = (np.sum(X[:, :3], axis=1) > 0).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Different configurations
    configs = [
        ("Batch GD", None),
        ("Mini-Batch (32)", 32),
        ("SGD", 1)
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, batch_size in configs:
        model = LogisticRegression(
            learning_rate=0.01,
            n_iterations=500,
            batch_size=batch_size,
            early_stopping=False,
            random_state=42
        )
        model.fit(X_scaled, y)

        # Plot loss
        ax1.plot(model.loss_history_, label=name, linewidth=2)

        # Final accuracy
        accuracy = model.score(X_scaled, y)
        print(f"   {name:20} - Accuracy: {accuracy:.4f}")

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Learning rate comparison
    learning_rates = [0.001, 0.01, 0.1, 0.5]

    for lr in learning_rates:
        model = LogisticRegression(
            learning_rate=lr,
            n_iterations=200,
            early_stopping=False,
            random_state=42
        )
        model.fit(X_scaled, y)
        ax2.plot(model.loss_history_, label=f'LR = {lr}', linewidth=2)

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Learning Rate Effect')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('images/04_convergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   [OK] Saved: images/04_convergence_comparison.png")


def plot_probability_calibration():
    """
    Example 5: Probability Calibration

    Analyzes how well predicted probabilities match actual outcomes.
    """
    print("\n[5/6] Probability Calibration...")

    np.random.seed(42)

    # Generate dataset
    n_samples = 1000
    X = np.random.randn(n_samples, 3)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(
        learning_rate=0.1,
        n_iterations=500,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Get probabilities
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Calibration curve
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fraction_positives = []
    mean_predicted_probs = []

    for i in range(n_bins):
        mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
        if np.sum(mask) > 0:
            fraction_positives.append(np.mean(y_test[mask]))
            mean_predicted_probs.append(np.mean(y_proba[mask]))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Calibration curve
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    ax1.plot(mean_predicted_probs, fraction_positives, 'o-',
             linewidth=2, markersize=8, label='Logistic Regression')
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title('Calibration Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Reliability diagram
    ax2.bar(bin_centers, fraction_positives, width=0.08, alpha=0.6,
            label='Actual Frequency', color='blue')
    ax2.plot(bin_centers, bin_centers, 'r--', linewidth=2, label='Perfect Calibration')
    ax2.set_xlabel('Predicted Probability Bin')
    ax2.set_ylabel('Actual Positive Fraction')
    ax2.set_title('Reliability Diagram')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('images/05_probability_calibration.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Metrics
    y_pred = model.predict(X_test_scaled)
    metrics = classification_report(y_test, y_pred, y_proba)

    print(f"   Test Accuracy: {metrics['Accuracy']:.4f}")
    print(f"   ROC AUC: {metrics['ROC AUC']:.4f}")
    print("   [OK] Saved: images/05_probability_calibration.png")


def plot_real_world_scenario():
    """
    Example 6: Real-World Scenario - Customer Churn Prediction

    Simulates a customer churn prediction scenario with imbalanced classes.
    """
    print("\n[6/6] Real-World Scenario: Customer Churn...")

    np.random.seed(42)

    # Simulate customer data (imbalanced: 20% churn rate)
    n_samples = 800

    # Features: age, tenure, monthly_charges, total_charges, support_calls
    age = np.random.randint(18, 70, n_samples)
    tenure = np.random.randint(0, 72, n_samples)
    monthly_charges = np.random.uniform(20, 100, n_samples)
    total_charges = tenure * monthly_charges + np.random.randn(n_samples) * 100
    support_calls = np.random.poisson(2, n_samples)

    X = np.column_stack([age, tenure, monthly_charges, total_charges, support_calls])

    # Churn probability (more support calls, less tenure -> higher churn)
    churn_score = -0.02 * tenure + 0.3 * support_calls + 0.01 * monthly_charges + np.random.randn(n_samples)
    y = (churn_score > np.percentile(churn_score, 80)).astype(int)  # 20% churn

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(
        learning_rate=0.01,
        n_iterations=1000,
        regularization='l2',
        alpha=0.1,
        early_stopping=True,
        patience=20,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    metrics = classification_report(y_test, y_pred, y_proba)

    # Plot
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Feature importance
    ax1 = fig.add_subplot(gs[0, :])
    feature_names = ['Age', 'Tenure', 'Monthly Charges', 'Total Charges', 'Support Calls']
    importance = np.abs(model.weights_)
    sorted_idx = np.argsort(importance)[::-1]

    ax1.barh(range(len(importance)), importance[sorted_idx], color='steelblue')
    ax1.set_yticks(range(len(importance)))
    ax1.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax1.set_xlabel('Absolute Coefficient Value')
    ax1.set_title('Feature Importance (Absolute Coefficients)')
    ax1.grid(True, alpha=0.3)

    # 2. Training history
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(model.loss_history_, 'b-', linewidth=2, label='Training Loss')
    if len(model.val_loss_history_) > 0:
        ax2.plot(model.val_loss_history_, 'r-', linewidth=2, label='Validation Loss')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'Training History (Stopped at {model.n_iter_})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Confusion matrix
    ax3 = fig.add_subplot(gs[1, 1])
    from metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    im = ax3.imshow(cm, cmap='Blues', interpolation='nearest')
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['No Churn', 'Churn'])
    ax3.set_yticklabels(['No Churn', 'Churn'])
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    ax3.set_title('Confusion Matrix')

    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=16)

    # 4. ROC Curve
    ax4 = fig.add_subplot(gs[1, 2])

    # Compute ROC manually
    thresholds = np.linspace(0, 1, 100)
    tpr_list = []
    fpr_list = []

    for threshold in thresholds:
        y_pred_t = (y_proba >= threshold).astype(int)
        tp = np.sum((y_test == 1) & (y_pred_t == 1))
        fn = np.sum((y_test == 1) & (y_pred_t == 0))
        tn = np.sum((y_test == 0) & (y_pred_t == 0))
        fp = np.sum((y_test == 0) & (y_pred_t == 1))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    ax4.plot(fpr_list, tpr_list, 'b-', linewidth=2, label=f'AUC = {metrics["ROC AUC"]:.3f}')
    ax4.plot([0, 1], [0, 1], 'r--', linewidth=2)
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC Curve')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Metrics table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    metrics_text = "Model Performance Metrics:\n\n"
    for name, value in metrics.items():
        metrics_text += f"{name:<20}: {value:>8.4f}\n"

    ax5.text(0.5, 0.5, metrics_text, ha='center', va='center',
            fontsize=12, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Customer Churn Prediction - Complete Analysis', fontsize=16, fontweight='bold')
    plt.savefig('images/06_real_world_scenario.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n   Model Performance:")
    print(f"   - Accuracy: {metrics['Accuracy']:.4f}")
    print(f"   - Precision: {metrics['Precision']:.4f}")
    print(f"   - Recall: {metrics['Recall']:.4f}")
    print(f"   - F1 Score: {metrics['F1 Score']:.4f}")
    print(f"   - ROC AUC: {metrics['ROC AUC']:.4f}")
    print("   [OK] Saved: images/06_real_world_scenario.png")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print(" " * 15 + "LOGISTIC REGRESSION - EXAMPLES")
    print("=" * 70)

    # Create output directory if needed
    Path("images").mkdir(exist_ok=True)

    try:
        plot_decision_boundary()
        plot_roc_curve()
        plot_regularization_effect()
        plot_convergence_comparison()
        plot_probability_calibration()
        plot_real_world_scenario()

        print("\n" + "=" * 70)
        print("All visualizations created successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
