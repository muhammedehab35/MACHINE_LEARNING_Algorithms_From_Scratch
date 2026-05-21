"""
Gaussian Naive Bayes Example
============================

Demonstrates GaussianNB classifier on the Iris dataset.
This example shows how to use GaussianNB for multi-class classification
with continuous features.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import naive_bayes module
sys.path.append(str(Path(__file__).parent.parent))
from naive_bayes import GaussianNB


def load_iris():
    """
    Load a simple version of the Iris dataset.

    Features: sepal length, sepal width, petal length, petal width
    Classes: 0 (setosa), 1 (versicolor), 2 (virginica)
    """
    # Simplified Iris dataset (150 samples)
    np.random.seed(42)

    # Class 0: Setosa
    setosa = np.column_stack([
        np.random.normal(5.0, 0.35, 50),  # sepal length
        np.random.normal(3.4, 0.38, 50),  # sepal width
        np.random.normal(1.5, 0.17, 50),  # petal length
        np.random.normal(0.2, 0.10, 50),  # petal width
    ])

    # Class 1: Versicolor
    versicolor = np.column_stack([
        np.random.normal(5.9, 0.51, 50),  # sepal length
        np.random.normal(2.8, 0.31, 50),  # sepal width
        np.random.normal(4.3, 0.47, 50),  # petal length
        np.random.normal(1.3, 0.20, 50),  # petal width
    ])

    # Class 2: Virginica
    virginica = np.column_stack([
        np.random.normal(6.5, 0.63, 50),  # sepal length
        np.random.normal(3.0, 0.32, 50),  # sepal width
        np.random.normal(5.6, 0.55, 50),  # petal length
        np.random.normal(2.0, 0.27, 50),  # petal width
    ])

    # Combine all classes
    X = np.vstack([setosa, versicolor, virginica])
    y = np.hstack([np.zeros(50), np.ones(50), np.full(50, 2)])

    # Shuffle the dataset
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    return X, y


def train_test_split(X, y, test_size=0.3, random_state=None):
    """Simple train-test split."""
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    n_test = int(n_samples * test_size)

    indices = np.random.permutation(n_samples)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def confusion_matrix(y_true, y_pred, n_classes):
    """Compute confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[int(true), int(pred)] += 1
    return cm


def classification_report(y_true, y_pred, class_names=None):
    """Generate classification report with precision, recall, and F1-score."""
    classes = np.unique(y_true)
    n_classes = len(classes)

    if class_names is None:
        class_names = [f"Class {int(c)}" for c in classes]

    print("\nClassification Report")
    print("=" * 70)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)

    precisions, recalls, f1_scores = [], [], []

    for i, c in enumerate(classes):
        # True positives, false positives, false negatives
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        support = np.sum(y_true == c)

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        print(f"{class_names[i]:<15} {precision:<12.3f} {recall:<12.3f} {f1:<12.3f} {support:<10}")

    # Macro average
    print("-" * 70)
    print(f"{'Macro Avg':<15} {np.mean(precisions):<12.3f} {np.mean(recalls):<12.3f} {np.mean(f1_scores):<12.3f} {len(y_true):<10}")

    # Weighted average
    weights = [np.sum(y_true == c) / len(y_true) for c in classes]
    weighted_precision = np.average(precisions, weights=weights)
    weighted_recall = np.average(recalls, weights=weights)
    weighted_f1 = np.average(f1_scores, weights=weights)
    print(f"{'Weighted Avg':<15} {weighted_precision:<12.3f} {weighted_recall:<12.3f} {weighted_f1:<12.3f} {len(y_true):<10}")
    print("=" * 70)


def main():
    """Main function demonstrating GaussianNB on Iris dataset."""
    print("Gaussian Naive Bayes - Iris Classification")
    print("=" * 70)

    # Load dataset
    print("\n1. Loading Iris dataset...")
    X, y = load_iris()
    print(f"   Dataset shape: {X.shape}")
    print(f"   Number of classes: {len(np.unique(y))}")
    print(f"   Class distribution: {np.bincount(y.astype(int))}")

    # Split dataset
    print("\n2. Splitting dataset (70% train, 30% test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # Train classifier
    print("\n3. Training Gaussian Naive Bayes classifier...")
    gnb = GaussianNB(var_smoothing=1e-9)
    gnb.fit(X_train, y_train)
    print("   Training completed!")

    # Display learned parameters
    print("\n4. Learned Parameters:")
    print("-" * 70)
    print(f"   Class priors: {gnb.class_prior_}")
    print(f"\n   Feature means per class:")
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    for i, class_label in enumerate(gnb.classes_):
        print(f"   Class {int(class_label)}:")
        for j, fname in enumerate(feature_names):
            print(f"      {fname}: mean={gnb.theta_[i, j]:.3f}, var={gnb.var_[i, j]:.3f}")

    # Make predictions on training set
    print("\n5. Training Set Performance:")
    print("-" * 70)
    y_train_pred = gnb.predict(X_train)
    train_accuracy = gnb.score(X_train, y_train)
    print(f"   Accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")

    # Make predictions on test set
    print("\n6. Test Set Performance:")
    print("-" * 70)
    y_test_pred = gnb.predict(X_test)
    test_accuracy = gnb.score(X_test, y_test)
    print(f"   Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    # Confusion matrix
    print("\n7. Confusion Matrix (Test Set):")
    print("-" * 70)
    cm = confusion_matrix(y_test, y_test_pred, len(gnb.classes_))
    print("   Predicted Class")
    print("              0    1    2")
    print("   Actual  +--------------")
    for i, row in enumerate(cm):
        print(f"      {i}    | {row[0]:2d}  {row[1]:2d}  {row[2]:2d}")

    # Classification report
    classification_report(y_test, y_test_pred, class_names=['Setosa', 'Versicolor', 'Virginica'])

    # Predict probabilities for a few samples
    print("\n8. Probability Predictions (First 5 Test Samples):")
    print("-" * 70)
    proba = gnb.predict_proba(X_test[:5])
    for i in range(5):
        print(f"   Sample {i+1}:")
        print(f"      True label: {int(y_test[i])}")
        print(f"      Predicted: {int(y_test_pred[i])}")
        print(f"      Probabilities: [Setosa: {proba[i, 0]:.4f}, "
              f"Versicolor: {proba[i, 1]:.4f}, Virginica: {proba[i, 2]:.4f}]")

    # Analyze misclassifications
    print("\n9. Misclassification Analysis:")
    print("-" * 70)
    misclassified = y_test != y_test_pred
    n_errors = np.sum(misclassified)
    print(f"   Total misclassifications: {n_errors} out of {len(y_test)}")

    if n_errors > 0:
        print(f"   Error rate: {n_errors / len(y_test) * 100:.2f}%")
        print("\n   Misclassified samples:")
        error_indices = np.where(misclassified)[0]
        for idx in error_indices[:5]:  # Show first 5 errors
            print(f"      Sample {idx}: True={int(y_test[idx])}, "
                  f"Predicted={int(y_test_pred[idx])}")
            print(f"         Features: {X_test[idx]}")
            print(f"         Probabilities: {gnb.predict_proba(X_test[idx:idx+1])[0]}")

    print("\n" + "=" * 70)
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
