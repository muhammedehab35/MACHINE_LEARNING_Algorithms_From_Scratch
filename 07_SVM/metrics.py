"""
Classification Metrics for SVM

Comprehensive metrics for evaluating SVM classification performance.

Author: ML Algorithms from Scratch
"""

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    cm : np.ndarray, shape (n_classes, n_classes)
        Confusion matrix.
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)

    cm = np.zeros((n_classes, n_classes), dtype=int)

    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))

    return cm


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy score."""
    return np.mean(y_true == y_pred)


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """
    Compute precision score.

    Precision = TP / (TP + FP)
    """
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """
    Compute recall score (sensitivity, true positive rate).

    Recall = TP / (TP + FN)
    """
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """
    Compute F1 score (harmonic mean of precision and recall).

    F1 = 2 * (precision * recall) / (precision + recall)
    """
    precision = precision_score(y_true, y_pred, pos_label)
    recall = recall_score(y_true, y_pred, pos_label)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """
    Compute specificity score (true negative rate).

    Specificity = TN / (TN + FP)
    """
    tn = np.sum((y_true != pos_label) & (y_pred != pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))

    if tn + fp == 0:
        return 0.0

    return tn / (tn + fp)


def matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Matthews Correlation Coefficient (MCC).

    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

    Returns value between -1 and +1:
    - +1: perfect prediction
    -  0: random prediction
    - -1: complete disagreement
    """
    # Assuming binary classification with labels 0 and 1
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    if denominator == 0:
        return 0.0

    return numerator / denominator


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Generate a classification report with all metrics.

    Returns
    -------
    report : dict
        Dictionary containing all classification metrics.
    """
    report = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'Specificity': specificity_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
    }

    return report


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray):
    """Print a formatted classification report."""
    report = classification_report(y_true, y_pred)

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)

    for metric, value in report.items():
        print(f"{metric:20}: {value:.6f}")

    print("=" * 50)


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    """Print a formatted confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 50)
    print("CONFUSION MATRIX")
    print("=" * 50)
    print(cm)
    print("=" * 50)


if __name__ == "__main__":
    # Example usage
    print("\nMetrics Example:")

    y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])

    print_confusion_matrix(y_true, y_pred)
    print_classification_report(y_true, y_pred)
