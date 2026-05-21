"""
Classification Metrics for Binary Classification

Comprehensive set of metrics to evaluate binary classification model performance.
All metrics are implemented from scratch using only NumPy.

Author: ML Algorithms from Scratch
"""

import numpy as np
from typing import Union, Tuple


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.

    Returns:
        [[TN, FP],
         [FN, TP]]
    """
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    return np.array([[tn, fp], [fn, tp]])


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy: (TP + TN) / Total

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    return np.mean(y_true == y_pred)


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate precision: TP / (TP + FP)

    Precision measures how many of the predicted positives are actually positive.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate recall (sensitivity, TPR): TP / (TP + FN)

    Recall measures how many of the actual positives were correctly identified.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate F1 score: 2 * (Precision * Recall) / (Precision + Recall)

    F1 is the harmonic mean of precision and recall.
    """
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)

    if prec + rec == 0:
        return 0.0

    return 2 * (prec * rec) / (prec + rec)


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate specificity (TNR): TN / (TN + FP)

    Specificity measures how many of the actual negatives were correctly identified.
    """
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if tn + fp == 0:
        return 0.0

    return tn / (tn + fp)


def roc_auc_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Calculate ROC AUC score using trapezoidal rule.

    AUC measures the area under the ROC curve.
    """
    # Sort by predicted probabilities
    desc_score_indices = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = y_true[desc_score_indices]

    # Calculate TPR and FPR at each threshold
    tpr_list = []
    fpr_list = []

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp = 0
    fp = 0

    for label in y_true_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1

        tpr = tp / n_pos
        fpr = fp / n_neg

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Add (0, 0) point
    tpr_list = [0] + tpr_list
    fpr_list = [0] + fpr_list

    # Calculate AUC using trapezoidal rule
    auc = 0.0
    for i in range(len(tpr_list) - 1):
        auc += (fpr_list[i + 1] - fpr_list[i]) * (tpr_list[i + 1] + tpr_list[i]) / 2

    return auc


def log_loss(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Calculate binary cross-entropy loss.

    Log Loss = -1/n Σ[y log(p) + (1-y) log(1-p)]
    """
    epsilon = 1e-15
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)

    return -np.mean(
        y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba)
    )


def matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Matthews Correlation Coefficient (MCC).

    MCC = (TP×TN - FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

    MCC is considered one of the best single metrics for binary classification.
    Range: [-1, 1], where 1 is perfect, 0 is random, -1 is total disagreement.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0]
    fn, tp = cm[1]

    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    if denominator == 0:
        return 0.0

    return numerator / denominator


def cohen_kappa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Cohen's Kappa score.

    Kappa = (p_o - p_e) / (1 - p_e)

    where p_o is observed agreement and p_e is expected agreement.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0]
    fn, tp = cm[1]

    n = tn + fp + fn + tp

    # Observed agreement
    p_o = (tp + tn) / n

    # Expected agreement
    p_e = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n * n)

    if p_e == 1:
        return 0.0

    return (p_o - p_e) / (1 - p_e)


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray = None
) -> dict:
    """
    Compute all classification metrics at once.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_pred : np.ndarray
        Predicted binary labels.
    y_pred_proba : np.ndarray, optional
        Predicted probabilities for positive class.

    Returns
    -------
    metrics : dict
        Dictionary containing all computed metrics.
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'Specificity': specificity_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Cohen Kappa': cohen_kappa_score(y_true, y_pred)
    }

    if y_pred_proba is not None:
        metrics['ROC AUC'] = roc_auc_score(y_true, y_pred_proba)
        metrics['Log Loss'] = log_loss(y_true, y_pred_proba)

    return metrics


def print_classification_report(metrics: dict, title: str = "Classification Metrics") -> None:
    """
    Pretty print classification metrics.

    Parameters
    ----------
    metrics : dict
        Dictionary of metrics.
    title : str
        Title to display.
    """
    print(f"\n{'=' * 50}")
    print(f"{title:^50}")
    print(f"{'=' * 50}")

    for metric_name, value in metrics.items():
        print(f"{metric_name:<20}: {value:>12.6f}")

    print(f"{'=' * 50}\n")


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print confusion matrix in a readable format."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0]
    fn, tp = cm[1]

    print("\nConfusion Matrix:")
    print(f"{'':>15} Predicted Neg  Predicted Pos")
    print(f"{'Actual Neg':<15} {tn:>13}  {fp:>13}")
    print(f"{'Actual Pos':<15} {fn:>13}  {tp:>13}\n")


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)

    # Generate test predictions
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 1, 1, 0])
    y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.85, 0.6, 0.75, 0.95, 0.15])

    # Compute metrics
    print("Individual Metrics:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_true, y_pred_proba):.4f}")

    # Print confusion matrix
    print_confusion_matrix(y_true, y_pred)

    # Full report
    metrics = classification_report(y_true, y_pred, y_pred_proba)
    print_classification_report(metrics)
