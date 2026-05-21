"""
K-Nearest Neighbors Module

Complete implementation of K-Nearest Neighbors for classification and regression.

Main Components:
- KNNClassifier: K-Nearest Neighbors for classification
- KNNRegressor: K-Nearest Neighbors for regression
- Metrics: Classification evaluation metrics
- Utils: Preprocessing utilities

Example:
    >>> from knn import KNNClassifier
    >>> from utils import StandardScaler, train_test_split
    >>>
    >>> model = KNNClassifier(n_neighbors=5, weights='distance')
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
"""

from .knn import KNNClassifier, KNNRegressor
from .metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    specificity_score,
    roc_auc_score,
    log_loss,
    matthews_corrcoef,
    cohen_kappa_score,
    classification_report,
    print_classification_report,
    print_confusion_matrix
)
from .utils import (
    StandardScaler,
    MinMaxScaler,
    train_test_split,
    polynomial_features
)

__all__ = [
    # Main models
    'KNNClassifier',
    'KNNRegressor',

    # Metrics
    'confusion_matrix',
    'accuracy_score',
    'precision_score',
    'recall_score',
    'f1_score',
    'specificity_score',
    'roc_auc_score',
    'log_loss',
    'matthews_corrcoef',
    'cohen_kappa_score',
    'classification_report',
    'print_classification_report',
    'print_confusion_matrix',

    # Utils
    'StandardScaler',
    'MinMaxScaler',
    'train_test_split',
    'polynomial_features',
]

__version__ = '1.0.0'
__author__ = 'ML Algorithms from Scratch'
