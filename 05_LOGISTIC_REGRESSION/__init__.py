"""
Logistic Regression Module

Binary classification using logistic regression with gradient descent.

Main Components:
- LogisticRegression: Binary classification model
- Metrics: 9 classification evaluation metrics
- Utils: Preprocessing utilities

Example:
    >>> from logistic_regression import LogisticRegression
    >>> from utils import StandardScaler, train_test_split
    >>>
    >>> model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
"""

from .logistic_regression import LogisticRegression
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
    # Main model
    'LogisticRegression',

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
