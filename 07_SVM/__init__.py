"""
Support Vector Machine Module

Complete implementation of Support Vector Machines for classification.

Main Components:
- SVMClassifier: Support Vector Machine classifier with multiple kernels
- Metrics: Classification evaluation metrics
- Utils: Preprocessing utilities

Example:
    >>> from svm import SVMClassifier
    >>> from utils import StandardScaler, train_test_split
    >>>
    >>> model = SVMClassifier(kernel='rbf', C=1.0)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
"""

from .svm import SVMClassifier
from .metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    specificity_score,
    matthews_corrcoef,
    classification_report,
    print_classification_report,
    print_confusion_matrix
)
from .utils import (
    StandardScaler,
    MinMaxScaler,
    train_test_split
)

__all__ = [
    # Main model
    'SVMClassifier',

    # Metrics
    'confusion_matrix',
    'accuracy_score',
    'precision_score',
    'recall_score',
    'f1_score',
    'specificity_score',
    'matthews_corrcoef',
    'classification_report',
    'print_classification_report',
    'print_confusion_matrix',

    # Utils
    'StandardScaler',
    'MinMaxScaler',
    'train_test_split',
]

__version__ = '1.0.0'
__author__ = 'ML Algorithms from Scratch'
