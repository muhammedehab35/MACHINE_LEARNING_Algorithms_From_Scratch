"""
Elastic Net Regression Module

A complete implementation of Elastic Net regression from scratch using only NumPy.

Main Components:
- ElasticNet: Main regression model with L1+L2 regularization
- Metrics: Comprehensive evaluation metrics
- Utils: Preprocessing utilities (StandardScaler, train_test_split, etc.)

Example:
    >>> from elastic_net import ElasticNet
    >>> from utils import StandardScaler, train_test_split
    >>>
    >>> # Your code here
    >>> model = ElasticNet(l1_ratio=0.5, alpha=1.0)
    >>> model.fit(X_train, y_train)
"""

from .elastic_net import ElasticNet
from .metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    adjusted_r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    max_error,
    evaluate_regression,
    print_metrics
)
from .utils import (
    StandardScaler,
    MinMaxScaler,
    train_test_split,
    polynomial_features
)

__all__ = [
    # Main model
    'ElasticNet',

    # Metrics
    'mean_squared_error',
    'root_mean_squared_error',
    'mean_absolute_error',
    'r2_score',
    'adjusted_r2_score',
    'mean_absolute_percentage_error',
    'explained_variance_score',
    'max_error',
    'evaluate_regression',
    'print_metrics',

    # Utils
    'StandardScaler',
    'MinMaxScaler',
    'train_test_split',
    'polynomial_features',
]

__version__ = '1.0.0'
__author__ = 'ML Algorithms from Scratch'
