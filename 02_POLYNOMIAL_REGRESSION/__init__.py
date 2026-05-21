"""
Polynomial Regression from Scratch
===================================

A complete implementation of Polynomial Regression with gradient descent optimization.

Main Components:
- PolynomialRegression: Main regression class with polynomial feature expansion
- Metrics: Evaluation functions (MSE, RMSE, MAE, R², etc.)
- Utils: Preprocessing tools (StandardScaler, MinMaxScaler, train_test_split)

Example usage:
    >>> from polynomial_regression import PolynomialRegression
    >>> from utils import StandardScaler, train_test_split
    >>> model = PolynomialRegression(degree=3, learning_rate=0.01)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
"""

from .polynomial_regression import PolynomialRegression
from .metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    adjusted_r2_score,
    mean_absolute_percentage_error,
    explained_variance_score
)
from .utils import (
    StandardScaler,
    MinMaxScaler,
    train_test_split
)

__version__ = '1.0.0'

__all__ = [
    'PolynomialRegression',
    'mean_squared_error',
    'root_mean_squared_error',
    'mean_absolute_error',
    'r2_score',
    'adjusted_r2_score',
    'mean_absolute_percentage_error',
    'explained_variance_score',
    'StandardScaler',
    'MinMaxScaler',
    'train_test_split'
]
