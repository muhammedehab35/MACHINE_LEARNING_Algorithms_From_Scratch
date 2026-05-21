"""
Linear Regression Module
========================

This module provides a complete implementation of Linear Regression
with multiple optimization methods and utilities.

Classes:
    LinearRegression: Main regression class with gradient descent

Functions:
    Various utility functions for metrics and preprocessing
"""

from .linear_regression import LinearRegression
from .metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    adjusted_r2_score
)
from .utils import StandardScaler

__all__ = [
    'LinearRegression',
    'mean_squared_error',
    'root_mean_squared_error',
    'mean_absolute_error',
    'r2_score',
    'adjusted_r2_score',
    'StandardScaler'
]

__version__ = '1.0.0'
