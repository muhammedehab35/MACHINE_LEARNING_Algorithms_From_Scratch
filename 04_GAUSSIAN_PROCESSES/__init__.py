"""
Gaussian Process Regression Module

A complete implementation of Gaussian Processes for regression.

Author: ML Algorithms from Scratch
"""

from .gp import GaussianProcessRegressor
from .kernels import (
    Kernel,
    RBFKernel,
    MaternKernel,
    RationalQuadraticKernel,
    PeriodicKernel,
    LinearKernel,
    WhiteNoiseKernel,
    KernelSum,
    KernelProduct
)
from .metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    max_error,
    mean_absolute_percentage_error,
    negative_log_predictive_density,
    continuous_ranked_probability_score,
    calibration_error,
    prediction_interval_coverage,
    regression_report,
    print_regression_report
)
from .utils import (
    StandardScaler,
    MinMaxScaler,
    train_test_split,
    make_regression,
    k_fold_split,
    cross_val_score,
    grid_search
)

__all__ = [
    # Core GP
    'GaussianProcessRegressor',

    # Kernels
    'Kernel',
    'RBFKernel',
    'MaternKernel',
    'RationalQuadraticKernel',
    'PeriodicKernel',
    'LinearKernel',
    'WhiteNoiseKernel',
    'KernelSum',
    'KernelProduct',

    # Metrics
    'mean_squared_error',
    'root_mean_squared_error',
    'mean_absolute_error',
    'r2_score',
    'explained_variance_score',
    'max_error',
    'mean_absolute_percentage_error',
    'negative_log_predictive_density',
    'continuous_ranked_probability_score',
    'calibration_error',
    'prediction_interval_coverage',
    'regression_report',
    'print_regression_report',

    # Utils
    'StandardScaler',
    'MinMaxScaler',
    'train_test_split',
    'make_regression',
    'k_fold_split',
    'cross_val_score',
    'grid_search',
]
