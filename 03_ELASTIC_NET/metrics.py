"""
Evaluation Metrics for Regression

Comprehensive set of metrics to evaluate regression model performance.
All metrics are implemented from scratch using only NumPy.

Author: ML Algorithms from Scratch
"""

import numpy as np
from typing import Union


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error (MSE).

    MSE = (1/n) Σ(y_true - y_pred)²

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    mse : float
        Mean squared error.
    """
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).

    RMSE = √(MSE)

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    rmse : float
        Root mean squared error.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    MAE = (1/n) Σ|y_true - y_pred|

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    mae : float
        Mean absolute error.
    """
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² (coefficient of determination).

    R² = 1 - (SS_res / SS_tot)

    where:
    - SS_res = Σ(y_true - y_pred)² (residual sum of squares)
    - SS_tot = Σ(y_true - mean(y_true))² (total sum of squares)

    R² = 1: Perfect fit
    R² = 0: Model performs as well as predicting the mean
    R² < 0: Model performs worse than predicting the mean

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    r2 : float
        R² score.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)


def adjusted_r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int
) -> float:
    """
    Calculate Adjusted R² score.

    Adjusted R² penalizes the addition of features that don't improve the model.

    Adjusted R² = 1 - [(1 - R²) * (n - 1) / (n - p - 1)]

    where:
    - n: number of samples
    - p: number of features

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values.
    n_features : int
        Number of features in the model.

    Returns
    -------
    adj_r2 : float
        Adjusted R² score.
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)

    if n <= n_features + 1:
        return r2

    return 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))


def mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    MAPE = (100/n) Σ|y_true - y_pred| / |y_true|

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values.
    epsilon : float, default=1e-10
        Small value to avoid division by zero.

    Returns
    -------
    mape : float
        Mean absolute percentage error (as percentage).
    """
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100


def explained_variance_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Explained Variance Score.

    EV = 1 - Var(y_true - y_pred) / Var(y_true)

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    ev : float
        Explained variance score.
    """
    var_residual = np.var(y_true - y_pred)
    var_true = np.var(y_true)

    if var_true == 0:
        return 0.0

    return 1 - (var_residual / var_true)


def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate maximum residual error.

    Max Error = max(|y_true - y_pred|)

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    max_err : float
        Maximum absolute error.
    """
    return np.max(np.abs(y_true - y_pred))


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int = None
) -> dict:
    """
    Compute all regression metrics at once.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values.
    n_features : int, optional
        Number of features (for adjusted R²).

    Returns
    -------
    metrics : dict
        Dictionary containing all computed metrics.
    """
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R²': r2_score(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'Explained Variance': explained_variance_score(y_true, y_pred),
        'Max Error': max_error(y_true, y_pred)
    }

    if n_features is not None:
        metrics['Adjusted R²'] = adjusted_r2_score(y_true, y_pred, n_features)

    return metrics


def print_metrics(metrics: dict, title: str = "Regression Metrics") -> None:
    """
    Pretty print regression metrics.

    Parameters
    ----------
    metrics : dict
        Dictionary of metrics (from evaluate_regression).
    title : str, default="Regression Metrics"
        Title to display.
    """
    print(f"\n{'=' * 50}")
    print(f"{title:^50}")
    print(f"{'=' * 50}")

    for metric_name, value in metrics.items():
        print(f"{metric_name:<20}: {value:>12.6f}")

    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)

    # Generate test data
    y_true = np.array([3, -0.5, 2, 7, 4.2])
    y_pred = np.array([2.5, 0.0, 2, 8, 4.5])

    # Compute individual metrics
    print("Individual Metrics:")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}")
    print(f"RMSE: {root_mean_squared_error(y_true, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"R²: {r2_score(y_true, y_pred):.4f}")
    print(f"MAPE: {mean_absolute_percentage_error(y_true, y_pred):.2f}%")

    # Compute all metrics
    all_metrics = evaluate_regression(y_true, y_pred, n_features=3)
    print_metrics(all_metrics, "All Regression Metrics")
