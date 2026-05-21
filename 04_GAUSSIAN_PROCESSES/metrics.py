"""
Regression Metrics for Gaussian Processes

Comprehensive metrics for evaluating GP regression performance.

Author: ML Algorithms from Scratch
"""

import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error (MSE).

    MSE = (1/n) * Σ(y_true - y_pred)²

    Parameters
    ----------
    y_true : np.ndarray
        True values.
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
    Root Mean Squared Error (RMSE).

    RMSE = sqrt((1/n) * Σ(y_true - y_pred)²)
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error (MAE).

    MAE = (1/n) * Σ|y_true - y_pred|
    """
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² (Coefficient of Determination).

    R² = 1 - (SS_res / SS_tot)

    where:
    - SS_res = Σ(y_true - y_pred)²
    - SS_tot = Σ(y_true - mean(y_true))²
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return 1 - (ss_res / ss_tot)


def explained_variance_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Explained Variance Score.

    EV = 1 - Var(y_true - y_pred) / Var(y_true)
    """
    numerator = np.var(y_true - y_pred)
    denominator = np.var(y_true)

    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0

    return 1 - (numerator / denominator)


def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Maximum absolute error.

    MaxError = max|y_true - y_pred|
    """
    return np.max(np.abs(y_true - y_pred))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Mean Absolute Percentage Error (MAPE).

    MAPE = (100/n) * Σ|y_true - y_pred| / |y_true|
    """
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def negative_log_predictive_density(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Negative Log Predictive Density (NLPD).

    Specific metric for probabilistic predictions.

    NLPD = -log p(y_true | y_mean, y_std)
         = 0.5 * log(2π * σ²) + 0.5 * (y_true - μ)² / σ²

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_mean : np.ndarray
        Predicted means.
    y_std : np.ndarray
        Predicted standard deviations.
    epsilon : float
        Small constant for numerical stability.

    Returns
    -------
    nlpd : float
        Negative log predictive density.
    """
    y_var = y_std ** 2 + epsilon
    log_term = 0.5 * np.log(2 * np.pi * y_var)
    error_term = 0.5 * (y_true - y_mean) ** 2 / y_var

    return np.mean(log_term + error_term)


def continuous_ranked_probability_score(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray
) -> float:
    """
    Continuous Ranked Probability Score (CRPS).

    For Gaussian distribution:
    CRPS = σ * (z * (2Φ(z) - 1) + 2φ(z) - 1/√π)

    where z = (y_true - μ) / σ, Φ is CDF, φ is PDF

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_mean : np.ndarray
        Predicted means.
    y_std : np.ndarray
        Predicted standard deviations.

    Returns
    -------
    crps : float
        Continuous ranked probability score.
    """
    from scipy.stats import norm

    z = (y_true - y_mean) / (y_std + 1e-10)

    phi_z = norm.pdf(z)  # PDF
    Phi_z = norm.cdf(z)  # CDF

    crps_values = y_std * (z * (2 * Phi_z - 1) + 2 * phi_z - 1 / np.sqrt(np.pi))

    return np.mean(crps_values)


def calibration_error(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    quantiles: np.ndarray = None
) -> float:
    """
    Calibration Error for predictive uncertainty.

    Measures how well predicted uncertainties match empirical frequencies.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_mean : np.ndarray
        Predicted means.
    y_std : np.ndarray
        Predicted standard deviations.
    quantiles : np.ndarray, optional
        Quantiles to check (default: [0.1, 0.2, ..., 0.9])

    Returns
    -------
    calib_error : float
        Average calibration error across quantiles.
    """
    from scipy.stats import norm

    if quantiles is None:
        quantiles = np.arange(0.1, 1.0, 0.1)

    errors = []

    for q in quantiles:
        # Expected: q fraction of points should fall below q-quantile
        z_q = norm.ppf(q)  # Inverse CDF
        threshold = y_mean + z_q * y_std

        # Actual fraction below threshold
        actual_fraction = np.mean(y_true <= threshold)

        # Calibration error for this quantile
        errors.append(abs(actual_fraction - q))

    return np.mean(errors)


def prediction_interval_coverage(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    confidence: float = 0.95
) -> float:
    """
    Prediction Interval Coverage.

    Fraction of true values within confidence interval.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_mean : np.ndarray
        Predicted means.
    y_std : np.ndarray
        Predicted standard deviations.
    confidence : float
        Confidence level (e.g., 0.95 for 95%).

    Returns
    -------
    coverage : float
        Fraction of points within interval.
    """
    from scipy.stats import norm

    # Compute confidence interval
    z_score = norm.ppf((1 + confidence) / 2)
    lower = y_mean - z_score * y_std
    upper = y_mean + z_score * y_std

    # Check coverage
    in_interval = (y_true >= lower) & (y_true <= upper)
    coverage = np.mean(in_interval)

    return coverage


def regression_report(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray = None
) -> dict:
    """
    Generate comprehensive regression report.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_mean : np.ndarray
        Predicted means.
    y_std : np.ndarray, optional
        Predicted standard deviations (for probabilistic metrics).

    Returns
    -------
    report : dict
        Dictionary with all metrics.
    """
    report = {
        'MSE': mean_squared_error(y_true, y_mean),
        'RMSE': root_mean_squared_error(y_true, y_mean),
        'MAE': mean_absolute_error(y_true, y_mean),
        'R²': r2_score(y_true, y_mean),
        'Explained Variance': explained_variance_score(y_true, y_mean),
        'Max Error': max_error(y_true, y_mean),
    }

    if y_std is not None:
        report['NLPD'] = negative_log_predictive_density(y_true, y_mean, y_std)
        report['95% Coverage'] = prediction_interval_coverage(y_true, y_mean, y_std, 0.95)
        report['Calibration Error'] = calibration_error(y_true, y_mean, y_std)

        try:
            report['CRPS'] = continuous_ranked_probability_score(y_true, y_mean, y_std)
        except ImportError:
            pass  # scipy not available

    return report


def print_regression_report(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray = None
):
    """Print formatted regression report."""
    report = regression_report(y_true, y_mean, y_std)

    print("\n" + "=" * 50)
    print("REGRESSION REPORT")
    print("=" * 50)

    for metric, value in report.items():
        print(f"{metric:20}: {value:.6f}")

    print("=" * 50)


if __name__ == "__main__":
    # Example usage
    print("\nMetrics Example:")

    np.random.seed(42)
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    y_std = np.array([0.2, 0.3, 0.2, 0.4, 0.3])

    print_regression_report(y_true, y_pred, y_std)
