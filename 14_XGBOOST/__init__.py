"""
XGBoost (Extreme Gradient Boosting) Implementation

An optimized gradient boosting algorithm with advanced regularization,
second-order optimization, and efficient tree construction.
"""

from .xgboost import XGBoostClassifier, XGBoostRegressor, XGBoostTree

__all__ = ['XGBoostClassifier', 'XGBoostRegressor', 'XGBoostTree']
__version__ = '1.0.0'
