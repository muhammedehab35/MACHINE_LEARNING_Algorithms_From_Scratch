"""
CatBoost (Categorical Boosting) — From Scratch Implementation

Yandex's gradient boosting with symmetric trees, ordered boosting,
and ordered target statistics for categorical features.
"""

from .catboost_scratch import (
    CatBoostClassifier,
    CatBoostRegressor,
    ObliviousTree,
    SplitCondition,
    ordered_target_statistics,
)

__all__ = [
    'CatBoostClassifier',
    'CatBoostRegressor',
    'ObliviousTree',
    'SplitCondition',
    'ordered_target_statistics',
]
__version__ = '1.0.0'
