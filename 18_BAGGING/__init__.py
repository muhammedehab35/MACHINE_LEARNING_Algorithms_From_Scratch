"""
Bagging (Bootstrap Aggregating) — From Scratch Implementation

Leo Breiman (1994): reduces variance by training multiple estimators
on independent bootstrap samples and aggregating predictions.
"""

from .bagging_scratch import (
    BaggingClassifier,
    BaggingRegressor,
    bootstrap_sample,
    oob_indices,
)

__all__ = [
    'BaggingClassifier',
    'BaggingRegressor',
    'bootstrap_sample',
    'oob_indices',
]
__version__ = '1.0.0'
