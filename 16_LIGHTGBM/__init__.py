"""
LightGBM (Light Gradient Boosting Machine) - From Scratch Implementation

Microsoft's highly efficient gradient boosting with histogram-based splits,
leaf-wise tree growth, and GOSS sampling.
"""

from .lightgbm_scratch import (
    LightGBMClassifier,
    LightGBMRegressor,
    LGBMTree,
    FeatureHistogram,
    goss_sample,
)

__all__ = [
    'LightGBMClassifier',
    'LightGBMRegressor',
    'LGBMTree',
    'FeatureHistogram',
    'goss_sample',
]
__version__ = '1.0.0'
