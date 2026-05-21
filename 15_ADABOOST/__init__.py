"""
AdaBoost (Adaptive Boosting) Implementation

An ensemble of decision stumps trained with adaptive sample weighting.
"""

from .adaboost import AdaBoostClassifier, DecisionStump

__all__ = ['AdaBoostClassifier', 'DecisionStump']
__version__ = '1.0.0'
