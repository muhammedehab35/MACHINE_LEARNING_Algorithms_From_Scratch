"""
Quadratic Discriminant Analysis (QDA) Implementation

A classifier that models each class with its own Gaussian distribution,
allowing for different covariance matrices and quadratic decision boundaries.
"""

from .qda import QuadraticDiscriminantAnalysis, RegularizedQDA

__all__ = ['QuadraticDiscriminantAnalysis', 'RegularizedQDA']
__version__ = '1.0.0'
