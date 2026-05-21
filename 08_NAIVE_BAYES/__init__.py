"""
Naive Bayes Classifiers Module
==============================

This module provides implementations of three Naive Bayes classifier variants:
- GaussianNB: For continuous features with Gaussian distributions
- MultinomialNB: For discrete count features (e.g., text classification)
- BernoulliNB: For binary features (presence/absence)

Example usage:
-------------
>>> from naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
>>> import numpy as np
>>>
>>> # Example with GaussianNB
>>> X = np.random.randn(100, 4)
>>> y = (X[:, 0] + X[:, 1] > 0).astype(int)
>>> clf = GaussianNB()
>>> clf.fit(X[:80], y[:80])
>>> accuracy = clf.score(X[80:], y[80:])
>>> print(f"Accuracy: {accuracy:.3f}")
"""

from .naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

__version__ = '1.0.0'
__author__ = 'ML Algorithms from Scratch'

__all__ = [
    'GaussianNB',
    'MultinomialNB',
    'BernoulliNB',
]
