"""
Decision Tree Classifier - From Scratch Implementation
======================================================

A complete implementation of a Decision Tree classifier using only NumPy,
following the CART (Classification and Regression Trees) algorithm.

Features:
---------
- Binary and multi-class classification
- Multiple split criteria (Gini, Entropy, Error)
- Pre-pruning and cost-complexity pruning
- Feature importance calculation
- Tree visualization
- sklearn-compatible API

Example Usage:
--------------
>>> from decision_tree import DecisionTreeClassifier
>>> import numpy as np
>>> X = np.array([[0, 0], [1, 1]])
>>> y = np.array([0, 1])
>>> clf = DecisionTreeClassifier(max_depth=3)
>>> clf.fit(X, y)
>>> clf.predict([[0.5, 0.5]])
array([0])

For more examples, see the examples/ directory.
"""

from .decision_tree import DecisionTreeClassifier, Node

__version__ = '1.0.0'
__author__ = 'ML Algorithms from Scratch'
__all__ = ['DecisionTreeClassifier', 'Node']
