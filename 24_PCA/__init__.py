"""
Principal Component Analysis (PCA) — From Scratch Implementation

Pearson (1901) / Hotelling (1933): orthonormal dimensionality reduction
via variance-maximising projections, computed through the economy SVD
of the mean-centred data matrix.
"""

from .pca_scratch import PCA, pca_svd

__all__ = ['PCA', 'pca_svd']
__version__ = '1.0.0'
