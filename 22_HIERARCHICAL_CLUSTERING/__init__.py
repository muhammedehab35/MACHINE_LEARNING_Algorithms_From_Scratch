"""
Agglomerative Hierarchical Clustering — From Scratch Implementation

Ward (1963) / Lance-Williams (1967): builds a dendrogram by iteratively
merging the two closest clusters under a chosen linkage criterion.
"""

from .hierarchical_scratch import (
    AgglomerativeClustering,
    linkage,
    fcluster,
    cophenetic_correlation,
    pairwise_distances,
)

__all__ = [
    'AgglomerativeClustering',
    'linkage',
    'fcluster',
    'cophenetic_correlation',
    'pairwise_distances',
]
__version__ = '1.0.0'
