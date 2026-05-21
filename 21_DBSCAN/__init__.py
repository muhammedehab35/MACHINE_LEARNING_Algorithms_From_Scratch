"""
DBSCAN — Density-Based Spatial Clustering of Applications with Noise
From Scratch Implementation

Ester et al. (1996): discovers clusters of arbitrary shape, handles noise,
requires no K upfront.
"""

from .dbscan_scratch import (
    DBSCAN,
    pairwise_distances,
    k_dist,
    cluster_stats,
)

__all__ = [
    'DBSCAN',
    'pairwise_distances',
    'k_dist',
    'cluster_stats',
]
__version__ = '1.0.0'
