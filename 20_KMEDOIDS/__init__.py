"""
K-Medoids Clustering (PAM) — From Scratch Implementation

PAM with BUILD initialisation, SWAP optimisation, multi-metric support
(Euclidean, Manhattan, Cosine), and outlier robustness.
"""

from .kmedoids_scratch import (
    KMedoids,
    pairwise_distances,
    elbow_costs,
)

__all__ = [
    'KMedoids',
    'pairwise_distances',
    'elbow_costs',
]
__version__ = '1.0.0'
