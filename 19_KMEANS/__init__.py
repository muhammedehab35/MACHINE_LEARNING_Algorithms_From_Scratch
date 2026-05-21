"""
K-Means Clustering — From Scratch Implementation

Lloyd's algorithm with K-Means++ seeding, OOB-free evaluation via
silhouette and Davies-Bouldin indices, and elbow method utilities.
"""

from .kmeans_scratch import (
    KMeans,
    euclidean_distances,
    inertia,
    silhouette_score,
    davies_bouldin_score,
    elbow_scores,
)

__all__ = [
    'KMeans',
    'euclidean_distances',
    'inertia',
    'silhouette_score',
    'davies_bouldin_score',
    'elbow_scores',
]
__version__ = '1.0.0'
