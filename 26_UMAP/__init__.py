"""
UMAP (Uniform Manifold Approximation and Projection) — From Scratch

McInnes, Healy & Melville (2018): nonlinear dimensionality reduction via
fuzzy simplicial sets and cross-entropy minimisation with SGD.
"""

from .umap_scratch import (
    UMAP,
    compute_fuzzy_simplicial_set,
    find_ab_params,
    umap_optimize_layout,
)

__all__ = [
    'UMAP',
    'compute_fuzzy_simplicial_set',
    'find_ab_params',
    'umap_optimize_layout',
]
__version__ = '1.0.0'
