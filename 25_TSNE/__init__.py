"""
t-SNE (t-distributed Stochastic Neighbor Embedding) — From Scratch

van der Maaten & Hinton (2008): nonlinear dimensionality reduction for
visualization via KL divergence minimisation between Gaussian (high-dim)
and Student-t (low-dim) pairwise similarity distributions.
"""

from .tsne_scratch import (
    TSNE,
    compute_joint_probabilities,
    student_t_affinities,
    tsne_gradient,
    kl_divergence,
)

__all__ = [
    'TSNE',
    'compute_joint_probabilities',
    'student_t_affinities',
    'tsne_gradient',
    'kl_divergence',
]
__version__ = '1.0.0'
