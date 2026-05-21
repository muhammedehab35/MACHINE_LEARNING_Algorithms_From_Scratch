"""
Comprehensive Tests for t-SNE
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsne_scratch import (
    TSNE, compute_joint_probabilities, student_t_affinities,
    tsne_gradient, kl_divergence, _pairwise_sq_distances,
    _cond_probs_and_entropy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blobs(centers, n=30, std=0.2, seed=42):
    np.random.seed(seed)
    return (
        np.vstack([np.random.randn(n, 2) * std + c for c in centers]),
        np.repeat(np.arange(len(centers)), n),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_shape():
    """embedding_ must have shape (n, n_components)."""
    print("Test 1: Output Shape")
    print("-" * 70)

    np.random.seed(0)
    X = np.random.randn(40, 5)
    for d in [1, 2]:
        tsne = TSNE(n_components=d, perplexity=10, n_iter=50, random_state=0)
        Z = tsne.fit_transform(X)
        assert Z.shape == (40, d), f"Expected (40, {d}), got {Z.shape}"
        print(f"   [OK] n_components={d}: embedding shape = {Z.shape}")
    return True


def test_embedding_attribute():
    """embedding_ attribute must exist and equal fit_transform output."""
    print("\nTest 2: embedding_ Attribute")
    print("-" * 70)

    np.random.seed(1)
    X = np.random.randn(30, 4)
    tsne = TSNE(n_components=2, perplexity=8, n_iter=50, random_state=1)
    Z = tsne.fit_transform(X)
    assert hasattr(tsne, 'embedding_')
    assert np.array_equal(Z, tsne.embedding_)
    print(f"   [OK] embedding_ shape: {tsne.embedding_.shape}")
    return True


def test_pairwise_distances_symmetry():
    """Pairwise squared distance matrix must be symmetric with zero diagonal."""
    print("\nTest 3: Pairwise Distance Matrix Symmetry")
    print("-" * 70)

    np.random.seed(2)
    X = np.random.randn(20, 6)
    D = _pairwise_sq_distances(X)
    assert D.shape == (20, 20)
    assert np.allclose(D, D.T, atol=1e-10)
    assert np.allclose(np.diag(D), 0.0, atol=1e-10)
    assert np.all(D >= 0)
    print(f"   [OK] D symmetric, zero diagonal, non-negative")
    return True


def test_joint_probabilities_symmetry():
    """P matrix must be symmetric."""
    print("\nTest 4: P Matrix Symmetry")
    print("-" * 70)

    np.random.seed(3)
    X = np.random.randn(20, 4)
    P = compute_joint_probabilities(X, perplexity=5)
    assert np.allclose(P, P.T, atol=1e-10), "P is not symmetric"
    assert np.all(P >= 0), "P has negative entries"
    print(f"   [OK] P is symmetric, shape={P.shape}, min={P.min():.2e}")
    return True


def test_joint_probabilities_sum():
    """P matrix must sum to 1."""
    print("\nTest 5: P Matrix Sums to 1")
    print("-" * 70)

    np.random.seed(4)
    X = np.random.randn(25, 3)
    P = compute_joint_probabilities(X, perplexity=5)
    total = P.sum()
    assert abs(total - 1.0) < 1e-6, f"P.sum() = {total}"
    print(f"   [OK] P.sum() = {total:.10f}")
    return True


def test_student_t_q_sums_to_one():
    """Q matrix must sum to 1 with zero diagonal."""
    print("\nTest 6: Q Matrix Sums to 1")
    print("-" * 70)

    np.random.seed(5)
    Y = np.random.randn(30, 2)
    Q, num = student_t_affinities(Y)
    assert abs(Q.sum() - 1.0) < 1e-10
    assert np.allclose(np.diag(Q), 0.0, atol=1e-12)
    assert np.allclose(np.diag(num), 0.0, atol=1e-12)
    print(f"   [OK] Q.sum() = {Q.sum():.12f}, diagonal = 0")
    return True


def test_perplexity_binary_search_accuracy():
    """Binary search must find beta so that H(P_i) = log(perplexity)."""
    print("\nTest 7: Binary Search Perplexity Accuracy")
    print("-" * 70)

    np.random.seed(6)
    X = np.random.randn(25, 4)
    D2 = _pairwise_sq_distances(X)

    for target_perp in [5.0, 10.0, 20.0]:
        target_H = np.log(target_perp)
        # Run binary search manually for one row
        d_i = D2[0].copy()
        d_i[0] = np.inf
        beta_lo, beta_hi = -np.inf, np.inf
        beta = 1.0
        for _ in range(50):
            H, p = _cond_probs_and_entropy(d_i, beta)
            diff = H - target_H
            if abs(diff) < 1e-5:
                break
            if diff > 0:
                beta_lo = beta
                beta = 2 * beta if beta_hi == np.inf else (beta + beta_hi) / 2
            else:
                beta_hi = beta
                beta = beta / 2 if beta_lo == -np.inf else (beta + beta_lo) / 2
        achieved_perp = np.exp(H)
        assert abs(achieved_perp - target_perp) / target_perp < 0.01, \
            f"target={target_perp}: achieved {achieved_perp:.4f}"
        print(f"   [OK] perplexity={target_perp}: achieved={achieved_perp:.4f}")
    return True


def test_kl_divergence_non_negative():
    """KL divergence must be non-negative."""
    print("\nTest 8: KL Divergence Non-Negative")
    print("-" * 70)

    np.random.seed(7)
    n = 20
    P = compute_joint_probabilities(np.random.randn(n, 4), perplexity=5)
    Y = np.random.randn(n, 2)
    Q, _ = student_t_affinities(Y)
    kl = kl_divergence(P, Q)
    assert kl >= 0, f"KL = {kl}"
    print(f"   [OK] KL divergence = {kl:.6f} >= 0")
    return True


def test_kl_decreases_over_iterations():
    """KL divergence must decrease from early phase to final phase."""
    print("\nTest 9: KL Divergence Decreases Over Training")
    print("-" * 70)

    np.random.seed(8)
    X = np.random.randn(60, 5)
    tsne = TSNE(n_components=2, perplexity=10, n_iter=300,
                n_iter_early_exag=100, random_state=8)
    tsne.fit(X)
    history = tsne.kl_divergence_history_
    assert len(history) > 0
    # KL at end should be less than at start (after early exaggeration)
    kl_start = history[len(history) // 4][1]
    kl_end = history[-1][1]
    assert kl_end < kl_start * 2, \
        f"KL did not decrease: start={kl_start:.4f}, end={kl_end:.4f}"
    print(f"   [OK] KL early={kl_start:.4f} -> final={kl_end:.4f}")
    return True


def test_gradient_shape():
    """Gradient must have same shape as Y."""
    print("\nTest 10: Gradient Shape")
    print("-" * 70)

    np.random.seed(9)
    n, d = 25, 2
    P = compute_joint_probabilities(np.random.randn(n, 5), perplexity=6)
    Y = np.random.randn(n, d)
    Q, num = student_t_affinities(Y)
    grad = tsne_gradient(P, Q, num, Y)
    assert grad.shape == (n, d), f"Expected ({n},{d}), got {grad.shape}"
    print(f"   [OK] gradient shape = {grad.shape}")
    return True


def test_random_state_reproducibility():
    """Same random_state must yield identical embeddings."""
    print("\nTest 11: Random State Reproducibility")
    print("-" * 70)

    np.random.seed(10)
    X = np.random.randn(40, 5)
    tsne1 = TSNE(n_components=2, perplexity=8, n_iter=100, random_state=42)
    tsne2 = TSNE(n_components=2, perplexity=8, n_iter=100, random_state=42)
    Z1 = tsne1.fit_transform(X)
    Z2 = tsne2.fit_transform(X)
    assert np.allclose(Z1, Z2, atol=1e-12), "Same seed gives different results"
    print(f"   [OK] Two runs with seed=42 give identical embeddings")
    return True


def test_pca_init():
    """init='pca' must produce valid output without error."""
    print("\nTest 12: PCA Initialization")
    print("-" * 70)

    np.random.seed(11)
    X = np.random.randn(30, 8)
    tsne = TSNE(n_components=2, perplexity=8, n_iter=50,
                init='pca', random_state=11)
    Z = tsne.fit_transform(X)
    assert Z.shape == (30, 2)
    assert np.all(np.isfinite(Z))
    print(f"   [OK] PCA init: embedding shape={Z.shape}, all finite")
    return True


def test_cluster_separation():
    """t-SNE must separate tight, well-separated 3-blob clusters."""
    print("\nTest 13: Cluster Separation on 3-Blob Data")
    print("-" * 70)

    np.random.seed(12)
    centers = [[8, 0], [-8, 0], [0, 8]]
    X, true = _blobs(centers, n=30, std=0.3, seed=12)

    tsne = TSNE(n_components=2, perplexity=8, n_iter=500,
                learning_rate=150, random_state=12)
    Z = tsne.fit_transform(X)

    # Cluster centroids in embedding
    centroids = np.array([Z[true == k].mean(axis=0) for k in range(3)])
    inter = np.min([np.linalg.norm(centroids[i] - centroids[j])
                    for i in range(3) for j in range(i + 1, 3)])
    intra = np.max([np.mean(np.linalg.norm(Z[true == k] - centroids[k], axis=1))
                    for k in range(3)])

    assert inter > intra * 1.5, \
        f"Clusters not separated: inter={inter:.3f}, intra={intra:.3f}"
    print(f"   [OK] inter-cluster dist={inter:.3f} >> intra={intra:.3f}")
    return True


def test_small_dataset():
    """t-SNE must work on very small datasets (n < perplexity)."""
    print("\nTest 14: Small Dataset (n=10, perplexity=3)")
    print("-" * 70)

    np.random.seed(13)
    X = np.random.randn(10, 4)
    tsne = TSNE(n_components=2, perplexity=3, n_iter=100, random_state=13)
    Z = tsne.fit_transform(X)
    assert Z.shape == (10, 2)
    assert np.all(np.isfinite(Z))
    print(f"   [OK] shape={Z.shape}, all finite")
    return True


def test_kl_history_length():
    """kl_divergence_history_ must have entries at correct intervals."""
    print("\nTest 15: KL History Length")
    print("-" * 70)

    np.random.seed(14)
    X = np.random.randn(20, 3)
    n_iter = 200
    tsne = TSNE(n_components=2, perplexity=5, n_iter=n_iter, random_state=14)
    tsne.fit(X)
    # Logged every 50 iters + final: roughly n_iter//50 + 1 entries
    assert len(tsne.kl_divergence_history_) >= n_iter // 50
    assert tsne.kl_divergence_ == tsne.kl_divergence_history_[-1][1]
    print(f"   [OK] {len(tsne.kl_divergence_history_)} KL entries, "
          f"final KL = {tsne.kl_divergence_:.5f}")
    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    print("Testing t-SNE")
    print("=" * 70)

    tests = [
        test_output_shape,
        test_embedding_attribute,
        test_pairwise_distances_symmetry,
        test_joint_probabilities_symmetry,
        test_joint_probabilities_sum,
        test_student_t_q_sums_to_one,
        test_perplexity_binary_search_accuracy,
        test_kl_divergence_non_negative,
        test_kl_decreases_over_iterations,
        test_gradient_shape,
        test_random_state_reproducibility,
        test_pca_init,
        test_cluster_separation,
        test_small_dataset,
        test_kl_history_length,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"   [FAIL] {test.__name__}: {e}")

    print("\n" + "=" * 70)
    print(f"Tests Passed: {passed}/{len(tests)}")
    print(f"Tests Failed: {failed}/{len(tests)}")
    if failed == 0:
        print("\n[OK] All tests passed successfully!")
        return True
    else:
        print(f"\n[FAIL] {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
