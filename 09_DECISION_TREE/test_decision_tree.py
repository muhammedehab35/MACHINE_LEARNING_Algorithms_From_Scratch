"""
Test Script for Decision Tree Classifier
========================================

Simple tests to verify the implementation works correctly.
"""

import numpy as np
from decision_tree import DecisionTreeClassifier


def test_binary_classification():
    """Test basic binary classification."""
    print("Test 1: Binary Classification")
    print("-" * 50)

    # Simple XOR-like problem
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [0.1, 0.1],
        [0.9, 0.1],
        [0.1, 0.9],
        [0.9, 0.9]
    ])
    y = np.array([0, 1, 1, 0, 0, 1, 1, 0])

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)

    predictions = clf.predict(X)
    accuracy = clf.score(X, y)

    print(f"Training accuracy: {accuracy:.4f}")
    print(f"Predictions: {predictions}")
    print(f"True labels: {y}")
    print(f"Match: {np.array_equal(predictions, y)}")

    assert accuracy >= 0.75, "Accuracy too low"
    print("[PASS] Test passed\n")


def test_multiclass_classification():
    """Test multi-class classification."""
    print("Test 2: Multi-class Classification")
    print("-" * 50)

    # Three clusters
    np.random.seed(42)
    n_per_class = 20

    X0 = np.random.randn(n_per_class, 2) + [-2, -2]
    X1 = np.random.randn(n_per_class, 2) + [2, -2]
    X2 = np.random.randn(n_per_class, 2) + [0, 2]

    X = np.vstack([X0, X1, X2])
    y = np.array([0]*n_per_class + [1]*n_per_class + [2]*n_per_class)

    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X, y)

    accuracy = clf.score(X, y)
    print(f"Training accuracy: {accuracy:.4f}")
    print(f"Number of classes: {clf.n_classes_}")
    print(f"Classes: {clf.classes_}")

    assert clf.n_classes_ == 3, "Wrong number of classes"
    assert accuracy >= 0.80, "Accuracy too low"
    print("[PASS] Test passed\n")


def test_different_criteria():
    """Test different split criteria."""
    print("Test 3: Different Split Criteria")
    print("-" * 50)

    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    criteria = ['gini', 'entropy', 'error']
    results = {}

    for criterion in criteria:
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=3, random_state=42)
        clf.fit(X, y)
        acc = clf.score(X, y)
        results[criterion] = acc
        print(f"{criterion:10s}: {acc:.4f}")

    # All criteria should work
    assert all(acc > 0.5 for acc in results.values()), "Some criteria failed"
    print("[PASS] Test passed\n")


def test_pruning():
    """Test pre-pruning parameters."""
    print("Test 4: Pre-pruning")
    print("-" * 50)

    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0]**2 + X[:, 1]**2 > 1).astype(int)

    # No pruning
    clf_full = DecisionTreeClassifier(random_state=42)
    clf_full.fit(X, y)

    # With pruning
    clf_pruned = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
    clf_pruned.fit(X, y)

    print(f"Full tree - Depth: {clf_full.get_depth()}, Leaves: {clf_full.get_n_leaves()}")
    print(f"Pruned tree - Depth: {clf_pruned.get_depth()}, Leaves: {clf_pruned.get_n_leaves()}")

    assert clf_pruned.get_depth() <= 3, "max_depth not working"
    assert clf_full.get_depth() > clf_pruned.get_depth(), "Pruning didn't reduce depth"
    print("[PASS] Test passed\n")


def test_feature_importance():
    """Test feature importance calculation."""
    print("Test 5: Feature Importance")
    print("-" * 50)

    np.random.seed(42)

    # Create data where feature 0 is most important
    X = np.random.randn(200, 3)
    y = (X[:, 0] > 0).astype(int)  # Only feature 0 matters

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)

    importances = clf.feature_importances_
    print(f"Feature importances: {importances}")
    print(f"Sum of importances: {np.sum(importances):.4f}")

    # Feature 0 should be most important
    assert np.argmax(importances) == 0, "Feature 0 should be most important"
    assert np.abs(np.sum(importances) - 1.0) < 1e-6, "Importances should sum to 1"
    print("[PASS] Test passed\n")


def test_predict_proba():
    """Test probability predictions."""
    print("Test 6: Probability Predictions")
    print("-" * 50)

    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    y = np.array([0, 1, 1, 0])

    clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf.fit(X, y)

    proba = clf.predict_proba(X)
    print(f"Probabilities shape: {proba.shape}")
    print(f"Sample probabilities:\n{proba[:2]}")

    # Check properties
    assert proba.shape == (len(X), 2), "Wrong probability shape"
    assert np.allclose(proba.sum(axis=1), 1.0), "Probabilities don't sum to 1"
    assert np.all(proba >= 0) and np.all(proba <= 1), "Probabilities out of range"
    print("[PASS] Test passed\n")


def test_tree_structure():
    """Test tree structure properties."""
    print("Test 7: Tree Structure")
    print("-" * 50)

    X = np.array([[i] for i in range(10)])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf.fit(X, y)

    depth = clf.get_depth()
    n_leaves = clf.get_n_leaves()

    print(f"Tree depth: {depth}")
    print(f"Number of leaves: {n_leaves}")
    print("\nTree structure:")
    clf.print_tree()

    assert depth <= 2, "Tree deeper than max_depth"
    assert n_leaves >= 2, "Tree should have at least 2 leaves"
    print("\n[PASS] Test passed\n")


def test_edge_cases():
    """Test edge cases."""
    print("Test 8: Edge Cases")
    print("-" * 50)

    # Single sample per class
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    assert clf.score(X, y) == 1.0, "Failed on minimal data"
    print("[PASS] Minimal data")

    # Pure node from start
    X = np.array([[i] for i in range(10)])
    y = np.zeros(10)
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    assert clf.get_n_leaves() == 1, "Should be single leaf for pure data"
    print("[PASS] Pure data")

    # High dimensional data
    X = np.random.randn(50, 20)
    y = (X[:, 0] > 0).astype(int)
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    assert clf.score(X, y) > 0.5, "Failed on high-dimensional data"
    print("[PASS] High-dimensional data")

    print("\n[PASS] All edge cases passed\n")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("Decision Tree Classifier - Test Suite")
    print("=" * 70)
    print()

    tests = [
        test_binary_classification,
        test_multiclass_classification,
        test_different_criteria,
        test_pruning,
        test_feature_importance,
        test_predict_proba,
        test_tree_structure,
        test_edge_cases
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] Test failed with error: {e}\n")
            failed += 1

    print("=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("\nAll tests passed successfully!")
    else:
        print(f"\n{failed} test(s) failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
