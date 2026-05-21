"""
Naive Bayes Variants Comparison
===============================

Compares all three Naive Bayes variants (GaussianNB, MultinomialNB, BernoulliNB)
on different types of datasets to demonstrate when each is most appropriate.
"""

import numpy as np
import sys
from pathlib import Path
import time

# Add parent directory to path to import naive_bayes module
sys.path.append(str(Path(__file__).parent.parent))
from naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


def create_continuous_dataset():
    """Create a dataset with continuous features (best for GaussianNB)."""
    np.random.seed(42)
    n_samples_per_class = 200

    # Class 0: centered at (2, 2)
    X0 = np.random.randn(n_samples_per_class, 2) * 0.5 + np.array([2, 2])

    # Class 1: centered at (-2, 2)
    X1 = np.random.randn(n_samples_per_class, 2) * 0.5 + np.array([-2, 2])

    # Class 2: centered at (0, -2)
    X2 = np.random.randn(n_samples_per_class, 2) * 0.5 + np.array([0, -2])

    X = np.vstack([X0, X1, X2])
    y = np.hstack([np.zeros(n_samples_per_class),
                   np.ones(n_samples_per_class),
                   np.full(n_samples_per_class, 2)])

    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def create_count_dataset():
    """Create a dataset with count features (best for MultinomialNB)."""
    np.random.seed(42)
    n_samples_per_class = 200

    # Class 0: high counts in first features
    X0 = np.random.poisson(lam=[10, 8, 2, 1, 0.5], size=(n_samples_per_class, 5))

    # Class 1: high counts in middle features
    X1 = np.random.poisson(lam=[1, 3, 10, 8, 2], size=(n_samples_per_class, 5))

    # Class 2: high counts in last features
    X2 = np.random.poisson(lam=[0.5, 1, 2, 8, 10], size=(n_samples_per_class, 5))

    X = np.vstack([X0, X1, X2])
    y = np.hstack([np.zeros(n_samples_per_class),
                   np.ones(n_samples_per_class),
                   np.full(n_samples_per_class, 2)])

    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def create_binary_dataset():
    """Create a dataset with binary features (best for BernoulliNB)."""
    np.random.seed(42)
    n_samples_per_class = 200

    # Class 0: high probability in first features
    X0 = np.random.binomial(1, p=[0.9, 0.8, 0.3, 0.2, 0.1], size=(n_samples_per_class, 5))

    # Class 1: high probability in middle features
    X1 = np.random.binomial(1, p=[0.2, 0.4, 0.9, 0.8, 0.3], size=(n_samples_per_class, 5))

    # Class 2: high probability in last features
    X2 = np.random.binomial(1, p=[0.1, 0.2, 0.3, 0.8, 0.9], size=(n_samples_per_class, 5))

    X = np.vstack([X0, X1, X2]).astype(np.float64)
    y = np.hstack([np.zeros(n_samples_per_class),
                   np.ones(n_samples_per_class),
                   np.full(n_samples_per_class, 2)])

    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def train_test_split(X, y, test_size=0.3, random_state=None):
    """Simple train-test split."""
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    n_test = int(n_samples * test_size)

    indices = np.random.permutation(n_samples)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def evaluate_classifier(clf, X_train, X_test, y_train, y_test, name):
    """Train and evaluate a classifier."""
    # Train
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Predict
    start_time = time.time()
    y_pred = clf.predict(X_test)
    predict_time = time.time() - start_time

    # Evaluate
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    return {
        'name': name,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_time': train_time,
        'predict_time': predict_time
    }


def print_results_table(results):
    """Print results in a formatted table."""
    print(f"{'Classifier':<20} {'Train Acc':<12} {'Test Acc':<12} {'Train Time':<12} {'Predict Time':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<20} {r['train_acc']:<12.4f} {r['test_acc']:<12.4f} "
              f"{r['train_time']:<12.6f} {r['predict_time']:<12.6f}")


def main():
    """Main function comparing all three Naive Bayes variants."""
    print("Naive Bayes Variants Comparison")
    print("=" * 80)

    # ============================================================
    # Test 1: Continuous Features Dataset
    # ============================================================
    print("\n" + "=" * 80)
    print("TEST 1: Continuous Features Dataset (Gaussian-distributed)")
    print("=" * 80)
    print("\nThis dataset has continuous features drawn from Gaussian distributions.")
    print("Expected: GaussianNB should perform best.")

    X, y = create_continuous_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(f"\nDataset shape: {X.shape}")
    print(f"Classes: {np.unique(y)}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"\nFeature statistics:")
    print(f"  Mean: {X.mean(axis=0)}")
    print(f"  Std:  {X.std(axis=0)}")

    results = []

    # GaussianNB
    print("\nTraining GaussianNB...")
    gnb = GaussianNB(var_smoothing=1e-9)
    results.append(evaluate_classifier(gnb, X_train, X_test, y_train, y_test, "GaussianNB"))

    # MultinomialNB (needs non-negative features)
    print("Training MultinomialNB...")
    # Shift data to make it non-negative
    X_train_pos = X_train - X_train.min() + 1e-10
    X_test_pos = X_test - X_test.min() + 1e-10
    mnb = MultinomialNB(alpha=1.0)
    results.append(evaluate_classifier(mnb, X_train_pos, X_test_pos, y_train, y_test, "MultinomialNB"))

    # BernoulliNB (binarize continuous features)
    print("Training BernoulliNB...")
    bnb = BernoulliNB(alpha=1.0, binarize=0.0)
    results.append(evaluate_classifier(bnb, X_train, X_test, y_train, y_test, "BernoulliNB"))

    print("\n" + "-" * 80)
    print("Results:")
    print_results_table(results)
    print("\nAnalysis: GaussianNB performs best because it's designed for continuous")
    print("          features with Gaussian distributions.")

    # ============================================================
    # Test 2: Count Features Dataset
    # ============================================================
    print("\n" + "=" * 80)
    print("TEST 2: Count Features Dataset (Discrete counts)")
    print("=" * 80)
    print("\nThis dataset has discrete count features (e.g., word counts).")
    print("Expected: MultinomialNB should perform best.")

    X, y = create_count_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(f"\nDataset shape: {X.shape}")
    print(f"Classes: {np.unique(y)}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"\nFeature statistics:")
    print(f"  Mean: {X.mean(axis=0)}")
    print(f"  Max:  {X.max(axis=0)}")
    print(f"  Unique values per feature: {[len(np.unique(X[:, i])) for i in range(X.shape[1])]}")

    results = []

    # GaussianNB
    print("\nTraining GaussianNB...")
    gnb = GaussianNB(var_smoothing=1e-9)
    results.append(evaluate_classifier(gnb, X_train, X_test, y_train, y_test, "GaussianNB"))

    # MultinomialNB
    print("Training MultinomialNB...")
    mnb = MultinomialNB(alpha=1.0)
    results.append(evaluate_classifier(mnb, X_train, X_test, y_train, y_test, "MultinomialNB"))

    # BernoulliNB (binarize counts)
    print("Training BernoulliNB...")
    bnb = BernoulliNB(alpha=1.0, binarize=0.5)
    results.append(evaluate_classifier(bnb, X_train, X_test, y_train, y_test, "BernoulliNB"))

    print("\n" + "-" * 80)
    print("Results:")
    print_results_table(results)
    print("\nAnalysis: MultinomialNB performs best because it's designed for count data")
    print("          and properly models the multinomial distribution.")

    # ============================================================
    # Test 3: Binary Features Dataset
    # ============================================================
    print("\n" + "=" * 80)
    print("TEST 3: Binary Features Dataset (Presence/Absence)")
    print("=" * 80)
    print("\nThis dataset has binary features (0 or 1).")
    print("Expected: BernoulliNB should perform best or tie with MultinomialNB.")

    X, y = create_binary_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(f"\nDataset shape: {X.shape}")
    print(f"Classes: {np.unique(y)}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"\nFeature statistics:")
    print(f"  Proportion of 1s per feature: {X.mean(axis=0)}")
    print(f"  Unique values: {np.unique(X)}")

    results = []

    # GaussianNB
    print("\nTraining GaussianNB...")
    gnb = GaussianNB(var_smoothing=1e-9)
    results.append(evaluate_classifier(gnb, X_train, X_test, y_train, y_test, "GaussianNB"))

    # MultinomialNB
    print("Training MultinomialNB...")
    mnb = MultinomialNB(alpha=1.0)
    results.append(evaluate_classifier(mnb, X_train, X_test, y_train, y_test, "MultinomialNB"))

    # BernoulliNB
    print("Training BernoulliNB...")
    bnb = BernoulliNB(alpha=1.0, binarize=None)
    results.append(evaluate_classifier(bnb, X_train, X_test, y_train, y_test, "BernoulliNB"))

    print("\n" + "-" * 80)
    print("Results:")
    print_results_table(results)
    print("\nAnalysis: BernoulliNB and MultinomialNB perform similarly because binary data")
    print("          can be modeled by both. BernoulliNB explicitly models presence/absence.")

    # ============================================================
    # Test 4: Mixed Features (All classifiers on same data)
    # ============================================================
    print("\n" + "=" * 80)
    print("TEST 4: Comparison on Mixed Features Dataset")
    print("=" * 80)
    print("\nTesting all classifiers on the same dataset with various feature types.")

    # Create a more complex dataset
    np.random.seed(42)
    n_samples = 600

    # Mix of continuous, count, and binary features
    continuous = np.random.randn(n_samples, 3)
    counts = np.random.poisson(lam=5, size=(n_samples, 3))
    binary = np.random.binomial(1, 0.5, size=(n_samples, 3))

    X = np.hstack([continuous, counts, binary])

    # Create labels based on feature combinations
    y = ((X[:, 0] > 0) & (X[:, 3] > 5)).astype(int) + \
        ((X[:, 1] > 0) & (X[:, 6] == 1)).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(f"\nDataset shape: {X.shape}")
    print(f"Feature types: 3 continuous, 3 count, 3 binary")
    print(f"Classes: {np.unique(y)}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    results = []

    # GaussianNB
    print("\nTraining GaussianNB...")
    gnb = GaussianNB(var_smoothing=1e-9)
    results.append(evaluate_classifier(gnb, X_train, X_test, y_train, y_test, "GaussianNB"))

    # MultinomialNB (shift to positive)
    print("Training MultinomialNB...")
    X_train_pos = X_train - X_train.min() + 1e-10
    X_test_pos = X_test - X_test.min() + 1e-10
    mnb = MultinomialNB(alpha=1.0)
    results.append(evaluate_classifier(mnb, X_train_pos, X_test_pos, y_train, y_test, "MultinomialNB"))

    # BernoulliNB (with binarization)
    print("Training BernoulliNB...")
    bnb = BernoulliNB(alpha=1.0, binarize=0.0)
    results.append(evaluate_classifier(bnb, X_train, X_test, y_train, y_test, "BernoulliNB"))

    print("\n" + "-" * 80)
    print("Results:")
    print_results_table(results)
    print("\nAnalysis: Performance depends on how well each classifier's assumptions")
    print("          match the actual data distribution.")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 80)
    print("SUMMARY: When to Use Each Variant")
    print("=" * 80)
    print("""
1. GaussianNB:
   - Best for: Continuous features that follow Gaussian (normal) distribution
   - Examples: Real-valued measurements, sensor data, physical measurements
   - Pros: Natural for continuous data, no preprocessing needed
   - Cons: Assumes Gaussian distribution, sensitive to outliers

2. MultinomialNB:
   - Best for: Discrete count features
   - Examples: Word counts in text, frequency data, histogram features
   - Pros: Handles count data naturally, good for text classification
   - Cons: Requires non-negative features, not suitable for negative values

3. BernoulliNB:
   - Best for: Binary features (presence/absence)
   - Examples: Document classification (word present/absent), binary attributes
   - Pros: Explicitly models binary features, efficient for sparse data
   - Cons: Information loss when binarizing count data, assumes independence

Key Takeaways:
- Choose based on your feature types, not just performance
- All variants assume feature independence (the "naive" assumption)
- Preprocessing matters: normalize for Gaussian, ensure non-negative for Multinomial
- BernoulliNB can be more stable than MultinomialNB for binary features
- Consider computational efficiency: all variants are fast, but BernoulliNB
  is often fastest for sparse binary data
    """)

    print("=" * 80)
    print("Comparison completed successfully!")


if __name__ == "__main__":
    main()
