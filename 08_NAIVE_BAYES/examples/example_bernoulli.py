"""
Bernoulli Naive Bayes Example
=============================

Demonstrates BernoulliNB classifier for binary document classification.
This example shows how BernoulliNB works with binary features (word presence/absence).
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import naive_bayes module
sys.path.append(str(Path(__file__).parent.parent))
from naive_bayes import BernoulliNB


class BinaryTextVectorizer:
    """
    Binary text vectorizer that converts documents to binary vectors.
    Each feature indicates presence (1) or absence (0) of a word.
    """

    def __init__(self, max_features=None):
        """
        Initialize vectorizer.

        Parameters
        ----------
        max_features : int or None
            Maximum number of features to keep (most frequent words)
        """
        self.max_features = max_features
        self.vocabulary_ = None
        self.feature_names_ = None

    def fit(self, documents):
        """Build vocabulary from documents."""
        from collections import Counter

        # Count word frequencies across all documents
        word_freq = Counter()
        for doc in documents:
            word_freq.update(doc.lower().split())

        # Select most frequent words
        if self.max_features:
            words = [w for w, _ in word_freq.most_common(self.max_features)]
        else:
            words = sorted(word_freq.keys())

        # Create vocabulary
        self.vocabulary_ = {word: idx for idx, word in enumerate(words)}
        self.feature_names_ = words

        return self

    def transform(self, documents):
        """Transform documents to binary vectors."""
        if self.vocabulary_ is None:
            raise ValueError("Vectorizer not fitted. Call fit() first.")

        n_docs = len(documents)
        n_features = len(self.vocabulary_)
        X = np.zeros((n_docs, n_features), dtype=np.float64)

        for i, doc in enumerate(documents):
            words = set(doc.lower().split())
            for word in words:
                if word in self.vocabulary_:
                    X[i, self.vocabulary_[word]] = 1.0

        return X

    def fit_transform(self, documents):
        """Fit and transform in one step."""
        self.fit(documents)
        return self.transform(documents)


def create_binary_document_dataset():
    """
    Create a binary document classification dataset.
    Two categories: positive and negative reviews
    """

    # Positive reviews
    positive_docs = [
        "excellent movie great acting wonderful story loved it",
        "amazing film fantastic performance brilliant direction loved every minute",
        "wonderful experience great story excellent cast highly recommend",
        "superb movie outstanding acting loved the plot",
        "fantastic film great direction wonderful cinematography enjoyed it",
        "excellent performance brilliant story amazing movie",
        "loved this film great acting wonderful experience",
        "amazing movie fantastic story excellent cast",
        "outstanding film wonderful performance great direction",
        "brilliant movie excellent acting loved every scene",
        "great film wonderful story fantastic experience",
        "excellent movie amazing performance highly recommend",
        "superb acting brilliant story loved this film",
        "fantastic movie great cast wonderful direction",
        "amazing film excellent story loved it very much",
        "wonderful movie great performance outstanding direction",
        "brilliant film fantastic acting excellent experience",
        "loved this movie amazing story great cast",
        "excellent film wonderful acting superb direction",
        "great movie fantastic performance brilliant story",
    ]

    # Negative reviews
    negative_docs = [
        "terrible movie bad acting awful story hated it",
        "boring film poor performance weak direction waste of time",
        "horrible experience bad story terrible cast avoid this",
        "awful movie disappointing acting hated the plot",
        "poor film bad direction terrible cinematography boring",
        "terrible performance weak story awful movie",
        "hated this film bad acting horrible experience",
        "boring movie poor story terrible cast",
        "disappointing film weak performance bad direction",
        "awful movie terrible acting hated every scene",
        "bad film poor story boring experience",
        "terrible movie disappointing performance avoid this",
        "poor acting weak story hated this film",
        "boring movie bad cast terrible direction",
        "awful film poor story hated it very much",
        "horrible movie bad performance disappointing direction",
        "weak film terrible acting poor experience",
        "hated this movie awful story bad cast",
        "terrible film poor acting disappointing direction",
        "bad movie boring performance weak story",
    ]

    # Combine all documents
    documents = positive_docs + negative_docs
    labels = np.array([1] * len(positive_docs) + [0] * len(negative_docs))

    # Shuffle
    indices = np.random.permutation(len(documents))
    documents = [documents[i] for i in indices]
    labels = labels[indices]

    return documents, labels, ['Negative', 'Positive']


def train_test_split(X, y, test_size=0.3, random_state=None):
    """Simple train-test split."""
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    n_test = int(n_samples * test_size)

    indices = np.random.permutation(n_samples)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    if isinstance(X, np.ndarray):
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    else:
        return [X[i] for i in train_idx], [X[i] for i in test_idx], y[train_idx], y[test_idx]


def confusion_matrix(y_true, y_pred):
    """Compute confusion matrix for binary classification."""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]])


def main():
    """Main function demonstrating BernoulliNB on document classification."""
    print("Bernoulli Naive Bayes - Binary Document Classification")
    print("=" * 80)

    # Create dataset
    print("\n1. Creating movie review classification dataset...")
    documents, labels, class_names = create_binary_document_dataset()
    print(f"   Total documents: {len(documents)}")
    print(f"   Classes: {class_names}")
    print(f"   Class distribution: {np.bincount(labels.astype(int))}")

    # Show sample documents
    print("\n2. Sample documents:")
    print("-" * 80)
    for i in range(2):
        print(f"   Positive review: \"{documents[labels == 1][i]}\"")
    for i in range(2):
        print(f"   Negative review: \"{documents[labels == 0][i]}\"")

    # Split dataset
    print("\n3. Splitting dataset (70% train, 30% test)...")
    docs_train, docs_test, y_train, y_test = train_test_split(
        documents, labels, test_size=0.3, random_state=42
    )
    print(f"   Training documents: {len(docs_train)}")
    print(f"   Test documents: {len(docs_test)}")

    # Vectorize text to binary features
    print("\n4. Converting text to binary feature vectors...")
    vectorizer = BinaryTextVectorizer(max_features=50)
    X_train = vectorizer.fit_transform(docs_train)
    X_test = vectorizer.transform(docs_test)
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"   Training matrix shape: {X_train.shape}")
    print(f"   Test matrix shape: {X_test.shape}")
    print(f"   Sample words: {vectorizer.feature_names_[:10]}")

    # Show sparsity
    sparsity_train = np.mean(X_train == 0)
    sparsity_test = np.mean(X_test == 0)
    print(f"   Training sparsity: {sparsity_train:.2%}")
    print(f"   Test sparsity: {sparsity_test:.2%}")

    # Train classifier with different alpha and binarize values
    print("\n5. Training Bernoulli Naive Bayes classifiers...")
    print("-" * 80)

    # Test different alpha values
    print("\n   Testing different smoothing parameters (alpha):")
    alphas = [0.01, 0.1, 1.0, 10.0]
    results = []

    for alpha in alphas:
        bnb = BernoulliNB(alpha=alpha, binarize=None)  # Already binary
        bnb.fit(X_train, y_train)

        train_acc = bnb.score(X_train, y_train)
        test_acc = bnb.score(X_test, y_test)

        results.append((alpha, train_acc, test_acc))
        print(f"      Alpha={alpha:6.2f}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

    # Use best alpha
    best_alpha = max(results, key=lambda x: x[2])[0]
    print(f"\n   Best alpha: {best_alpha}")

    # Test binarization threshold on non-binary data
    print("\n   Testing binarization threshold (on count data):")
    # Create count-based features
    from collections import Counter
    X_train_counts = np.zeros_like(X_train)
    for i, doc in enumerate(docs_train):
        word_counts = Counter(doc.lower().split())
        for word, count in word_counts.items():
            if word in vectorizer.vocabulary_:
                X_train_counts[i, vectorizer.vocabulary_[word]] = count

    for threshold in [0.0, 0.5, 1.0]:
        bnb = BernoulliNB(alpha=best_alpha, binarize=threshold)
        bnb.fit(X_train_counts, y_train)
        acc = bnb.score(X_train_counts, y_train)
        print(f"      Binarize threshold={threshold}: Accuracy={acc:.4f}")

    # Train final model
    print("\n6. Training final model...")
    bnb = BernoulliNB(alpha=best_alpha, binarize=None)
    bnb.fit(X_train, y_train)

    # Display learned parameters
    print("\n7. Learned Parameters:")
    print("-" * 80)
    print(f"   Class log priors: {bnb.class_prior_}")
    print(f"   Prior probabilities:")
    for i, class_name in enumerate(class_names):
        print(f"      {class_name}: {np.exp(bnb.class_prior_[i]):.4f}")

    # Most indicative features per class
    print("\n8. Most indicative words per class:")
    print("-" * 80)

    # For each class, find words with highest probability
    for i, class_name in enumerate(class_names):
        log_probs = bnb.feature_log_prob_[i, :]
        probs = np.exp(log_probs)

        # Get indices of top features
        top_indices = np.argsort(probs)[-10:][::-1]
        top_words = [vectorizer.feature_names_[idx] for idx in top_indices]
        top_probs = [probs[idx] for idx in top_indices]

        print(f"\n   {class_name} (P(word present|{class_name})):")
        for word, prob in zip(top_words, top_probs):
            print(f"      {word:15s}: {prob:.4f}")

    # Features most different between classes
    print("\n9. Most discriminative features:")
    print("-" * 80)
    prob_diff = np.exp(bnb.feature_log_prob_[1, :]) - np.exp(bnb.feature_log_prob_[0, :])

    # Words indicating positive
    top_positive_idx = np.argsort(prob_diff)[-10:][::-1]
    print("\n   Words indicating POSITIVE:")
    for idx in top_positive_idx:
        word = vectorizer.feature_names_[idx]
        p_pos = np.exp(bnb.feature_log_prob_[1, idx])
        p_neg = np.exp(bnb.feature_log_prob_[0, idx])
        print(f"      {word:15s}: P(word|pos)={p_pos:.4f}, P(word|neg)={p_neg:.4f}, diff={prob_diff[idx]:.4f}")

    # Words indicating negative
    top_negative_idx = np.argsort(prob_diff)[:10]
    print("\n   Words indicating NEGATIVE:")
    for idx in top_negative_idx:
        word = vectorizer.feature_names_[idx]
        p_pos = np.exp(bnb.feature_log_prob_[1, idx])
        p_neg = np.exp(bnb.feature_log_prob_[0, idx])
        print(f"      {word:15s}: P(word|pos)={p_pos:.4f}, P(word|neg)={p_neg:.4f}, diff={prob_diff[idx]:.4f}")

    # Make predictions on test set
    print("\n10. Test Set Performance:")
    print("-" * 80)
    y_test_pred = bnb.predict(X_test)
    test_accuracy = bnb.score(X_test, y_test)
    print(f"   Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    # Confusion matrix
    print("\n11. Confusion Matrix (Test Set):")
    print("-" * 80)
    cm = confusion_matrix(y_test, y_test_pred)
    print("                Predicted")
    print("                Neg    Pos")
    print("   Actual  +---------------")
    print(f"      Neg  | {cm[0, 0]:4d}  {cm[0, 1]:4d}")
    print(f"      Pos  | {cm[1, 0]:4d}  {cm[1, 1]:4d}")

    # Calculate metrics
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\n12. Detailed Metrics:")
    print("-" * 80)
    print(f"   Precision (Positive): {precision:.4f}")
    print(f"   Recall (Positive):    {recall:.4f}")
    print(f"   F1-Score (Positive):  {f1:.4f}")
    print(f"   Specificity:          {specificity:.4f}")

    # Predict on new documents
    print("\n13. Predicting on New Documents:")
    print("-" * 80)

    new_docs = [
        "excellent movie loved the acting and story",
        "terrible film boring and disappointing",
        "great performance wonderful experience",
        "awful movie waste of time hated it",
        "fantastic story brilliant direction amazing",
    ]

    X_new = vectorizer.transform(new_docs)
    predictions = bnb.predict(X_new)
    probabilities = bnb.predict_proba(X_new)

    for i, (doc, pred, probs) in enumerate(zip(new_docs, predictions, probabilities)):
        print(f"\n   Document {i+1}: \"{doc}\"")
        print(f"   Predicted: {class_names[int(pred)]}")
        print(f"   Probabilities: Negative={probs[0]:.4f}, Positive={probs[1]:.4f}")
        print(f"   Confidence: {max(probs):.4f}")

    # Feature presence analysis
    print("\n14. Feature Presence Analysis for New Documents:")
    print("-" * 80)
    for i, doc in enumerate(new_docs[:2]):
        print(f"\n   Document: \"{doc}\"")
        print(f"   Present features:")
        present_features = np.where(X_new[i] > 0)[0]
        for feat_idx in present_features[:10]:
            word = vectorizer.feature_names_[feat_idx]
            p_neg = np.exp(bnb.feature_log_prob_[0, feat_idx])
            p_pos = np.exp(bnb.feature_log_prob_[1, feat_idx])
            print(f"      {word:15s}: P(word|neg)={p_neg:.4f}, P(word|pos)={p_pos:.4f}")

    print("\n" + "=" * 80)
    print("Example completed successfully!")


if __name__ == "__main__":
    np.random.seed(42)
    main()
