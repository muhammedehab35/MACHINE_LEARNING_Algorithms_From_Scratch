"""
Multinomial Naive Bayes Example
===============================

Demonstrates MultinomialNB classifier for text classification.
This example simulates a simple text classification task similar to
20 newsgroups or topic classification.
"""

import numpy as np
import sys
from pathlib import Path
from collections import Counter

# Add parent directory to path to import naive_bayes module
sys.path.append(str(Path(__file__).parent.parent))
from naive_bayes import MultinomialNB


class SimpleTextVectorizer:
    """
    Simple text vectorizer that converts documents to word count vectors.
    Similar to CountVectorizer from sklearn.
    """

    def __init__(self, max_features=None, min_df=1):
        """
        Initialize vectorizer.

        Parameters
        ----------
        max_features : int or None
            Maximum number of features to keep (most frequent words)
        min_df : int
            Minimum document frequency (words must appear in at least min_df documents)
        """
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary_ = None
        self.feature_names_ = None

    def fit(self, documents):
        """Build vocabulary from documents."""
        # Count document frequency for each word
        word_doc_freq = Counter()
        for doc in documents:
            unique_words = set(doc.lower().split())
            word_doc_freq.update(unique_words)

        # Filter by minimum document frequency
        words = [word for word, freq in word_doc_freq.items() if freq >= self.min_df]

        # Sort by frequency and limit to max_features
        word_freq = Counter()
        for doc in documents:
            word_freq.update(doc.lower().split())

        if self.max_features:
            words = [w for w, _ in word_freq.most_common(self.max_features) if w in words]
        else:
            words = sorted(words)

        # Create vocabulary
        self.vocabulary_ = {word: idx for idx, word in enumerate(words)}
        self.feature_names_ = words

        return self

    def transform(self, documents):
        """Transform documents to word count matrix."""
        if self.vocabulary_ is None:
            raise ValueError("Vectorizer not fitted. Call fit() first.")

        n_docs = len(documents)
        n_features = len(self.vocabulary_)
        X = np.zeros((n_docs, n_features), dtype=np.int64)

        for i, doc in enumerate(documents):
            word_counts = Counter(doc.lower().split())
            for word, count in word_counts.items():
                if word in self.vocabulary_:
                    X[i, self.vocabulary_[word]] = count

        return X

    def fit_transform(self, documents):
        """Fit and transform in one step."""
        self.fit(documents)
        return self.transform(documents)


def create_text_dataset():
    """
    Create a simple text classification dataset.
    Three categories: sports, technology, politics
    """

    # Sports documents
    sports_docs = [
        "football game team player win championship league match season score",
        "basketball player dunk score team championship win season game",
        "soccer goal team match player football league tournament win",
        "tennis player match tournament win game set championship score",
        "baseball team player game season win championship league score",
        "hockey team player game win season championship league match",
        "cricket match team player score win tournament league game",
        "golf player tournament championship win score match game season",
        "football league team player championship season game win match score",
        "basketball championship team player game win season score league",
        "soccer tournament team player match goal football win championship",
        "tennis championship player win match game tournament season",
        "baseball league player team game score championship win season",
        "hockey season team player game championship league win match",
        "cricket player team score match win game league tournament",
    ]

    # Technology documents
    tech_docs = [
        "computer software programming code algorithm data system application",
        "internet web network server cloud computing data technology",
        "smartphone mobile app software technology device system data",
        "artificial intelligence machine learning algorithm data neural network",
        "database system data server software technology application cloud",
        "computer hardware processor memory system technology device chip",
        "software development code programming application system algorithm",
        "cybersecurity network system data encryption software technology",
        "cloud computing server data software technology network system",
        "machine learning algorithm artificial intelligence data neural network",
        "programming language code software development system application",
        "data science algorithm software technology system computing analytics",
        "blockchain technology network data system computing application",
        "quantum computing processor technology system algorithm data",
        "robotics artificial intelligence system technology algorithm machine",
    ]

    # Politics documents
    politics_docs = [
        "government election president policy vote democracy parliament law",
        "senate congress legislation bill vote policy law government",
        "political party election campaign vote candidate democracy",
        "president executive government policy administration law decision",
        "parliament legislation government policy vote law democracy decision",
        "election vote candidate campaign political party democracy government",
        "foreign policy government president international diplomacy relations",
        "legislation congress senate bill law policy vote government",
        "democracy government election vote political party parliament policy",
        "campaign election candidate vote political party government democracy",
        "government policy president administration law executive decision",
        "senate vote legislation bill congress government policy law",
        "political election vote candidate party campaign democracy government",
        "president government executive policy decision law administration",
        "parliament democracy government policy legislation vote law decision",
    ]

    # Combine all documents
    documents = sports_docs + tech_docs + politics_docs
    labels = np.array([0] * len(sports_docs) + [1] * len(tech_docs) + [2] * len(politics_docs))

    # Shuffle
    indices = np.random.permutation(len(documents))
    documents = [documents[i] for i in indices]
    labels = labels[indices]

    return documents, labels, ['Sports', 'Technology', 'Politics']


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


def confusion_matrix(y_true, y_pred, n_classes):
    """Compute confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[int(true), int(pred)] += 1
    return cm


def main():
    """Main function demonstrating MultinomialNB on text classification."""
    print("Multinomial Naive Bayes - Text Classification")
    print("=" * 80)

    # Create dataset
    print("\n1. Creating text classification dataset...")
    documents, labels, class_names = create_text_dataset()
    print(f"   Total documents: {len(documents)}")
    print(f"   Classes: {class_names}")
    print(f"   Class distribution: {np.bincount(labels.astype(int))}")

    # Split dataset
    print("\n2. Splitting dataset (70% train, 30% test)...")
    docs_train, docs_test, y_train, y_test = train_test_split(
        documents, labels, test_size=0.3, random_state=42
    )
    print(f"   Training documents: {len(docs_train)}")
    print(f"   Test documents: {len(docs_test)}")

    # Vectorize text
    print("\n3. Converting text to word count vectors...")
    vectorizer = SimpleTextVectorizer(max_features=100)
    X_train = vectorizer.fit_transform(docs_train)
    X_test = vectorizer.transform(docs_test)
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"   Training matrix shape: {X_train.shape}")
    print(f"   Test matrix shape: {X_test.shape}")
    print(f"   Sample words: {vectorizer.feature_names_[:10]}")

    # Train classifier with different alpha values
    print("\n4. Training Multinomial Naive Bayes classifiers with different alpha...")
    print("-" * 80)

    alphas = [0.01, 0.1, 1.0, 10.0]
    results = []

    for alpha in alphas:
        mnb = MultinomialNB(alpha=alpha)
        mnb.fit(X_train, y_train)

        train_acc = mnb.score(X_train, y_train)
        test_acc = mnb.score(X_test, y_test)

        results.append((alpha, train_acc, test_acc))
        print(f"   Alpha={alpha:6.2f}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

    # Use best alpha
    best_alpha = max(results, key=lambda x: x[2])[0]
    print(f"\n   Best alpha: {best_alpha}")

    # Train final model
    print("\n5. Training final model...")
    mnb = MultinomialNB(alpha=best_alpha)
    mnb.fit(X_train, y_train)

    # Display learned parameters
    print("\n6. Learned Parameters:")
    print("-" * 80)
    print(f"   Class log priors: {mnb.class_prior_}")
    print(f"   Prior probabilities:")
    for i, class_name in enumerate(class_names):
        print(f"      {class_name}: {np.exp(mnb.class_prior_[i]):.4f}")

    # Top features per class
    print("\n7. Top 10 words per class:")
    print("-" * 80)
    for i, class_name in enumerate(class_names):
        # Get feature log probabilities for this class
        log_probs = mnb.feature_log_prob_[i, :]
        # Get indices of top features
        top_indices = np.argsort(log_probs)[-10:][::-1]
        top_words = [vectorizer.feature_names_[idx] for idx in top_indices]
        top_probs = [np.exp(log_probs[idx]) for idx in top_indices]

        print(f"\n   {class_name}:")
        for word, prob in zip(top_words, top_probs):
            print(f"      {word:15s}: {prob:.4f}")

    # Make predictions on test set
    print("\n8. Test Set Performance:")
    print("-" * 80)
    y_test_pred = mnb.predict(X_test)
    test_accuracy = mnb.score(X_test, y_test)
    print(f"   Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    # Confusion matrix
    print("\n9. Confusion Matrix (Test Set):")
    print("-" * 80)
    cm = confusion_matrix(y_test, y_test_pred, len(class_names))
    print("   Predicted Class")
    print("              Sports  Tech  Politics")
    print("   Actual  +-------------------------")
    for i, (row, class_name) in enumerate(zip(cm, class_names)):
        print(f"   {class_name:8s} | {row[0]:4d}   {row[1]:4d}   {row[2]:4d}")

    # Classification report
    print("\n10. Per-class Metrics:")
    print("-" * 80)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)

    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"{class_name:<15} {precision:<12.3f} {recall:<12.3f} {f1:<12.3f}")

    # Predict on new documents
    print("\n11. Predicting on New Documents:")
    print("-" * 80)

    new_docs = [
        "football team won the championship match yesterday",
        "new programming language released for software development",
        "president announced new government policy today",
        "basketball player scored winning points in final game",
        "artificial intelligence algorithm improves data processing",
    ]

    X_new = vectorizer.transform(new_docs)
    predictions = mnb.predict(X_new)
    probabilities = mnb.predict_proba(X_new)

    for i, (doc, pred, probs) in enumerate(zip(new_docs, predictions, probabilities)):
        print(f"\n   Document {i+1}: \"{doc}\"")
        print(f"   Predicted: {class_names[int(pred)]}")
        print(f"   Probabilities:")
        for j, class_name in enumerate(class_names):
            print(f"      {class_name}: {probs[j]:.4f}")

    # Analyze misclassifications
    print("\n12. Misclassification Analysis:")
    print("-" * 80)
    misclassified = y_test != y_test_pred
    n_errors = np.sum(misclassified)
    print(f"   Total misclassifications: {n_errors} out of {len(y_test)}")

    if n_errors > 0:
        print(f"   Error rate: {n_errors / len(y_test) * 100:.2f}%")
        print("\n   Misclassified documents:")
        error_indices = np.where(misclassified)[0]
        for idx in error_indices[:3]:  # Show first 3 errors
            print(f"\n      Document: \"{docs_test[idx]}\"")
            print(f"      True class: {class_names[int(y_test[idx])]}")
            print(f"      Predicted: {class_names[int(y_test_pred[idx])]}")
            probs = mnb.predict_proba(X_test[idx:idx+1])[0]
            print(f"      Probabilities: {', '.join([f'{cn}: {p:.3f}' for cn, p in zip(class_names, probs)])}")

    print("\n" + "=" * 80)
    print("Example completed successfully!")


if __name__ == "__main__":
    np.random.seed(42)
    main()
