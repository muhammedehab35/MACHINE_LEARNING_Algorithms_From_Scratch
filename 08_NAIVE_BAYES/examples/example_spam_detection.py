"""
Email Spam Detection using Multinomial Naive Bayes
==================================================

A real-world example demonstrating spam email classification using MultinomialNB.
This example simulates a practical spam detection system with feature engineering
and performance analysis.
"""

import numpy as np
import sys
from pathlib import Path
import re
from collections import Counter

# Add parent directory to path to import naive_bayes module
sys.path.append(str(Path(__file__).parent.parent))
from naive_bayes import MultinomialNB


class EmailFeatureExtractor:
    """
    Extract features from email text for spam detection.
    Combines word frequencies with additional spam-indicative features.
    """

    def __init__(self, max_features=100):
        """
        Initialize feature extractor.

        Parameters
        ----------
        max_features : int
            Maximum number of word features to keep
        """
        self.max_features = max_features
        self.vocabulary_ = None
        self.feature_names_ = None
        self.additional_features_ = [
            'num_exclamations',
            'num_questions',
            'num_capitals',
            'has_money_symbol',
            'num_digits',
            'avg_word_length',
            'num_words'
        ]

    def _extract_text_features(self, text):
        """Extract additional features from text."""
        features = {}

        # Count punctuation
        features['num_exclamations'] = text.count('!')
        features['num_questions'] = text.count('?')

        # Count capital letters
        features['num_capitals'] = sum(1 for c in text if c.isupper())

        # Check for money symbols
        features['has_money_symbol'] = int(any(c in text for c in ['$', '£', '€']))

        # Count digits
        features['num_digits'] = sum(1 for c in text if c.isdigit())

        # Word statistics
        words = text.split()
        features['num_words'] = len(words)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0

        return features

    def fit(self, emails):
        """Build vocabulary from emails."""
        # Count word frequencies
        word_freq = Counter()
        for email in emails:
            # Clean and tokenize
            words = re.findall(r'\b[a-z]+\b', email.lower())
            word_freq.update(words)

        # Select most frequent words
        words = [w for w, _ in word_freq.most_common(self.max_features)]

        # Create vocabulary
        self.vocabulary_ = {word: idx for idx, word in enumerate(words)}
        self.feature_names_ = words + self.additional_features_

        return self

    def transform(self, emails):
        """Transform emails to feature vectors."""
        if self.vocabulary_ is None:
            raise ValueError("Extractor not fitted. Call fit() first.")

        n_emails = len(emails)
        n_text_features = len(self.vocabulary_)
        n_additional = len(self.additional_features_)
        n_features = n_text_features + n_additional

        X = np.zeros((n_emails, n_features), dtype=np.float64)

        for i, email in enumerate(emails):
            # Word counts
            words = re.findall(r'\b[a-z]+\b', email.lower())
            word_counts = Counter(words)

            for word, count in word_counts.items():
                if word in self.vocabulary_:
                    X[i, self.vocabulary_[word]] = count

            # Additional features
            add_features = self._extract_text_features(email)
            for j, feat_name in enumerate(self.additional_features_):
                X[i, n_text_features + j] = add_features[feat_name]

        return X

    def fit_transform(self, emails):
        """Fit and transform in one step."""
        self.fit(emails)
        return self.transform(emails)


def create_spam_dataset():
    """
    Create a simulated spam email dataset.
    Returns emails and labels (0=ham, 1=spam)
    """

    # Ham (legitimate) emails
    ham_emails = [
        "Hi John, can we schedule a meeting for next week to discuss the project progress?",
        "Dear colleague, please find attached the quarterly report for your review.",
        "Thanks for your email. I will get back to you by tomorrow with the details.",
        "Meeting scheduled for 3pm in conference room B. Please confirm your attendance.",
        "Could you please send me the updated presentation slides before Friday?",
        "Looking forward to seeing you at the conference next month.",
        "Please review the attached document and let me know if you have any questions.",
        "Thank you for your contribution to the team project. Great work!",
        "Can you help me with the database query I mentioned yesterday?",
        "The client meeting went well. I will send you the notes shortly.",
        "Please submit your timesheets by end of day Friday.",
        "Would you like to grab lunch tomorrow to discuss the new initiative?",
        "Congratulations on your promotion! Well deserved.",
        "The system maintenance is scheduled for this weekend. Plan accordingly.",
        "Can we reschedule our call to Thursday afternoon instead?",
        "Please find the meeting minutes attached for your reference.",
        "Thank you for attending the workshop. Feedback form is attached.",
        "The project deadline has been extended to next month.",
        "Could you please review my code before I merge the pull request?",
        "Team building event is planned for next Friday. Hope you can join.",
        "Your order has been shipped and will arrive in 3-5 business days.",
        "Reminder: annual performance reviews are due by month end.",
        "Looking forward to our collaboration on the new project.",
        "The training session was very informative. Thank you for organizing.",
        "Please update your contact information in the employee directory.",
        "Can you share the link to the documentation you mentioned?",
        "The conference call link is in the calendar invite.",
        "Great presentation today! The team really appreciated it.",
        "Please confirm receipt of this email at your earliest convenience.",
        "I have reviewed the proposal and have some suggestions to share.",
    ]

    # Spam emails
    spam_emails = [
        "CONGRATULATIONS!!! You have WON $1,000,000! Click here NOW to claim your prize!!!",
        "Make $5000 per week working from home! No experience needed! Act NOW!!!",
        "URGENT: Your account will be suspended! Click here immediately to verify!",
        "Limited time offer! Buy cheap medications online! Save 90% TODAY!!!",
        "You have been selected for a FREE iPhone! Claim it now before it expires!!!",
        "Weight loss miracle! Lose 20 pounds in 2 weeks! ORDER NOW!!!",
        "MAKE MONEY FAST! Work from home! $10,000 per month GUARANTEED!!!",
        "Congratulations! You have won a FREE vacation to Hawaii! Click to claim!!!",
        "URGENT: Your package is waiting! Click here to schedule delivery NOW!!!",
        "Get rich quick! Secret method revealed! Download FREE ebook!!!",
        "AMAZING OFFER!!! Luxury watches at 99% discount! Buy now before sold out!!!",
        "You have inherited $5,000,000 from a distant relative! Contact us NOW!!!",
        "FREE credit report! Check your score NOW! Limited time offer!!!",
        "WORK FROM HOME!!! Earn $500 per day!!! No skills required!!! Start TODAY!!!",
        "URGENT: Tax refund waiting! Claim your $2000 refund NOW!!!",
        "Miracle cure! Eliminate pain in 24 hours! Order TODAY and save 50%!!!",
        "CONGRATULATIONS! You are our 1 millionth visitor! Claim your prize!!!",
        "Make money online! Join thousands earning $1000 daily! Sign up FREE!!!",
        "URGENT RESPONSE NEEDED! Your account has been compromised! Click NOW!!!",
        "Amazing investment opportunity! Double your money in 30 days GUARANTEED!!!",
        "You have been pre-approved for a $50,000 loan! Apply NOW!!!",
        "LOSE WEIGHT FAST!!! Secret formula! Results in 7 days! Buy NOW!!!",
        "FREE samples! Try before you buy! Limited quantities! Order TODAY!!!",
        "Your computer has a virus! Download our software NOW to fix it!!!",
        "CONGRATULATIONS! You have won lottery! Claim $10,000,000 TODAY!!!",
        "Meet singles in your area! FREE registration! Join NOW!!!",
        "URGENT: Your subscription will expire! Renew NOW to avoid interruption!!!",
        "Get college degree online! No classes required! Order NOW!!!",
        "AMAZING DEAL!!! Designer bags 90% off! Limited stock! Buy TODAY!!!",
        "You have been chosen! Exclusive offer just for you! Click NOW!!!",
    ]

    # Combine and shuffle
    emails = ham_emails + spam_emails
    labels = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

    indices = np.random.permutation(len(emails))
    emails = [emails[i] for i in indices]
    labels = labels[indices]

    return emails, labels


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
    """Main function demonstrating spam detection."""
    print("Email Spam Detection using Multinomial Naive Bayes")
    print("=" * 80)

    # Create dataset
    print("\n1. Creating spam email dataset...")
    emails, labels = create_spam_dataset()
    print(f"   Total emails: {len(emails)}")
    print(f"   Ham (legitimate): {np.sum(labels == 0)}")
    print(f"   Spam: {np.sum(labels == 1)}")

    # Show examples
    print("\n2. Sample emails:")
    print("-" * 80)
    ham_indices = np.where(labels == 0)[0]
    spam_indices = np.where(labels == 1)[0]
    print("\n   HAM example:")
    print(f"   \"{emails[ham_indices[0]]}\"")
    print("\n   SPAM example:")
    print(f"   \"{emails[spam_indices[0]]}\"")

    # Split dataset
    print("\n3. Splitting dataset (70% train, 30% test)...")
    emails_train, emails_test, y_train, y_test = train_test_split(
        emails, labels, test_size=0.3, random_state=42
    )
    print(f"   Training emails: {len(emails_train)}")
    print(f"   Test emails: {len(emails_test)}")

    # Extract features
    print("\n4. Extracting features from emails...")
    extractor = EmailFeatureExtractor(max_features=50)
    X_train = extractor.fit_transform(emails_train)
    X_test = extractor.transform(emails_test)

    print(f"   Feature vector size: {X_train.shape[1]}")
    print(f"   Word features: {len(extractor.vocabulary_)}")
    print(f"   Additional features: {len(extractor.additional_features_)}")
    print(f"   Sample word features: {list(extractor.vocabulary_.keys())[:10]}")
    print(f"   Additional features: {extractor.additional_features_}")

    # Analyze feature statistics
    print("\n5. Feature statistics:")
    print("-" * 80)
    print(f"   Training set sparsity: {np.mean(X_train == 0):.2%}")
    print(f"   Average non-zero features per email: {np.mean(np.sum(X_train > 0, axis=1)):.1f}")

    # Train classifier with different alpha values
    print("\n6. Training classifiers with different smoothing parameters...")
    print("-" * 80)

    alphas = [0.01, 0.1, 1.0, 10.0]
    results = []

    for alpha in alphas:
        clf = MultinomialNB(alpha=alpha)
        clf.fit(X_train, y_train)

        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        results.append((alpha, train_acc, test_acc))
        print(f"   Alpha={alpha:6.2f}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

    # Use best alpha
    best_alpha = max(results, key=lambda x: x[2])[0]
    print(f"\n   Best alpha: {best_alpha}")

    # Train final model
    print("\n7. Training final spam detector...")
    spam_detector = MultinomialNB(alpha=best_alpha)
    spam_detector.fit(X_train, y_train)

    # Analyze learned parameters
    print("\n8. Most indicative features for SPAM:")
    print("-" * 80)

    # Get log probabilities for each class
    spam_log_prob = spam_detector.feature_log_prob_[1, :]  # spam class
    ham_log_prob = spam_detector.feature_log_prob_[0, :]   # ham class

    # Calculate log likelihood ratio
    log_ratio = spam_log_prob - ham_log_prob

    # Top spam indicators
    top_spam_idx = np.argsort(log_ratio)[-15:][::-1]
    print("\n   Top 15 SPAM indicators:")
    for idx in top_spam_idx:
        if idx < len(extractor.vocabulary_):
            feature = extractor.feature_names_[idx]
            ratio = np.exp(log_ratio[idx])
            print(f"      {feature:20s}: {ratio:8.2f}x more likely in spam")

    # Top ham indicators
    top_ham_idx = np.argsort(log_ratio)[:15]
    print("\n   Top 15 HAM indicators:")
    for idx in top_ham_idx:
        if idx < len(extractor.vocabulary_):
            feature = extractor.feature_names_[idx]
            ratio = np.exp(-log_ratio[idx])
            print(f"      {feature:20s}: {ratio:8.2f}x more likely in ham")

    # Make predictions on test set
    print("\n9. Test Set Performance:")
    print("-" * 80)
    y_pred = spam_detector.predict(X_test)
    y_proba = spam_detector.predict_proba(X_test)
    test_accuracy = spam_detector.score(X_test, y_test)

    print(f"   Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    # Confusion matrix
    print("\n10. Confusion Matrix:")
    print("-" * 80)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    print("                Predicted")
    print("                Ham    Spam")
    print("   Actual  +-----------------")
    print(f"      Ham  | {tn:4d}   {fp:4d}")
    print(f"      Spam | {fn:4d}   {tp:4d}")

    # Calculate detailed metrics
    print("\n11. Detailed Performance Metrics:")
    print("-" * 80)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f"   Precision (spam detection):  {precision:.4f}")
    print(f"   Recall (spam detection):     {recall:.4f}")
    print(f"   F1-Score:                    {f1_score:.4f}")
    print(f"   Specificity:                 {specificity:.4f}")
    print(f"   False Positive Rate:         {false_positive_rate:.4f}")
    print(f"   False Negative Rate:         {false_negative_rate:.4f}")

    print("\n   Interpretation:")
    print(f"   - {precision*100:.1f}% of emails classified as spam are actually spam")
    print(f"   - {recall*100:.1f}% of actual spam emails are correctly identified")
    print(f"   - {fp} legitimate emails incorrectly marked as spam (False Positives)")
    print(f"   - {fn} spam emails incorrectly marked as legitimate (False Negatives)")

    # Analyze prediction confidence
    print("\n12. Prediction Confidence Analysis:")
    print("-" * 80)

    spam_confidence = y_proba[:, 1]  # Probability of being spam
    high_confidence = spam_confidence > 0.9
    low_confidence = (spam_confidence > 0.4) & (spam_confidence < 0.6)

    print(f"   High confidence predictions (>90%): {np.sum(high_confidence)}")
    print(f"   Low confidence predictions (40-60%): {np.sum(low_confidence)}")
    print(f"   Average confidence: {np.mean(np.max(y_proba, axis=1)):.4f}")

    # Test on new emails
    print("\n13. Testing on New Emails:")
    print("-" * 80)

    new_emails = [
        "Hi Sarah, can we reschedule our meeting to next Tuesday?",
        "CONGRATULATIONS!!! You WON $1000000! Click NOW to claim!!!",
        "Please find the attached quarterly report for review.",
        "URGENT!!! Your account will be closed! Verify immediately!!!",
        "Thanks for the document. I will review it this week.",
        "MAKE MONEY FAST!!! Work from home!!! $5000 per week!!!",
    ]

    X_new = extractor.transform(new_emails)
    predictions = spam_detector.predict(X_new)
    probabilities = spam_detector.predict_proba(X_new)

    for i, (email, pred, probs) in enumerate(zip(new_emails, predictions, probabilities)):
        label = "SPAM" if pred == 1 else "HAM"
        confidence = probs[int(pred)]

        print(f"\n   Email {i+1}:")
        print(f"   Text: \"{email[:60]}{'...' if len(email) > 60 else ''}\"")
        print(f"   Prediction: {label} (confidence: {confidence:.4f})")
        print(f"   Spam probability: {probs[1]:.4f}")

    # Analyze false positives and false negatives
    print("\n14. Error Analysis:")
    print("-" * 80)

    # False positives (ham classified as spam)
    fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
    if len(fp_indices) > 0:
        print(f"\n   False Positives ({len(fp_indices)} emails):")
        for i, idx in enumerate(fp_indices[:3]):
            print(f"\n      Example {i+1}:")
            print(f"      \"{emails_test[idx]}\"")
            print(f"      Spam probability: {y_proba[idx, 1]:.4f}")

    # False negatives (spam classified as ham)
    fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]
    if len(fn_indices) > 0:
        print(f"\n   False Negatives ({len(fn_indices)} emails):")
        for i, idx in enumerate(fn_indices[:3]):
            print(f"\n      Example {i+1}:")
            print(f"      \"{emails_test[idx]}\"")
            print(f"      Spam probability: {y_proba[idx, 1]:.4f}")

    # Recommendations
    print("\n15. System Recommendations:")
    print("-" * 80)
    print("""
   For Production Deployment:

   1. Confidence Thresholds:
      - High confidence (>90%): Automatically filter
      - Medium confidence (60-90%): Mark as suspicious
      - Low confidence (<60%): Require manual review

   2. Feature Engineering Improvements:
      - Add domain reputation features
      - Include sender history
      - Analyze email headers
      - Check for phishing patterns
      - Monitor attachment types

   3. Model Improvements:
      - Regularly retrain with new spam patterns
      - Implement user feedback loop
      - Consider ensemble methods
      - Add real-time feature updates

   4. Performance Optimization:
      - Minimize false positives (user frustration)
      - Balance between precision and recall
      - Monitor concept drift over time
    """)

    print("\n" + "=" * 80)
    print("Spam detection example completed successfully!")


if __name__ == "__main__":
    np.random.seed(42)
    main()
