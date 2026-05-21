# Naive Bayes Quick Usage Guide

## Installation

No installation required! Just ensure you have NumPy:

```bash
pip install numpy
```

## Quick Start

### 1. GaussianNB - For Continuous Features

```python
import numpy as np
from naive_bayes import GaussianNB

# Create sample data
X_train = np.random.randn(100, 4)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

X_test = np.random.randn(20, 4)
y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

# Train
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")

# Get probabilities
proba = clf.predict_proba(X_test)
print(f"Class probabilities: {proba[0]}")
```

### 2. MultinomialNB - For Count Features

```python
import numpy as np
from naive_bayes import MultinomialNB

# Create count data (e.g., word counts)
X_train = np.random.randint(0, 10, size=(100, 50))
y_train = (X_train[:, 0] > 5).astype(int)

X_test = np.random.randint(0, 10, size=(20, 50))
y_test = (X_test[:, 0] > 5).astype(int)

# Train with Laplace smoothing
clf = MultinomialNB(alpha=1.0)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")
```

### 3. BernoulliNB - For Binary Features

```python
import numpy as np
from naive_bayes import BernoulliNB

# Create binary data (0s and 1s)
X_train = np.random.binomial(1, 0.5, size=(100, 20))
y_train = (X_train.sum(axis=1) > 10).astype(int)

X_test = np.random.binomial(1, 0.5, size=(20, 20))
y_test = (X_test.sum(axis=1) > 10).astype(int)

# Train
clf = BernoulliNB(alpha=1.0, binarize=None)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")
```

## Text Classification Example

```python
import numpy as np
from naive_bayes import MultinomialNB

# Simple text to word count conversion
def text_to_counts(texts, vocab):
    """Convert texts to word count vectors."""
    X = np.zeros((len(texts), len(vocab)))
    for i, text in enumerate(texts):
        words = text.lower().split()
        for word in words:
            if word in vocab:
                X[i, vocab[word]] += 1
    return X

# Sample documents
train_texts = [
    "great movie excellent acting",
    "terrible movie bad acting",
    "wonderful film loved it",
    "awful film hated it",
]
train_labels = np.array([1, 0, 1, 0])  # 1=positive, 0=negative

# Build vocabulary
all_words = set()
for text in train_texts:
    all_words.update(text.lower().split())
vocab = {word: idx for idx, word in enumerate(all_words)}

# Convert to counts
X_train = text_to_counts(train_texts, vocab)

# Train classifier
clf = MultinomialNB(alpha=1.0)
clf.fit(X_train, train_labels)

# Test on new documents
test_texts = [
    "great film excellent",
    "terrible awful movie"
]
X_test = text_to_counts(test_texts, vocab)
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

for text, pred, proba in zip(test_texts, predictions, probabilities):
    sentiment = "Positive" if pred == 1 else "Negative"
    print(f"Text: '{text}'")
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {proba[pred]:.3f}\n")
```

## Parameter Tuning

### GaussianNB Parameters

```python
# var_smoothing: prevents division by zero
clf = GaussianNB(var_smoothing=1e-9)  # default
clf = GaussianNB(var_smoothing=1e-6)  # more smoothing
```

### MultinomialNB Parameters

```python
# alpha: Laplace smoothing parameter
clf = MultinomialNB(alpha=0.1)   # less smoothing
clf = MultinomialNB(alpha=1.0)   # default (Laplace smoothing)
clf = MultinomialNB(alpha=10.0)  # more smoothing
```

### BernoulliNB Parameters

```python
# alpha: smoothing parameter
# binarize: threshold for converting to binary
clf = BernoulliNB(alpha=1.0, binarize=0.0)   # threshold at 0
clf = BernoulliNB(alpha=1.0, binarize=None)  # assume already binary
clf = BernoulliNB(alpha=1.0, binarize=0.5)   # threshold at 0.5
```

## Common Methods

All classifiers share these methods:

```python
# Fit the model
clf.fit(X_train, y_train)

# Predict classes
y_pred = clf.predict(X_test)

# Predict class probabilities
proba = clf.predict_proba(X_test)

# Predict log probabilities (more stable)
log_proba = clf.predict_log_proba(X_test)

# Calculate accuracy
accuracy = clf.score(X_test, y_test)

# Access learned parameters
print(clf.classes_)              # class labels
print(clf.class_prior_)          # class prior probabilities
```

## Running Examples

### Test Individual Examples

```bash
# Gaussian example (Iris classification)
python examples/example_gaussian.py

# Multinomial example (text classification)
python examples/example_multinomial.py

# Bernoulli example (document classification)
python examples/example_bernoulli.py

# Compare all variants
python examples/example_comparison.py

# Real-world spam detection
python examples/example_spam_detection.py
```

### Test the Module

```bash
# Run built-in tests
python naive_bayes.py
```

## Troubleshooting

### 1. ValueError: X must be 2D array

**Problem:** Input must be a 2D array, not 1D.

**Solution:**
```python
# Wrong
X = np.array([1, 2, 3, 4])

# Correct
X = np.array([[1, 2, 3, 4]])  # Add outer brackets
# or
X = X.reshape(1, -1)
```

### 2. ValueError: Input X must contain non-negative values (MultinomialNB)

**Problem:** MultinomialNB requires non-negative features.

**Solution:**
```python
# Shift data to be non-negative
X = X - X.min() + 1e-10
```

### 3. Warnings about zero variance (GaussianNB)

**Problem:** A feature has zero variance in a class.

**Solution:**
```python
# Increase smoothing
clf = GaussianNB(var_smoothing=1e-6)  # default is 1e-9
```

### 4. Poor probability estimates

**Problem:** Probabilities are too extreme (near 0 or 1).

**Solution:**
```python
# Use predict instead of predict_proba for classification
# Or increase smoothing parameter (alpha)
clf = MultinomialNB(alpha=10.0)  # more smoothing
```

## Performance Tips

### 1. Feature Preprocessing

```python
# For GaussianNB: normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# For MultinomialNB: ensure non-negative
X_train = np.abs(X_train)

# For BernoulliNB: binarize features
# Option 1: Use binarize parameter
clf = BernoulliNB(binarize=0.5)

# Option 2: Manual binarization
X_train = (X_train > 0.5).astype(float)
```

### 2. Handling Imbalanced Classes

```python
# Check class distribution
unique, counts = np.unique(y_train, return_counts=True)
print(f"Class distribution: {dict(zip(unique, counts))}")

# Consider resampling or adjusting class priors
# Naive Bayes naturally handles imbalance through class priors
```

### 3. Feature Selection

```python
# For text: limit vocabulary size
max_features = 1000  # keep top 1000 words

# For all: remove features with low variance
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_train = selector.fit_transform(X_train)
X_test = selector.transform(X_test)
```

## Common Use Cases

### 1. Text Classification
- **Best choice:** MultinomialNB or BernoulliNB
- **Reason:** Text features are typically word counts or binary presence

### 2. Spam Detection
- **Best choice:** MultinomialNB
- **Reason:** Word frequency matters, count-based features

### 3. Sentiment Analysis
- **Best choice:** MultinomialNB or BernoulliNB
- **Reason:** Both work well; choose based on whether counts matter

### 4. Medical Diagnosis (Continuous Features)
- **Best choice:** GaussianNB
- **Reason:** Medical measurements are typically continuous

### 5. Image Classification (Pixel Values)
- **Best choice:** GaussianNB with preprocessing
- **Reason:** Pixel values are continuous, normalize first

### 6. Categorical Data
- **Best choice:** MultinomialNB with one-hot encoding
- **Reason:** One-hot encoded categories are count-like

## Comparison with Sklearn

Our implementation is compatible with sklearn's API:

```python
# Our implementation
from naive_bayes import GaussianNB as MyGaussianNB
my_clf = MyGaussianNB()

# Sklearn
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB
sklearn_clf = SklearnGaussianNB()

# Both work the same way
my_clf.fit(X_train, y_train)
sklearn_clf.fit(X_train, y_train)

# Results should be nearly identical
print(f"Our accuracy: {my_clf.score(X_test, y_test):.4f}")
print(f"Sklearn accuracy: {sklearn_clf.score(X_test, y_test):.4f}")
```

## Further Reading

- See `README.md` for mathematical details
- Check `examples/` directory for complete examples
- Read inline code documentation for implementation details
